"""Conservative request reservations against an operator-audited tariff.

The policy is trusted configuration, not proof of a provider's billing behavior.
Input limits include provider framing, cached tokens and all billable input;
output limits must include reasoning. No reservation is refunded, even on error.
"""
import json
import math
from decimal import Decimal, InvalidOperation, localcontext, ROUND_CEILING
from pathlib import Path
from uuid import uuid4

from onecli.budget import RequestBudget, BudgetError
from onecli.providers.base import ProviderError
from onecli.providers.openai_wire import OpenAIWire


class RequestAdmission:
    def __init__(self, policy, ledger, limit_usd, *, create=False):
        required = {'version', 'base_url', 'model', 'input_token_bound',
                    'output_token_bound', 'input_usd_per_million',
                    'output_usd_per_million', 'audit_evidence'}
        if not isinstance(policy, dict) or set(policy) != required:
            raise ValueError('exact audited request budget policy required')
        if type(policy['version']) is not int or policy['version'] != 1:
            raise ValueError('unsupported request budget policy')
        for key in ('base_url', 'model', 'audit_evidence'):
            if not isinstance(policy[key], str) or not policy[key].strip():
                raise ValueError('request budget policy identity/evidence required')
        for key in ('input_token_bound', 'output_token_bound'):
            if type(policy[key]) is not int or not 0 < policy[key] <= 10**9:
                raise ValueError('positive finite token bounds required')
        rates = []
        for key in ('input_usd_per_million', 'output_usd_per_million'):
            if type(policy[key]) not in (str, int, float):
                raise ValueError('invalid tariff')
            try:
                rate = Decimal(str(policy[key]))
            except InvalidOperation as exc:
                raise ValueError('invalid tariff') from exc
            if not rate.is_finite() or rate < 0 or rate > 10**9:
                raise ValueError('invalid tariff')
            rates.append(rate)
        self.policy = dict(policy)
        self.rates = rates
        self.ledger = RequestBudget(ledger, limit_usd, create=create)

    @classmethod
    def load(cls, policy_path, ledger_path, limit_usd, *, create=False):
        return cls(json.loads(Path(policy_path).read_text()), ledger_path, limit_usd, create=create)

    def bind(self, provider):
        # Subclasses could bypass the audited wire hooks.
        if (type(provider) is not OpenAIWire
                or provider.base_url != self.policy['base_url'].rstrip('/')
                or provider.model != self.policy['model'] or provider.extra_headers):
            raise ValueError('request budget forbids unaudited provider route')
        existing = provider.request_gate
        if getattr(existing, '_request_admission', None) is self:
            return
        if getattr(existing, '_request_admission', None) is not None:
            raise ValueError('provider already bound to another request budget')
        def gate(body):
            if (provider.base_url != self.policy['base_url'].rstrip('/')
                    or provider.model != self.policy['model'] or provider.extra_headers):
                raise ProviderError('request budget: provider route changed')
            if existing is not None:
                existing(body)
            self.admit(body)
        gate._request_admission = self
        provider.request_gate = gate
        provider.reject_redirects = True

    def admit(self, body):
        allowed = {'model', 'messages', 'temperature', 'max_tokens',
                   'max_completion_tokens', 'stream', 'stream_options',
                   'tools', 'tool_choice', 'response_format'}
        if not isinstance(body, dict) or set(body) - allowed:
            raise ProviderError('request budget: unsupported request parameters')
        if body.get('model') != self.policy['model']:
            raise ProviderError('request budget: model changed')
        caps = [body[k] for k in ('max_tokens', 'max_completion_tokens') if k in body]
        if (len(caps) != 1 or type(caps[0]) is not int
                or not 0 < caps[0] <= self.policy['output_token_bound']):
            raise ProviderError('request budget: audited output cap required')
        messages = body.get('messages')
        if (not isinstance(messages, list) or not messages
                or any(not isinstance(m, dict)
                       or not isinstance(m.get('content', ''), (str, type(None)))
                       or set(m) - {'role', 'content', 'tool_calls', 'tool_call_id', 'name'}
                       for m in messages)):
            raise ProviderError('request budget: only text requests are audited')
        for message in messages:
            calls = message.get('tool_calls', [])
            if not isinstance(calls, list):
                raise ProviderError('request budget: invalid tool calls')
            for call in calls:
                if (not isinstance(call, dict) or set(call) != {'id', 'type', 'function'}
                        or call['type'] != 'function'
                        or not isinstance(call['function'], dict)
                        or set(call['function']) != {'name', 'arguments'}
                        or any(not isinstance(v, str) for v in call['function'].values())):
                    raise ProviderError('request budget: unsupported tool call')
        tools = body.get('tools', [])
        if not isinstance(tools, list) or any(
                not isinstance(t, dict) or set(t) != {'type', 'function'}
                or t['type'] != 'function' or not isinstance(t['function'], dict)
                or set(t['function']) - {'name', 'description', 'parameters', 'strict'}
                for t in tools):
            raise ProviderError('request budget: only client function tools are audited')
        if body.get('stream_options', {}) not in ({}, {'include_usage': True}):
            raise ProviderError('request budget: unsupported stream options')
        if body.get('tool_choice', 'auto') not in ('auto', 'none', 'required'):
            raise ProviderError('request budget: unsupported tool selection')
        if body.get('response_format', {}) not in ({}, {'type': 'json_object'}):
            raise ProviderError('request budget: unsupported response format')
        if type(body.get('stream', False)) is not bool:
            raise ProviderError('request budget: invalid stream flag')
        temperature = body.get('temperature', 0)
        if type(temperature) not in (int, float) or not math.isfinite(temperature):
            raise ProviderError('request budget: invalid temperature')
        # Reserve the FULL input bound, never a characters-to-tokens estimate.
        with localcontext() as ctx:
            ctx.prec = max(80, *(len(r.as_tuple().digits) + 30 for r in self.rates))
            ctx.rounding = ROUND_CEILING
            cost = (self.rates[0] * self.policy['input_token_bound']
                    + self.rates[1] * caps[0]) / Decimal(1_000_000)
        try:
            # New ID per actual HTTP attempt: retries spend another reservation.
            self.ledger.reserve(uuid4().hex, max(cost, Decimal('0.000000001')))
        except BudgetError as exc:
            raise ProviderError(f'request budget: {exc}') from exc

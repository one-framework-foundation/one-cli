"""Named task masks shared by one and one-chat. No provider selection or code."""
import json
import os
from onecli.evals.corpus import MaskInfo, load_masks


def catalog():
    masks = {
        'general': MaskInfo('general', 'Answer directly. For substantial work, act, verify and give one concise outcome.',
            source='builtin', loop_policy={'version': 1, 'progress_policy': {'enabled': True, 'seconds': 30, 'tool_calls': 5}}),
        'build': MaskInfo('build', 'Inspect the relevant code, implement the requested change, and run proportionate verification. Keep working through recoverable failures. Report the change, evidence and remaining limits without repeating the tool trace.',
            source='builtin', loop_policy={'version': 1, 'max_steps': 30, 'repeat_limit': 3,
                'empty_retries': 1, 'progress_policy': {'enabled': True, 'seconds': 30, 'tool_calls': 5}}),
        'review': MaskInfo('review', 'Review local files. Lead with concrete findings, severity and file locations. Distinguish a clean review from missing evidence. Do not change files.',
            source='builtin', execution_policy={'kind': 'review', 'max_steps': 12},
            loop_policy={'version': 1, 'max_steps': 12, 'repeat_limit': 2,
                'empty_retries': 1, 'progress_policy': {'enabled': True, 'seconds': 45, 'tool_calls': 8}}),
    }
    for name, required in {
        'general': ['User request and relevant verified evidence'],
        'build': ['Acceptance criteria', 'Relevant implementation', 'Relevant tests and verification results'],
        'review': ['Diff or explicit review scope', 'Surrounding implementation', 'Review criteria'],
    }.items():
        masks[name].context_contract = {'version': 1, 'required_knowledge': required,
            'retrieval': 'workspace_files', 'refresh': 'each_request', 'token_budget': 8192}
    if os.environ.get('ONE_MASKS_JSON'):
        configured, error = load_masks()
        if error:
            raise ValueError(error)
        masks.update(configured)
    return masks


def select(name):
    if not name or name == 'off':
        return None
    masks = catalog()
    if name not in masks:
        raise ValueError(f'unknown mask {name!r}; available: ' + ', '.join(sorted(masks)))
    mask = masks[name]
    if mask.execution_policy and mask.execution_policy.get('kind') == 'debug':
        raise ValueError('debug execution policy requires a trusted host verifier; use the embedding API')
    return mask


def options(name):
    mask = select(name)
    return mask.agent_options(enforce=True) if mask else {}


def listing(active=None):
    rows = [f'Active mask: {active or "off"}', 'Masks (select with /mask NAME; /mask off disables):']
    for name, mask in sorted(catalog().items()):
        policy = mask.execution_policy or {}
        mode = policy.get('kind', 'host permissions')
        rows.append(f'  {name}: {mode}; {mask.source}')
    if active and active != 'off':
        mask = select(active)
        rows.append('Loop policy: ' + json.dumps(mask.loop_policy or {}, sort_keys=True))
        rows.append('Context contract: ' + json.dumps(mask.context_contract or {}, sort_keys=True))
        rows.append('Execution policy: ' + json.dumps(mask.execution_policy or {}, sort_keys=True))
    return '\n'.join(rows)

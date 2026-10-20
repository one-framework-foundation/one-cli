import json
from onecli.agent import Agent
from onecli.providers.base import ToolCall, Turn


def test_disallowed_url_is_restriction_not_step_budget(tmp_path):
    (tmp_path / '.one-run-limits.json').write_text(json.dumps({
        'max_steps': 5, 'max_output_tokens': 512, 'max_request_bytes': 100000,
        'max_total_request_bytes': 500000, 'max_input_tokens': 50000,
        'urls': ['https://allowed.example/']}))
    class Provider:
        def chat(self, messages, tools=None, **kwargs):
            return Turn(tool_calls=[ToolCall(id='denied', name='web_fetch',
                arguments={'url': 'https://example.com'})])
    result = Agent(Provider(), str(tmp_path)).run('Read the allowed source.')
    assert result.stopped == 'restricted', result.error
    assert 'without execution' in result.error
    assert result.messages[-1]['tool_call_id'] == 'denied'
    assert 'Not executed' in result.messages[-1]['content']

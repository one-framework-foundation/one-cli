import io
import pytest
from onecli import chat as chatlib, config, session, tools
from onecli.providers.base import Turn, ToolCall


@pytest.mark.parametrize('mode', ['off', 'auto', 'all'])
def test_interrupted_chat_request_is_persisted_and_resumed(tmp_path, monkeypatch, mode):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(session, 'SESSION_DIR', tmp_path/'sessions')
    c = chatlib.Chat(config.Backend(name='fixture',model='fixture',base_url='http://127.0.0.1:1',wire='openai'),out=io.StringIO(),err=io.StringIO(),stream=False)
    c.tools = mode
    class Interrupted:
        def chat(self, *args, **kwargs):
            raise KeyboardInterrupt
    c._wire = Interrupted()
    with pytest.raises(KeyboardInterrupt):
        chatlib.turn(c, 'original unfinished request', echo=False)
    assert any(m.get('content') == 'original unfinished request' for m in c.messages)
    assert session.load_messages(c.sid) == c.messages
    c.messages = []
    chatlib.dispatch(c, '/resume '+c.sid)
    class Resumed:
        def chat(self, messages, *args, **kwargs):
            assert any(m.get('content') == 'original unfinished request' for m in messages)
            assert messages[-1]['content'] == 'new correction'
            return Turn(text='continued')
    c._wire = Resumed()
    chatlib.turn(c, 'new correction', echo=False)
    assert c.last_answer == 'continued'


def test_interrupted_chat_tool_retains_unknown_outcome(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(session,'SESSION_DIR',tmp_path/'sessions')
    c=chatlib.Chat(config.Backend(name='fixture',model='fixture',base_url='http://127.0.0.1:1',wire='openai'),out=io.StringIO(),err=io.StringIO(),stream=False)
    c.tools='auto'
    class Provider:
        def chat(self,*args,**kwargs):
            return Turn(tool_calls=[ToolCall(id='read1',name='read',arguments={'path':'x'})])
    c._wire=Provider()
    def interrupt(*args,**kwargs):raise KeyboardInterrupt
    monkeypatch.setattr(tools,'run',interrupt)
    with pytest.raises(KeyboardInterrupt):
        chatlib.turn(c, "read x", echo=False)
    result=[m for m in c.messages if m.get('role')=='tool']
    assert len(result)==1 and result[0]['tool_call_id']=='read1'
    assert 'Outcome unknown' in result[0]['content']
    assert session.load_messages(c.sid)==c.messages

import pathlib
import os
import signal
import pytest
from onecli import terminal, permissions
from onecli.agent import Agent
from onecli.providers.base import Turn, ToolCall


def test_double_enter_preserves_order_and_draft(monkeypatch):
    sent=[]
    monkeypatch.setattr(terminal.os, 'kill', lambda pid,sig: sent.append(sig))
    ui = terminal.Surface()
    ui.busy = True
    ui.allow_steer = True
    ui.draft = "original addition"
    ui.submit()
    ui.draft = "newest correction"
    ui.submit()
    ui.submit()
    assert sent == [signal.SIGINT]
    assert ui.pending_messages() == ['original addition','newest correction']
    ui.draft = "unsent"
    ui.cursor = 6
    assert ui.read() == 'original addition\n\nnewest correction'
    assert ui.draft == 'unsent'
    assert ui.pending_messages() == []


def test_queue_command_is_local_while_busy():
    ui = terminal.Surface()
    ui.busy = True
    ui.draft = "first"
    ui.submit()
    ui.draft = "/queue"
    ui.submit()
    assert ui.turns.qsize() == 1
    assert 'first' in '\n'.join(ui.lines)


def test_double_enter_does_not_bypass_question_or_execute_command(monkeypatch):
    sent = []
    monkeypatch.setattr(terminal.os, "kill", lambda *args: sent.append(args))
    ui = terminal.Surface()
    ui.busy = True
    ui.allow_steer = True
    ui.draft = "/exit"
    ui.submit()
    ui.submit()
    assert not sent
    ui.answering = True
    ui.draft = "yes"
    ui.submit()
    ui.submit()
    assert not sent
    assert ui.answers.get_nowait() == 'yes'


class InterruptedProvider:
    def chat(self, *args, **kwargs):
        raise KeyboardInterrupt()


def test_interrupted_provider_retains_original_request(tmp_path):
    agent=Agent(InterruptedProvider(),str(tmp_path))
    with pytest.raises(KeyboardInterrupt):
        agent.run("finish original task")
    assert any(m.get('content') == 'finish original task' for m in agent.interrupted_messages)


def test_interrupted_tool_closes_wire_with_unknown_outcome(tmp_path,monkeypatch):
    from onecli import tools
    class Provider:
        def chat(self,*args,**kwargs):
            return Turn(tool_calls=[ToolCall(id='call1',name='read',arguments={'path':'x'})])
    def interrupt(*args,**kwargs): raise KeyboardInterrupt()
    monkeypatch.setattr(tools,'run',interrupt)
    agent=Agent(Provider(),str(tmp_path))
    with pytest.raises(KeyboardInterrupt):
        agent.run("read x")
    results=[m for m in agent.interrupted_messages if m.get('role')=='tool']
    assert len(results)==1 and results[0]['tool_call_id']=='call1'
    assert 'Outcome unknown' in results[0]['content']


def test_identity_and_workspace_instructions_are_loaded(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace/'AGENTS.md').write_text('workspace instruction')
    context=permissions.project_context(str(workspace),home=str(tmp_path/'home'))
    assert 'workspace instruction' in context


def test_real_signal_retains_request_and_steers_in_subprocess(tmp_path):
    import subprocess
    import sys
    script=r'''
import threading,time
from onecli import terminal
from onecli.agent import Agent
from onecli.providers.base import Turn
ui=terminal.Surface();ui.busy=True;ui.allow_steer=True
started=threading.Event()
class Slow:
 def chat(self,*a,**kw):
  started.set();time.sleep(120);return Turn(text='too late')
a=Agent(Slow(),'.')
def steer():
 assert started.wait(30)
 ui.draft='new correction';ui.submit();ui.submit()
t=threading.Thread(target=steer);t.start()
try:
 a.run('original task')
 raise AssertionError('not interrupted')
except KeyboardInterrupt: pass
t.join(30)
class Next:
 def chat(self,messages,*args,**kw):
  assert any(m.get('content')=='original task' for m in messages)
  assert messages[-1]['content']=='new correction'
  return Turn(text='continued')
result=Agent(Next(),'.').run(ui.read(),a.interrupted_messages)
assert result.text=='continued'
print('real SIGINT resumed with original and newest requests')
'''
    # cwd is a tmp dir, so the repo is not on the child's sys.path by
    # accident of the working directory. Put it there explicitly.
    env=dict(os.environ,PYTHONPATH=str(pathlib.Path(__file__).resolve().parents[1]))
    result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,timeout=180,cwd=tmp_path,env=env)
    assert result.returncode == 0,result.stdout+result.stderr


def test_queue_is_discoverable():
    from onecli import commands
    assert '/queue' in commands.names_for()


def test_prior_tool_id_does_not_mask_interrupted_result(tmp_path,monkeypatch):
    from onecli import tools
    prior=[{'role':'system','content':'test'},
           {'role':'assistant','content':'','tool_calls':[{'id':'same','type':'function','function':{'name':'read','arguments':'{}'}}]},
           {'role':'tool','tool_call_id':'same','content':'old result'}]
    class Provider:
        def chat(self,*args,**kwargs):
            return Turn(tool_calls=[ToolCall(id='same',name='read',arguments={'path':'x'})])
    def interrupt(*args,**kwargs): raise KeyboardInterrupt()
    monkeypatch.setattr(tools,'run',interrupt)
    a=Agent(Provider(),str(tmp_path))
    with pytest.raises(KeyboardInterrupt):
        a.run("new request", prior)
    results=[m for m in a.interrupted_messages if m.get('role')=='tool']
    assert len(results)==2 and 'Outcome unknown' in results[-1]['content']


def test_enter_again_steers_after_any_delay(monkeypatch):
    """The banner says 'Enter again to steer' with no deadline; slow SSH must still steer."""
    sent=[]
    monkeypatch.setattr(terminal.os, 'kill', lambda pid,sig: sent.append(sig))
    clock=[1000.0]
    monkeypatch.setattr(terminal.time, 'monotonic', lambda: clock[0])
    ui = terminal.Surface()
    ui.busy = True
    ui.allow_steer = True
    ui.draft = "correction"
    ui.submit()
    clock[0]+=5.0
    ui.submit()
    assert sent == [signal.SIGINT]


def test_esc_stop_does_not_start_the_queued_message(monkeypatch):
    monkeypatch.setattr(terminal.os, 'kill', lambda *a: None)
    ui = terminal.Surface()
    ui.busy = ui.allow_steer = True
    ui.draft = 'queued behind the run'
    ui.submit()
    ui.key('\x1b')
    ui.turns.put(EOFError())
    with pytest.raises(EOFError):
        ui.read()
    assert ui.draft == 'queued behind the run'
    assert ui.turns.empty()
    assert any('queued message not sent' in line for line in ui.lines)


def test_forced_steer_still_runs_the_queue(monkeypatch):
    monkeypatch.setattr(terminal.os, 'kill', lambda *a: None)
    ui = terminal.Surface()
    ui.busy = ui.allow_steer = True
    ui.draft = 'steer me'
    ui.submit()
    ui.submit()  # empty Enter while busy: deliberate steer
    assert ui.read() == 'steer me'

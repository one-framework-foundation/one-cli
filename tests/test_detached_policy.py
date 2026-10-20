from unittest.mock import patch
from onecli import tasks, loopguard

class Running:
    def __init__(self): self.returncode = None
    def poll(self): return self.returncode

def test_phase_does_not_complete_process(tmp_path):
    proc = Running()
    sig = tmp_path / "signals"
    sig.write_text("PHASE: Execute\nPHASE: Verify\nPHASE: Done\n")
    task = tasks.DetachedTask('1','test',proc=proc,signals_path=sig)
    task.poll()
    assert task.status == 'running'
    proc.returncode = 7
    assert task.poll()
    assert task.status == 'failed' and task.exit_code == 7

def test_exit_zero_empty_is_not_success(tmp_path):
    proc = Running()
    proc.returncode = 0
    task = tasks.DetachedTask('1','test',proc=proc)
    task.poll()
    assert task.status == 'failed'

def test_detached_prompt_has_policy(tmp_path, monkeypatch):
    monkeypatch.delenv('ONE_DETACHED_OWNER',raising=False)
    class Proc:
        pid=1234567
    with patch('subprocess.Popen',return_value=Proc()):
        task=tasks.TaskManager(root=tmp_path).spawn('test',prompt='Inspect the assigned file',cmd=['fixture'])
    body=task.prompt_path.read_text()
    assert 'CODE SIMPLIFICATION' in body and 'PHASES:' in body
    assert 'spending' in body and 'Inspect the assigned file' in body

def test_detached_policy_allows_bounded_nested_workers():
    from onecli import offload_policy
    assert 'Delegate bounded subtasks' in offload_policy.INSTRUCTIONS
    assert 'do not add further worker' in offload_policy.INSTRUCTIONS
    assert 'Do not silently switch providers' in offload_policy.INSTRUCTIONS
    assert 'Do not silently switch providers or launch nested workers' not in offload_policy.INSTRUCTIONS

def test_cli_child_disables_implicit_fallback():
    cmd,_=tasks.build_runner_cmd('one','test')
    assert '--no-fallback' in cmd

def test_default_tool_deadline_is_bounded():
    assert 0 < loopguard.DEADLINE <= 600

def test_zero_timeout_does_not_bypass_deadline():
    from onecli import tools
    with patch.dict(tools.REGISTRY,{'probe':type('Tool',(),{'parameters':{'properties':{'timeout':{'type':'number'}}}})()}):
        assert loopguard.LoopGuard(deadline=30).clamp('probe',{'timeout':0})['timeout']==30



def test_fractional_short_timeout_is_preserved():
    from onecli import tools
    with patch.dict(tools.REGISTRY,{'probe':type('Tool',(),{'parameters':{'properties':{'timeout':{'type':'number'}}}})()}):
        assert loopguard.LoopGuard(deadline=30).clamp('probe',{'timeout':.5})['timeout']==.5

def test_detached_policy_targets_checks():
    from onecli import offload_policy
    assert 'TARGETED VERIFICATION' in offload_policy.INSTRUCTIONS
    assert 'STEERING' not in offload_policy.INSTRUCTIONS

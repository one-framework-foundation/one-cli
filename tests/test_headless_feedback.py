import pytest
import json
import socket

from onecli.agent import Agent
from onecli.providers.base import Turn
from onecli.turn_feedback import TurnFeedback


class Provider:
    def __init__(self, parent):
        self.parent = parent
        self.seen = []

    def chat(self, messages, tools=None, **kwargs):
        self.seen.append([dict(m) for m in messages])
        if len(self.seen) == 1:
            self.parent.sendall(json.dumps({'id': 'change-1', 'text': 'Use purple'}).encode() + b'\n')
            return Turn(text='blue')
        assert any(m.get('content') == 'Use purple' for m in messages)
        return Turn(text='purple')


def test_feedback_during_final_generation_continues_same_task(tmp_path):
    parent, child = socket.socketpair()
    parent.settimeout(1)
    feedback = TurnFeedback(child.detach())
    provider = Provider(parent)
    result = Agent(provider, str(tmp_path), turn_feedback=feedback).run('Build page')
    assert result.text == 'purple'
    assert len(result.steps) == 2
    assert any(m.get('content') == 'Build page' for m in provider.seen[1])
    assert json.loads(parent.recv(1024)) == {'id': 'change-1', 'accepted': True}
    assert feedback.closed
    parent.close()


def test_budget_does_not_acknowledge_input_without_another_model_step(tmp_path):
    parent, child = socket.socketpair()
    parent.settimeout(1)
    feedback = TurnFeedback(child.detach())
    result = Agent(Provider(parent), str(tmp_path), max_steps=1, turn_feedback=feedback).run('Build page')
    assert result.text == 'blue'
    try:
        received = parent.recv(1024)
    except ConnectionResetError:
        received = b''
    assert received == b''
    assert feedback.closed
    parent.close()


def test_input_budget_rejection_does_not_acknowledge_feedback(tmp_path):
    parent, child = socket.socketpair()
    parent.settimeout(1)
    feedback = TurnFeedback(child.detach())
    parent.sendall(json.dumps({'id': 'budget-blocked', 'text': 'Use purple'}).encode()+b'\n')
    provider = Provider(parent)
    result = Agent(provider, str(tmp_path), max_input_tokens=1,
                   turn_feedback=feedback).run('Build page')
    assert result.stopped == 'budget_input'
    assert provider.seen == []
    try:
        received = parent.recv(1024)
    except ConnectionResetError:
        received = b''
    assert received == b''
    assert feedback.closed
    parent.close()


def test_auth_failure_keeps_mid_request_feedback_unacknowledged(tmp_path):
    from onecli.providers.base import ProviderError

    parent, child = socket.socketpair()
    parent.settimeout(1)
    feedback=TurnFeedback(child.detach())
    class FailedAuth:
        def chat(self,messages,tools=None,**kwargs):
            parent.sendall(json.dumps({'id':'pending','text':'Use purple'}).encode()+b'\n')
            raise ProviderError('HTTP 401: authentication_failed')
    result=Agent(FailedAuth(),str(tmp_path),turn_feedback=feedback).run('Build page')
    assert result.stopped=='error'
    assert '401' in result.error
    assert any(m.get('content')=='Build page' for m in result.messages)
    assert feedback.closed
    try:
        received=parent.recv(1024)
    except ConnectionResetError:
        received=b''
    assert received==b''
    parent.close()


def test_feedback_fd_rejected_with_request_budget(tmp_path, capsys, monkeypatch):
    from onecli import cli, config
    monkeypatch.setattr(config, 'resolve', lambda **kw: pytest.fail('backend resolution'))
    parent, child = socket.socketpair()
    try:
        argv = ['--feedback-fd', str(child.fileno()), '--request-budget-policy', 'p.json',
                '--workspace', str(tmp_path), '-p', 'hi']
        assert cli.main(argv) == 2
        assert 'budget' in capsys.readouterr().err
    finally:
        parent.close()
        child.close()


def test_feedback_fd_must_be_a_socket(tmp_path, capsys, monkeypatch):
    from onecli import cli, config
    monkeypatch.setattr(config, 'resolve', lambda **kw: pytest.fail('backend resolution'))
    for fd in (987654, 0):
        assert cli.main(['--feedback-fd', str(fd), '--workspace', str(tmp_path), '-p', 'hi']) == 2
        assert 'not a usable socket' in capsys.readouterr().err

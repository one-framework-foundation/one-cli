from types import SimpleNamespace
from onecli.config import Backend
from onecli.model_picker import open_picker, handle_picker


def test_switch_retains_history_and_credentials():
    a = Backend('small', 'm1', 'https://user:secret@host/v1?key=hidden', api_key='private')
    b = Backend('large', 'm2', a.base_url, api_key='private')
    history = [{'role': 'user', 'content': 'keep'}]
    owner = SimpleNamespace(backend=a, messages=history, rebuild=False)
    text = open_picker(owner, [a, b])
    assert all(secret not in text for secret in ('secret', 'hidden', 'private'))
    assert len(owner.model_picker.keys) == 1
    assert 'Models' in handle_picker(owner, '1')
    assert owner.backend is a
    assert 'retained' in handle_picker(owner, '2')
    assert owner.backend is b and owner.backend.api_key == 'private'
    assert owner.messages is history and owner.rebuild


def test_invalid_back_cancel_command_escape():
    a = Backend('a', 'm', 'http://host/v1')
    owner = SimpleNamespace(backend=a, rebuild=False)
    open_picker(owner, [a])
    assert 'Choose' in handle_picker(owner, '999')
    handle_picker(owner, '1')
    assert 'Providers' in handle_picker(owner, 'back')
    assert 'cancelled' in handle_picker(owner, 'cancel')
    assert owner.backend is a and not owner.rebuild
    open_picker(owner, [a])
    assert handle_picker(owner, '/exit') is None
    assert owner.model_picker is None


def test_empty_catalog():
    assert 'No configured' in open_picker(SimpleNamespace(backend=None), [])


def test_chooser_is_visible_in_the_status_line_and_clears(monkeypatch):
    from onecli import terminal
    ui = terminal.Surface()
    monkeypatch.setattr(terminal, 'ACTIVE', ui)
    a = Backend('a', 'm', 'http://host/v1')
    owner = SimpleNamespace(backend=a, rebuild=False)
    assert ui.status_text() == 'ready'
    open_picker(owner, [a])
    assert 'choosing a model' in ui.status_text()
    assert 'Choose a listed number' in handle_picker(owner, 'hello')
    assert 'choosing a model' in ui.status_text(), 'a rejected line keeps the chooser open'
    handle_picker(owner, 'cancel')
    assert ui.status_text() == 'ready'
    open_picker(owner, [a])
    handle_picker(owner, '/help')
    assert ui.status_text() == 'ready'

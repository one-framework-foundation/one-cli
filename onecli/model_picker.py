"""A local, two-stage menu over configured backends. No model requests.

Each model shows runner's live status (onecli.presence); an offline one is
listed but cannot be chosen, and the check is repeated at the moment of
choosing so a lane that came back is selectable without a restart.
"""
from urllib.parse import urlsplit

from onecli import presence


class ModelPicker:
    def __init__(self, backends, current):
        self.current = current
        self.groups = {}
        for backend in backends:
            url = urlsplit(backend.base_url)
            # Never render credentials or query parameters from endpoint URLs.
            key = (url.scheme, url.hostname, url.port, url.path.rstrip('/'), backend.wire)
            self.groups.setdefault(key, []).append(backend)
        self.keys = list(self.groups)
        self.selected = None

    def render(self):
        lines = [f"Current: {self.current.name} ({self.current.model})"]
        if self.selected is None:
            lines.append("Providers")
            for i, key in enumerate(self.keys, 1):
                _, host, port, path, wire = key
                endpoint = f"{host or 'local'}{':' + str(port) if port else ''}{path}"
                names = ', '.join(b.name for b in self.groups[key])
                lines.append(f"  {i}. {endpoint} [{wire}] - {names}")
        else:
            lines.append("Models")
            for i, b in enumerate(self.groups[self.selected], 1):
                mark = ' *' if (b.name, b.model) == (self.current.name, self.current.model) else ''
                state = presence.status(b)
                tag = '' if state in (presence.ONLINE, presence.UNCHECKED) else f' [{state}]'
                lines.append(f"  {i}. {b.model} ({b.name}){mark}{tag}")
        lines.append("Choose a number; back returns to providers; cancel closes.")
        return '\n'.join(lines)

    def choose(self, value):
        value = value.strip().lower()
        if value in ('cancel', 'q', 'quit', '0'):
            return None, True, 'Model selection cancelled.'
        if value == 'back':
            self.selected = None
            return None, False, self.render()
        options = self.keys if self.selected is None else self.groups[self.selected]
        if not value.isdecimal() or not 1 <= int(value) <= len(options):
            return None, False, 'Choose a listed number, back, or cancel.'
        choice = options[int(value) - 1]
        if self.selected is None:
            self.selected = choice
            return None, False, self.render()
        refused = presence.refusal(choice)
        if refused:
            return None, False, refused
        return choice, True, f'Switched to {choice.name} ({choice.model}); conversation retained.'


CUE = 'choosing a model: type a number, back or cancel'


def _cue(label):
    """Tell the full-screen surface (if any) that the prompt is in chooser mode."""
    from onecli import terminal
    if terminal.ACTIVE is not None:
        terminal.ACTIVE.mode = label


def open_picker(owner, backends):
    if not backends:
        return 'No configured backends found.'
    presence.prime(backends)
    owner.model_picker = ModelPicker(backends, owner.backend)
    _cue(CUE)
    return owner.model_picker.render()


def handle_picker(owner, line):
    """Return menu text, or None to let a slash command take over."""
    picker = getattr(owner, 'model_picker', None)
    if picker is None:
        return None
    if line.startswith('/'):
        owner.model_picker = None
        _cue('')
        return None
    backend, done, text = picker.choose(line)
    if done:
        owner.model_picker = None
        _cue('')
    if backend is not None:
        if hasattr(owner, 'use'):
            owner.use(backend)
        else:
            owner.backend = backend
            owner.rebuild = True
    return text

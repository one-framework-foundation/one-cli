import json
import pytest
from onecli import session, terminal

@pytest.fixture(autouse=True)
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr(session, 'SESSION_DIR', tmp_path)

def create(sid, workspace='/project', title=''):
    session.append(sid, {'kind':'metadata', 'workspace':workspace, 'title':title})
    session.append(sid, {'kind':'messages', 'messages':[{'role':'user','content':'repair widgets'}]})

def test_search_names_openings_workspace_and_limit():
    create('old', '/project', 'Release notes')
    create('new', '/elsewhere', 'Other')
    assert session.search('RELEASE', '/project', 1)[0]['session'] == 'old'
    assert len(session.search('widgets')) == 2
    assert not session.search('absent')

def test_fork_independent_and_retains_plan():
    create('original')
    session.append('original', {'kind':'todos', 'items':[{'text':'pending'}]})
    before = session.path_for('original').read_bytes()
    target = session.fork('original', 'experiment')
    session.rename(target, 'renamed')
    assert session.path_for('original').read_bytes() == before
    assert session.load_messages(target) == session.load_messages('original')
    assert session.load_records(target, 'todos') == session.load_records('original', 'todos')
    assert session.metadata(target)['parent'] == 'original'
    assert session.search('renamed')[0]['session'] == target

def test_reject_paths_and_missing_rename():
    with pytest.raises(ValueError):
        session.load_messages("../secret")
    with pytest.raises(FileNotFoundError):
        session.rename("missing", "name")


def test_draft_roundtrip_and_clear():
    session.save_draft('draft', 'hello\nworld')
    assert session.load_draft('draft') == 'hello\nworld'
    assert session.path_for('draft').with_suffix('.draft.json').stat().st_mode & 0o777 == 0o600
    session.save_draft('draft', '')
    assert session.load_draft('draft') == ''

def test_terminal_restores_and_never_persists_permission_answer():
    session.save_draft('draft', 'unfinished')
    surface = terminal.Surface('test', None)
    surface.restore_draft('draft')
    assert surface.draft == 'unfinished'
    surface.saved_draft = (surface.draft, surface.cursor)
    surface.answering = True
    surface.draft = 'sensitive permission answer'
    surface.persist_draft()
    assert session.load_draft('draft') == 'unfinished'
    surface.answering = False
    surface.draft = ''
    surface.persist_draft()
    assert session.load_draft('draft') == ''

def test_legacy_and_truncated_transcripts():
    session.append('old', {'kind':'messages','messages':[{'role':'user','content':'legacy'}]})
    with session.path_for("old").open("a") as f:
        f.write("null\n{bad")
    assert session.search('legacy')[0]['session'] == 'old'
    assert session.search('legacy', '/project') == []

def test_checkpoints_grow_linearly_and_resume_after_rewrite():
    sid = 'linear'
    messages = []
    sizes = {}
    for turn in range(1, 101):
        messages.append({'role': 'user', 'content': 'x' * 200})
        messages.append({'role': 'assistant', 'content': 'y' * 800})
        session.append(sid, {'kind': 'messages', 'messages': messages})
        if turn in (50, 100):
            sizes[turn] = session.path_for(sid).stat().st_size
    assert sizes[100] < sizes[50] * 2.5
    assert session.load_messages(sid) == messages
    assert len(session.load_records(sid, 'messages')) == 100

    # A compacted or edited transcript replaces the prior state.
    replacement = [messages[0], {'role': 'assistant', 'content': 'revised'}]
    session.append(sid, {'kind': 'messages', 'messages': replacement})
    assert session.load_messages(sid) == replacement


def test_resuming_legacy_snapshot_then_appending_delta():
    sid = 'legacy'
    old = [{'role': 'user', 'content': 'old'}]
    session.path_for(sid).write_text(json.dumps({'kind': 'messages', 'messages': old}) + '\n')
    current = old + [{'role': 'assistant', 'content': 'new'}]
    session.append(sid, {'kind': 'messages', 'messages': current})
    current.append({'role': 'user', 'content': 'again'})
    session.append(sid, {'kind': 'messages', 'messages': current})
    assert session.load_messages(sid) == current


def test_partial_write_and_external_change_force_a_complete_checkpoint():
    sid = 'interrupted'
    first = [{'role': 'user', 'content': 'first'}]
    session.append(sid, {'kind': 'messages', 'messages': first})
    with session.path_for(sid).open('a') as fh:
        fh.write('{"kind": "messages_delta", "start": 1')
    second = first + [{'role': 'assistant', 'content': 'second'}]
    session.append(sid, {'kind': 'messages', 'messages': second})
    assert session.load_messages(sid) == second
    assert json.loads(session.path_for(sid).read_text().splitlines()[-1])['kind'] == 'messages'

    with session.path_for(sid).open('a') as fh:
        fh.write(json.dumps({'kind': 'messages', 'messages': first}) + '\n')
    session.append(sid, {'kind': 'messages', 'messages': second})
    assert session.load_messages(sid) == second
    assert json.loads(session.path_for(sid).read_text().splitlines()[-1])['kind'] == 'messages'


@pytest.mark.parametrize('record', [
    {'kind': 'metadata', 'title': 'renamed'},
    {'kind': 'todos', 'items': []},
])
def test_external_rewrite_before_other_record_invalidates_message_cache(record):
    sid = 'external'
    first = [{'role': 'user', 'content': 'first'}]
    session.append(sid, {'kind': 'messages', 'messages': first})
    external = [{'role': 'user', 'content': 'external rewrite'}]
    with session.path_for(sid).open('a') as fh:
        fh.write(json.dumps({'kind': 'messages', 'messages': external}) + '\n')

    session.append(sid, record)
    current = first + [{'role': 'assistant', 'content': 'second'}]
    session.append(sid, {'kind': 'messages', 'messages': current})

    assert session.load_messages(sid) == current
    assert session.load_records(sid, 'messages')[-1]['messages'] == current
    assert json.loads(session.path_for(sid).read_text().splitlines()[-1])['kind'] == 'messages'


@pytest.mark.parametrize("value", [None, [], "draft", 42, True])
def test_draft_wrong_json_shape_is_recoverable(value):
    session.path_for("wrongshape").with_suffix(".draft.json").write_text(json.dumps(value))
    assert session.load_draft("wrongshape") == ""


def test_opening_line_survives_non_text_content():
    sid = 'nulcontent'
    session.append(sid, {'kind': 'messages', 'messages': [{'role': 'user', 'content': None}]})
    assert session.opening_line(sid) == ''


def test_delta_record_without_cached_prefix_does_not_crash():
    sid = 'cold-delta'
    session.append(sid, {'kind': 'messages_delta', 'version': 1, 'start': 0,
                         'messages': [{'role': 'user', 'content': 'hi'}]})
    assert session.path_for(sid).is_file()

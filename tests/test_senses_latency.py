from types import SimpleNamespace
from unittest.mock import patch
import pytest
from onecli.senses import eye
from onecli.senses.base import Outcome

@pytest.fixture(autouse=True)
def clean():
    eye._FAILURE_UNTIL.clear()
    yield
    eye._FAILURE_UNTIL.clear()

def backends():
    return [SimpleNamespace(name=str(i), base_url='http://fixture.invalid', model=str(i)) for i in range(3)]

def run():
    out=Outcome(modality='image')
    text=eye._describe_rung(out,'fixture.png','image/png',object(),'describe')
    return out,text

def test_failure_skipped_across_images_and_retried_after_expiry():
    now = [10.0]
    seen = []

    def describe(path,mime,b,prompt,**kw):
        seen.append(b.name)
        if b.name == "0":
            raise RuntimeError("model failed to load")
        return 'A white card with black lettering.'
    with patch.object(eye.time,'monotonic',lambda:now[0]), patch.object(eye.caps_mod,'vision_backends',lambda c:backends()),patch.object(eye,'describe_image',describe):
        assert run()[1]
        out, text = run()
        assert text
        assert seen==['0','1','1']
        assert 'cooling down' in out.rungs[0].skipped
        now[0] = 71
        assert run()[1]
        assert seen==['0','1','1','0','1']

def test_attempts_use_remaining_budget_and_stop():
    now = [0.0]
    timeouts = []

    def describe(*args,timeout):
        timeouts.append(timeout)
        now[0] += timeout
        raise TimeoutError('timed out')
    with patch.object(eye.time,'monotonic',lambda:now[0]),patch.object(eye.caps_mod,'vision_backends',lambda c:backends()),patch.object(eye,'describe_image',describe),patch.object(eye,'DESCRIBE_TIMEOUT',15),patch.object(eye,'DESCRIBE_BUDGET',22):
        out,text=run()
        assert not text
        assert timeouts == [15, 7]
        assert 'budget exhausted' in out.rungs[-1].skipped

def test_unseeing_answer_does_not_quarantine_model_for_other_images():
    seen=[]
    def describe(*args,**kw):
        seen.append(1)
        return "I cannot see the image."

    with patch.object(eye.caps_mod,'vision_backends',lambda c:backends()[:1]),patch.object(eye,'describe_image',describe):
        assert not run()[1]
        assert not run()[1]
        assert len(seen) == 2
        assert not eye._FAILURE_UNTIL


def test_empty_candidate_list_is_explicit():
    with patch.object(eye.caps_mod,'vision_backends',lambda c:[]):
        out, text = run()
        assert not text
        assert 'no reachable backend' in out.rungs[0].skipped


def test_bad_file_does_not_quarantine_provider():
    with patch.object(eye.caps_mod,'vision_backends',lambda c:backends()[:1]),patch.object(eye,'describe_image',side_effect=FileNotFoundError('missing attachment')):
        assert not run()[1]
        assert not eye._FAILURE_UNTIL

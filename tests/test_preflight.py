"""What a run refuses BEFORE it spends anything.

Both cases here were found by pointing the CLI at something a person mistypes.
The cost of getting them wrong is not a bad error message - it is a whole paid
run that answers from nothing and then exits 0, which is the one failure a
caller cannot detect.
"""

from __future__ import annotations

import os

import pytest

from onecli import cli, config


# --- the workspace ---------------------------------------------------------

def test_a_missing_workspace_is_refused(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        cli._check_workspace(str(tmp_path / "typo"))


def test_a_workspace_that_is_a_file_is_refused(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("hi")
    with pytest.raises(ValueError, match="not a directory"):
        cli._check_workspace(str(f))


@pytest.mark.skipif(os.getuid() == 0, reason="root reads a mode-000 directory")
def test_an_unreadable_workspace_is_refused(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o000)
    try:
        with pytest.raises(ValueError, match="not readable"):
            cli._check_workspace(str(locked))
    finally:
        locked.chmod(0o755)


def test_a_real_workspace_passes(tmp_path):
    cli._check_workspace(str(tmp_path))


def test_the_check_runs_before_the_first_request(tmp_path, monkeypatch, capsys):
    """The whole point: exit 2 with nobody having been paid.

    `run_once` is monkeypatched to explode - if the guard is removed, the run
    reaches it and the test fails with that explosion rather than with exit 2,
    which is exactly the failure being guarded against.
    """
    def never(*a, **k):  # pragma: no cover - reaching this IS the failure
        raise AssertionError("a request went out for a workspace that is not there")

    monkeypatch.setattr(cli, "run_once", never)
    code = cli.main(["-m", "whatever", "-w", str(tmp_path / "typo"), "hi"])
    assert code == 2
    assert "does not exist" in capsys.readouterr().err


# --- the backend name ------------------------------------------------------

def _dropin(directory, name, model):
    (directory / f"{name}.env").write_text(
        f"ONE_PROVIDER_MODEL={model}\n"
        "OPENAI_BASE_URL=http://127.0.0.1:1/v1\nONE_PROVIDER_KEY=x\n")


@pytest.fixture
def providers(tmp_path, monkeypatch):
    d = tmp_path / "providers"
    d.mkdir()
    monkeypatch.setattr(config, "PROVIDER_DIR", d)
    monkeypatch.setenv(config.PROVIDER_DIR_ENV, "")
    monkeypatch.setenv(config.LLAMACPP_HOSTS_ENV, "")
    # No llama-server discovery: this is about the drop-ins and nothing else.
    monkeypatch.setattr(config, "llamacpp_backends", lambda *a, **k: [])
    return d


def test_an_ambiguous_name_is_refused_rather_than_guessed(providers):
    _dropin(providers, "qwen-fast", "qwen-a")
    _dropin(providers, "qwen-big", "qwen-b")
    with pytest.raises(RuntimeError) as exc:
        config.resolve(name="qwen")
    assert "matches 2 backends" in str(exc.value)
    # Both are named, so the person can pick without going to look.
    assert "qwen-fast" in str(exc.value) and "qwen-big" in str(exc.value)


def test_a_unique_substring_is_still_shorthand(providers):
    _dropin(providers, "qwen-fast", "qwen-a")
    _dropin(providers, "kimi", "k3")
    assert config.resolve(name="qwe").name == "qwen-fast"


def test_an_exact_name_wins_over_a_substring_of_another(providers):
    """`-m qwen` with a backend literally called `qwen` is not ambiguous."""
    _dropin(providers, "qwen", "qwen-a")
    _dropin(providers, "qwen-big", "qwen-b")
    assert config.resolve(name="qwen").name == "qwen"


def test_no_match_still_lists_what_there_is(providers):
    _dropin(providers, "kimi", "k3")
    with pytest.raises(RuntimeError, match="no backend matching"):
        config.resolve(name="zzz")


def test_a_tier_name_is_not_fuzzy_matched_into_an_unrelated_local_model(providers):
    """`-m opus` must not land on a local model that merely has "opus" in its id.

    A local model whose id merely contains the word was found by the substring
    fallback as the unique match for `opus`, so `one -m opus ...` ran the whole
    job on that model, exit 0, with nothing in the output naming the swap. The
    caller had asked for the top tier. A name that is an ESCALATION TIER is not
    a backend shorthand, and the honest answer is the error, not the nearest
    model whose marketing name contains it.
    """
    _dropin(providers, "local-a", "example-opus-finetune")
    with pytest.raises(RuntimeError, match="escalation tier"):
        config.resolve(name="opus")


def test_a_tier_name_still_resolves_when_a_backend_really_carries_it(providers):
    """The reserved names guard the FUZZY path only - an exact backend wins."""
    _dropin(providers, "opus", "some-model")
    assert config.resolve(name="opus").name == "opus"


def test_default_resolve_uses_discovery_rather_than_a_shipped_preference(providers, monkeypatch):
    """No backend is preferred by name: with no request, discovery decides."""
    monkeypatch.delenv("ONE_DEFAULT_BACKEND", raising=False)
    monkeypatch.delenv("ONE_DEFAULT_MODEL", raising=False)
    _dropin(providers, "first-lane", "model-one")
    _dropin(providers, "second-lane", "model-two")
    assert config.resolve().name == "first-lane"
    monkeypatch.setenv("ONE_DEFAULT_BACKEND", "second-lane")
    assert config.resolve().name == "second-lane"


def test_generic_deepseek_name_resolves_to_preferred_v4_1_flash(providers):
    _dropin(providers, "dashscope", "deepseek-v4-flash")
    _dropin(providers, "opencode-go-deepseek-v4.1-flash", "deepseek-v4.1-flash")
    assert config.resolve(name="deepseek").model == "deepseek-v4.1-flash"



def test_an_explicit_backend_name_is_never_overridden_by_a_preference(providers):
    _dropin(providers, "aaa-provider", "deepseek-v4.1-flash")
    _dropin(providers, "deepseek-vision", "deepseek-flash")
    assert config.resolve(name="deepseek-vision").name == "deepseek-vision"
    assert config.resolve(model="deepseek-flash").name == "deepseek-vision"

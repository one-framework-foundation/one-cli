import json

import pytest

from onecli.serving import catalog, pull


def _model(file="m.gguf", **kw):
    return catalog.Model(name="m", repo="o/r", file=file, **kw)


@pytest.mark.parametrize("bad", ["../x.gguf", "a/../../x.gguf", "/etc/passwd"])
def test_pull_refuses_traversal_and_absolute(tmp_path, bad):
    with pytest.raises(pull.PullError, match="unsafe file"):
        pull.plan(_model(bad), tmp_path)
    with pytest.raises(pull.PullError, match="unsafe file"):
        pull.pull(_model(bad), tmp_path)


def test_pull_refuses_hf_rfilename_draft(tmp_path):
    m = _model(draft=catalog.Artifact(file="../mtp.gguf"))
    with pytest.raises(pull.PullError, match=r"\.\./mtp\.gguf"):
        pull.pull(m, tmp_path)


def test_pull_refuses_symlink_escape(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    (root / "link").symlink_to(out)
    with pytest.raises(pull.PullError, match="outside"):
        pull.pull(_model("link/m.gguf"), root)
    with pytest.raises(pull.PullError, match="outside"):
        pull.remove(_model("link/m.gguf"), root)
    assert not list(out.iterdir())


def test_safe_join_allows_nested_inside(tmp_path):
    assert catalog.safe_join(tmp_path, "sub/m.gguf") == tmp_path / "sub/m.gguf"


def _user_catalog(tmp_path, monkeypatch, payload):
    p = tmp_path / "models.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(catalog, "USER_CATALOG", p)


GOOD = {"name": "good", "repo": "o/r", "file": "good.gguf", "size": 5}


@pytest.mark.parametrize("bad,reason", [
    ({"name": "nokey", "repo": "o/r"}, "missing key"),
    ({"name": "sz", "repo": "o/r", "file": "a.gguf", "size": "big"}, "invalid literal"),
    ({"name": "ty", "repo": "o/r", "file": 5}, "file is not"),
    ({"name": "trav", "repo": "o/r", "file": "../a.gguf"}, "traversal"),
    ({"name": "abs", "repo": "o/r", "file": "/a.gguf"}, "absolute"),
    ({"name": "dr", "repo": "o/r", "file": "a.gguf",
      "draft": {"file": "../d.gguf"}}, "traversal"),
    ("notadict", "#"),
])
def test_malformed_entry_skipped_with_warning(tmp_path, monkeypatch, capsys,
                                              bad, reason):
    _user_catalog(tmp_path, monkeypatch, {"models": [bad, GOOD]})
    names = [m.name for m in catalog.load()]
    assert "good" in names
    assert catalog.find("good").file == "good.gguf"
    label = bad["name"] if isinstance(bad, dict) else "#0"
    assert label not in names
    err = capsys.readouterr().err
    assert "warning" in err and label in err and reason in err


@pytest.mark.parametrize("payload", [[GOOD], {"models": {"a": 1}}, "x", 3])
def test_wrong_shape_catalog_skipped_with_warning(tmp_path, monkeypatch,
                                                  capsys, payload):
    _user_catalog(tmp_path, monkeypatch, payload)
    assert catalog.load()  # packaged catalog still lists
    assert "skipped" in capsys.readouterr().err

import io
import json
from onecli import config
from onecli.senses import capabilities as c,eye

def test_no_implicit_paid_route(monkeypatch):
 monkeypatch.delenv('ONE_CLI_VISION_BACKENDS',raising=False)
 monkeypatch.setattr(c,'local_vision_backends',lambda caps: [])
 monkeypatch.setattr(config,'discover',lambda: (_ for _ in ()).throw(AssertionError('implicit paid discovery')))
 assert c.vision_backends(c.Capabilities())==[]

def test_local_model_metadata_and_size_gate(monkeypatch):
    monkeypatch.setenv("ONE_SENSES_OLLAMA_ENDPOINTS", "http://127.0.0.1:11434")
    caps = c.Capabilities()
    monkeypatch.setattr(
        c,
        "_get_json",
        lambda *a, **k: {
            "models": [
                {"name": "blind", "size": 20},
                {"name": "eyes", "size": 30},
                {"name": "huge", "size": 9_000_000_000},
            ]
        },
    )
    calls = []

    def show(req, **kw):
        model = json.loads(req.data)["model"]
        calls.append(model)
        return io.BytesIO(
            json.dumps(
                {"capabilities": ["vision"] if model == "eyes" else ["completion"]}
            ).encode()
        )

    monkeypatch.setattr(c.urllib.request, "urlopen", show)
    bs = c.local_vision_backends(caps)
    assert [b.model for b in bs] == ["eyes"]
    assert calls == ["eyes", "blind"]
    assert caps.declares(bs[0], "vision")
    bs[0].model = "other"
    assert not caps.declares(bs[0], "vision")

def test_describer_headers_and_root_url(monkeypatch,tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"fixture")
    b = config.Backend(
        "eyes", "eyes", "https://example.invalid", headers={"X-Endpoint": "required"}
    )

    def reply(req, **kw):
        assert req.full_url == "https://example.invalid/v1/chat/completions"
        assert req.get_header("X-endpoint") == "required"
        body = json.loads(req.data)
        assert body["model"] == "eyes" and body["messages"][0]["content"][1][
            "image_url"
        ]["url"].startswith("data:image/png;base64,")
        return io.BytesIO(
            b'{"choices":[{"message":{"content":"A red circle on white."}}]}'
        )

    monkeypatch.setattr(eye.urllib.request, "urlopen", reply)
    assert eye.describe_image(str(path), "image/png", b) == "A red circle on white."

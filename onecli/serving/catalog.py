"""The model list: what we have served, and how to get it.

Two halves that answer different questions:

- the CURATED list (`models.json`) - weights that were pulled, hashed and
  verified on a local server, with the drafter that belongs to each. This
  is what a menu offers, because every row in it is known to work.
- LIVE search over HuggingFace, for everything else. A repo found this way has
  not been verified locally and is marked as such; it is a lead, not a promise.

Nothing here downloads. `pull.py` does that.
"""

from __future__ import annotations

import http.client
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .. import secureio
from . import hostfacts

HF_API = "https://huggingface.co/api"
HF_RESOLVE = "https://huggingface.co/{repo}/resolve/main/{path}"
CATALOG_PATH = Path(__file__).with_name("models.json")

# A user catalog, so a host can add a model without editing the package.
USER_CATALOG = Path.home() / ".one-cli" / "models.json"

UA = "one-cli/serving"


def unsafe_reason(rel: object) -> str:
    """Why `rel` cannot be a file name under the models root, or "" if it can."""
    if not isinstance(rel, str) or not rel:
        return "file is not a non-empty string"
    if "\x00" in rel:
        return "file contains a NUL byte"
    if rel.startswith(("/", "\\")) or Path(rel).is_absolute():
        return "absolute path"
    if ".." in rel.replace("\\", "/").split("/"):
        return "'..' path traversal"
    return ""


def safe_join(root: Path, rel: str) -> Path:
    """`root / rel`, or ValueError if it would land outside `root`.

    Checks the name, then the real path, so a symlink inside the root that
    points out of it is refused as well as a `../` name.
    """
    why = unsafe_reason(rel)
    if why:
        raise ValueError(f"unsafe file {rel!r}: {why}")
    path = root / rel
    real_root = Path(os.path.realpath(root))
    real = Path(os.path.realpath(path))
    if real != real_root and real_root not in real.parents:
        raise ValueError(f"unsafe file {rel!r}: resolves to {real}, "
                         f"outside {real_root}")
    return path


@dataclass
class Artifact:
    """One file that has to be on disk for a model to serve."""
    file: str
    repo: str = ""
    size: int = 0
    sha256: str = ""
    spec_type: str = ""
    note: str = ""

    def url(self, parent_repo: str) -> str:
        repo = self.repo or parent_repo
        return HF_RESOLVE.format(repo=repo, path=urllib.parse.quote(self.file))


@dataclass
class Model:
    name: str
    repo: str
    file: str
    size: int = 0
    sha256: str = ""
    params_b: float = 0.0
    quant: str = ""
    role: str = "chat"
    context: int = 0
    alias: str = ""
    draft: Artifact | None = None
    mmproj: Artifact | None = None
    curated: bool = True
    source: str = "huggingface"

    def artifacts(self) -> list[Artifact]:
        out = [Artifact(file=self.file, repo=self.repo, size=self.size,
                        sha256=self.sha256)]
        if self.draft:
            out.append(self.draft)
        if self.mmproj:
            out.append(self.mmproj)
        return out

    def url(self) -> str:
        return HF_RESOLVE.format(repo=self.repo,
                                 path=urllib.parse.quote(self.file))

    def total_bytes(self) -> int:
        return sum(a.size for a in self.artifacts())

    def path(self, root: Path | None = None) -> Path:
        root = root or hostfacts.models_dir()
        return safe_join(root, self.file)

    def draft_path(self, root: Path | None = None) -> Path | None:
        if not self.draft:
            return None
        root = root or hostfacts.models_dir()
        return safe_join(root, self.draft.file)

    def present(self, root: Path | None = None) -> bool:
        """On disk AND the right size. A truncated pull is not a model."""
        p = self.path(root)
        if not p.is_file():
            return False
        if self.size and p.stat().st_size != self.size:
            return False
        return True

    def draft_present(self, root: Path | None = None) -> bool:
        p = self.draft_path(root)
        if p is None or not p.is_file():
            return False
        if self.draft and self.draft.size and p.stat().st_size != self.draft.size:
            return False
        return True


def _artifact(raw: dict | None) -> Artifact | None:
    if not raw:
        return None
    why = unsafe_reason(raw["file"])
    if why:
        raise ValueError(f"unsafe file {raw['file']!r}: {why}")
    return Artifact(file=raw["file"], repo=raw.get("repo", ""),
                    size=int(raw.get("size", 0)), sha256=raw.get("sha256", ""),
                    spec_type=raw.get("spec_type", ""), note=raw.get("note", ""))


def _model(raw: dict) -> Model:
    for key in ("name", "repo", "file"):
        if not isinstance(raw[key], str) or not raw[key]:
            raise ValueError(f"{key} is not a non-empty string")
    why = unsafe_reason(raw["file"])
    if why:
        raise ValueError(f"unsafe file {raw['file']!r}: {why}")
    return Model(
        name=raw["name"], repo=raw["repo"], file=raw["file"],
        size=int(raw.get("size", 0)), sha256=raw.get("sha256", ""),
        params_b=float(raw.get("params_b", 0) or 0), quant=raw.get("quant", ""),
        role=raw.get("role", "chat"), context=int(raw.get("context", 0) or 0),
        alias=raw.get("alias") or raw["name"],
        draft=_artifact(raw.get("draft")), mmproj=_artifact(raw.get("mmproj")),
    )


def _read(path: Path) -> list[Model]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as exc:
        print(f"warning: {path}: skipped unreadable catalog: {exc}", file=sys.stderr)
        return []
    entries = raw.get("models", []) if isinstance(raw, dict) else None
    if not isinstance(entries, list):
        print(f"warning: {path}: skipped, expected an object with a "
              f"'models' list", file=sys.stderr)
        return []
    out = []
    for i, m in enumerate(entries):
        try:
            out.append(_model(m))
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            label = m.get("name", f"#{i}") if isinstance(m, dict) else f"#{i}"
            reason = (f"missing key {exc}" if isinstance(exc, KeyError)
                      else str(exc))
            print(f"warning: {path}: skipped entry {label!r}: {reason}",
                  file=sys.stderr)
    return out


def load() -> list[Model]:
    """Curated models, package first, then locally added ones.

    A user entry with the same name REPLACES the packaged one rather than
    shadowing it in a second list - one name, one answer, so a menu and a serve
    command cannot disagree about which file they mean.
    """
    by_name: dict[str, Model] = {m.name: m for m in _read(CATALOG_PATH)}
    for m in _read(USER_CATALOG):
        by_name[m.name] = m
    return list(by_name.values())


def find(name: str) -> Model | None:
    """Resolve by name, then by alias, then by unambiguous prefix."""
    models = load()
    for m in models:
        if m.name == name or m.alias == name:
            return m
    hits = [m for m in models if m.name.startswith(name)]
    if len(hits) == 1:
        return hits[0]
    for m in models:
        if m.file == name:
            return m
    return None


def local(root: Path | None = None) -> list[tuple[Path, int]]:
    """Every GGUF on this host, catalogued or not, biggest first."""
    root = root or hostfacts.models_dir()
    out: list[tuple[Path, int]] = []
    try:
        for p in sorted(root.glob("*.gguf")):
            out.append((p, p.stat().st_size))
    except OSError:
        return []
    return sorted(out, key=lambda r: -r[1])


def _get(url: str, timeout: float = 12.0) -> dict | list | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except (http.client.HTTPException, urllib.error.URLError, OSError, ValueError):
        return None


def repo_files(repo: str) -> list[dict]:
    """The GGUF files in a repo, with sizes when the API reports them."""
    data = _get(f"{HF_API}/models/{repo}?blobs=true")
    if not isinstance(data, dict):
        return []
    out = []
    for sib in data.get("siblings", []) or []:
        name = sib.get("rfilename", "")
        if not name.endswith(".gguf"):
            continue
        out.append({"file": name, "size": int(sib.get("size", 0) or 0),
                    "sha256": (sib.get("lfs") or {}).get("sha256", "")})
    return out


def search(query: str, limit: int = 15) -> list[dict]:
    """Live GGUF repos on HuggingFace matching a query.

    Sorted by downloads because on a list of unknown quantizers that is the
    only signal we have about which one people actually load.
    """
    q = urllib.parse.urlencode({"search": query, "filter": "gguf",
                                "sort": "downloads", "direction": "-1",
                                "limit": limit})
    data = _get(f"{HF_API}/models?{q}")
    if not isinstance(data, list):
        return []
    return [{"repo": d.get("modelId") or d.get("id", ""),
             "downloads": int(d.get("downloads", 0) or 0),
             "likes": int(d.get("likes", 0) or 0),
             "curated": False}
            for d in data if d.get("modelId") or d.get("id")]


def from_repo(repo: str, file: str = "", name: str = "") -> Model | None:
    """Build a Model from a live repo, picking a file when none is named.

    The pick is deliberate rather than alphabetical: prefer Q4_K_M, then Q4,
    then the largest single-part file. A repo full of shards has no single
    file to serve, so a `-00001-of-` name is skipped instead of half-pulled.
    """
    files = repo_files(repo)
    if not files:
        return None
    drafts = [f for f in files if f["file"].lower().startswith("mtp-")]
    mmprojs = [f for f in files if f["file"].lower().startswith("mmproj")]
    body = [f for f in files
            if f not in drafts and f not in mmprojs
            and "-of-" not in f["file"]]
    if file:
        chosen = next((f for f in files if f["file"] == file), None)
    else:
        chosen = (next((f for f in body if "Q4_K_M" in f["file"]), None)
                  or next((f for f in body if "Q4" in f["file"]), None)
                  or (max(body, key=lambda f: f["size"]) if body else None))
    if not chosen:
        return None
    stem = chosen["file"].rsplit(".gguf", 1)[0]
    return Model(
        name=name or stem.lower(), repo=repo, file=chosen["file"],
        size=chosen["size"], sha256=chosen["sha256"],
        quant=_quant_of(chosen["file"]), alias=name or stem.lower(),
        curated=False,
        draft=(Artifact(file=drafts[0]["file"], repo=repo,
                        size=drafts[0]["size"], sha256=drafts[0]["sha256"],
                        spec_type="draft-mtp") if drafts else None),
        mmproj=(Artifact(file=mmprojs[0]["file"], repo=repo,
                         size=mmprojs[0]["size"], sha256=mmprojs[0]["sha256"])
                if mmprojs else None),
    )


_QUANTS = ("Q2_K", "Q3_K_M", "Q3_K_S", "Q4_K_M", "Q4_K_S", "Q4_0", "Q5_K_M",
           "Q5_0", "Q6_K", "Q8_0", "MXFP4", "BF16", "F16")


def _quant_of(filename: str) -> str:
    up = filename.upper()
    for q in _QUANTS:
        if q in up:
            return q
    return ""


def add_user_model(model: Model) -> Path:
    """Record a live-found model in the local catalog.

    So the second time it is a named row in the menu rather than a repo id
    somebody has to remember.
    """
    USER_CATALOG.parent.mkdir(parents=True, exist_ok=True)
    try:
        raw = json.loads(USER_CATALOG.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raw = {"version": 1, "models": []}
    except (OSError, ValueError) as exc:
        raise ValueError(f"refusing to overwrite unreadable catalog {USER_CATALOG}: {exc}") from exc
    if (not isinstance(raw, dict) or not isinstance(raw.get("models"), list)
            or any(not isinstance(m, dict) for m in raw["models"])):
        raise ValueError(f"refusing to overwrite invalid catalog {USER_CATALOG}")
    entry = {"name": model.name, "repo": model.repo, "file": model.file,
             "size": model.size, "sha256": model.sha256, "quant": model.quant,
             "role": model.role, "alias": model.alias or model.name}
    if model.context:
        entry["context"] = model.context
    if model.draft:
        entry["draft"] = {"file": model.draft.file, "repo": model.draft.repo,
                          "size": model.draft.size, "sha256": model.draft.sha256,
                          "spec_type": model.draft.spec_type or "draft-mtp"}
    if model.mmproj:
        entry["mmproj"] = {"file": model.mmproj.file, "repo": model.mmproj.repo,
                           "size": model.mmproj.size,
                           "sha256": model.mmproj.sha256}
    raw["models"] = [m for m in raw.get("models", [])
                     if m.get("name") != model.name] + [entry]
    secureio.atomic_write_text(USER_CATALOG, json.dumps(raw, indent=2) + "\n")
    return USER_CATALOG

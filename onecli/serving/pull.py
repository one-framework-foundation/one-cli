"""Fetch weights, resumably, and refuse to serve bytes that are not the model.

One rule runs through this file: a file is only a model once its size and its
hash match what the source says. Anything else is set aside under a .bad name
rather than deleted, so a failed pull can be inspected instead of silently
retried forever.
"""

from __future__ import annotations

import hashlib
import http.client
import os
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import hostfacts
from .catalog import Artifact, Model, safe_join

CHUNK = 1024 * 1024
UA = "one-cli/pull"


@dataclass
class Progress:
    file: str
    done: int = 0
    total: int = 0
    rate_mb_s: float = 0.0
    resumed: bool = False

    @property
    def share(self) -> float:
        return self.done / self.total if self.total else 0.0


class PullError(RuntimeError):
    pass


def _dest(root: Path, art: Artifact) -> Path:
    try:
        return safe_join(root, art.file)
    except ValueError as exc:
        raise PullError(str(exc)) from exc


def _request(url: str, start: int = 0) -> urllib.request.Request:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if start:
        req.add_header("Range", f"bytes={start}-")
    return req


def _sha256(path: Path, on_progress=None) -> str:
    h = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    with path.open("rb") as fh:
        while True:
            block = fh.read(CHUNK * 8)
            if not block:
                break
            h.update(block)
            done += len(block)
            if on_progress:
                on_progress(Progress(file=path.name, done=done, total=total))
    return h.hexdigest()


def verify(path: Path, size: int = 0, sha256: str = "",
           on_progress=None) -> tuple[bool, str]:
    """Is this file the artifact it claims to be?

    Size first because it is free and catches the common failure (a truncated
    or still-running download). The hash is only worth its minutes once the
    size already agrees.
    """
    if not path.is_file():
        return False, "not on disk"
    actual = path.stat().st_size
    if size and actual != size:
        return False, f"size {actual} != {size}"
    if sha256:
        got = _sha256(path, on_progress)
        if got != sha256:
            return False, f"sha256 {got[:12]} != {sha256[:12]}"
        return True, "size and sha256 verified"
    if size:
        return True, "size verified, source publishes no hash"
    return True, "present, source publishes neither size nor hash"


def fetch(url: str, dest: Path, size: int = 0, sha256: str = "",
          on_progress=None, resume: bool = True) -> Path:
    """Download one artifact to `dest`, resuming a partial `.part` if there is one."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    ok, _why = verify(dest, size, sha256, on_progress)
    if ok:
        return dest

    part = dest.with_suffix(dest.suffix + ".part")
    start = part.stat().st_size if (resume and part.is_file()) else 0
    if start and size and start >= size:
        # A .part at or past the full size is not a resume point, it is junk.
        part.unlink()
        start = 0

    mode = "ab" if start else "wb"
    try:
        with urllib.request.urlopen(_request(url, start), timeout=60) as r:
            if start and r.status != 206:
                # The server ignored the Range header; restarting is the only
                # correct answer, appending would corrupt the file.
                start = 0
                mode = "wb"
            total = size or (int(r.headers.get("Content-Length", 0) or 0) + start)
            done = start
            t0 = time.time()
            with part.open(mode) as fh:
                while True:
                    block = r.read(CHUNK)
                    if not block:
                        break
                    fh.write(block)
                    done += len(block)
                    if on_progress:
                        dt = max(time.time() - t0, 0.001)
                        on_progress(Progress(
                            file=dest.name, done=done, total=total,
                            rate_mb_s=round((done - start) / dt / 1e6, 1),
                            resumed=bool(start)))
    except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
        raise PullError(f"{dest.name}: {exc}") from exc

    ok, why = verify(part, size, sha256, on_progress)
    if not ok:
        bad = part.with_suffix(f".bad.{int(time.time())}")
        part.rename(bad)
        raise PullError(f"{dest.name}: {why} - set aside as {bad.name}")
    part.replace(dest)
    return dest


def free_bytes(root: Path) -> int:
    st = os.statvfs(root)
    return st.f_bavail * st.f_frsize


def plan(model: Model, root: Path | None = None,
         with_draft: bool = True) -> list[tuple[Artifact, Path, bool]]:
    """What a pull would fetch: (artifact, destination, already present)."""
    root = root or hostfacts.models_dir()
    out = []
    for art in model.artifacts():
        if art.spec_type and not with_draft:
            continue
        dest = _dest(root, art)
        ok, _ = verify(dest, art.size)
        out.append((art, dest, ok))
    return out


def pull(model: Model, root: Path | None = None, with_draft: bool = True,
         on_progress=None, log=None) -> list[Path]:
    """Fetch every artifact this model needs, drafter included.

    The drafter comes down by default and that is deliberate: it is under a
    tenth of the weights and it is what makes the fast recipe measurable at
    all. A pull that skips it produces a host where the fastest configuration
    cannot even be tried.
    """
    root = root or hostfacts.models_dir()
    root.mkdir(parents=True, exist_ok=True)
    todo = [(a, d) for a, d, ok in plan(model, root, with_draft) if not ok]
    need = sum(a.size for a, _ in todo)
    have = free_bytes(root)
    if need and have and need > have * 0.95:
        raise PullError(
            f"{need / 1e9:.1f} GB to fetch, {have / 1e9:.1f} GB free on "
            f"{root} - not starting a pull that cannot finish")
    got = []
    for art, dest in todo:
        if log:
            log(f"fetching {art.file} ({art.size / 1e9:.1f} GB)")
        fetch(art.url(model.repo), dest, art.size, art.sha256,
              on_progress=on_progress)
        got.append(dest)
    return got


def remove(model: Model, root: Path | None = None) -> list[Path]:
    """Delete a model's files. Never called without an explicit request."""
    root = root or hostfacts.models_dir()
    gone = []
    for art in model.artifacts():
        p = _dest(root, art)
        if p.is_file():
            trash = shutil.which("trash")
            if trash:
                status = os.spawnvp(os.P_WAIT, trash, [trash, str(p)])
                if status != 0 or p.exists():
                    raise PullError(f"trash failed to remove {p} (exit {status})")
            else:
                p.unlink()
            gone.append(p)
    return gone

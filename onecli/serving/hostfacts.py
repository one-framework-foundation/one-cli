"""What this machine actually is, measured rather than remembered.

Every other module here refuses to guess. A recipe is only offered if the
hardware and the runner that would execute it were both observed on THIS host,
in this process - never read from a note about what the box used to be.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Where GGUF weights live, in the order we look. First existing wins; a host
# that keeps them somewhere else names it in ONE_MODELS_DIR.
MODEL_DIR_ENV = "ONE_MODELS_DIR"
MODEL_DIRS = (
    "/mnt/models/gguf",
    "/models",
    "~/models",
    "~/.cache/llama.cpp",
)

# The llama.cpp server image we launch through docker when no native binary is
# on PATH. Pinned so a tuned recipe stays reproducible: a measurement is only
# valid for the runner that produced it.
DOCKER_IMAGE = "ghcr.io/ggml-org/llama.cpp:server-cuda13"


@dataclass
class Gpu:
    name: str = ""
    memory_mb: int = 0
    count: int = 1


@dataclass
class HostFacts:
    hostname: str = ""
    system: str = ""
    arch: str = ""
    cpus: int = 0
    ram_mb: int = 0
    gpu: Gpu = field(default_factory=Gpu)
    unified_memory: bool = False
    models_dir: str = ""
    models_free_gb: float = 0.0
    llama_server: str = ""     # path to a native binary, "" when absent
    docker: bool = False
    docker_image: str = ""
    ollama: str = ""

    def runner(self) -> str:
        """The string a tuning entry is keyed against.

        A measured speedup belongs to a (host, gpu, runner) triple. Change the
        runner and the number is a claim about a build that is no longer here,
        so it is recorded and compared, never assumed stable.
        """
        if self.llama_server:
            return f"native:{self.llama_server}"
        if self.docker:
            return f"docker:{self.docker_image or DOCKER_IMAGE}"
        return "none"

    def can_serve(self) -> bool:
        return bool(self.llama_server or self.docker)

    def to_json(self) -> dict:
        d = asdict(self)
        return d


def _run(cmd: list[str], timeout: float = 6.0) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip()


def _nvidia() -> Gpu:
    raw = _run(["nvidia-smi", "--query-gpu=name,memory.total",
                "--format=csv,noheader"])
    if not raw:
        return Gpu()
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    name, _, mem = lines[0].partition(",")
    mb = 0
    m = re.search(r"(\d+)", mem)
    if m:
        mb = int(m.group(1))
    return Gpu(name=name.strip(), memory_mb=mb, count=len(lines))


def _ram_mb() -> int:
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        # macOS
        raw = _run(["sysctl", "-n", "hw.memsize"])
        return int(raw) // (1024 * 1024) if raw.isdigit() else 0
    m = re.search(r"MemTotal:\s+(\d+) kB", text)
    return int(m.group(1)) // 1024 if m else 0


def _free_gb(path: Path) -> float:
    try:
        st = os.statvfs(path)
    except OSError:
        return 0.0
    return round(st.f_bavail * st.f_frsize / (1024 ** 3), 1)


def models_dir() -> Path:
    """The weights directory for this host, created if it is the fallback.

    Looked up rather than configured, because the answer differs per box and a
    wrong path is how a 17 GB pull lands on the wrong disk.
    """
    named = os.environ.get(MODEL_DIR_ENV, "").strip()
    if named:
        p = Path(named).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p
    for cand in MODEL_DIRS:
        p = Path(cand).expanduser()
        if p.is_dir():
            return p
    p = Path.home() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _unified(gpu: Gpu, ram_mb: int) -> bool:
    """True when GPU and system memory are the same pool.

    It changes what a recipe may ask for: on a GB10 or an Apple chip a 27B at
    Q4 plus a drafter is fine, on a discrete 24 GB card the same pair does not
    fit and the drafter recipe must not be offered.
    """
    name = gpu.name.upper()
    if "GB10" in name or "GH200" in name or "GB200" in name:
        return True
    if platform.system() == "Darwin":
        return True
    return False


def collect() -> HostFacts:
    md = models_dir()
    gpu = _nvidia()
    ram = _ram_mb()
    facts = HostFacts(
        hostname=socket.gethostname(),
        system=platform.system(),
        arch=platform.machine(),
        cpus=os.cpu_count() or 0,
        ram_mb=ram,
        gpu=gpu,
        unified_memory=_unified(gpu, ram),
        models_dir=str(md),
        models_free_gb=_free_gb(md),
        llama_server=shutil.which("llama-server") or "",
        docker=bool(shutil.which("docker")) and _docker_alive(),
        ollama=shutil.which("ollama") or "",
    )
    if facts.docker:
        facts.docker_image = DOCKER_IMAGE
    return facts


def _docker_alive() -> bool:
    """docker on PATH is not docker usable - an unreachable daemon is common."""
    return _run(["docker", "info", "--format", "{{.ServerVersion}}"],
                timeout=8.0) != ""


def usable_memory_mb(facts: HostFacts) -> int:
    """How much a model may occupy on this host.

    On unified memory the pool is system RAM, minus a working margin. On a
    discrete card it is the card, whatever the box has.
    """
    if facts.unified_memory:
        return max(0, int(facts.ram_mb * 0.85))
    return facts.gpu.memory_mb


def describe(facts: HostFacts) -> list[str]:
    """Human lines for the menus, in the order a person wants them."""
    gpu = facts.gpu.name or "no GPU detected"
    if facts.gpu.memory_mb:
        gpu += f" ({facts.gpu.memory_mb // 1024} GB)"
    if facts.unified_memory:
        gpu += " unified"
    rows = [
        f"host      {facts.hostname} ({facts.system}/{facts.arch})",
        f"gpu       {gpu}",
        f"memory    {facts.ram_mb // 1024} GB system, "
        f"{usable_memory_mb(facts) // 1024} GB usable for weights",
        f"models    {facts.models_dir} ({facts.models_free_gb} GB free)",
        f"runner    {facts.runner()}",
    ]
    if facts.ollama:
        rows.append(f"ollama    {facts.ollama}")
    return rows


def main(argv: list[str] | None = None) -> int:
    facts = collect()
    print(json.dumps(facts.to_json(), indent=2))
    return 0

"""Load, validate and resolve the three YAML configs plus .env.

Sole owner of configuration. Imports nothing heavy (no torch, no vllm), so it stays
usable on a CPU-only login shell.

The design principle: every difference between the source's two workers
(07_rewrite "wiki" and 09_Distill "distill") is expressed here as DATA, not as a branch
in the worker. There is no `if pass == "distill"` anywhere in this package -- the
asymmetry lives in `prompt_defs` in configs/data.yaml. That is the only way it survives
a future refactor.
"""
from __future__ import annotations

import json
import os
import re
import string
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Built by concatenation so this module never matches its own placeholder scan.
_OPEN = "<" * 3
_CLOSE = ">" * 3
PLACEHOLDER_RE = re.compile(_OPEN + r"(TIANJIAN|WYTRO)\s*:\s*(.*?)" + _CLOSE, re.S)

# Exactly the kwargs the source passed to LLM(), minus `model` which the code injects
# from the resolved local model directory. Anything else in configs/vllm.yaml `engine`
# is a configuration error, not a feature.
ALLOWED_ENGINE_KEYS = {
    "tensor_parallel_size",
    "dtype",
    "gpu_memory_utilization",
    "max_model_len",
}

VALID_MODES = {"grounded", "wrap"}
VALID_TRIMS = {"wiki", "distill", "wrap"}

# The literal marker the grounded prompts interpolate the document into.
# source: 07_rewrite/rewrite_worker.py:49
TEXT_PLACEHOLDER = "[TEXT]"
# Every wrap prompt ends with this; the document is concatenated after it.
# source: 07_rewrite/rewrite_worker.py:50-51
WRAP_SUFFIX = "\n\nPassage:\n"


def stop(msg: str) -> "NoReturn":  # noqa: F821
    """Fail the way the source failed: a loud STOP and a non-zero exit."""
    print(f"\n*** STOP: {msg}\n", file=sys.stderr, flush=True)
    raise SystemExit(2)


# --------------------------------------------------------------------------- .env
def load_env(repo_root: Path) -> dict:
    """Parse .env as plain KEY=VALUE. No shell evaluation, ever.

    Values are NOT exported here -- callers decide what reaches the environment, so a
    token cannot leak into a subprocess that did not ask for it.
    """
    out = {}
    p = Path(repo_root) / ".env"
    if not p.exists():
        return out
    for i, raw in enumerate(p.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            stop(f".env line {i} is not KEY=VALUE: {raw!r}")
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


# --------------------------------------------------------------------------- specs
@dataclass(frozen=True)
class PromptSpec:
    id: str                    # "p1".."p4"
    arm: str
    mode: str                  # grounded | wrap
    trim: str                  # wiki | distill | wrap
    path: Path
    text: str
    input_drop: int | None     # None => derive max_model_len - max_tokens
    expected_overhead: int


@dataclass(frozen=True)
class ArmSpec:
    name: str
    repo_id: str
    revision: str
    rewrite: bool
    prompts: tuple = ()


@dataclass(frozen=True)
class JobSpec:
    job_id: str                # "wrap-inspired__p3"
    arm: str
    prompt: PromptSpec
    input_dir: Path            # data_root/shards/<arm>
    output_dir: Path           # out_root/raw/<arm>/<prompt_id>


@dataclass
class Config:
    repo_root: Path
    cluster: dict
    data: dict
    vllm: dict
    env: dict
    paths: dict = field(default_factory=dict)
    arms: tuple = ()

    # -- convenience accessors, so callers never index raw dicts --
    @property
    def num_gpus(self) -> int:
        return int(self.cluster["compute"]["num_gpus"])

    @property
    def gpu_ids(self) -> list:
        g = self.cluster["compute"]["gpu_ids"]
        if g == "auto" or g is None:
            return list(range(self.num_gpus))
        return [int(x) for x in g]

    @property
    def max_model_len(self) -> int:
        return int(self.vllm["engine"]["max_model_len"])

    @property
    def max_tokens(self) -> int:
        return int(self.vllm["sampling"]["max_tokens"])

    @property
    def text_column(self) -> str:
        return self.data["sharding"]["text_column"]

    @property
    def compression(self) -> str:
        return self.cluster["runtime"]["output_compression"]

    @property
    def shard_suffix(self) -> str:
        return ".jsonl.zst" if self.compression == "zstd" else ".jsonl"

    def arm(self, name: str) -> ArmSpec:
        for a in self.arms:
            if a.name == name:
                return a
        stop(f"unknown arm {name!r}; known: {[a.name for a in self.arms]}")

    def shards_dir(self, arm: str) -> Path:
        return self.paths["data_root"] / "shards" / arm

    def raw_dir(self, arm: str, prompt_id: str) -> Path:
        return self.paths["out_root"] / "raw" / arm / prompt_id

    def shuffled_dir(self, arm: str, prompt_id: str) -> Path:
        return self.paths["out_root"] / "shuffled" / arm / prompt_id

    def trimmed_dir(self, arm: str, prompt_id: str) -> Path:
        return self.paths["out_root"] / "trimmed" / arm / prompt_id


# --------------------------------------------------------------------------- helpers
def find_placeholders(text: str):
    """Yield (klass, description) for every placeholder marker in `text`."""
    for m in PLACEHOLDER_RE.finditer(text):
        yield m.group(1), " ".join(m.group(2).split())


def _walk_scalars(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_scalars(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _walk_scalars(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        yield path, obj


def assert_no_placeholders(obj, origin: str) -> None:
    """Refuse to run while any blank remains, naming the exact YAML key."""
    bad = []
    for key, val in _walk_scalars(obj):
        for klass, desc in find_placeholders(val):
            bad.append(f"  {origin}: {key}\n      {klass}: {desc}")
    if bad:
        stop(
            f"{len(bad)} unfilled placeholder(s) in {origin}:\n"
            + "\n".join(bad)
            + "\n\n  Run:  python scripts/check_placeholders.py"
        )


class _SafeDict(dict):
    def __missing__(self, k):
        stop(f"cluster.yaml references ${{{k}}}, which is not a known path key")


def _expand_paths(raw: dict) -> dict:
    """Resolve ${key} references between path entries, then make absolute.

    Iterative rather than recursive: a handful of passes settles any acyclic reference
    graph, and a leftover '${' after that is a cycle, which we name explicitly.
    """
    vals = {k: str(v) for k, v in raw.items()}
    for _ in range(len(vals) + 1):
        changed = False
        for k, v in list(vals.items()):
            if "${" in v:
                new = string.Template(v).substitute(_SafeDict(vals))
                if new != v:
                    vals[k], changed = new, True
        if not changed:
            break
    for k, v in vals.items():
        if "${" in v:
            stop(f"cluster.yaml paths.{k} has an unresolved or circular reference: {v}")
    return {k: Path(v).expanduser().resolve() for k, v in vals.items()}


def resolve_drop_threshold(prompt: PromptSpec, max_model_len: int, max_tokens: int):
    """Return (threshold, was_derived).

    This three-line function is the whole point of the port. The source ran two workers
    that were byte-identical except for this:

      07_rewrite  --input-drop 30720          (fixed, tuned to the wiki prompt)
      09_Distill  --input-drop unset          -> max_model_len - max_tokens = 28672,
                                                 so every kept doc gets the full
                                                 4096-token output budget

    source: 09_Distill/rewrite_worker.py:204-207
    """
    if prompt.input_drop is not None:
        return int(prompt.input_drop), False
    return int(max_model_len) - int(max_tokens), True


# --------------------------------------------------------------------------- load
def load_config(config_root: Path | str | None = None) -> Config:
    repo_root = Path(config_root) if config_root else Path(__file__).resolve().parents[2]
    repo_root = repo_root.resolve()
    cdir = repo_root / "configs"
    for name in ("cluster.yaml", "data.yaml", "vllm.yaml"):
        if not (cdir / name).exists():
            stop(f"missing config file: {cdir / name}")

    cluster = yaml.safe_load((cdir / "cluster.yaml").read_text())
    data = yaml.safe_load((cdir / "data.yaml").read_text())
    vllm = yaml.safe_load((cdir / "vllm.yaml").read_text())
    env = load_env(repo_root)

    assert_no_placeholders(cluster, "configs/cluster.yaml")
    assert_no_placeholders(data, "configs/data.yaml")
    assert_no_placeholders(vllm, "configs/vllm.yaml")
    assert_no_placeholders(env, ".env")

    # ---- engine args: exactly the source's set, nothing more ----
    engine_keys = set(vllm["engine"])
    if engine_keys != ALLOWED_ENGINE_KEYS:
        extra = sorted(engine_keys - ALLOWED_ENGINE_KEYS)
        missing = sorted(ALLOWED_ENGINE_KEYS - engine_keys)
        stop(
            "configs/vllm.yaml `engine` must contain exactly the four kwargs the source "
            f"passed to LLM().\n  unexpected: {extra}\n  missing:    {missing}\n"
            "  Adding an engine arg the source never passed changes generation behaviour "
            "and destroys comparability. See docs/SOURCE_INVENTORY.md section 3."
        )
    if int(vllm["engine"]["tensor_parallel_size"]) != 1:
        stop(
            "tensor_parallel_size must be 1 (source parity). Data parallelism here is N "
            "independent single-GPU processes, exactly as the source's 8-way SLURM array "
            "did it -- not vLLM's data_parallel_size, which the source never used."
        )

    # ---- sampling: refuse anything the source did not set ----
    allowed_sampling = {"temperature", "top_p", "max_tokens"}
    extra_sampling = set(vllm["sampling"]) - allowed_sampling
    if extra_sampling:
        stop(
            f"configs/vllm.yaml `sampling` has keys the source never set: "
            f"{sorted(extra_sampling)}. The source used greedy decoding with no stop "
            "sequences, no seed and no penalties (07_rewrite/README.md:23)."
        )

    # ---- paths ----
    paths = _expand_paths(cluster["paths"])

    # ---- compute ----
    try:
        n_gpus = int(cluster["compute"]["num_gpus"])
    except (TypeError, ValueError):
        stop(f"cluster.yaml compute.num_gpus must be an integer, got "
             f"{cluster['compute']['num_gpus']!r}")
    if n_gpus < 1:
        stop(f"cluster.yaml compute.num_gpus must be >= 1, got {n_gpus}")
    cluster["compute"]["num_gpus"] = n_gpus
    cluster["compute"]["cpu_workers"] = int(cluster["compute"]["cpu_workers"])
    cluster["compute"]["shuffle_mem_bytes"] = int(cluster["compute"]["shuffle_mem_bytes"])

    g = cluster["compute"]["gpu_ids"]
    if g != "auto" and g is not None and len(list(g)) != n_gpus:
        stop(f"cluster.yaml compute.gpu_ids has {len(list(g))} entries but "
             f"num_gpus is {n_gpus}")

    if cluster["runtime"]["output_compression"] not in ("zstd", "none"):
        stop("cluster.yaml runtime.output_compression must be 'zstd' or 'none'")
    if cluster["scheduler"]["kind"] not in ("bash", "slurm"):
        stop("cluster.yaml scheduler.kind must be 'bash' or 'slurm'")

    # ---- arms and prompts ----
    defs = data["prompt_defs"]
    arms = []
    for a in data["arms"]:
        prompts = []
        for p in a["prompts"]:
            d = defs.get(p["def"])
            if d is None:
                stop(f"arm {a['name']}: prompt {p['id']} references unknown "
                     f"prompt_def {p['def']!r}")
            ppath = repo_root / p["file"]
            if not ppath.exists():
                stop(f"arm {a['name']}: prompt file not found: {ppath}")
            text = ppath.read_text()

            mode, trim = d["mode"], d["trim"]
            if mode not in VALID_MODES:
                stop(f"{a['name']}/{p['id']}: mode must be one of {sorted(VALID_MODES)}")
            if trim not in VALID_TRIMS:
                stop(f"{a['name']}/{p['id']}: trim must be one of {sorted(VALID_TRIMS)}")

            # The source's own startup guards, ported.
            # source: 07_rewrite/rewrite_worker.py:178-186
            if mode == "grounded":
                n = text.count(TEXT_PLACEHOLDER)
                if n != 1:
                    stop(f"{a['name']}/{p['id']} ({ppath.name}): grounded prompt must "
                         f"contain exactly one {TEXT_PLACEHOLDER} placeholder, found {n}")
                if trim == "wrap":
                    stop(f"{a['name']}/{p['id']}: trim 'wrap' is only valid with "
                         "mode 'wrap'")
            else:
                if not text.endswith(WRAP_SUFFIX):
                    stop(f"{a['name']}/{p['id']} ({ppath.name}): wrap prompt must end "
                         f"with {WRAP_SUFFIX!r} -- the document is concatenated directly "
                         "after it (source: rewrite_worker.py:50-51)")
                if TEXT_PLACEHOLDER in text:
                    stop(f"{a['name']}/{p['id']}: wrap prompts concatenate the document; "
                         f"they must not contain {TEXT_PLACEHOLDER}")
                if trim != "wrap":
                    stop(f"{a['name']}/{p['id']}: mode 'wrap' requires trim 'wrap'")

            prompts.append(PromptSpec(
                id=p["id"], arm=a["name"], mode=mode, trim=trim, path=ppath, text=text,
                input_drop=(None if d["input_drop"] is None else int(d["input_drop"])),
                expected_overhead=int(d["expected_overhead"]),
            ))

        if a["rewrite"] and not prompts:
            stop(f"arm {a['name']} has rewrite: true but no prompts")
        if not a["rewrite"] and prompts:
            stop(f"arm {a['name']} is a control arm (rewrite: false) but declares "
                 f"{len(prompts)} prompt(s); controls are never rewritten")

        arms.append(ArmSpec(name=a["name"], repo_id=a["repo_id"],
                            revision=a.get("revision") or "main",
                            rewrite=bool(a["rewrite"]), prompts=tuple(prompts)))

    cfg = Config(repo_root=repo_root, cluster=cluster, data=data, vllm=vllm, env=env,
                 paths=paths, arms=tuple(arms))

    # ---- the job count is a hard assertion, not a comment ----
    n_jobs = sum(len(a.prompts) for a in arms if a.rewrite)
    expected = int(data["expected_jobs"])
    if n_jobs != expected:
        stop(f"configs/data.yaml enumerates {n_jobs} rewrite jobs but expected_jobs is "
             f"{expected}. A typo here would silently skip a whole pass over an arm.")
    if not data.get("semantics", {}).get("full_cross_product"):
        stop("configs/data.yaml semantics.full_cross_product must be true: every prompt "
             "rewrites the ENTIRE dataset for its arm.")
    return cfg


# --------------------------------------------------------------------------- jobs
def enumerate_jobs(cfg: Config) -> list:
    """The 12 jobs, in a stable order (arm order in data.yaml, then prompt id)."""
    jobs = []
    for a in cfg.arms:
        if not a.rewrite:
            continue
        for p in a.prompts:
            jobs.append(JobSpec(
                job_id=f"{a.name}__{p.id}", arm=a.name, prompt=p,
                input_dir=cfg.shards_dir(a.name),
                output_dir=cfg.raw_dir(a.name, p.id),
            ))
    return jobs


def get_job(cfg: Config, arm: str, prompt_id: str) -> JobSpec:
    for j in enumerate_jobs(cfg):
        if j.arm == arm and j.prompt.id == prompt_id:
            return j
    stop(f"no job for arm={arm!r} prompt={prompt_id!r}; known jobs: "
         f"{[j.job_id for j in enumerate_jobs(cfg)]}")


def describe_jobs(cfg: Config) -> str:
    """Human-readable enumeration, printed by preflight so it can be eyeballed."""
    lines = [f"{'#':>3}  {'JOB':32s} {'MODE':9s} {'TRIM':8s} {'DROP':>7s}  PROMPT"]
    for i, j in enumerate(enumerate_jobs(cfg), 1):
        drop, derived = resolve_drop_threshold(j.prompt, cfg.max_model_len, cfg.max_tokens)
        lines.append(
            f"{i:3d}  {j.job_id:32s} {j.prompt.mode:9s} {j.prompt.trim:8s} "
            f"{drop:7d}{'*' if derived else ' '} {j.prompt.path.relative_to(cfg.repo_root)}"
        )
    lines.append("     (* = derived as max_model_len - max_tokens, source: 09_Distill)")
    return "\n".join(lines)

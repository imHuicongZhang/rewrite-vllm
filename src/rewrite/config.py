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

# The canonical style list and its ORDER live in wrap_styles.py, which also owns the seeded
# assignment. Imported rather than duplicated: the order is part of the reproducible seed,
# so two copies that could drift apart would be a silent corpus corruption.
from .wrap_styles import WRAP_STYLES

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

# grounded   -> one prompt, [TEXT] interpolated. Nine of the ten jobs.
# wrap_multi -> ONE job carrying FOUR style prompts, one chosen per document by a seeded
#               RNG (src/rewrite/wrap_styles.py). wrap-inspired's styled pass only.
#               The old plain "wrap" mode -- one style per job, four jobs -- is gone with
#               the four-pass design. See docs/DESIGN_DELTA.md section 2.
VALID_MODES = {"grounded", "wrap_multi"}
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
class StyleSpec:
    """One of the four wrap styles. Only used by mode wrap_multi."""
    style: str                 # easy | hard | wiki | qa
    path: Path
    text: str
    expected_overhead: int


@dataclass(frozen=True)
class PromptSpec:
    id: str                    # "p1" | "p2"
    arm: str
    mode: str                  # grounded | wrap_multi
    trim: str                  # wiki | distill | wrap
    path: Path | None          # None for wrap_multi -- the text lives in `styles`
    text: str | None           # None for wrap_multi
    input_drop: int | None     # None => derive max_model_len - max_tokens
    expected_overhead: int | None   # None for wrap_multi -- per style, see `styles`
    # Four entries for wrap_multi, empty otherwise. ORDER IS PART OF THE SEED: the style
    # RNG draws an index into it. See src/rewrite/wrap_styles.py.
    styles: tuple = ()
    # Measured output/input token ratio in llama-2 tokens, and the resulting estimate.
    # Provenance: docs/DESIGN_DELTA.md section 5.
    r: float = 0.0
    est_output_tokens: int = 0

    def texts(self) -> list:
        """Every distinct prompt text this job can emit. One entry, or four for wrap_multi."""
        return [st.text for st in self.styles] if self.styles else [self.text]

    def overheads(self) -> list:
        """(label, text, expected_overhead) for every prompt text, for the parity gate."""
        if self.styles:
            return [(f"{self.id}:{st.style}", st.text, st.expected_overhead)
                    for st in self.styles]
        return [(self.id, self.text, self.expected_overhead)]


@dataclass(frozen=True)
class ArmSpec:
    """One rewritten arm. Every arm is rewritten -- there is no control arm any more.

    The raw side of the corpus (the shared 20B core and the 50B quality-base control) is
    deliberately NOT modelled here: neither consumes GPU time and neither is downloaded.
    Both are documented in the header comment of configs/data.yaml so the token accounting
    stays complete. See docs/DESIGN_DELTA.md section 3.
    """
    name: str
    subdir: str                # folder inside the single gated HF repo
    docs: int                  # remainder document count, as uploaded
    source_tokens_llama2: int  # remainder token count -- what the GPU consumes, x n_prompts
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

    def check_gpu_arch(self, cc: str, gpu_name: str = "") -> tuple:
        """Is this compute capability allowed for this workload?

        Returns (ok, level, message) where level is 'ok' | 'warn' | 'fail'.
        Enforced by preflight AND by every worker, because preflight can be skipped.
        """
        con = self.data["compute_constraints"]
        allowed = [str(a) for a in con["allowed_gpu_arch"]]
        majors = [int(m) for m in con.get("allowed_gpu_arch_major", [])]
        reason = con.get("reason", "")
        if cc in allowed:
            return True, "ok", f"{gpu_name} {cc} allowed"
        try:
            major = int(cc.split("_")[1][:-1]) if len(cc.split("_")[1]) > 1 \
                else int(cc.split("_")[1])
        except (IndexError, ValueError):
            major = -1
        if major in majors:
            return True, "warn", (
                f"{gpu_name} reports {cc}, which is not explicitly in "
                f"{allowed} but is the same architecture family. Allowed, but tell Wytro "
                f"what card this is.")
        return False, "fail", (
            f"{gpu_name} is {cc}, which this workload does not allow (allowed: {allowed}"
            + (f"; family {majors}" if majors else "") + f").\n"
            f"  Reason: {reason}\n"
            f"  Deselect this GPU in cluster.yaml compute.gpu_ids rather than widening the "
            f"allow-list. If you believe the list is wrong, ask Wytro -- changing it "
            f"changes what the experiment measures.")

    @property
    def text_column(self) -> str:
        return self.data["sharding"]["text_column"]

    # ---- the single gated input repo ----
    @property
    def hf_repo_id(self) -> str:
        return self.data["hf"]["repo_id"]

    @property
    def hf_revision(self) -> str:
        return self.data["hf"]["revision"]

    def hf_data_files(self, arm: str) -> str:
        """The explicit glob for one arm's folder inside the single repo.

        Addressed by glob and NOT by HF config name on purpose: the Hub card for this
        dataset declares no `configs:` block, so the auto-converter exposes a single config
        `default`/`train` globbing every folder. load_dataset(repo_id) would silently mix
        shared-core into all five arms. See docs/DESIGN_DELTA.md section 4.
        """
        return self.data["hf"]["data_files_template"].format(subdir=self.arm(arm).subdir)

    @property
    def est_output_tokens_total(self) -> int:
        """Summed per-arm estimate. Replaces the old est_output_tokens_per_arm scalar,
        which could not express a 2.8x spread across arms."""
        return sum(p.est_output_tokens for a in self.arms for p in a.prompts)

    @property
    def est_output_rows_total(self) -> int:
        return sum(a.docs * len(a.prompts) for a in self.arms)

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

    sa = cluster["compute"].get("shard_assignment", "dynamic")
    if sa not in ("dynamic", "static"):
        stop("cluster.yaml compute.shard_assignment must be 'dynamic' or 'static'")
    cluster["compute"]["shard_assignment"] = sa

    if cluster["runtime"]["output_compression"] not in ("zstd", "none"):
        stop("cluster.yaml runtime.output_compression must be 'zstd' or 'none'")
    if cluster["scheduler"]["kind"] not in ("bash", "slurm"):
        stop("cluster.yaml scheduler.kind must be 'bash' or 'slurm'")

    # ---- the single gated input repo ----
    hf = data.get("hf") or {}
    for k in ("repo_id", "revision", "data_files_template"):
        if not hf.get(k):
            stop(f"configs/data.yaml hf.{k} is required. The input is ONE HuggingFace "
                 "repo with one folder per arm, not six flat repos.")

    # ---- arms and prompts ----
    defs = data["prompt_defs"]
    arms = []
    seen_subdirs = {}
    for a in data["arms"]:
        for k in ("subdir", "docs", "source_tokens_llama2"):
            if a.get(k) in (None, ""):
                stop(f"arm {a['name']}: missing required key {k!r}. docs and "
                     "source_tokens_llama2 make the workload auditable from data.yaml "
                     "alone and feed the disk estimate.")
        if a["subdir"] in seen_subdirs:
            stop(f"arms {seen_subdirs[a['subdir']]!r} and {a['name']!r} both point at "
                 f"subdir {a['subdir']!r}; each arm must name a distinct folder")
        seen_subdirs[a["subdir"]] = a["name"]

        prompts = []
        for p in a["prompts"]:
            d = defs.get(p["def"])
            if d is None:
                stop(f"arm {a['name']}: prompt {p['id']} references unknown "
                     f"prompt_def {p['def']!r}")

            mode, trim = d["mode"], d["trim"]
            if mode not in VALID_MODES:
                stop(f"{a['name']}/{p['id']}: mode must be one of {sorted(VALID_MODES)}")
            if trim not in VALID_TRIMS:
                stop(f"{a['name']}/{p['id']}: trim must be one of {sorted(VALID_TRIMS)}")

            ppath = ptext = None
            styles = []

            if mode == "grounded":
                if not p.get("file"):
                    stop(f"{a['name']}/{p['id']}: mode 'grounded' requires a prompt file")
                ppath = repo_root / p["file"]
                if not ppath.exists():
                    stop(f"arm {a['name']}: prompt file not found: {ppath}")
                ptext = ppath.read_text()
                # The source's own startup guards, ported.
                # source: 07_rewrite/rewrite_worker.py:178-186
                n = ptext.count(TEXT_PLACEHOLDER)
                if n != 1:
                    stop(f"{a['name']}/{p['id']} ({ppath.name}): grounded prompt must "
                         f"contain exactly one {TEXT_PLACEHOLDER} placeholder, found {n}")
                if trim == "wrap":
                    stop(f"{a['name']}/{p['id']}: trim 'wrap' is only valid with "
                         "mode 'wrap_multi'")
            else:
                # ---- wrap_multi: ONE job, FOUR style prompts, one per document ----
                if p.get("file"):
                    stop(f"{a['name']}/{p['id']}: mode 'wrap_multi' takes its prompts "
                         "from prompt_defs.<def>.styles, not from a `file:` key")
                if trim != "wrap":
                    stop(f"{a['name']}/{p['id']}: mode 'wrap_multi' requires trim 'wrap'")
                spec_styles = d.get("styles") or []
                names = [st["style"] for st in spec_styles]
                if names != WRAP_STYLES:
                    stop(f"{a['name']}/{p['id']}: prompt_def {p['def']!r} styles are "
                         f"{names}, expected exactly {WRAP_STYLES} IN THAT ORDER. The "
                         "order is part of the reproducible seed -- the style RNG draws "
                         "an index into it, so reordering silently changes which document "
                         "gets which style (source: 07_rewrite/rewrite_worker.py:39).")
                for st in spec_styles:
                    spath = repo_root / st["file"]
                    if not spath.exists():
                        stop(f"arm {a['name']}: wrap style prompt file not found: {spath}")
                    stext = spath.read_text()
                    if not stext.endswith(WRAP_SUFFIX):
                        stop(f"{a['name']}/{p['id']} ({spath.name}): wrap prompt must end "
                             f"with {WRAP_SUFFIX!r} -- the document is concatenated "
                             "directly after it (source: rewrite_worker.py:50-51)")
                    if TEXT_PLACEHOLDER in stext:
                        stop(f"{a['name']}/{p['id']} ({spath.name}): wrap prompts "
                             f"concatenate the document; they must not contain "
                             f"{TEXT_PLACEHOLDER}")
                    styles.append(StyleSpec(style=st["style"], path=spath, text=stext,
                                            expected_overhead=int(st["expected_overhead"])))
                if len({st.text for st in styles}) != len(styles):
                    stop(f"{a['name']}/{p['id']}: two wrap styles have identical prompt "
                         "text; the style assignment would be unrecoverable")

            prompts.append(PromptSpec(
                id=p["id"], arm=a["name"], mode=mode, trim=trim, path=ppath, text=ptext,
                input_drop=(None if d["input_drop"] is None else int(d["input_drop"])),
                expected_overhead=(None if mode == "wrap_multi"
                                   else int(d["expected_overhead"])),
                styles=tuple(styles),
                r=float(p.get("r") or 0.0),
                est_output_tokens=int(p.get("est_output_tokens") or 0),
            ))

        if not prompts:
            stop(f"arm {a['name']} declares no prompts; every arm is rewritten")

        arms.append(ArmSpec(name=a["name"], subdir=a["subdir"],
                            docs=int(a["docs"]),
                            source_tokens_llama2=int(a["source_tokens_llama2"]),
                            prompts=tuple(prompts)))

    cfg = Config(repo_root=repo_root, cluster=cluster, data=data, vllm=vllm, env=env,
                 paths=paths, arms=tuple(arms))

    # ---- GPU architecture constraint ----
    cc = data.get("compute_constraints")
    if not cc or not cc.get("allowed_gpu_arch"):
        stop("configs/data.yaml must define compute_constraints.allowed_gpu_arch. It is "
             "part of the experiment definition, not an optional setting.")
    if not all(str(a).startswith("sm_") for a in cc["allowed_gpu_arch"]):
        stop("compute_constraints.allowed_gpu_arch entries must look like 'sm_100'")

    # ---- the job count is a hard assertion, not a comment ----
    n_jobs = sum(len(a.prompts) for a in arms)
    expected = int(data["expected_jobs"])
    if n_jobs != expected:
        stop(f"configs/data.yaml enumerates {n_jobs} rewrite jobs but expected_jobs is "
             f"{expected}. A typo here would silently skip a whole pass over an arm.")
    if not data.get("semantics", {}).get("full_coverage"):
        stop("configs/data.yaml semantics.full_coverage must be true: every prompt covers "
             "EVERY document of its arm exactly once. (This replaced full_cross_product in "
             "round 4: wrap-inspired's styled pass covers every document once using one of "
             "four styles per document, which is full coverage but not a cross product.)")
    return cfg


# --------------------------------------------------------------------------- jobs
def enumerate_jobs(cfg: Config) -> list:
    """The 10 jobs, in a stable order (arm order in data.yaml, then prompt id).

    Five arms x two prompts. wrap-inspired's p1 is ONE job that emits four different style
    prompts, one per document -- not four jobs. See docs/DESIGN_DELTA.md section 2.
    """
    jobs = []
    for a in cfg.arms:
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
    lines = [f"{'#':>3}  {'JOB':28s} {'MODE':11s} {'TRIM':8s} {'DROP':>7s}  PROMPT"]
    for i, j in enumerate(enumerate_jobs(cfg), 1):
        drop, derived = resolve_drop_threshold(j.prompt, cfg.max_model_len, cfg.max_tokens)
        if j.prompt.styles:
            # One job, four prompts. Show every one -- a reader must be able to see that
            # this single line of the table is where four style prompts live.
            head = (f"{i:3d}  {j.job_id:28s} {j.prompt.mode:11s} {j.prompt.trim:8s} "
                    f"{drop:7d}{'*' if derived else ' '} "
                    f"{len(j.prompt.styles)} styles, one per document:")
            lines.append(head)
            for st in j.prompt.styles:
                lines.append(f"{'':>3}  {'':28s} {'':11s} {'':8s} {'':>7s}   "
                             f"{st.style:5s} {st.path.relative_to(cfg.repo_root)}")
        else:
            lines.append(
                f"{i:3d}  {j.job_id:28s} {j.prompt.mode:11s} {j.prompt.trim:8s} "
                f"{drop:7d}{'*' if derived else ' '} "
                f"{j.prompt.path.relative_to(cfg.repo_root)}")
    lines.append("     (* = derived as max_model_len - max_tokens, source: 09_Distill)")
    return "\n".join(lines)

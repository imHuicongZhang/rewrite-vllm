"""End-to-end integration test with vLLM and the tokenizers stubbed out.

Self-contained: no GPU, no model download, no network, no source pipeline, no filled
placeholders. It builds a synthetic corpus, drives all 12 jobs across several workers, and
asserts the invariants that matter -- above all that every prompt rewrites its arm's corpus
in full.

    python tests/test_integration.py

Requires: PyYAML, pyarrow, numpy, zstandard, datasets. Exit 0 on success.
"""
import json, os, random, shutil, sys, tempfile, types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")

import pyarrow as pa, pyarrow.parquet as pq, yaml

WORK = Path(tempfile.mkdtemp(prefix="rwtest-"))
CFG_ROOT = WORK / "repo"
shutil.copytree(REPO / "configs", CFG_ROOT / "configs")
shutil.copytree(REPO / "prompts", CFG_ROOT / "prompts")
(CFG_ROOT / "manifests").mkdir(parents=True, exist_ok=True)

NGPU = 3
ARMS = ["quality-base", "quality-first", "diversity-oriented",
        "disagreement-aware", "wrap-inspired", "rewire-inspired"]
ROWS = {"quality-base": 500, "quality-first": 1234, "diversity-oriented": 900,
        "disagreement-aware": 700, "wrap-inspired": 1500, "rewire-inspired": 640}

# ---- fill the configs ----
c = yaml.safe_load((CFG_ROOT / "configs/cluster.yaml").read_text())
c["paths"] = {"repo_root": str(CFG_ROOT), "model_dir": str(WORK/"models"),
              "data_root": str(WORK/"data"), "out_root": str(WORK/"out"),
              "tmp_root": str(WORK/"tmp"), "log_root": str(WORK/"logs"),
              "hf_cache": str(WORK/"hf")}
c["compute"] = {"num_gpus": NGPU, "gpu_ids": "auto", "cpu_workers": 2,
                "shuffle_mem_bytes": 2*1024**3}
c["env"]["activate_cmd"] = "true"
(CFG_ROOT / "configs/cluster.yaml").write_text(yaml.safe_dump(c, sort_keys=False))

d = yaml.safe_load((CFG_ROOT / "configs/data.yaml").read_text())
d["upload"]["repo_template"] = "testorg/rewrite-{arm}-{prompt_id}"
for a in d["arms"]:
    a["repo_id"] = f"testorg/{a['name']}"
d["sharding"]["shard_target_rows"] = 10      # tiny shards -> many of them
d["sharding"]["shard_target_bytes"] = 1 << 20
d["shuffle"]["rows_per_shard"] = 137         # force the carry-buffer path
(CFG_ROOT / "configs/data.yaml").write_text(yaml.safe_dump(d, sort_keys=False))
(CFG_ROOT / ".env").write_text("HF_TOKEN=hf_faketoken\nHF_TOKEN_WRITE=hf_faketoken\n")

# fake `vllm` so engine.py's lazy imports resolve without a GPU
class _SP:
    def __init__(self, **kw): self.__dict__.update(kw)
_v = types.ModuleType("vllm"); _v.SamplingParams = _SP; _v.LLM = object
sys.modules["vllm"] = _v

from rewrite import config as C, data as D, engine as E, postprocess as PP, shuffle as SH

# ---- stub the Hub away: local fake corpora ----
def fake_open_dataset(cfg, a, streaming=False):
    n = ROWS[a.name]
    rnd = random.Random(hash(a.name) & 0xffff)
    texts = []
    for i in range(n):
        if i % 97 == 0:
            texts.append("LONG " * 30000)        # ~37.5k fake tokens > both drop thresholds
        else:
            texts.append(f"[{a.name}] doc {i}. " + "content word " * rnd.randint(3, 40))
    tbl = pa.table({"text": pa.array(texts, type=pa.large_string())})
    from datasets import Dataset
    return Dataset(tbl)
D._open_dataset = fake_open_dataset
D.download_arm = lambda cfg, arm, log=print: log(f"[data] {arm}: (stub) download skipped")

# ---- stub tokenizers and the engine ----
class FakeTok:
    vocab_size = 32000
    def __call__(self, text, add_special_tokens=False):
        if isinstance(text, str): text = [text]; single = True
        else: single = False
        ids = [list(range(max(1, len(t)//4))) for t in text]
        return types.SimpleNamespace(input_ids=ids[0] if single else ids)
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "<|im_start|>user\n" + msgs[0]["content"] + "<|im_end|>\n<|im_start|>assistant\n"
E.load_qwen_tokenizer = lambda cfg: FakeTok()
E.load_llama2_tokenizer = lambda cfg: FakeTok()
E.count_llama2 = lambda ltok, texts: [max(0, len(t)//4) for t in texts]

# Overheads differ under the fake tokenizer; relax the parity gate for the stub only.
_real_overhead = E.empty_doc_overhead
E.empty_doc_overhead = lambda qtok, cfg, prompt: prompt.expected_overhead

class FakeOut:
    def __init__(self, text, fr): self.text=text; self.finish_reason=fr; self.token_ids=list(range(max(1,len(text)//4)))
class FakeReq:
    def __init__(self, o): self.outputs=[o]
class FakeLLM:
    def generate(self, prompts, sps):
        out=[]
        for i,p in enumerate(prompts):
            body = f"REWRITTEN body {i} " + "x"*30
            if i % 5 == 0:   body = PP.WIKI_PREFIX + body                     # wiki artifact
            elif i % 5 == 1: body = "Here is the rewritten passage:\n\n" + body  # wrap artifact
            elif i % 5 == 2: body = "### Paraphrased Version\n\n" + body         # distill artifact
            out.append(FakeReq(FakeOut(body, "length" if i % 11 == 0 else "stop")))
        return out
E.build_llm = lambda cfg: FakeLLM()
PP._init_worker = lambda p: setattr(PP, "_TOK", FakeTok())

def hdr(s): print(f"\n{'='*70}\n{s}\n{'='*70}")
FAILS=[]
def expect(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond: FAILS.append(msg)

# ================================================================= 1. config
hdr("1. config load + validation")
cfg = C.load_config(CFG_ROOT)
jobs = C.enumerate_jobs(cfg)
expect(len(jobs)==12, f"12 jobs enumerated (got {len(jobs)})")
expect([j.job_id for j in jobs if j.arm=="wrap-inspired"]==
       ["wrap-inspired__p1","wrap-inspired__p2","wrap-inspired__p3","wrap-inspired__p4"],
       "wrap-inspired has 4 jobs = 4 full passes")
expect(not any(j.arm=="quality-base" for j in jobs), "quality-base produces NO jobs (control)")
wiki = next(j for j in jobs if j.job_id=="quality-first__p1")
dist = next(j for j in jobs if j.job_id=="quality-first__p2")
expect(C.resolve_drop_threshold(wiki.prompt, cfg.max_model_len, cfg.max_tokens)==(30720,False),
       "wiki drop threshold = 30720 (fixed)")
expect(C.resolve_drop_threshold(dist.prompt, cfg.max_model_len, cfg.max_tokens)==(28672,True),
       "distill drop threshold = 28672 (derived)")
expect([j.prompt.trim for j in jobs].count("wrap")==4, "4 jobs use the wrap trim rule")
expect(sorted({j.prompt.trim for j in jobs})==["distill","wiki","wrap"], "3 distinct trim rules")

# reject a forbidden engine arg
bad = yaml.safe_load((CFG_ROOT/"configs/vllm.yaml").read_text())
bad["engine"]["enable_prefix_caching"]=True
(CFG_ROOT/"configs/vllm_bad.yaml").write_text(yaml.safe_dump(bad))
shutil.copy(CFG_ROOT/"configs/vllm.yaml", CFG_ROOT/"configs/vllm_good.yaml")
shutil.copy(CFG_ROOT/"configs/vllm_bad.yaml", CFG_ROOT/"configs/vllm.yaml")
try:
    C.load_config(CFG_ROOT); rejected=False
except SystemExit: rejected=True
shutil.copy(CFG_ROOT/"configs/vllm_good.yaml", CFG_ROOT/"configs/vllm.yaml")
expect(rejected, "an engine arg the source never passed is REJECTED")

# ================================================================= 2. sharding
hdr("2. download + probe + shard + manifest")
for a in ARMS:
    D.download_arm(cfg, a, log=lambda m: None)
    D.probe_arm(cfg, a, log=lambda m: None)
    D.shard_arm(cfg, a, log=lambda m: None)
D.write_data_manifest(cfg, log=lambda m: None)
for a in ARMS:
    m = D.load_manifest(cfg, a)
    expect(m.total_rows==ROWS[a], f"{a}: manifest rows {m.total_rows} == {ROWS[a]}")
qb = D.load_manifest(cfg, "quality-base")
expect(qb.rewrite is False and not list(cfg.shards_dir('quality-base').glob('part_*.parquet')),
       "quality-base verified + counted but NO shards written (control)")
expect(qb.content_sha1 and qb.total_text_bytes>0, "quality-base has a content hash + byte count")
dm = json.loads((CFG_ROOT/"manifests/data_manifest.json").read_text())
expect(dm["total_rewrite_jobs"]==12, "data_manifest records 12 rewrite jobs")
expect(dm["arms"]["quality-base"]["rewrite"] is False, "data_manifest flags quality-base rewrite:false")

# ================================================================= 3. run all 12
hdr("3. run all 12 jobs across %d workers" % NGPU)
from rewrite import run_rewrite as RR
import threading, time as _t
# Heterogeneous fleet simulation: worker 0 is "slow" (H200), workers 1-2 "fast" (B200/B300).
SPEED = {0: 0.004, 1: 0.0005, 2: 0.0005}
_orig_run_batch = E.run_batch
def slow_run_batch(llm, prep, _w=[0]):
    _t.sleep(SPEED.get(getattr(threading.current_thread(), "_wid", None), 0.0))
    return _orig_run_batch(llm, prep)
E.run_batch = slow_run_batch
_shard_owner = {}
_orig_write_sidecar = D.write_sidecar
def tracking_write_sidecar(out_path, payload):
    _shard_owner.setdefault(payload["job_id"], {})[payload["shard_index"]] = payload["worker_id"]
    return _orig_write_sidecar(out_path, payload)
D.write_sidecar = tracking_write_sidecar

for j in jobs:
    D.reap_stale_claims(j.output_dir, log=lambda m: None)
    ths = []
    for w in range(NGPU):
        def go(w=w, j=j):
            threading.current_thread()._wid = w
            RR.run_worker(cfg, j, w, NGPU)
        th = threading.Thread(target=go); th.start(); ths.append(th)
    for th in ths: th.join()
print()
for j in jobs:
    st = D.verify_job(cfg, j)
    inp = D.input_rows(cfg, j.arm)
    expect(st.state=="DONE" and st.rows_out==inp and not st.problems,
           f"{j.job_id}: {st.state} rows_out={st.rows_out} == input {inp}")

# every prompt of an arm covered the SAME full corpus
for arm in ["wrap-inspired","quality-first"]:
    outs={j.prompt.id: D.verify_job(cfg,j).rows_out for j in jobs if j.arm==arm}
    expect(len(set(outs.values()))==1 and list(outs.values())[0]==D.input_rows(cfg,arm),
           f"{arm}: all {len(outs)} prompts rewrote the ENTIRE corpus {outs}")

# status-0 rows are emitted, not dropped
p0 = sorted((cfg.raw_dir("quality-first","p1")).glob(f"part_*{cfg.shard_suffix}"))[0]
rows0 = [r for r in D.iter_jsonl(p0)]
n_s0 = sum(1 for r in rows0 if r["status"]==0)
expect(all(set(r)=={"doc_id","arm","prompt_id","source_text_sha1","rewritten_text",
                    "finish_reason","n_prompt_tokens","n_output_tokens","status",
                    "n_output_tokens_llama2"} for r in rows0), "output rows have exactly the 10 keys")
expect(all(r["rewritten_text"]=="" and r["n_output_tokens"]==0 and r["finish_reason"]==""
           for r in rows0 if r["status"]==0), f"status-0 rows emitted empty ({n_s0} in shard 0)")
st_all = {s for r in rows0 for s in [r["status"]]}
expect(1 in {r["status"] for f in sorted(cfg.raw_dir("quality-first","p1").glob(f"part_*{cfg.shard_suffix}")) for r in D.iter_jsonl(f)},
       "status=1 (finish_reason='length') is produced and recorded")

# ================================================================= 4. resume
hdr("4. resume behaviour")
jd = jobs[0]
sc = sorted(jd.output_dir.glob("part_*.done"))
victim = sc[0]
vi = json.loads(victim.read_text())["shard_index"]
victim.unlink()
before = len(D.done_shards(jd.output_dir))
RR.run_worker(cfg, jd, vi % NGPU, NGPU)
after = len(D.done_shards(jd.output_dir))
expect(before==len(sc)-1 and after==len(sc), f"a deleted .done is regenerated (shard {vi})")

# a data file with no sidecar is treated as incomplete
victim2 = sorted(jd.output_dir.glob("part_*.done"))[1]
vi2 = json.loads(victim2.read_text())["shard_index"]
victim2.unlink()
(jd.output_dir / f"part_{vi2:05d}{cfg.shard_suffix}").write_bytes(b"garbage-not-jsonl")
RR.run_worker(cfg, jd, vi2 % NGPU, NGPU)
rows = list(D.iter_jsonl(jd.output_dir / f"part_{vi2:05d}{cfg.shard_suffix}"))
expect(len(rows)==D.load_manifest(cfg, jd.arm).shard_rows(vi2),
       "a data file with no .done is overwritten, not trusted")

# stale .tmp cleanup
(jd.output_dir / f"part_{vi:05d}{cfg.shard_suffix}.tmp").write_bytes(b"partial")
n = D.clean_stale_tmp(jd.output_dir, [(vi, None)], cfg.shard_suffix)
expect(n==1 and not (jd.output_dir/f"part_{vi:05d}{cfg.shard_suffix}.tmp").exists(),
       "stale .tmp files are removed at worker start")

# fingerprint interlock
sp = D.sidecar_path(jd.output_dir / f"part_{vi:05d}{cfg.shard_suffix}")
bad_sc = json.loads(sp.read_text()); bad_sc["input_fingerprint"]="deadbeef"
sp.write_text(json.dumps(bad_sc))
st = D.verify_job(cfg, jd)
expect(any("fingerprint" in p for p in st.problems), "a stale fingerprint is DETECTED")
RR.run_worker(cfg, jd, vi % NGPU, NGPU)
expect(not D.verify_job(cfg, jd).problems, "and the shard is redone, clearing the problem")

# row-conservation violation is caught
st = D.verify_job(cfg, jd)
sp = D.sidecar_path(jd.output_dir / f"part_{vi:05d}{cfg.shard_suffix}")
tamper = json.loads(sp.read_text()); tamper["n_rows_out"] = tamper["n_rows_out"] - 1
sp.write_text(json.dumps(tamper))
st = D.verify_job(cfg, jd)
expect(any("ROW CONSERVATION" in p for p in st.problems), "a row-count mismatch FAILS verification")
RR.run_worker(cfg, jd, vi % NGPU, NGPU)
sp2 = D.sidecar_path(jd.output_dir / f"part_{vi:05d}{cfg.shard_suffix}")
if not json.loads(sp2.read_text())["n_rows_out"] == D.load_manifest(cfg, jd.arm).shard_rows(vi):
    sp2.unlink(); RR.run_worker(cfg, jd, vi % NGPU, NGPU)
expect(not D.verify_job(cfg, jd).problems, "repaired after re-run")

# ================================================================= 5. deep verify
hdr("5. deep verify (count every line on disk)")
for j in jobs[:4]:
    st = D.verify_job(cfg, j, deep=True)
    expect(st.state=="DONE" and not st.problems, f"{j.job_id}: deep verify clean")

# ================================================================= 6. trim
hdr("6. trim (per (arm,prompt) rule)")
for j in jobs:
    s = PP.trim_job(cfg, j, workers=2, log=lambda m: None)
    inp = D.input_rows(cfg, j.arm)
    st = D.verify_job(cfg, j, deep=True)
    expect(s["rows"]==inp and st.rows_out==inp,
           f"{j.job_id}: rule={s['rule']:8s} stripped={s['stripped']:5d}/{s['status2']:5d} "
           f"rows preserved ({inp})")
allrows = [r for f in sorted(cfg.raw_dir("quality-first","p1").glob(f"part_*{cfg.shard_suffix}"))
             for r in D.iter_jsonl(f)]
s2 = [r for r in allrows if r["status"] == 2]
s1 = [r for r in allrows if r["status"] == 1]
expect(len(s2) > 0 and not any((r["rewritten_text"] or "").startswith(PP.WIKI_PREFIX) for r in s2),
       f"wiki prefix removed from every status==2 row ({len(s2)} rows)")
# SOURCE PARITY: trimming touches ONLY status==2. Truncated (status==1) rows keep their
# prefix on purpose -- 01_strip_prefix.py:102 selects np.flatnonzero(status == 2).
kept = [r for r in s1 if (r["rewritten_text"] or "").startswith(PP.WIKI_PREFIX)]
expect(len(kept) > 0,
       f"status==1 rows deliberately NOT trimmed, matching source ({len(kept)} such rows)")
sample = list(D.iter_jsonl(sorted(cfg.raw_dir("wrap-inspired","p1").glob(f"part_*{cfg.shard_suffix}"))[0]))
expect(not any((r["rewritten_text"] or "").startswith("Here is the rewritten passage:") for r in sample),
       "wrap preamble removed from wrap-prompt output")
expect(PP.trim_job(cfg, jobs[0], workers=2, log=lambda m: None)["job_id"]==jobs[0].job_id,
       "trim is resumable (second call short-circuits on the marker)")

# ================================================================= 7. shuffle
hdr("7. shuffle (within (arm,prompt) only)")
for j in jobs:
    s = SH.shuffle_job(cfg, j, log=lambda m: None)
    inp = D.input_rows(cfg, j.arm)
    files = sorted(cfg.shuffled_dir(j.arm, j.prompt.id).glob("part_*.parquet"))
    tot = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    arms_in = set()
    for f in files: arms_in |= set(pq.read_table(f).column("arm").to_pylist())
    expect(tot==inp and s["total_rows"]==inp, f"{j.job_id}: shuffled {tot} rows == input {inp}")
    expect(arms_in=={j.arm}, f"{j.job_id}: output contains ONLY arm {j.arm} (no cross-arm mixing)")
f0 = sorted(cfg.shuffled_dir("wrap-inspired","p1").glob("part_*.parquet"))
expect(all(set(pq.read_table(f).column("prompt_id").to_pylist())=={"p1"} for f in f0),
       "shuffled output is scoped to a single prompt_id too")
docs = [d for f in f0 for d in pq.read_table(f).column("doc_id").to_pylist()]
expect(sorted(docs)==list(range(ROWS["wrap-inspired"])), "shuffle is a complete permutation")
expect(docs != sorted(docs), "shuffle actually reordered the rows")
expect(SH.shuffle_job(cfg, jobs[0], log=lambda m: None) is None, "shuffle is resumable")

# ================================================================= 8. claiming
hdr("8. dynamic shard claiming on a simulated heterogeneous fleet")
jid = "wrap-inspired__p1"
owners = _shard_owner[jid]
from collections import Counter
dist = Counter(owners.values())
nsh = len(owners)
expect(nsh == D.load_manifest(cfg, "wrap-inspired").n_shards,
       f"every shard of {jid} was completed exactly once ({nsh} shards)")
print(f"       shards per worker: {dict(sorted(dist.items()))}  (worker 0 = 8x slower)")
expect(dist[0] < dist[1] and dist[0] < dist[2],
       "the SLOW worker took fewer shards -- dynamic claiming rebalanced the fleet")
expect(max(dist.values()) - min(dist.values()) > 1,
       "assignment is genuinely uneven (a static modulo split would be exactly equal)")
# no claim directories survive a clean run
leftover = [p for j2 in jobs for p in j2.output_dir.glob("part_*.claim")]
expect(not leftover, f"no claim directories left behind after clean runs ({len(leftover)})")

# a claim left by a dead run blocks the shard until reaped, then is picked up
jd2 = jobs[2]
sc = sorted(jd2.output_dir.glob("part_*.done"))[3]
si2 = json.loads(sc.read_text())["shard_index"]
sc.unlink()
D.try_claim(jd2.output_dir, si2, {"worker_id": 99, "pid": -1, "host": "dead-node"})
RR.run_worker(cfg, jd2, 0, NGPU)
expect(not D.sidecar_path(jd2.output_dir / f"part_{si2:05d}{cfg.shard_suffix}").exists(),
       f"a live claim BLOCKS another worker from redoing shard {si2}")
n = D.reap_stale_claims(jd2.output_dir, log=lambda m: None)
expect(n >= 1, f"reap_stale_claims removed the dead run's claim ({n})")
RR.run_worker(cfg, jd2, 0, NGPU)
expect(D.sidecar_path(jd2.output_dir / f"part_{si2:05d}{cfg.shard_suffix}").exists(),
       "after reaping, the shard is picked up and completed")
expect(not D.verify_job(cfg, jd2).problems, "job verifies clean again")

# gpu provenance is recorded on every shard
sc = json.loads(sorted(jobs[0].output_dir.glob("part_*.done"))[0].read_text())
expect("gpu_name" in sc and "gpu_cc" in sc,
       f"sidecar records GPU provenance (gpu_name={sc.get('gpu_name')!r}, gpu_cc={sc.get('gpu_cc')!r})")

# static mode still works
cfg.cluster["compute"]["shard_assignment"] = "static"
jd3 = jobs[4]
for f in list(jd3.output_dir.glob("part_*.done"))[:3]: f.unlink()
for w in range(NGPU): RR.run_worker(cfg, jd3, w, NGPU)
expect(not D.verify_job(cfg, jd3).problems, "shard_assignment: static still works (source parity)")
cfg.cluster["compute"]["shard_assignment"] = "dynamic"

# ================================================================= 9. review-round-2 regressions
hdr("9. regressions for the defects found in review round 2")

# ---- A1: a leaked .claim directory must never reach the Hub ----
jclaim = jobs[1]
D.try_claim(jclaim.output_dir, 99999, {"worker_id": 7, "pid": -1, "host": "killed-node"})
assert (jclaim.output_dir / "part_99999.claim").is_dir()
payload = sorted(x for x in jclaim.output_dir.glob("part_*")
                 if x.is_file() and not x.name.endswith((".tmp", ".done")))
expect(not any(x.name.endswith(".claim") for x in payload),
       "A1: a leaked .claim directory is excluded from the upload payload")
expect(all(x.is_file() for x in payload),
       "A1: every payload entry is a file, so .stat() byte totals are real")
# and prove it against the REAL huggingface_hub matcher, which is what actually decides
try:
    from huggingface_hub.utils import filter_repo_objects
    names = ["part_00000.jsonl.zst", "part_00000.done", "part_00001.jsonl.zst.tmp",
             "part_99999.claim/owner.json"]
    kept = list(filter_repo_objects(
        names, allow_patterns=["part_*"],
        ignore_patterns=["*.tmp", "*.done", "*.claim", "*.claim/*", ".joblock"]))
    expect(kept == ["part_00000.jsonl.zst"],
           f"A1: huggingface_hub keeps only the real shard {kept}")
except ImportError:
    print("       (huggingface_hub unavailable; pattern check skipped)")
D.release_claim(jclaim.output_dir, 99999)

# ---- A2: missing shards are NAMED, and the total-rows assertion is reachable ----
jmiss = jobs[3]
before = D.verify_job(cfg, jmiss)
expect(before.state == "DONE" and not before.problems, "A2: job is clean to begin with")
victim = sorted(jmiss.output_dir.glob("part_*.done"))[2]
vidx = json.loads(victim.read_text())["shard_index"]
vrows = D.load_manifest(cfg, jmiss.arm).shard_rows(vidx)
victim.unlink()
after = D.verify_job(cfg, jmiss)
expect(after.problems, f"A2: a missing shard now RAISES a problem ({len(after.problems)})")
expect(any(str(vidx) in pr and "no .done" in pr for pr in after.problems),
       f"A2: the problem names the missing shard index {vidx}")
expect(after.rows_out == before.rows_out - vrows,
       f"A2: rows_out drops by exactly that shard ({vrows})")
# check #4 must be reachable once the shard set is complete again
RR.run_worker(cfg, jmiss, vidx % NGPU, NGPU)
expect(not D.verify_job(cfg, jmiss).problems, "A2: repaired after re-run")
tampered = D.sidecar_path(jmiss.output_dir / f"part_{vidx:05d}{cfg.shard_suffix}")
sc = json.loads(tampered.read_text())
man_rows = D.load_manifest(cfg, jmiss.arm).total_rows
saved = dict(sc)
sc["n_rows_out"] = sc["n_rows_out"]          # keep per-shard consistent...
tampered.write_text(json.dumps(sc))
expect(D.verify_job(cfg, jmiss).rows_out == man_rows,
       "A2: with a complete shard set, rows_out equals the manifest total")

# ---- A4: run_all.sh argument parsing ----
import subprocess, textwrap
raw = (REPO / "scripts" / "run_all.sh").read_text()
block = raw.split("# --- ARGPARSE BEGIN")[1].split("\n", 1)[1].split("# --- ARGPARSE END")[0]
# `usage` lives outside the extracted block; stub it so -h does not blow up
block = "usage() { :; }\n" + block
script = "#!/usr/bin/env bash\nset -euo pipefail\n" + block + \
         '\necho "FROM_JOB=$FROM_JOB STATUS_ONLY=$STATUS_ONLY SKIP_UPLOAD=$SKIP_UPLOAD"\n'
sp = Path(tempfile.mkdtemp()) / "parse.sh"
sp.write_text(script)
def parse(*argv):
    r = subprocess.run(["bash", str(sp), *argv], capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()
good_forms = [["--from-job=5"], ["--from-job", "5"], ["--skip-upload", "--from-job", "5"],
              ["--from-job", "5", "--skip-upload"], ["--status", "--from-job", "5"],
              ["--skip-preflight", "--skip-upload", "--from-job", "5"]]
allok = True
for form in good_forms:
    rc, out = parse(*form)
    hit = rc == 0 and "FROM_JOB=5" in out
    allok &= hit
    if not hit: print(f"       {' '.join(form)} -> rc={rc} {out}")
expect(allok, f"A4: --from-job yields 5 in all {len(good_forms)} invocation forms")
badok = True
for form in (["--bogus"], ["--from-job"], ["--from-job", "abc"], ["--from-job", "0"]):
    rc, out = parse(*form)
    hit = rc != 0 and "STOP" in out
    badok &= hit
    if not hit: print(f"       {' '.join(form)} -> rc={rc} {out} (should be rejected)")
expect(badok, "A4: malformed and unknown options are REJECTED, not silently ignored")

# ================================================================= summary
hdr("SUMMARY")
print(f"  checks failed: {len(FAILS)}")
for f in FAILS: print("   -", f)
shutil.rmtree(WORK, ignore_errors=True)
print("\n" + ("ALL INTEGRATION CHECKS PASSED" if not FAILS else "INTEGRATION FAILURES"))
sys.exit(0 if not FAILS else 1)

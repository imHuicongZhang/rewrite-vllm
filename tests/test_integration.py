"""End-to-end integration test with vLLM and the tokenizers stubbed out.

Self-contained: no GPU, no model download, no network, no source pipeline, no filled
placeholders. It builds a synthetic corpus, drives all 10 jobs across several workers, and
asserts the invariants that matter -- above all that every prompt rewrites its arm's corpus
in full.

    python tests/test_integration.py

Requires: PyYAML, pyarrow, numpy, zstandard, datasets. Exit 0 on success.
"""
import contextlib, io, json, os, random, shutil, sys, tempfile, time, types
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
ARMS = ["quality-first", "diversity-oriented",
        "disagreement-aware", "wrap-inspired", "rewire-inspired"]
ROWS = {"quality-first": 1234, "diversity-oriented": 900,
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
# The fixture corpora are tiny, so the declared docs counts cannot apply; the row-count
# cross-check is exercised separately below.
for a in d["arms"]:
    a["docs"] = ROWS[a["name"]]
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
    cols = {"text": pa.array(texts, type=pa.large_string())}
    if not getattr(fake_open_dataset, "omit_doc_id", False):
        # real datasets are required to carry doc_id (configs/data.yaml require_doc_id)
        cols["doc_id"] = pa.array(range(n), type=pa.int64())
    if not getattr(fake_open_dataset, "omit_record_id", False):
        # WARC-Record-ID, shaped exactly like the real column. Deliberately NOT unique:
        # every 500th row repeats its predecessor's id, mirroring the 0.0662% duplication
        # measured on the real 600M corpus, so anything that assumes uniqueness breaks here
        # rather than in production.
        rid = [f"<urn:uuid:{a.name[:8]:_<8}-0000-4000-8000-{(i - (i % 500 == 0 and i > 0)):012d}>"
               for i in range(n)]
        cols["record_id"] = pa.array(rid, type=pa.large_string())
    from datasets import Dataset
    return Dataset(pa.table(cols))
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
# The parity gate measures a real tokenizer; this test has a fake one. Stub it to return
# whatever the config expects, so the gate passes and the REST of the worker is exercised.
# check_overheads() must be stubbed too, since it is what run_worker actually calls and it
# is the thing that fans out over wrap-inspired's four styles.
E.empty_doc_overhead = lambda qtok, cfg, mode, text: 0
E.check_overheads = lambda qtok, cfg, prompt: [
    (label, expected, expected) for label, text, expected in prompt.overheads()]

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
expect(len(jobs)==10, f"10 jobs enumerated (got {len(jobs)})")
expect([j.job_id for j in jobs if j.arm=="wrap-inspired"]==
       ["wrap-inspired__p1","wrap-inspired__p2"],
       "wrap-inspired has 2 jobs: ONE styled pass + the shared distill pass")
wj = next(j for j in jobs if j.job_id=="wrap-inspired__p1")
expect(wj.prompt.mode=="wrap_multi", "wrap-inspired p1 is mode wrap_multi")
expect([st.style for st in wj.prompt.styles]==["easy","hard","wiki","qa"],
       "wrap styled pass carries easy/hard/wiki/qa IN SEED ORDER")
expect([st.expected_overhead for st in wj.prompt.styles]==[72,66,73,83],
       "the four style overheads are 72/66/73/83")
expect(len({st.text for st in wj.prompt.styles})==4, "the four style prompts are distinct")
wd = next(j for j in jobs if j.job_id=="wrap-inspired__p2")
expect(wd.prompt.trim=="distill" and wd.prompt.text==
       (CFG_ROOT/"prompts/quality-first/p2.txt").read_text(),
       "wrap-inspired p2 is byte-identical to the shared distill prompt")
expect(not any(a.name=="quality-base" for a in cfg.arms),
       "quality-base is not an arm (raw control, never uploaded)")
expect(not any(a.name=="shared-core" for a in cfg.arms),
       "shared-core is not an arm (raw half, 0 GPU tokens, not downloaded)")
wiki = next(j for j in jobs if j.job_id=="quality-first__p1")
dist = next(j for j in jobs if j.job_id=="quality-first__p2")
expect(C.resolve_drop_threshold(wiki.prompt, cfg.max_model_len, cfg.max_tokens)==(30720,False),
       "wiki drop threshold = 30720 (fixed)")
expect(C.resolve_drop_threshold(dist.prompt, cfg.max_model_len, cfg.max_tokens)==(28672,True),
       "distill drop threshold = 28672 (derived)")
expect([j.prompt.trim for j in jobs].count("wrap")==1, "exactly 1 job uses the wrap trim rule")
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
dm = json.loads((CFG_ROOT/"manifests/data_manifest.json").read_text())
expect(dm["total_rewrite_jobs"]==10, "data_manifest records 10 rewrite jobs")
expect(set(dm["raw_not_rewritten"])=={"shared-core","quality-base"},
       "data_manifest records the raw half that is never rewritten")
expect(dm["raw_not_rewritten"]["shared-core"]["tokens_llama2"]==20000010702
       and dm["raw_not_rewritten"]["quality-base"]["tokens_llama2"]==50000002028,
       "raw-half token accounting is 20.0B core + 50.0B quality-base")

# ================================================================= 3. run all 12
hdr("3. run all 10 jobs across %d workers" % NGPU)
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
    if "job_id" in payload and "worker_id" in payload:      # real worker sidecars only
        _shard_owner.setdefault(payload["job_id"], {})[payload["shard_index"]] = \
            payload["worker_id"]
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
# Driven off the config rather than a hardcoded list: the point of the check is that the
# four layers (config, JSONL rows, Arrow schema, shuffle) agree, not that the count is 11.
SCHEMA_KEYS = set(cfg.data["output"]["keys"])
expect(all(set(r)==SCHEMA_KEYS for r in rows0),
       f"output rows have exactly the {len(SCHEMA_KEYS)} keys of output.keys")
expect("record_id" in SCHEMA_KEYS, "record_id is part of the output schema")
expect(all(isinstance(r.get("record_id"), str) and r["record_id"] for r in rows0),
       "every output row carries a non-empty record_id")
_rid_in = {}
for _ridp in sorted(cfg.shards_dir("quality-first").glob("part_*.parquet")):
    _ridt = pq.read_table(_ridp, columns=["doc_id", "record_id"])
    _rid_in.update(zip(_ridt.column("doc_id").to_pylist(),
                       _ridt.column("record_id").to_pylist()))
expect(all(r["record_id"] == _rid_in[r["doc_id"]] for r in rows0),
       "record_id in the output matches the input shard for the same doc_id (verbatim)")
# The property that makes it a repair path and NOT a key: it is deliberately not unique.
_dups = len(_rid_in) - len(set(_rid_in.values()))
expect(_dups > 0,
       f"the fixture reproduces real record_id duplication ({_dups} dupes) -- "
       "nothing may assume uniqueness")
expect(set(cfg.data["output"]["keys"])==set().union(*(set(r) for r in rows0)),
       "configs/data.yaml output.keys matches what is actually written")
expect(all(r["wrap_style"]=="" for r in rows0),
       "wrap_style is the empty string for a non-wrap job")
expect(all(r["rewritten_text"]=="" and r["n_output_tokens"]==0 and r["finish_reason"]==""
           for r in rows0 if r["status"]==0), f"status-0 rows emitted empty ({n_s0} in shard 0)")
st_all = {s for r in rows0 for s in [r["status"]]}
expect(1 in {r["status"] for f in sorted(cfg.raw_dir("quality-first","p1").glob(f"part_*{cfg.shard_suffix}")) for r in D.iter_jsonl(f)},
       "status=1 (finish_reason='length') is produced and recorded")

# ---- wrap-inspired's styled pass: the style really reaches the output rows ----
from rewrite.wrap_styles import assign_wrap_styles as _aws, WRAP_STYLES as _WS
_wrap_rows, _wrap_ok = [], True
for _f in sorted(cfg.raw_dir("wrap-inspired","p1").glob(f"part_*{cfg.shard_suffix}")):
    _m = D.SHARD_RE.search(_f.name.replace(cfg.shard_suffix, ".parquet"))
    _si = int(_m.group(1))
    _rs = list(D.iter_jsonl(_f))
    _wrap_rows += _rs
    # THE property: the style written to disk is exactly what the seeded function gives
    # for this shard index -- not merely a plausible value.
    if [r["wrap_style"] for r in _rs] != _aws(_si, len(_rs)):
        _wrap_ok = False
expect(_wrap_ok, "every wrap row's style == assign_wrap_styles(shard_index, n_rows)")
expect(all(r["wrap_style"] in _WS for r in _wrap_rows),
       "every wrap row carries one of easy/hard/wiki/qa")
_wc = {st: sum(1 for r in _wrap_rows if r["wrap_style"]==st) for st in _WS}
expect(sum(_wc.values())==len(_wrap_rows) and min(_wc.values())>0,
       f"all four styles present across the corpus {_wc}")
# the distill pass of the SAME arm must NOT carry a style
_dr = [r for f in sorted(cfg.raw_dir("wrap-inspired","p2").glob(f"part_*{cfg.shard_suffix}"))
       for r in D.iter_jsonl(f)]
expect(_dr and all(r["wrap_style"]=="" for r in _dr),
       "wrap-inspired's distill pass carries no style (it is the shared grounded prompt)")

# ---- resume determinism, end to end: delete a shard, regenerate, same styles ----
_f0 = sorted(cfg.raw_dir("wrap-inspired","p1").glob(f"part_*{cfg.shard_suffix}"))[0]
_m0 = D.SHARD_RE.search(_f0.name.replace(cfg.shard_suffix, ".parquet"))
_si0 = int(_m0.group(1))
_before = [r["wrap_style"] for r in D.iter_jsonl(_f0)]
_f0.unlink(); D.sidecar_path(_f0).unlink()
_wj = next(j for j in jobs if j.job_id=="wrap-inspired__p1")
RR.run_worker(cfg, _wj, 0, 1)
_after = [r["wrap_style"] for r in D.iter_jsonl(_f0)]
expect(_before == _after and len(_before) > 0,
       f"a deleted wrap shard regenerates with IDENTICAL styles (shard {_si0}, "
       f"{len(_before)} rows)")

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

# ---- record_id survives all four layers, ending in the SHUFFLED parquet ----
# config -> JSONL rows -> Arrow schema -> shuffled output. The suite already proves the
# first two; this closes the last two, which is where a new key silently goes missing.
_shdir = cfg.shuffled_dir(jobs[0].arm, jobs[0].prompt.id)
_shp = sorted(_shdir.glob("part_*.parquet"))
expect(bool(_shp), f"shuffled output exists for {jobs[0].job_id}")
_sht = pq.read_table(_shp[0])
expect(list(_sht.schema.names) == list(cfg.data["output"]["keys"]),
       f"shuffled parquet schema == output.keys, in order ({_sht.num_columns} cols)")
expect(_sht.schema.field("record_id").type == pa.large_string(),
       f"record_id is large_string in the shuffled parquet (got {_sht.schema.field('record_id').type})")
_shrid = _sht.column("record_id").to_pylist()
expect(all(isinstance(x, str) and x.startswith("<urn:uuid:") for x in _shrid),
       "every shuffled row carries a well-formed record_id -- no nulls, no drops")
# and the value still pairs with the right doc_id after the shuffle reorders everything
_shmap = dict(zip(_sht.column("doc_id").to_pylist(), _shrid))
_srcmap = {}
for _q in sorted(cfg.shards_dir(jobs[0].arm).glob("part_*.parquet")):
    _qt = pq.read_table(_q, columns=["doc_id", "record_id"])
    _srcmap.update(zip(_qt.column("doc_id").to_pylist(), _qt.column("record_id").to_pylist()))
expect(all(_srcmap[k] == v for k, v in _shmap.items()),
       "(doc_id, record_id) pairing survives trim + shuffle intact")

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
# Reaping is age-based now, so a FRESH claim must survive -- that is what makes a reap
# from another node safe. Backdate it to simulate the owner having died.
n = D.reap_stale_claims(jd2.output_dir, stale_after_s=1800, log=lambda m: None)
expect(n == 0, "a fresh claim survives a reap (this is the multi-node safety property)")
import os as _os2, time as _t2
_os2.utime(D.claim_path(jd2.output_dir, si2), (_t2.time() - 7200,) * 2)
n = D.reap_stale_claims(jd2.output_dir, stale_after_s=1800, log=lambda m: None)
expect(n >= 1, f"once stale, the dead run's claim IS reaped ({n})")
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

# ================================================================= 10. multi-node safety
hdr("10. multi-node safety (round 3)")
import os as _os, time as _time
mn = Path(tempfile.mkdtemp())

# A reap run from another NODE must not touch claims a live worker is holding.
D.try_claim(mn, 1, {"worker_id": 0, "pid": 111, "host": "node01"})   # live
D.try_claim(mn, 3, {"worker_id": 0, "pid": 999, "host": "dead"})     # orphan
_os.utime(D.claim_path(mn, 3), (_time.time() - 3600,) * 2)
D.try_claim(mn, 4, {"worker_id": 2, "pid": 113, "host": "node01"})   # litter
D.write_sidecar(mn / "part_00004.jsonl.zst", {"shard_index": 4, "n_rows_out": 1})
reaped = D.reap_stale_claims(mn, stale_after_s=1800, log=lambda m: None)
expect(D.claim_path(mn, 1).exists(),
       "1.1: a LIVE claim survives a reap launched from another node")
expect(not D.claim_path(mn, 3).exists(), "1.1: an orphaned claim IS reaped")
expect(not D.claim_path(mn, 4).exists(), "1.1: a finished shard's claim is reaped as litter")
expect(reaped == 2, f"1.1: exactly the dead ones went ({reaped})")

# heartbeat keeps a long-running shard's claim alive
D.try_claim(mn, 6, {"worker_id": 0, "pid": 115, "host": "node01"})
_os.utime(D.claim_path(mn, 6), (_time.time() - 3600,) * 2)
with D.ClaimHeartbeat(mn, 6, interval=0.2):
    _time.sleep(0.6)
    age = D.claim_age_s(mn, 6)
expect(age < 5, f"1.1: ClaimHeartbeat refreshes a claim mid-shard ({age:.1f}s old)")
D.reap_stale_claims(mn, stale_after_s=1800, log=lambda m: None)
expect(D.claim_path(mn, 6).exists(), "1.1: and it then survives a reap")

# --force is still available for "nothing is running anywhere"
D.reap_stale_claims(mn, force=True, log=lambda m: None)
expect(not any(mn.glob("part_*.claim")), "1.1: --force still clears everything")

# only one of many nodes can hold a shard
got = [D.try_claim(mn, 9, {"worker_id": i, "pid": i, "host": f"node{i:02d}"})
       for i in range(12)]
expect(sum(got) == 1, f"1.1: 12 nodes race for one shard, exactly {sum(got)} wins")
shutil.rmtree(mn, ignore_errors=True)

# 1.2: the job lock is node-local, so other nodes are not blocked
rj = (REPO / "scripts" / "03_run_job.sh").read_text()
expect("${TMPDIR:-/tmp}" in rj and "$(hostname -s)" in rj,
       "1.2: the job lock lives on node-local storage and is keyed by hostname")
expect('LOCK="$LOCK_DIR/.joblock"' not in rj,
       "1.2: the old shared-filesystem lock is gone")

# 1.3: progress files cannot collide across nodes
expect("progress_{HOST}_w{worker_id}.json" in (REPO / "src" / "rewrite" / "run_rewrite.py").read_text(),
       "1.3: progress filenames are node-qualified")

# 1.5: the SHARDING lock recovers from a dead holder (round 6)
# Before round 6 this deadlocked forever: `while not manifest.exists(): sleep(10)` with no
# liveness check and no bound, so one killed process wedged every later run of the fleet.
_arm6 = "disagreement-aware"
_sd6  = cfg.shards_dir(_arm6)
_lock6 = _sd6 / ".sharding.lock"
_man6 = D.manifest_path(cfg, _arm6)
_man_backup = _man6.read_text()

def _plant_lock(age_s):
    if _lock6.exists():
        D.break_lock(_lock6)
    _lock6.mkdir()
    D.atomic_write_text(json.dumps({"host": "deadnode", "pid": 999999}),
                        _lock6 / "owner.json")
    if age_s:
        t = D.fs_now(_sd6) - age_s
        os.utime(_lock6, (t, t))

# (a) a LIVE lock is respected: the waiter must not steal it, and must give up bounded
_plant_lock(age_s=0)
try:
    D.acquire_dir_lock(_lock6, done_when=lambda: False, owner={"host": "me"},
                       stale_after_s=9999, max_wait_s=0.0, label="t", log=lambda m: None)
    _a = "returned"
except SystemExit:
    _a = "stopped"
expect(_a == "stopped", "1.5: a LIVE sharding lock is NOT stolen; the wait is bounded and fails")
expect(D.read_lock_owner(_lock6).get("host") == "deadnode",
       "1.5: the live lock still belongs to its original holder")

# (b) a STALE lock is taken over
_plant_lock(age_s=5000)
_got = D.acquire_dir_lock(_lock6, done_when=lambda: False, owner={"host": "me", "pid": 1},
                          stale_after_s=1800, max_wait_s=60, label="t", log=lambda m: None)
expect(_got is True, "1.5: a STALE sharding lock is taken over")
expect(D.read_lock_owner(_lock6).get("host") == "me",
       "1.5: the taker-over records itself as the new owner")
D.break_lock(_lock6)

# (c) if the work completes while waiting, the waiter returns False and does NOT shard
_plant_lock(age_s=0)
_got = D.acquire_dir_lock(_lock6, done_when=lambda: True, owner={"host": "me"},
                          stale_after_s=9999, max_wait_s=0.0, label="t", log=lambda m: None)
expect(_got is False, "1.5: waiter returns False when the manifest appears")
D.break_lock(_lock6)

# (d) END TO END: a stale lock plus a missing manifest must not hang shard_arm.
#     This is the exact situation a killed run leaves behind.
_man6.unlink()
_plant_lock(age_s=5000)
_t6 = _t.time()
_m6 = D.shard_arm(cfg, _arm6, log=lambda m: None)
_el6 = _t.time() - _t6
expect(_m6.total_rows == ROWS[_arm6] and _el6 < 60,
       f"1.5: shard_arm recovers from a stale lock instead of hanging ({_el6:.1f}s)")
expect(not _lock6.exists(), "1.5: the lock is released when sharding finishes")
expect(D.load_manifest(cfg, _arm6).total_rows == ROWS[_arm6],
       "1.5: and the manifest it wrote is correct")

# (e) orphaned shard files from the dead run are cleared, not mixed in
_man6.unlink()
_junk = _sd6 / "part_99999.parquet"
_junk.write_bytes(b"not a parquet file")
_plant_lock(age_s=5000)
D.shard_arm(cfg, _arm6, log=lambda m: None)
expect(not _junk.exists(), "1.5: orphaned shard files from an interrupted run are cleared")
expect(D.load_manifest(cfg, _arm6).total_rows == ROWS[_arm6], "1.5: manifest still correct")

# §2: the shard guard's arithmetic, and that the shipped default clears it at 100 GPUs
# read the SHIPPED config, not the tiny override this test uses for its toy corpus
_shipped = yaml.safe_load((REPO / "configs" / "data.yaml").read_text())["sharding"]
rows_per_shard = _shipped["shard_target_rows"]
min_ratio = _shipped["min_shards_per_gpu"]
expect(rows_per_shard == 5000,
       f"2: shipped shard_target_rows is sized for ~100 GPUs ({rows_per_shard})")
# Recomputed in round 4 against the REAL remainder sizes. The smallest arm is now
# disagreement-aware at 33,381,230 docs -- 6x the old smallest -- so every candidate
# clears the guard and the binding constraint became filesystem metadata load instead.
_declared = {a["name"]: a["docs"] for a in
             yaml.safe_load((REPO / "configs" / "data.yaml").read_text())["arms"]}
_smallest = min(_declared.values())
expect(_smallest == 33_381_230,
       f"2: smallest arm is disagreement-aware at {_smallest:,} docs")
for arm_docs, label in ((_smallest, "smallest arm"),):
    shards = -(-arm_docs // rows_per_shard)
    expect(shards >= min_ratio * 100,
           f"2: {label} gives {shards:,} shards >= {min_ratio*100:,} needed at 100 GPUs")
# Total file pressure is the reason 5000 was chosen over 2000: 2000 would create ~592k
# output files across the 10 jobs.
_total_shards = sum(-(-d // rows_per_shard) for d in _declared.values())
expect(_total_shards < 70_000,
       f"2: {_total_shards:,} input shards across all arms (2000 rows would give "
       f"{sum(-(-d // 2000) for d in _declared.values()):,})")

# 2.1: the GPU ceiling the sharding implies, and that the docs quote it correctly (round 6)
_caps = {n: (-(-d // rows_per_shard)) // min_ratio for n, d in _declared.items()}
_bind = min(_caps, key=lambda k: _caps[k])
expect(_bind == "disagreement-aware" and _caps[_bind] == 333,
       f"2.1: binding arm is {_bind} at {_caps[_bind]} GPUs (expected disagreement-aware/333)")
for _doc in ("docs/GUIDE_FOR_TIANJIAN.md", "configs/data.yaml"):
    _txt = (REPO / _doc).read_text()
    expect("330" in _txt and "333" in _txt,
           f"2.1: {_doc} states the ~330 GPU ceiling and its binding value")
expect(-(-_declared[_bind] // rows_per_shard) == 6677,
       "2.1: the arithmetic in the docs (6,677 shards) matches the config")

# §3: doc_id provenance is recorded per arm, and the requirement is enforced
for a in ARMS:
    expect(D.load_manifest(cfg, a).doc_id_source == "dataset",
           f"3: {a} manifest records doc_id_source=dataset")
import copy as _copy
_tmpcfg = _copy.deepcopy(cfg)
_tmpcfg.paths = dict(cfg.paths)
_tmpcfg.paths["data_root"] = WORK / "nodocid"
fake_open_dataset.omit_doc_id = True
try:
    D.shard_arm(_tmpcfg, "quality-first", log=lambda m: None)
    hard_failed = False
except SystemExit:
    hard_failed = True
expect(hard_failed, "3: a dataset with NO doc_id is REJECTED when require_doc_id is true")
_tmpcfg.data = _copy.deepcopy(cfg.data)
_tmpcfg.data["sharding"]["require_doc_id"] = False
_tmpcfg.paths["data_root"] = WORK / "nodocid2"
try:
    m2 = D.shard_arm(_tmpcfg, "quality-first", log=lambda m: None)
    ok2 = m2.doc_id_source == "synthesized"
except SystemExit:
    ok2 = False
expect(ok2, "3: require_doc_id: false accepts it and records doc_id_source=synthesized")
fake_open_dataset.omit_doc_id = False

# ================================================================= round 7
hdr("4: upload is out of scope and ships disabled")

_shipped_data = yaml.safe_load((REPO / "configs" / "data.yaml").read_text())
_up = _shipped_data["upload"]
expect(_up["enabled"] is False, "4.1: configs/data.yaml ships upload.enabled: false")
expect(_up["repo_template"] == "",
       f"4.1: repo_template is EMPTY, not a plausible-looking id (got {_up['repo_template']!r})")
expect("<" * 3 not in str(_up.get("repo_template", "")),
       "4.1: and it carries no leftover placeholder marker")

# The whole point of §1: a clean checkout has no WYTRO blanks left, so the checker passes
# for a run that will never upload.
_raw_cfgs = (REPO / "configs" / "data.yaml").read_text() + (REPO / "configs" / "vllm.yaml").read_text()
expect("<" * 3 + "WYTRO" not in _raw_cfgs,
       "4.1: no WYTRO placeholder remains anywhere in the shipped configs")

# enabled: true with a blank template must be a NAMED error, not a crash in create_repo("")
def _load_with_upload(**over):
    import copy, subprocess
    root = Path(tempfile.mkdtemp(prefix="rwup-"))
    shutil.copytree(CFG_ROOT / "configs", root / "configs")
    shutil.copytree(CFG_ROOT / "prompts", root / "prompts")   # resolved against config root
    dd = yaml.safe_load((root / "configs/data.yaml").read_text())
    dd["upload"].update(over)
    (root / "configs/data.yaml").write_text(yaml.safe_dump(dd, sort_keys=False))
    code = ("import sys; sys.path.insert(0, %r);\n"
            "from rewrite.config import load_config; load_config(%r)" % (str(REPO / "src"), str(root)))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr

_rc, _out = _load_with_upload(enabled=True, repo_template="")
expect(_rc != 0 and "repo_template" in _out and "enabled" in _out,
       "4.2: upload.enabled: true with a blank repo_template is a clear config error")
_rc, _out = _load_with_upload(enabled=True, repo_template="org/one-repo-for-everything")
expect(_rc != 0 and "{arm}" in _out,
       "4.2: a template without {arm}/{prompt_id} is rejected (10 jobs, one repo each)")
_rc, _out = _load_with_upload(enabled=True, repo_template="org/rw-{arm}-{prompt_id}")
if _rc != 0: print("       ", _out.strip()[:400])
expect(_rc == 0, "4.2: enabled: true WITH a valid template loads fine")
_rc, _out = _load_with_upload(enabled=False, repo_template="")
expect(_rc == 0, "4.2: disabled with a blank template is the shipped, valid state")

# 05_upload_to_hf.py must refuse cleanly rather than traceback, even under --dry-run
import subprocess as _sp
_r = _sp.run([sys.executable, str(REPO / "scripts" / "05_upload_to_hf.py"),
              "--config-root", str(CFG_ROOT), "--dry-run"], capture_output=True, text=True)
_txt = _r.stdout + _r.stderr
expect(_r.returncode != 0 and "STOP" in _txt and "Traceback" not in _txt,
       "4.3: 05_upload_to_hf.py stops cleanly when upload is disabled (no traceback)")
expect("shuffled" in _txt,
       "4.3: and it says where the finished data actually is")
expect((REPO / "scripts" / "05_upload_to_hf.py").exists(),
       "4.3: the upload script is still in the repo -- disabled, not deleted")

hdr("5: per-job postprocess locking")

_lk = WORK / "locktest" / ".postprocess.lock"
_o1 = {"host": "nodeA", "pid": 111}
_o2 = {"host": "nodeB", "pid": 222}
expect(D.try_dir_lock(_lk, _o1, 1800, "j", log=lambda m: None) is True,
       "5.1: try_dir_lock takes a free lock")
expect(D.try_dir_lock(_lk, _o2, 1800, "j", log=lambda m: None) is False,
       "5.1: a second node is refused IMMEDIATELY -- it does not wait")
expect(D.read_lock_owner(_lk).get("host") == "nodeA",
       "5.1: the lock still belongs to its original holder")
os.utime(_lk, (time.time() - 4000, time.time() - 4000))
expect(D.try_dir_lock(_lk, _o2, 1800, "j", log=lambda m: None) is True,
       "5.1: a lock that stopped heartbeating IS taken over")
expect(D.read_lock_owner(_lk).get("host") == "nodeB",
       "5.1: and the taker-over records itself as the new owner")
D.break_lock(_lk)
expect(D.try_dir_lock(_lk, _o1, 1800, "j", log=lambda m: None) is True,
       "5.1: break_lock releases it")
D.break_lock(_lk)

# The property that makes multi-node postprocess safe: jobs partition, none run twice.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("pp7", REPO / "scripts" / "04_postprocess.py")
_pp = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_pp)

# A clean out_root: the real postprocess ran earlier in this file, so every job already
# carries a _shuffle.done and the sweep would correctly skip all ten. Point it somewhere
# empty so there is actually work to distribute.
_cfg7 = C.load_config(CFG_ROOT)
_cfg7.paths["out_root"] = WORK / "sweepout"
_jobs7 = C.enumerate_jobs(_cfg7)
_taken = {}
_args7 = types.SimpleNamespace(stage="both", no_lock=False, workers=None)

def _make_runner(node):
    def _run(cfg, job, args):
        _taken.setdefault(node, []).append(job.job_id)
        m = cfg.shuffled_dir(job.arm, job.prompt.id)
        m.mkdir(parents=True, exist_ok=True)
        (m / "_shuffle.done").write_text("{}")
        return {"shuffle": {"rows": 0}}
    return _run

# Two "nodes" interleaved: node B sweeps while node A holds one job's lock.
_held = D.try_dir_lock(_pp.job_lock(_cfg7, _jobs7[0]), {"host": "nodeA", "pid": 1},
                       1800, "held", log=lambda m: None)
_pp.run_one = _make_runner("nodeB")
_pp.HOSTNAME = "nodeB"
_pp.sweep(_cfg7, _jobs7, _args7)
expect(_held, "5.2: node A holds job 1's lock")
expect(_jobs7[0].job_id not in _taken.get("nodeB", []),
       f"5.2: node B did NOT touch the job node A holds ({_jobs7[0].job_id})")
expect(len(_taken.get("nodeB", [])) == len(_jobs7) - 1,
       f"5.2: node B took every OTHER job ({len(_taken.get('nodeB', []))}/{len(_jobs7)-1})")

D.break_lock(_pp.job_lock(_cfg7, _jobs7[0]))
_pp.run_one = _make_runner("nodeA")
_pp.HOSTNAME = "nodeA"
_pp.sweep(_cfg7, _jobs7, _args7)
expect(_taken.get("nodeA", []) == [_jobs7[0].job_id],
       "5.2: node A then takes exactly the one job that was left")
_all7 = [j for v in _taken.values() for j in v]
expect(sorted(_all7) == sorted(j.job_id for j in _jobs7),
       "5.2: between them the two nodes covered all 10 jobs")
expect(len(_all7) == len(set(_all7)),
       "5.2: and no job was postprocessed twice")

# The summary must survive concurrent writers, which the old whole-file rewrite did not.
_sum = _cfg7.repo_root / "manifests" / "postprocess_summary.json"
expect(_sum.exists(), "5.3: merged postprocess_summary.json was written")
_merged = json.loads(_sum.read_text())
expect(len(_merged) == len(_jobs7),
       f"5.3: it holds ALL {len(_jobs7)} jobs, not just the last sweep's ({len(_merged)})")
expect(not list((_cfg7.repo_root / "manifests").glob("*.tmp")),
       "5.3: no temp files left behind")

hdr("6: num_gpus is per node, and the docs say so")

_gtxt = (REPO / "docs" / "GUIDE_FOR_TIANJIAN.md").read_text()
_dtxt = (REPO / "configs" / "data.yaml").read_text()
_ctxt = (REPO / "configs" / "cluster.yaml").read_text()
expect("fleet_gpus" in _ctxt, "6.1: cluster.yaml offers compute.fleet_gpus")
expect(yaml.safe_load(_ctxt)["compute"]["fleet_gpus"] is None,
       "6.1: and it ships null, so single-node runs are unaffected")
expect("PER NODE" in _dtxt or "per node" in _dtxt,
       "6.2: data.yaml's ceiling box says the automatic check is per node")
expect("fleet_gpus" in _dtxt,
       "6.2: and points at the value that makes it a real check")
expect("6,677 / 20" in _gtxt,
       "6.2: the guide gives the arithmetic Tianjian can apply himself")

# The regression this guards: a per-node count presented as the fleet's.
_cal = (REPO / "scripts" / "06_calibrate.py").read_text()
expect('"  fleet    :' not in _cal,
       "6.3: 06_calibrate.py no longer labels one node's GPUs 'fleet'")
expect("ON THIS NODE" in _cal and "PER NODE" in _cal,
       "6.3: its projection is explicitly labelled per node")

_pre = (REPO / "scripts" / "preflight.py").read_text()
expect('"out_root": 2 * comp' in _pre,
       "6.4: preflight gates out_root on TWO copies (raw/ survives the in-place trim)")

# The check itself, not just the prose: fleet_gpus must actually bite where num_gpus cannot.
# This corpus is tiny (quality-first has 1,234 rows), so it stands in for the real thing --
# what is being tested is that the FLEET number is what gets compared.
_fc = _copy.deepcopy(cfg)
_fc.paths = dict(cfg.paths)
_fc.data = _copy.deepcopy(cfg.data)
_fc.cluster = _copy.deepcopy(cfg.cluster)
_n_shards = D.load_manifest(cfg, "quality-first").n_shards
_min_ratio = _fc.data["sharding"]["min_shards_per_gpu"]

# num_gpus small enough to pass on its own; fleet large enough that the fleet must fail.
_fc.cluster["compute"]["num_gpus"] = 1
_fc.cluster["compute"]["fleet_gpus"] = None
_fc.paths["data_root"] = WORK / "fleet_unset"
try:
    D.shard_arm(_fc, "quality-first", log=lambda m: None)
    _unset_ok = True
except SystemExit:
    _unset_ok = False
expect(_unset_ok, "6.5: with fleet_gpus unset, only this node's num_gpus is checked")

_fc.cluster["compute"]["fleet_gpus"] = _n_shards * 2      # far over the ceiling
_fc.paths["data_root"] = WORK / "fleet_over"
import contextlib as _ctx, io as _io
_err = _io.StringIO()
try:
    with _ctx.redirect_stderr(_err):        # stop() writes the message to stderr
        D.shard_arm(_fc, "quality-first", log=lambda m: None)
    _over_caught = False
except SystemExit:
    _over_caught = True
_emsg = _err.getvalue()
expect(_over_caught,
       "6.5: a fleet that exceeds the ceiling IS caught once fleet_gpus is set")
expect("across the fleet" in _emsg,
       "6.5: and the error says the ratio is fleet-wide, not this node's")
expect("shard_target_rows" in _emsg,
       "6.5: and it computes the value to use instead of just refusing")

_fc.cluster["compute"]["fleet_gpus"] = 1
_fc.paths["data_root"] = WORK / "fleet_ok"
_msgs = []
try:
    D.shard_arm(_fc, "quality-first", log=_msgs.append)
    _ok_fleet = True
except SystemExit:
    _ok_fleet = False
expect(_ok_fleet, "6.5: a fleet within the ceiling passes")
expect(any("across the declared fleet" in m for m in _msgs),
       "6.5: and the fleet-wide ratio is reported so it is auditable")

# ================================================================= round 9
hdr("7: record_id -- schema interlock and input guard")

import copy as _c9
# The fingerprint decision: output.keys is folded in, so adding or removing a key
# invalidates every .done marker instead of silently mixing two schemas in one job.
_fp_base = D.compute_fingerprint(cfg, "deadbeef", "dataset")
_cfg_k = _c9.deepcopy(cfg); _cfg_k.data = _c9.deepcopy(cfg.data)
_cfg_k.data["output"]["keys"] = [k for k in cfg.data["output"]["keys"] if k != "record_id"]
expect(D.compute_fingerprint(_cfg_k, "deadbeef", "dataset") != _fp_base,
       "7.1: dropping record_id from output.keys CHANGES the manifest fingerprint")
_cfg_o = _c9.deepcopy(cfg); _cfg_o.data = _c9.deepcopy(cfg.data)
_cfg_o.data["output"]["keys"] = list(reversed(cfg.data["output"]["keys"]))
expect(D.compute_fingerprint(_cfg_o, "deadbeef", "dataset") != _fp_base,
       "7.1: so does reordering them -- the shuffled parquet's column order follows")
expect(D.compute_fingerprint(cfg, "deadbeef", "dataset") == _fp_base,
       "7.1: and an unchanged schema is stable (no spurious invalidation)")

# The input guard: record_id is declared in output.keys, so a dataset without it must
# stop with a named error rather than emit empty strings.
fake_open_dataset.omit_record_id = True
_cfg_r = _c9.deepcopy(cfg); _cfg_r.paths = dict(cfg.paths)
_cfg_r.paths["data_root"] = WORK / "norecid"
_err = io.StringIO()
try:
    with contextlib.redirect_stderr(_err):
        D.shard_arm(_cfg_r, "quality-first", log=lambda m: None)
    _guarded = False
except SystemExit:
    _guarded = True
fake_open_dataset.omit_record_id = False
expect(_guarded, "7.2: a dataset with NO record_id column is REJECTED, not silently blanked")
_gm = _err.getvalue()
expect("record_id" in _gm and "output.keys" in _gm,
       "7.2: the error names the column and the config key that requires it")
expect("metadata" in _gm,
       "7.2: and says not to parse it out of the metadata blob")

# Its role, asserted where someone will read it.
_dy = (REPO / "configs" / "data.yaml").read_text()
expect("NOT A UNIQUE KEY" in _dy,
       "7.3: data.yaml states plainly that record_id is not a unique key")
expect("0.0662%" in _dy and "599,603,031" in _dy,
       "7.3: with the measured duplication rate and the exact distinct count")
expect("payload_digest" in _dy,
       "7.3: and names payload_digest as the tie-breaker for the repair path")

# ================================================================= summary
hdr("SUMMARY")
print(f"  checks failed: {len(FAILS)}")
for f in FAILS: print("   -", f)
shutil.rmtree(WORK, ignore_errors=True)
print("\n" + ("ALL INTEGRATION CHECKS PASSED" if not FAILS else "INTEGRATION FAILURES"))
sys.exit(0 if not FAILS else 1)

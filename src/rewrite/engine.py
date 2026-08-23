"""vLLM engine construction, chat templating, prompt assembly, and batch generation.

The only module that imports vllm or torch.

Everything here is a direct transcription of the source worker's inner loop
(07_rewrite/rewrite_worker.py and its 09_Distill fork, which differ only in the drop
threshold). Values are never invented: if it is not in configs/vllm.yaml, it is not
passed, and configs/vllm.yaml is not editable configuration -- it is a record.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .config import (TEXT_PLACEHOLDER, WRAP_SUFFIX, ALLOWED_ENGINE_KEYS, Config,
                     PromptSpec, stop)


def set_source_env(cfg: Config) -> None:
    """Environment the source set on every compute node.

    source: 07_rewrite/rewrite_worker.py:29-32 and sbatch_template.sh:31-38.
    `setdefault`, so an explicit override from the caller still wins.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM",
                          "true" if cfg.cluster["runtime"]["tokenizers_parallelism"]
                          else "false")
    if cfg.cluster["runtime"]["hf_hub_offline_after_download"]:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", str(cfg.paths["hf_cache"]))


def model_dir(cfg: Config) -> Path:
    return cfg.paths["model_dir"] / cfg.vllm["model"]["local_subdir"]


def llama2_dir(cfg: Config) -> Path:
    return cfg.paths["model_dir"] / cfg.vllm["llama2_tokenizer"]["local_subdir"]


# --------------------------------------------------------------------------- tokenizers
def load_qwen_tokenizer(cfg: Config):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(model_dir(cfg)))


def load_llama2_tokenizer(cfg: Config):
    """The tokenizer that produced the source's `rewritten_tokens` column.

    source: 07_rewrite/sbatch_template.sh:25 (a local dir whose tokenizer_report.json
    records source_repo = unsloth/llama-2-7b, LlamaTokenizer, vocab_size 32000).
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(llama2_dir(cfg)))
    exp = int(cfg.vllm["llama2_tokenizer"]["expected_vocab_size"])
    got = getattr(tok, "vocab_size", None)
    if got is not None and int(got) != exp:
        stop(f"Llama-2 tokenizer vocab_size is {got}, expected {exp}. Output token "
             "counts would not be comparable to the source's.")
    return tok


def count_llama2(ltok, texts: list) -> list:
    """len(tok(text, add_special_tokens=False).input_ids), batched.

    source: 07_rewrite/rewrite_worker.py:193-194 (`n_llama`). The source called this once
    per document; batching is value-identical for a fast tokenizer and is the difference
    between minutes and hours at this scale. Verified by
    `scripts/preflight.py --verify-tokenizer-batching`.
    """
    if not texts:
        return []
    enc = ltok([t or "" for t in texts], add_special_tokens=False)
    return [len(x) for x in enc.input_ids]


# --------------------------------------------------------------------------- prompts
def build_content(mode: str, doc_text, prompt_text: str) -> str:
    """The user-message content for one document, pre chat-template.

    source: 07_rewrite/rewrite_worker.py:45-51, verbatim.
    """
    doc_text = doc_text or ""
    if mode == "grounded":
        return prompt_text.replace(TEXT_PLACEHOLDER, doc_text)
    # wrap: instruction string already ends with "Passage:\n"
    return prompt_text + doc_text


def template(qtok, cfg: Config, content: str) -> str:
    """source: 07_rewrite/rewrite_worker.py:285-286.

    A single user message. No system message is authored -- Qwen's own chat template
    injects "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    `tokenize=False`, so a rendered STRING is what reaches llm.generate().
    """
    ch = cfg.vllm["chat"]
    return qtok.apply_chat_template(
        [{"role": ch["role"], "content": content}],
        tokenize=ch["tokenize"], add_generation_prompt=ch["add_generation_prompt"])


def count_qwen(qtok, finals: list) -> list:
    """source: 07_rewrite/rewrite_worker.py:287 -- len(qtok(final, add_special_tokens=False))"""
    if not finals:
        return []
    return [len(x) for x in qtok(finals, add_special_tokens=False).input_ids]


def empty_doc_overhead(qtok, cfg: Config, prompt: PromptSpec) -> int:
    """Templated token count with an EMPTY document.

    source: 09_Distill/rewrite_worker.py:231-237, which printed exactly this and logged
    `OVERHEAD(empty-doc templated tokens)=185` for the distill prompt.

    This single integer proves the prompt file, the chat template and the tokenizer are
    all the ones the source used. It is asserted before any GPU work begins.
    """
    final = template(qtok, cfg, build_content(prompt.mode, "", prompt.text))
    return len(qtok(final, add_special_tokens=False).input_ids)


# --------------------------------------------------------------------------- engine
def build_llm(cfg: Config):
    """LLM(...) built from a whitelist, so nothing can leak in.

    source: 07_rewrite/rewrite_worker.py:240-241
        LLM(model=..., tensor_parallel_size=1, dtype="bfloat16",
            gpu_memory_utilization=..., max_model_len=...)

    Everything else runs at the vLLM 0.22.0 defaults the source silently relied on; they
    are recorded in configs/vllm.yaml under `inherited_defaults_do_not_pass`.
    """
    from vllm import LLM
    eng = {k: cfg.vllm["engine"][k] for k in sorted(ALLOWED_ENGINE_KEYS)}
    md = model_dir(cfg)
    if not (md / "config.json").exists():
        stop(f"model not found at {md} (no config.json). "
             "Run scripts/01_download_model.py first.")
    return LLM(model=str(md), **eng)


def build_sampling(cfg: Config, n_in: int):
    """Per-document SamplingParams.

    source: 07_rewrite/rewrite_worker.py:294-296
        max_new = min(max_tokens, max_model_len - n_in)
        SamplingParams(temperature=0, top_p=1.0, max_tokens=max(1, max_new))

    Note the consequence the source documented: under the wiki prompt's fixed 30720 drop
    threshold, a document at exactly n_in=30720 receives max_tokens=2048, not 4096 --
    silently reduced output room that surfaces only as status=1.
    """
    from vllm import SamplingParams
    s = cfg.vllm["sampling"]
    max_new = min(int(s["max_tokens"]), cfg.max_model_len - int(n_in))
    return SamplingParams(temperature=s["temperature"], top_p=s["top_p"],
                          max_tokens=max(1, max_new))


# --------------------------------------------------------------------------- batching
@dataclass
class Prepared:
    n_rows: int
    n_in_list: list      # templated Qwen token count per row
    keep_idx: list       # row indices that will be generated
    keep_prompts: list   # templated strings for those rows
    keep_sp: list        # SamplingParams for those rows


def prepare_batch(qtok, cfg: Config, texts: list, prompt: PromptSpec,
                  drop_threshold: int) -> Prepared:
    """Build prompts and decide status=0 by templated length.

    source: 07_rewrite/rewrite_worker.py:279-296, transcribed.

    Documents are DROPPED, never truncated -- there is no document truncation anywhere in
    the source. The decision is made on the real templated n_in, not by additivity.
    """
    n_rows = len(texts)
    finals = [template(qtok, cfg, build_content(prompt.mode, t, prompt.text))
              for t in texts]
    n_in_list = count_qwen(qtok, finals)

    keep_idx, keep_prompts, keep_sp = [], [], []
    for j in range(n_rows):
        if n_in_list[j] > drop_threshold:
            continue                      # status=0
        keep_idx.append(j)
        keep_prompts.append(finals[j])
        keep_sp.append(build_sampling(cfg, n_in_list[j]))
    return Prepared(n_rows, n_in_list, keep_idx, keep_prompts, keep_sp)


@dataclass
class BatchResult:
    rewritten: list
    finish_reason: list
    status: list
    n_output_tokens: list


def run_batch(llm, prep: Prepared) -> BatchResult:
    """ONE llm.generate() call for the whole shard, as the source did -- this is what
    lets vLLM's continuous batching and prefix caching do their job.

    source: 07_rewrite/rewrite_worker.py:298-312.
    Defaults below are the status=0 values (:299-302).
    """
    n = prep.n_rows
    rewritten = [""] * n
    finish_reason = [""] * n
    status = [0] * n
    n_out = [0] * n

    if prep.keep_prompts:
        outputs = llm.generate(prep.keep_prompts, prep.keep_sp)
        for o, j in zip(outputs, prep.keep_idx):
            g = o.outputs[0]
            rewritten[j] = g.text
            finish_reason[j] = g.finish_reason or ""
            status[j] = 1 if g.finish_reason == "length" else 2
            n_out[j] = len(g.token_ids)
    return BatchResult(rewritten, finish_reason, status, n_out)

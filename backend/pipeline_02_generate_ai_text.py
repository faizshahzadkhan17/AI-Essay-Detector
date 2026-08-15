"""
Pipeline stage 2: generate synthetic AI training data with small local
instruct models (no external API calls). Two things get produced:

  1. "self_ai_fresh" -- full essays generated from scratch across two
     models x two prompt styles x the prompt bank in essay_prompts.py.
     This diversifies the AI side of the dataset beyond whatever models
     produced the DAIGT AI essays (which we don't fully control), and
     covers the personal-narrative register real admissions essays use
     that DAIGT (persuasive/argumentative, PERSUADE-sourced) under-covers.

  2. "self_ai_polished" -- the hardest and most realistic case: a real
     human essay with SOME paragraphs rewritten by a model ("polished")
     and others left verbatim. Ground truth is exact because we choose
     which chunks to rewrite ourselves.

Every document is stored as a list of (chunk_text, label) pairs so stage
3 can split each chunk into sentences independently and propagate the
label without any ambiguity about which sentences are AI vs human.

Generation is BATCHED (left-padded) since a 1.5B model on a 6GB GPU is
throughput-bound, not compute-bound, at batch size 1 -- batching gets
several times more essays/minute out of the same GPU.

Models used: Qwen2.5-1.5B-Instruct and TinyLlama-1.1B-Chat-v1.0. Both are
small enough to run on a 6GB GPU in fp16 and are not gated on the Hub.
"""

import gc
import random
import re

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR, RANDOM_SEED
from essay_prompts import ALL_PROMPTS, GENERATION_STYLES, POLISH_STYLES

random.seed(RANDOM_SEED)

GEN_MODELS = {
    "qwen2.5-1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "tinyllama-1.1b-chat": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

FRESH_OUT = PROCESSED_DATA_DIR / "self_ai_fresh_chunks.parquet"
POLISHED_OUT = PROCESSED_DATA_DIR / "self_ai_polished_chunks.parquet"

N_POLISH_SOURCE_ESSAYS = 220
GEN_BATCH_SIZE = 8
FRESH_MAX_NEW_TOKENS = 550
POLISH_MAX_NEW_TOKENS = 180


def load_model(model_key: str):
    name = GEN_MODELS[model_key]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16 if device == "cuda" else torch.float32)
    model.to(device)
    model.eval()
    return tok, model, device


def chat_generate_batch(tok, model, device, user_messages: list[str], max_new_tokens: int) -> list[str]:
    prompts = [
        tok.apply_chat_template([{"role": "user", "content": m}], tokenize=False, add_generation_prompt=True)
        for m in user_messages
    ]
    inputs = tok(prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            repetition_penalty=1.15,
            pad_token_id=tok.pad_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    texts = tok.batch_decode(out[:, input_len:], skip_special_tokens=True)
    return [t.strip() for t in texts]


def clean_essay_text(text: str) -> str:
    text = re.sub(r'^"|"$', "", text.strip())
    text = re.sub(r"^(Title|Essay|Prompt Response)\s*:.*\n", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^#+\s*.*\n", "", text)
    return text.strip()


def batched(items: list, n: int):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def generate_fresh_essays():
    if FRESH_OUT.exists():
        print(f"[skip] {FRESH_OUT} exists")
        return
    rows = []
    for model_key in GEN_MODELS:
        tok, model, device = load_model(model_key)

        jobs = []  # (user_msg, style_key, prompt_domain)
        for style_key, style_template in GENERATION_STYLES.items():
            for prompt_text, prompt_domain in ALL_PROMPTS:
                jobs.append((style_template.format(prompt=prompt_text), style_key, prompt_domain))
        random.shuffle(jobs)

        for batch in batched(jobs, GEN_BATCH_SIZE):
            msgs = [j[0] for j in batch]
            try:
                outputs = chat_generate_batch(tok, model, device, msgs, FRESH_MAX_NEW_TOKENS)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print("  OOM on batch, skipping")
                continue
            for (_, style_key, prompt_domain), essay in zip(batch, outputs):
                essay = clean_essay_text(essay)
                if len(essay.split()) < 80:
                    continue
                doc_id = f"selfai_{model_key}_{style_key}_{prompt_domain}_{len(rows)}"
                rows.append({
                    "doc_id": doc_id, "group_id": doc_id, "chunk_idx": 0,
                    "chunk_text": essay, "label": 1, "bucket": "self_ai_fresh",
                    "gen_model": model_key, "prompt_style": style_key, "prompt_domain": prompt_domain,
                })
            print(f"  [{model_key}] {len(rows)} fresh AI essays so far...")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_parquet(FRESH_OUT, index=False)
    print(f"Saved {len(df)} fresh AI essay chunks -> {FRESH_OUT}")


def split_into_paragraph_chunks(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) >= 2:
        return paras
    from sentence_utils import split_sentences
    sents = split_sentences(text)
    chunks = [" ".join(sents[i:i + 3]) for i in range(0, len(sents), 3)]
    return chunks if chunks else [text]


def generate_polished_mixed_docs():
    if POLISHED_OUT.exists():
        print(f"[skip] {POLISHED_OUT} exists")
        return
    daigt = pd.read_parquet(RAW_DATA_DIR / "daigt.parquet")
    human = daigt[daigt["generated"] == 0].sample(
        n=min(N_POLISH_SOURCE_ESSAYS, (daigt["generated"] == 0).sum()), random_state=RANDOM_SEED
    ).reset_index(drop=True)

    # pre-split every source essay into chunks and decide which get polished,
    # BEFORE loading any model, so both models just work through a job queue
    doc_plan = []  # dict per doc: essay_uid, chunks (list[str]), polish_idxs (set), polish_style
    for _, row in human.iterrows():
        chunks = split_into_paragraph_chunks(row["text"])
        if len(chunks) < 2:
            continue
        n_to_polish = max(1, int(len(chunks) * random.uniform(0.4, 0.7)))
        polish_idxs = set(random.sample(range(len(chunks)), min(n_to_polish, len(chunks))))
        doc_plan.append({
            "essay_uid": row["essay_uid"], "chunks": chunks, "polish_idxs": polish_idxs,
            "polish_style": random.choice(list(POLISH_STYLES.keys())),
        })

    model_keys = list(GEN_MODELS.keys())
    assignments = {plan["essay_uid"]: model_keys[i % len(model_keys)] for i, plan in enumerate(doc_plan)}

    all_rows = {}  # doc_id -> list of chunk row dicts (indexed by chunk_idx later)
    for model_key in GEN_MODELS:
        tok, model, device = load_model(model_key)
        my_plans = [p for p in doc_plan if assignments[p["essay_uid"]] == model_key]

        # build a flat job queue of (essay_uid, chunk_idx, chunk_text, style)
        jobs = []
        for plan in my_plans:
            for i in plan["polish_idxs"]:
                if len(plan["chunks"][i].split()) >= 15:
                    jobs.append((plan["essay_uid"], i, plan["chunks"][i], plan["polish_style"]))

        results = {}  # (essay_uid, chunk_idx) -> polished text
        for batch in batched(jobs, GEN_BATCH_SIZE):
            msgs = [POLISH_STYLES[j[3]].format(chunk=j[2]) for j in batch]
            try:
                outputs = chat_generate_batch(tok, model, device, msgs, POLISH_MAX_NEW_TOKENS)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print("  OOM on polish batch, skipping")
                continue
            for (essay_uid, chunk_idx, _, _), polished in zip(batch, outputs):
                polished = clean_essay_text(polished)
                if len(polished.split()) >= 8:
                    results[(essay_uid, chunk_idx)] = polished
            print(f"  [{model_key}] polished {len(results)}/{len(jobs)} chunks so far...")

        for plan in my_plans:
            doc_id = f"selfaipol_{plan['essay_uid']}_{model_key}"
            out_chunks = []
            any_success = False
            for i, chunk in enumerate(plan["chunks"]):
                polished = results.get((plan["essay_uid"], i))
                if polished:
                    out_chunks.append({
                        "doc_id": doc_id, "group_id": plan["essay_uid"], "chunk_idx": i,
                        "chunk_text": polished, "label": 1, "bucket": "self_ai_polished",
                        "gen_model": model_key, "prompt_style": plan["polish_style"], "prompt_domain": "polish",
                    })
                    any_success = True
                else:
                    out_chunks.append({
                        "doc_id": doc_id, "group_id": plan["essay_uid"], "chunk_idx": i,
                        "chunk_text": chunk, "label": 0, "bucket": "self_ai_polished",
                        "gen_model": "human", "prompt_style": "n/a", "prompt_domain": "n/a",
                    })
            if any_success:
                all_rows[doc_id] = out_chunks

        del model
        gc.collect()
        torch.cuda.empty_cache()

    rows = [r for chunks in all_rows.values() for r in chunks]
    df = pd.DataFrame(rows)
    df.to_parquet(POLISHED_OUT, index=False)
    print(f"Saved {len(df)} chunks across {len(all_rows)} polished-mixed docs -> {POLISHED_OUT}")


if __name__ == "__main__":
    generate_fresh_essays()
    generate_polished_mixed_docs()

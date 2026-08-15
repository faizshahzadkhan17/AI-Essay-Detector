"""
Signal layer: model-ASSISTED, not model-JUDGED.

We run the essay through a small local causal language model (GPT-2) and
read off purely statistical quantities — per-token log-probability and
rank under the model's own next-token distribution. Nothing here asks the
model "is this AI-written?"; the model never sees that question and has
no output channel for an opinion. It only ever computes P(next token |
previous tokens), the same quantity it was pretrained to compute. Those
numbers are then aggregated into per-sentence and per-document features
that get handed to the (separately trained) decision layer, exactly like
any other feature.

Also includes a Binoculars-style cross-model ratio: comparing a
sentence's log-probability under two related models gives a second,
better-calibrated statistic, since it partially cancels out the fact
that some text is simply intrinsically easy/hard to predict.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MAX_MODEL_TOKENS, SIGNAL_MODEL_NAME, SIGNAL_MODEL_NAME_B, SLIDING_WINDOW_STRIDE
from sentence_utils import split_sentences_with_spans

SIGNAL_FEATURE_NAMES = [
    "sig_mean_nll",
    "sig_perplexity",
    "sig_mean_rank",
    "sig_median_rank",
    "sig_frac_top1",
    "sig_frac_top10",
    "sig_frac_top50",
    "sig_surprisal_std",
    "sig_max_surprisal",
    "sig_doc_perplexity",
]

BINOCULARS_FEATURE_NAMES = ["sig_binoculars_ratio"]


@lru_cache(maxsize=4)
def _load(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return tokenizer, model, device


@dataclass
class TokenStats:
    nll: list  # per-token negative log-likelihood (surprisal, in nats)
    rank: list  # per-token rank of the actual token (0 = model's top pick)


@torch.no_grad()
def _token_level_stats(text: str, model_name: str = SIGNAL_MODEL_NAME) -> tuple[list[int], TokenStats]:
    """Run the LM once over `text` and return, for every token position
    (except the first, which has no left context), its offset-mapping
    start char, negative log-likelihood, and rank.
    """
    tokenizer, model, device = _load(model_name)
    enc = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=False)
    input_ids = enc["input_ids"][0]
    offsets = enc["offset_mapping"][0].tolist()
    n_tokens = input_ids.shape[0]

    nll_by_pos: dict[int, float] = {}
    rank_by_pos: dict[int, int] = {}

    window = MAX_MODEL_TOKENS
    stride = SLIDING_WINDOW_STRIDE
    start = 0
    while start < n_tokens:
        end = min(start + window, n_tokens)
        chunk = input_ids[start:end].unsqueeze(0).to(device)
        logits = model(chunk).logits[0]  # (chunk_len, vocab)
        log_probs = F.log_softmax(logits.float(), dim=-1)

        # position i's logits predict token i+1 (local index). Skip any
        # global target already scored by a previous (overlapping) window.
        for local_pred_idx in range(0, end - start - 1):
            global_target = start + local_pred_idx + 1
            if global_target in nll_by_pos:
                continue
            actual_token = input_ids[global_target].item()
            lp = log_probs[local_pred_idx]
            token_logprob = lp[actual_token].item()
            nll_by_pos[global_target] = -token_logprob
            rank_by_pos[global_target] = int((lp > token_logprob).sum().item())

        if end == n_tokens:
            break
        start += stride

    positions = sorted(nll_by_pos.keys())
    starts = [offsets[p][0] for p in positions]
    stats = TokenStats(nll=[nll_by_pos[p] for p in positions], rank=[rank_by_pos[p] for p in positions])
    return starts, stats


def score_document(text: str, model_name: str = SIGNAL_MODEL_NAME) -> tuple[list[str], list[dict], dict]:
    """Return (sentences, per_sentence_signal_feature_dicts, doc_level_dict)."""
    normalized, spans = split_sentences_with_spans(text)
    if not spans:
        return [], [], {}

    token_starts, stats = _token_level_stats(normalized, model_name)

    # assign each scored token to a sentence index by its char start offset
    sent_token_idx: list[list[int]] = [[] for _ in spans]
    span_ends = [e for _, e, _ in spans]
    cur = 0
    for i, char_start in enumerate(token_starts):
        while cur < len(spans) - 1 and char_start >= span_ends[cur]:
            cur += 1
        sent_token_idx[cur].append(i)

    doc_nll = stats.nll
    doc_perplexity = float(torch.tensor(doc_nll).mean().exp()) if doc_nll else float("nan")

    per_sentence = []
    for idxs in sent_token_idx:
        if not idxs:
            per_sentence.append({name: 0.0 for name in SIGNAL_FEATURE_NAMES})
            continue
        nlls = torch.tensor([stats.nll[i] for i in idxs])
        ranks = torch.tensor([float(stats.rank[i]) for i in idxs])
        mean_nll = nlls.mean().item()
        feats = {
            "sig_mean_nll": mean_nll,
            "sig_perplexity": float(torch.exp(nlls.mean())),
            "sig_mean_rank": ranks.mean().item(),
            "sig_median_rank": ranks.median().item(),
            "sig_frac_top1": (ranks < 1).float().mean().item(),
            "sig_frac_top10": (ranks < 10).float().mean().item(),
            "sig_frac_top50": (ranks < 50).float().mean().item(),
            "sig_surprisal_std": nlls.std().item() if len(nlls) > 1 else 0.0,
            "sig_max_surprisal": nlls.max().item(),
            "sig_doc_perplexity": doc_perplexity,
        }
        per_sentence.append(feats)

    doc_level = {
        "doc_perplexity": doc_perplexity,
        "doc_mean_rank": float(torch.tensor([float(r) for r in stats.rank]).mean()) if stats.rank else float("nan"),
    }

    sentences = [s for _, _, s in spans]
    return sentences, per_sentence, doc_level


@torch.no_grad()
def _mean_nll_whole(text: str, model_name: str) -> float:
    tokenizer, model, device = _load(model_name)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_MODEL_TOKENS)
    input_ids = enc["input_ids"].to(device)
    if input_ids.shape[1] < 2:
        return float("nan")
    out = model(input_ids, labels=input_ids)
    return out.loss.item()


def binoculars_ratio(text: str, observer: str = SIGNAL_MODEL_NAME, performer: str = SIGNAL_MODEL_NAME_B) -> float:
    """Simplified Binoculars-style score: ratio of a text's perplexity
    under a smaller "observer" model to its perplexity under a larger
    "performer" model. Human text tends to be relatively harder for the
    small model than the large one (ratio closer to 1 / higher);
    AI-generated text is often disproportionately easy for both,
    compressing the ratio. This is a simplified stand-in for the full
    Binoculars method (which compares cross-entropy between the two
    models' output *distributions* at every position, not just the two
    perplexities) — cheaper to compute, same intuition, weaker
    calibration.
    """
    nll_observer = _mean_nll_whole(text, observer)
    nll_performer = _mean_nll_whole(text, performer)
    if nll_performer == 0:
        return float("nan")
    return nll_observer / nll_performer

"""
Feature layer: hand-engineered stylometric signals. Zero model calls.
Everything here is arithmetic over the raw text, computed once at the
document level and once per sentence (relative to its document).

This is the layer that gives the classifier signals that are cheap,
deterministic, and easy to explain in a sentence to a non-technical
reader ("this sentence is far shorter than the rest of the essay").
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import re
import statistics

from sentence_utils import split_sentences, word_tokenize

# Words/phrases disproportionately over-used by instruction-tuned LLMs in
# essay-style writing. Curated from published analyses of ChatGPT stylistic
# tics (e.g. "delve", "tapestry", "boasts", "moreover") plus common
# five-paragraph-essay transition boilerplate. This is a *feature*, not a
# rule: the classifier learns how much weight it deserves.
_AI_TELL_WORDS = {
    "delve", "delves", "delving", "tapestry", "boasts", "showcase",
    "showcases", "showcasing", "underscore", "underscores", "underscoring",
    "furthermore", "moreover", "additionally", "notably", "importantly",
    "ultimately", "overarching", "multifaceted", "intricate", "intricacies",
    "paramount", "pivotal", "invaluable", "indelible", "testament",
    "navigate", "navigating", "landscape", "realm", "foster", "fosters",
    "fostering", "endeavor", "endeavors", "cultivate", "cultivates",
    "holistic", "robust", "leverage", "leveraging", "synergy",
    "in conclusion", "in summary", "to summarize", "it is important to",
    "it is worth noting", "it is essential to", "plays a crucial role",
    "plays a significant role", "in today's society", "in today's world",
}

_CONTRACTION_RE = re.compile(r"\b\w+'(s|t|re|ve|ll|d|m)\b", re.IGNORECASE)
_TRANSITION_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _AI_TELL_WORDS) + r")\b",
    re.IGNORECASE,
)

FEATURE_NAMES = [
    # --- sentence-local ---
    "sent_len_words",
    "sent_avg_word_len",
    "sent_comma_rate",
    "sent_semicolon_dash_rate",
    "sent_punct_density",
    "sent_contraction_rate",
    "sent_ai_tell_count",
    "sent_unique_word_ratio",
    "sent_local_bigram_overlap",
    # --- sentence-relative-to-document (burstiness) ---
    "sent_len_zscore_in_doc",
    "sent_len_abs_dev_ratio",
    # --- document-level context, broadcast to every sentence ---
    "doc_sentence_count",
    "doc_mean_sentence_len",
    "doc_sentence_len_cv",
    "doc_mattr",
    "doc_bigram_repetition_rate",
    "doc_trigram_repetition_rate",
    "doc_contraction_rate",
    "doc_ai_tell_rate",
    "doc_comma_per_sentence",
]


def _ngrams(words: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(w.lower() for w in words[i : i + n]) for i in range(len(words) - n + 1)]


def _repetition_rate(words: list[str], n: int) -> float:
    grams = _ngrams(words, n)
    if len(grams) < 2:
        return 0.0
    unique = len(set(grams))
    return 1.0 - (unique / len(grams))


def _mattr(words: list[str], window: int = 40) -> float:
    """Moving-average type-token ratio: TTR averaged over fixed-size
    windows so the metric is comparable across essays of different
    lengths (raw TTR shrinks mechanically as a document gets longer)."""
    words = [w.lower() for w in words]
    if len(words) < 5:
        if not words:
            return 0.0
        return len(set(words)) / len(words)
    if len(words) <= window:
        return len(set(words)) / len(words)
    ratios = []
    for i in range(0, len(words) - window + 1, max(1, window // 2)):
        chunk = words[i : i + window]
        ratios.append(len(set(chunk)) / len(chunk))
    return statistics.mean(ratios) if ratios else len(set(words)) / len(words)


@dataclass
class DocumentContext:
    sentences: list[str]
    sentence_word_lists: list[list[str]] = field(default_factory=list)
    sentence_lengths: list[int] = field(default_factory=list)
    all_words: list[str] = field(default_factory=list)
    all_bigrams_counter: Counter = field(default_factory=Counter)

    mean_sentence_len: float = 0.0
    std_sentence_len: float = 0.0
    mattr: float = 0.0
    bigram_repetition_rate: float = 0.0
    trigram_repetition_rate: float = 0.0
    contraction_rate: float = 0.0
    ai_tell_rate: float = 0.0
    comma_per_sentence: float = 0.0


def build_document_context(sentences: list[str]) -> DocumentContext:
    ctx = DocumentContext(sentences=sentences)
    ctx.sentence_word_lists = [word_tokenize(s) for s in sentences]
    ctx.sentence_lengths = [max(len(w), 1) for w in ctx.sentence_word_lists]
    ctx.all_words = [w for wl in ctx.sentence_word_lists for w in wl]

    if len(ctx.sentence_lengths) >= 2:
        ctx.mean_sentence_len = statistics.mean(ctx.sentence_lengths)
        ctx.std_sentence_len = statistics.pstdev(ctx.sentence_lengths)
    elif ctx.sentence_lengths:
        ctx.mean_sentence_len = float(ctx.sentence_lengths[0])
        ctx.std_sentence_len = 0.0

    ctx.mattr = _mattr(ctx.all_words)
    ctx.bigram_repetition_rate = _repetition_rate(ctx.all_words, 2)
    ctx.trigram_repetition_rate = _repetition_rate(ctx.all_words, 3)
    ctx.all_bigrams_counter = Counter(_ngrams(ctx.all_words, 2))

    full_text = " ".join(sentences)
    n_words = max(len(ctx.all_words), 1)
    ctx.contraction_rate = len(_CONTRACTION_RE.findall(full_text)) / n_words * 100
    ctx.ai_tell_rate = len(_TRANSITION_RE.findall(full_text)) / n_words * 100
    ctx.comma_per_sentence = full_text.count(",") / max(len(sentences), 1)

    return ctx


def compute_sentence_features(index: int, ctx: DocumentContext) -> dict:
    sentence = ctx.sentences[index]
    words = ctx.sentence_word_lists[index]
    n_words = max(len(words), 1)
    n_chars = max(len(sentence), 1)

    unique_ratio = len(set(w.lower() for w in words)) / n_words

    # fraction of this sentence's distinct bigrams that also occur
    # somewhere else in the document (i.e. doc-wide count exceeds the
    # count contributed by this sentence alone)
    sent_bigram_counts = Counter(_ngrams(words, 2))
    if sent_bigram_counts:
        reused = sum(1 for bg, c in sent_bigram_counts.items() if ctx.all_bigrams_counter[bg] > c)
        overlap = reused / len(sent_bigram_counts)
    else:
        overlap = 0.0

    if ctx.std_sentence_len > 1e-6:
        len_z = (len(words) - ctx.mean_sentence_len) / ctx.std_sentence_len
    else:
        len_z = 0.0
    len_abs_dev_ratio = abs(len(words) - ctx.mean_sentence_len) / max(ctx.mean_sentence_len, 1.0)

    avg_word_len = statistics.mean(len(w) for w in words) if words else 0.0

    feats = {
        "sent_len_words": len(words),
        "sent_avg_word_len": avg_word_len,
        "sent_comma_rate": sentence.count(",") / n_chars * 100,
        "sent_semicolon_dash_rate": (sentence.count(";") + sentence.count("—") + sentence.count(" - ")) / n_chars * 100,
        "sent_punct_density": sum(1 for c in sentence if c in ".,;:!?—-") / n_chars,
        "sent_contraction_rate": len(_CONTRACTION_RE.findall(sentence)) / n_words * 100,
        "sent_ai_tell_count": len(_TRANSITION_RE.findall(sentence)),
        "sent_unique_word_ratio": unique_ratio,
        "sent_local_bigram_overlap": overlap,
        "sent_len_zscore_in_doc": len_z,
        "sent_len_abs_dev_ratio": len_abs_dev_ratio,
        "doc_sentence_count": len(ctx.sentences),
        "doc_mean_sentence_len": ctx.mean_sentence_len,
        "doc_sentence_len_cv": (ctx.std_sentence_len / ctx.mean_sentence_len) if ctx.mean_sentence_len else 0.0,
        "doc_mattr": ctx.mattr,
        "doc_bigram_repetition_rate": ctx.bigram_repetition_rate,
        "doc_trigram_repetition_rate": ctx.trigram_repetition_rate,
        "doc_contraction_rate": ctx.contraction_rate,
        "doc_ai_tell_rate": ctx.ai_tell_rate,
        "doc_comma_per_sentence": ctx.comma_per_sentence,
    }
    return feats


def extract_document_features(text: str) -> tuple[list[str], list[dict]]:
    """Convenience entry point: text -> (sentences, per-sentence feature dicts)."""
    sentences = split_sentences(text)
    if not sentences:
        return [], []
    ctx = build_document_context(sentences)
    return sentences, [compute_sentence_features(i, ctx) for i in range(len(sentences))]

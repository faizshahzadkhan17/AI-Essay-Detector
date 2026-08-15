"""
Sentence splitting shared by the signal layer, feature layer, and dataset
builder. Deliberately a plain rule-based splitter (no NLTK/spaCy model) so
that everything upstream of the trained classifier stays inspectable code,
not another model's judgment call.
"""

import re

# Common abbreviations that end in a period but do not end a sentence.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g",
    "i.e", "no", "fig", "vol", "gen", "rep", "sen", "col", "gov", "lt",
    "capt", "u.s", "u.k", "u.n", "a.m", "p.m", "ph.d", "b.a", "m.a",
}

_SENTENCE_END_RE = re.compile(r"([.!?]+)(['\")\]]*)(\s+)(?=[A-Z0-9\"'(])")
_WORD_RE = re.compile(r"[A-Za-z']+")


def split_sentences(text: str) -> list[str]:
    """Split a passage into sentences, preserving original wording.

    Rule-based on punctuation + capitalization, with an abbreviation
    guard. Not linguistically perfect (no model is, on messy essay text)
    but transparent and good enough for sentence-level attribution.
    """
    _, spans = split_sentences_with_spans(text)
    return [s for _, _, s in spans]


def word_tokenize(text: str) -> list[str]:
    """Lightweight word tokenizer (letters + apostrophes only)."""
    return _WORD_RE.findall(text)


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace/newlines to single spaces. Extracted so
    that dataset-building code can normalize chunks the exact same way
    before joining them, guaranteeing split_sentences_with_spans() is
    deterministic and idempotent across re-splits of reconstructed text.
    """
    text = text.strip()
    if not text:
        return ""
    normalized = re.sub(r"[ \t]+", " ", text)
    normalized = re.sub(r"\n{2,}", "\n\n", normalized)
    normalized = normalized.replace("\n\n", " ").replace("\n", " ")
    normalized = re.sub(r" {2,}", " ", normalized)
    return normalized


def reconstruct_doc_text(chunk_texts: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Join whitespace-normalized chunks with a single space, returning
    the joined text and each chunk's (start, end) char range within it.
    Used by the dataset builder so a document assembled from labeled
    chunks (e.g. human paragraphs + AI-polished paragraphs) can be
    re-split into sentences exactly the same way live inference splits
    a pasted essay -- one code path, no train/inference skew.
    """
    parts: list[str] = []
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for raw in chunk_texts:
        norm = normalize_whitespace(raw)
        if not norm:
            ranges.append((cursor, cursor))
            continue
        if cursor > 0:
            parts.append(" ")
            cursor += 1
        start = cursor
        parts.append(norm)
        cursor += len(norm)
        ranges.append((start, cursor))
    return "".join(parts), ranges


def split_sentences_with_spans(text: str) -> tuple[str, list[tuple[int, int, str]]]:
    """Like split_sentences, but also returns the normalized text used and
    each sentence's (start, end) character span within that normalized
    text. Used by the signal layer to map GPT-2 token offsets back to
    sentences without re-tokenizing per sentence (which would lose
    cross-sentence context for the language model).
    """
    normalized = normalize_whitespace(text)
    if not normalized:
        return "", []

    spans: list[tuple[int, int]] = []
    last = 0
    for match in _SENTENCE_END_RE.finditer(normalized):
        candidate_end = match.end(2)
        before_period = normalized[max(0, match.start() - 15):match.start(1)]
        last_word = before_period.strip().split(" ")[-1].lower().rstrip(".")
        if last_word in _ABBREVIATIONS or len(last_word) == 1:
            continue
        start = last
        while start < candidate_end and normalized[start] == " ":
            start += 1
        spans.append((start, candidate_end))
        last = match.end()

    tail_start = last
    while tail_start < len(normalized) and normalized[tail_start] == " ":
        tail_start += 1
    if tail_start < len(normalized):
        spans.append((tail_start, len(normalized)))

    result = [(s, e, normalized[s:e]) for s, e in spans if normalized[s:e].strip()]
    return normalized, result

"""Canonical, ordered feature list shared by training and the live app.
Signal-layer features first, then hand-engineered style features -- the
exact order used to build every numpy array fed to the classifier.

Whole-document aggregate features (the "doc_*" family, plus the one
signal-layer equivalent "sig_doc_perplexity") are deliberately EXCLUDED
from the classifier's input, even though signal_layer.py/features.py
still compute them. Reason, found via live testing on a mixed
human/AI-polished essay: these features are constant across every
sentence in a document, so a linear model given enough of them learns to
classify the *document* and broadcasts that one verdict to every
sentence -- every sentence came back at ~100% AI regardless of its own
content, because one or two doc-level features (first
doc_bigram_repetition_rate/doc_trigram_repetition_rate, then after
removing those, doc_ai_tell_rate) had outsized coefficients and are
identical for every sentence in the same document. Some are also
confounded with our own synthetic-data generation settings
(repetition_penalty=1.15 during local generation mechanically suppresses
bigram/trigram repetition in the self_ai_* buckets).

Every excluded doc_* feature has a properly-scoped sentence-level
(or sentence-relative-to-document) analog that stays in the feature set:
doc_ai_tell_rate -> sent_ai_tell_count, doc_contraction_rate ->
sent_contraction_rate, doc_comma_per_sentence -> sent_comma_rate,
doc_mattr -> sent_unique_word_ratio, doc_sentence_len_cv ->
sent_len_zscore_in_doc / sent_len_abs_dev_ratio. So this isn't a loss of
signal, just forcing that signal to be attributed to the sentence it
actually came from -- which is the entire point of a sentence-level
detector.
"""

from features import FEATURE_NAMES as STYLE_FEATURE_NAMES
from signal_layer import SIGNAL_FEATURE_NAMES

ALL_FEATURE_NAMES = [
    n for n in list(SIGNAL_FEATURE_NAMES) + list(STYLE_FEATURE_NAMES)
    if not n.startswith("doc_") and n != "sig_doc_perplexity"
]

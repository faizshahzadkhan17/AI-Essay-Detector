"""
Pipeline stage 4: run the signal layer (GPT-2) and feature layer (hand
-engineered stylometrics) over every document, producing one feature
row per sentence. This is the only stage that calls the language model.

Both signal_layer.score_document() and features.extract_document_features()
independently call the same deterministic sentence splitter on the same
doc_text, so they -- and the sentence_dataset table built in stage 3 --
all agree on sentence order without any string-matching: we just index
by position.
"""

import time

import pandas as pd

import features
import signal_layer
from config import PROCESSED_DATA_DIR
from feature_schema import ALL_FEATURE_NAMES

DOC_MANIFEST_PATH = PROCESSED_DATA_DIR / "doc_manifest.parquet"
SENTENCE_PATH = PROCESSED_DATA_DIR / "sentence_dataset.parquet"
ELL_DOC_MANIFEST_PATH = PROCESSED_DATA_DIR / "ell_doc_manifest.parquet"
ELL_SENTENCE_PATH = PROCESSED_DATA_DIR / "ell_sentence_dataset.parquet"

FEATURES_OUT = PROCESSED_DATA_DIR / "sentence_features.parquet"
ELL_FEATURES_OUT = PROCESSED_DATA_DIR / "ell_sentence_features.parquet"


def extract_for_corpus(doc_manifest: pd.DataFrame, sentence_df: pd.DataFrame, label_note: str) -> pd.DataFrame:
    rows = []
    t0 = time.time()
    n_docs = len(doc_manifest)
    sentence_groups = {doc_id: g for doc_id, g in sentence_df.groupby("doc_id", sort=False)}
    for i, doc in enumerate(doc_manifest.itertuples()):
        doc_id = doc.doc_id
        doc_text = doc.doc_text
        sub = sentence_groups.get(doc_id)
        if sub is None or sub.empty:
            continue
        try:
            _, sig_feats, _ = signal_layer.score_document(doc_text)
            _, style_feats = features.extract_document_features(doc_text)
        except Exception as e:
            print(f"  [{label_note}] doc {doc_id} failed: {e}")
            continue

        for srow in sub.itertuples():
            idx = srow.sentence_order
            if idx >= len(sig_feats) or idx >= len(style_feats):
                continue
            combined = {**sig_feats[idx], **style_feats[idx]}
            combined["sentence_uid"] = srow.sentence_uid
            combined["doc_id"] = doc_id
            combined["label"] = srow.label
            combined["bucket"] = srow.bucket
            combined["split"] = getattr(srow, "split", "n/a")
            combined["gen_model"] = srow.gen_model
            combined["sentence_text"] = srow.sentence_text
            rows.append(combined)

        if (i + 1) % 250 == 0 or (i + 1) == n_docs:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{label_note}] {i+1}/{n_docs} docs ({rate:.1f} docs/sec, {len(rows)} sentences so far)")

    df = pd.DataFrame(rows)
    missing = [c for c in ALL_FEATURE_NAMES if c not in df.columns]
    if missing:
        raise RuntimeError(f"Feature extraction missing columns: {missing}")
    return df


def main():
    doc_manifest = pd.read_parquet(DOC_MANIFEST_PATH)
    sentence_df = pd.read_parquet(SENTENCE_PATH)
    print(f"Extracting features for {len(doc_manifest)} main-corpus docs...")
    main_feats = extract_for_corpus(doc_manifest, sentence_df, "main")
    main_feats.to_parquet(FEATURES_OUT, index=False)
    print(f"Saved {len(main_feats)} sentence feature rows -> {FEATURES_OUT}")

    if ELL_SENTENCE_PATH.exists() and ELL_DOC_MANIFEST_PATH.exists():
        ell_sentences = pd.read_parquet(ELL_SENTENCE_PATH)
        ell_docs = pd.read_parquet(ELL_DOC_MANIFEST_PATH)
        print(f"Extracting features for {len(ell_docs)} ELL fairness-probe docs...")
        ell_feats = extract_for_corpus(ell_docs, ell_sentences, "ell")
        ell_feats.to_parquet(ELL_FEATURES_OUT, index=False)
        print(f"Saved {len(ell_feats)} ELL sentence feature rows -> {ELL_FEATURES_OUT}")


if __name__ == "__main__":
    main()

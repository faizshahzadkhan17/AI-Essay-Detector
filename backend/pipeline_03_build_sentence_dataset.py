"""
Pipeline stage 3: combine every source into one sentence-level dataset.

Every document (from DAIGT, or self-generated) is represented as a list
of (chunk_text, label) rows -- for a plain human or AI essay that's a
single chunk; for a polished-mixed essay it's alternating human/AI
paragraphs. This stage:

  1. reconstructs each document's full text by joining its chunks
     (sentence_utils.reconstruct_doc_text), tracking each chunk's char
     range in the joined text,
  2. splits the FULL document into sentences with the exact same
     splitter live inference uses (so training and inference never see
     different sentence boundaries for the same text),
  3. labels each resulting sentence by which chunk range its midpoint
     falls in.

This also assigns a document-grouped train/test split so no sentence
from the same source essay appears on both sides -- important because
the polished-mixed documents and the standalone human documents can
share a source essay (see pipeline_02).

The ELL (non-native English) essays are processed separately and never
assigned to train/test -- they exist only as a held-out fairness probe,
used in pipeline_05's honesty report.
"""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR, RANDOM_SEED
from sentence_utils import reconstruct_doc_text, split_sentences_with_spans

N_HUMAN_STANDALONE = 1700
N_AI_STANDALONE = 1700
N_ELL_SAMPLE = 350
MIN_SENTENCE_WORDS = 3

DOC_CHUNKS_OUT = PROCESSED_DATA_DIR / "doc_chunks.parquet"
SENTENCE_OUT = PROCESSED_DATA_DIR / "sentence_dataset.parquet"
ELL_SENTENCE_OUT = PROCESSED_DATA_DIR / "ell_sentence_dataset.parquet"
ELL_DOC_MANIFEST_OUT = PROCESSED_DATA_DIR / "ell_doc_manifest.parquet"
DOC_MANIFEST_OUT = PROCESSED_DATA_DIR / "doc_manifest.parquet"


def load_self_generated() -> pd.DataFrame:
    frames = []
    fresh_path = PROCESSED_DATA_DIR / "self_ai_fresh_chunks.parquet"
    polished_path = PROCESSED_DATA_DIR / "self_ai_polished_chunks.parquet"
    if fresh_path.exists():
        frames.append(pd.read_parquet(fresh_path))
    if polished_path.exists():
        frames.append(pd.read_parquet(polished_path))
    if not frames:
        raise FileNotFoundError("Run pipeline_02_generate_ai_text.py first.")
    return pd.concat(frames, ignore_index=True)


def load_daigt_chunks(exclude_essay_uids: set) -> pd.DataFrame:
    daigt = pd.read_parquet(RAW_DATA_DIR / "daigt.parquet")
    daigt = daigt[~daigt["essay_uid"].isin(exclude_essay_uids)]

    human = daigt[daigt["generated"] == 0]
    ai = daigt[daigt["generated"] == 1]
    human = human.sample(n=min(N_HUMAN_STANDALONE, len(human)), random_state=RANDOM_SEED)
    ai = ai.sample(n=min(N_AI_STANDALONE, len(ai)), random_state=RANDOM_SEED)

    rows = []
    for _, r in pd.concat([human, ai]).iterrows():
        rows.append({
            "doc_id": r["essay_uid"], "group_id": r["essay_uid"], "chunk_idx": 0,
            "chunk_text": r["text"], "label": int(r["generated"]),
            "bucket": "daigt_human" if r["generated"] == 0 else "daigt_ai",
            "gen_model": "human" if r["generated"] == 0 else "daigt_mixed_llms",
            "prompt_style": "n/a", "prompt_domain": "n/a",
        })
    return pd.DataFrame(rows)


def load_ell_chunks() -> pd.DataFrame:
    ell = pd.read_parquet(RAW_DATA_DIR / "ell_nonnative.parquet")
    text_col = [c for c in ell.columns if "text" in c.lower() and "id" not in c.lower()][0]
    ell = ell.sample(n=min(N_ELL_SAMPLE, len(ell)), random_state=RANDOM_SEED)
    rows = []
    for _, r in ell.iterrows():
        rows.append({
            "doc_id": r["essay_uid"], "group_id": r["essay_uid"], "chunk_idx": 0,
            "chunk_text": r[text_col], "label": 0, "bucket": "ell_nonnative",
            "gen_model": "human_nonnative", "prompt_style": "n/a", "prompt_domain": "n/a",
        })
    return pd.DataFrame(rows)


def _label_for_midpoint(mid: int, chunk_ranges: list[tuple[int, int]]) -> int:
    for i, (s, e) in enumerate(chunk_ranges):
        if s <= mid < e:
            return i
    return len(chunk_ranges) - 1


def chunks_to_sentence_and_doc_tables(chunks_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sentence_rows = []
    doc_rows = []
    for doc_id, group in chunks_df.sort_values("chunk_idx").groupby("doc_id", sort=False):
        group = group.reset_index(drop=True)
        doc_text, chunk_ranges = reconstruct_doc_text(group["chunk_text"].tolist())
        if not doc_text:
            continue
        _, spans = split_sentences_with_spans(doc_text)

        doc_rows.append({
            "doc_id": doc_id, "group_id": group.loc[0, "group_id"], "bucket": group.loc[0, "bucket"],
            "doc_text": doc_text, "n_sentences": len(spans),
        })

        for order, (s, e, sent_text) in enumerate(spans):
            if len(sent_text.split()) < MIN_SENTENCE_WORDS:
                continue
            mid = (s + e) // 2
            chunk_i = _label_for_midpoint(mid, chunk_ranges)
            chunk_row = group.loc[chunk_i]
            sentence_rows.append({
                "doc_id": doc_id, "group_id": chunk_row["group_id"],
                "sentence_order": order, "char_start": s, "char_end": e,
                "sentence_text": sent_text, "label": int(chunk_row["label"]),
                "bucket": chunk_row["bucket"], "gen_model": chunk_row["gen_model"],
                "prompt_style": chunk_row["prompt_style"], "prompt_domain": chunk_row["prompt_domain"],
            })

    if not sentence_rows:
        raise RuntimeError("chunks_to_sentence_and_doc_tables produced zero sentences -- check chunk_text column/content upstream.")
    sentence_df = pd.DataFrame(sentence_rows)
    sentence_df["sentence_uid"] = sentence_df["doc_id"] + "_s" + sentence_df["sentence_order"].astype(str)
    doc_df = pd.DataFrame(doc_rows)
    return sentence_df, doc_df


def assign_split(sentence_df: pd.DataFrame, doc_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    doc_groups = doc_df[["doc_id", "group_id"]].drop_duplicates()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    train_idx, test_idx = next(gss.split(doc_groups, groups=doc_groups["group_id"]))
    train_docs = set(doc_groups.iloc[train_idx]["doc_id"])
    sentence_df["split"] = sentence_df["doc_id"].apply(lambda d: "train" if d in train_docs else "test")
    doc_df["split"] = doc_df["doc_id"].apply(lambda d: "train" if d in train_docs else "test")
    return sentence_df, doc_df


def main():
    self_gen = load_self_generated()
    polish_source_uids = set(self_gen.loc[self_gen["bucket"] == "self_ai_polished", "group_id"])

    daigt_chunks = load_daigt_chunks(exclude_essay_uids=polish_source_uids)
    all_chunks = pd.concat([daigt_chunks, self_gen], ignore_index=True)
    all_chunks.to_parquet(DOC_CHUNKS_OUT, index=False)
    print(f"doc_chunks: {len(all_chunks)} rows across {all_chunks['doc_id'].nunique()} docs")
    print(all_chunks.groupby("bucket")["doc_id"].nunique())

    sentence_df, doc_df = chunks_to_sentence_and_doc_tables(all_chunks)
    sentence_df, doc_df = assign_split(sentence_df, doc_df)

    sentence_df.to_parquet(SENTENCE_OUT, index=False)
    doc_df.to_parquet(DOC_MANIFEST_OUT, index=False)

    print(f"\nsentence_dataset: {len(sentence_df)} sentences across {doc_df.shape[0]} docs")
    print(sentence_df.groupby(["split", "label"]).size())
    print("\nby bucket:")
    print(sentence_df.groupby(["bucket", "label"]).size())

    ell_chunks = load_ell_chunks()
    ell_sentences, ell_docs = chunks_to_sentence_and_doc_tables(ell_chunks)
    ell_sentences.to_parquet(ELL_SENTENCE_OUT, index=False)
    ell_docs.to_parquet(ELL_DOC_MANIFEST_OUT, index=False)
    print(f"\nell_sentence_dataset (held out, fairness probe only): {len(ell_sentences)} sentences from {ell_docs.shape[0]} essays")


if __name__ == "__main__":
    main()

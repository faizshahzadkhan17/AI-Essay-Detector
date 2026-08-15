"""
Pipeline stage 1: fetch raw source data.

Downloads two public, no-auth-required Hugging Face datasets:

  - zeyadusf/daigt      -- a processed mirror of the well-known "DAIGT"
    collection built for the Kaggle "LLM - Detect AI Generated Text"
    competition: human essays from the PERSUADE 2.0 corpus (6th-12th
    grade student argumentative writing) paired with AI-generated essays
    on the same prompts from multiple LLMs. Apache-2.0 licensed mirror.

  - tasksource/english-grading -- mirrors the Kaggle "Feedback Prize -
    English Language Learning" competition: essays written by 8th-12th
    grade English Language Learners (non-native English speakers). Used
    ONLY as a held-out fairness probe (never trained on) to test whether
    the detector over-flags non-native writing patterns.

Both are cached to backend/data/raw/ as parquet so later stages don't
re-download.
"""

import pandas as pd
from datasets import load_dataset

from config import RAW_DATA_DIR

DAIGT_OUT = RAW_DATA_DIR / "daigt.parquet"
ELL_OUT = RAW_DATA_DIR / "ell_nonnative.parquet"


def fetch_daigt():
    if DAIGT_OUT.exists():
        print(f"[skip] {DAIGT_OUT} already exists")
        return
    print("Downloading zeyadusf/daigt ...")
    ds = load_dataset("zeyadusf/daigt")
    frames = []
    for split in ds.keys():
        df = ds[split].to_pandas()
        df["hf_split"] = split
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full = full.drop_duplicates(subset=["text"]).reset_index(drop=True)
    full["essay_uid"] = [f"daigt_{i:06d}" for i in range(len(full))]
    full.to_parquet(DAIGT_OUT, index=False)
    print(f"Saved {len(full)} rows -> {DAIGT_OUT}")
    print(full["generated"].value_counts())


def fetch_ell():
    if ELL_OUT.exists():
        print(f"[skip] {ELL_OUT} already exists")
        return
    print("Downloading tasksource/english-grading ...")
    ds = load_dataset("tasksource/english-grading")
    split = "train" if "train" in ds else list(ds.keys())[0]
    df = ds[split].to_pandas()
    df = df.drop_duplicates(subset=[df.columns[df.columns.str.contains("text", case=False)][0]]).reset_index(drop=True)
    df["essay_uid"] = [f"ell_{i:06d}" for i in range(len(df))]
    df.to_parquet(ELL_OUT, index=False)
    print(f"Saved {len(df)} rows -> {ELL_OUT}")
    print(df.columns.tolist())


if __name__ == "__main__":
    fetch_daigt()
    fetch_ell()

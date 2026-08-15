# Dataset

Status: **v1 complete.** All numbers below are the actual built dataset
(see build log at the bottom), not projections.

## Composition

The dataset is built entirely from public, no-authentication sources
plus text we generate ourselves locally. Every document ends up as a
list of `(chunk_text, label)` pairs (label 0 = human, 1 = AI), which get
split into sentences and combined into one sentence-level table
(`backend/data/processed/sentence_dataset.parquet`).

| Bucket | Source | Label(s) | Purpose |
|---|---|---|---|
| `daigt_human` | [zeyadusf/daigt](https://huggingface.co/datasets/zeyadusf/daigt) (HF mirror of the Kaggle "LLM – Detect AI Generated Text" competition data; human side sourced from the [PERSUADE 2.0](https://github.com/scrosseye/persuade_corpus_2.0) corpus of 6th–12th grade student argumentative essays) | 0 (all sentences human) | Bulk of human training signal |
| `daigt_ai` | same mirror | 1 (all sentences AI) | Bulk of AI training signal, spans multiple original LLMs (the mirror does not preserve which specific model produced each essay — see Known Gaps) |
| `self_ai_fresh` | Generated locally with `Qwen2.5-1.5B-Instruct` and `TinyLlama-1.1B-Chat-v1.0`, 2 prompt styles each, across a hand-written bank of Common-App-style personal-narrative and PERSUADE-style persuasive prompts (`backend/essay_prompts.py`) | 1 (all sentences AI) | Adds personal-narrative-register AI essays (DAIGT skews persuasive/argumentative) and a second, controlled set of generator models |
| `self_ai_polished` | A sample of `daigt_human` essays (reserved separately from the `daigt_human` bucket above — never both) with 40–70% of paragraphs rewritten by one of the two local models ("light copy-edit" or "heavy rewrite" prompt), the rest left verbatim | Mixed — 0 for untouched paragraphs, 1 for polished ones, at **sentence-level ground truth** | The realistic, hardest case: a human essay that's been AI-polished, not fully AI-written. This is the case the product is actually built to catch. |
| `ell_nonnative` | [tasksource/english-grading](https://huggingface.co/datasets/tasksource/english-grading) (HF mirror of the Kaggle "Feedback Prize – English Language Learning" competition data: essays by 8th–12th grade English Language Learners, each essay hand-scored 1–5 on grammar/syntax/vocabulary/etc.) | 0 (human, non-native English) | **Held out of training entirely.** Used only as a fairness probe: does the detector over-flag non-native English patterns as AI? See HONESTY_REPORT.md. |

## Splitting

Train/test split is **document-grouped**, not sentence-grouped: an
80/20 `GroupShuffleSplit` over documents, grouped by `group_id`. For a
`self_ai_polished` document, `group_id` is the source essay's ID from
`daigt_human` — this matters because a polished-mixed doc reuses real
sentences from a source essay; without grouping, the same human sentence
could appear in both train and test, inflating apparent accuracy. The
`daigt_human` essays used as polish sources are excluded from the plain
`daigt_human` standalone bucket so no sentence is duplicated across two
different documents in the dataset.

`ell_nonnative` is never in train or test — it's a separate held-out
probe, loaded fresh at evaluation time.

## Known gaps / coverage limitations

- **DAIGT source-model identity isn't preserved by the HF mirror used.**
  We know (from the competition's public documentation) that the AI side
  spans multiple LLMs (GPT-3.5/4, PaLM, LLaMA, Falcon, Cohere, and
  others across different DAIGT dataset versions), but can't break out
  per-row which model produced which essay. The `self_ai_fresh` and
  `self_ai_polished` buckets partly compensate by adding two models we
  do control and can attribute.
- **Topic skew**: the human essays are dominated by PERSUADE's 15
  argumentative prompts (school policy topics like phones-in-class,
  uniforms, community service). Real admissions essays are more
  personal-narrative. `self_ai_fresh`/`self_ai_polished` intentionally
  add personal-narrative prompts, but the human side of that register is
  comparatively under-represented.
- **Age/population skew**: PERSUADE essays are from US 6th–12th graders,
  skewing younger and more school-assignment-like than a graduating
  senior's polished admissions essay.
- **Non-native English coverage**: handled as a dedicated held-out probe
  (`ell_nonnative`) rather than blended into training, specifically so
  we can measure — not accidentally learn to penalize — non-native
  writing patterns. See HONESTY_REPORT.md for the result.
- **Two local generator models**: `Qwen2.5-1.5B-Instruct` and
  `TinyLlama-1.1B-Chat-v1.0` were chosen for being small enough to run
  on a 6GB laptop GPU, not for being representative of what a real
  applicant would use (more likely ChatGPT/GPT-4-class models). The
  DAIGT bucket is the main source of larger-model AI text; our
  self-generated text adds diversity but skews toward smaller/weaker
  generators.
- **Rule-based sentence splitter**: no ML sentence segmentation model is
  used anywhere in the pipeline (deliberately — see the app's hard
  constraint on explainability), so occasional mis-splits on unusual
  punctuation are possible and propagate into both training labels and
  live inference identically.
- **93% of documents are uniformly-labeled** (whole essay human or whole
  essay AI) — only the `self_ai_polished` bucket has genuine within
  -document label variation. This turned out to matter a lot: see
  HONESTY_REPORT.md's first section for how it initially let the
  classifier shortcut to document-level classification instead of real
  sentence-level attribution, and why several whole-document features
  were excluded from the final model as a result. A larger
  `self_ai_polished` bucket would be the highest-value next addition to
  this dataset.

## Build log

Final counts from the actual run (see `backend/pipeline_0N_*.py`, each
idempotent / resumable via cached parquet files):

| Bucket | Documents | Sentences (human / AI) |
|---|---|---|
| `daigt_human` | 1,700 | 33,325 / 0 |
| `daigt_ai` | 1,700 | 0 / 27,645 |
| `self_ai_fresh` | 139 | 0 / 2,521 |
| `self_ai_polished` | 210 | 2,296 / 2,192 |
| **Train/test total** | **3,749** | **35,621 / 32,358** (67,979 total) |
| `ell_nonnative` (held out, never trained on) | 350 | 5,837 / 0 |

Train: 2,999 documents / 54,217 sentences. Test: 750 documents / 13,762
sentences (document-grouped 80/20 split — see "Splitting" above).

Generation models used for `self_ai_fresh` / `self_ai_polished`:
`Qwen/Qwen2.5-1.5B-Instruct` and `TinyLlama/TinyLlama-1.1B-Chat-v1.0`,
run locally via `transformers`, batched generation on a 6GB GPU. Prompt
bank and polish-style prompts are in `backend/essay_prompts.py`.

Two things worth calling out explicitly, found *during* this build
rather than assumed beforehand:

- The `self_ai_fresh` yield (139 of 140 attempted generations) was
  close to 100% — almost nothing got filtered by the length/quality
  guard in `pipeline_02`.
- The DAIGT AI-generated side (27,645 sentences) includes at least one
  visibly degenerate example (an essay that devolves into a repeated
  stutter token) and at least one with garbled emoji-name artifacts
  (see HONESTY_REPORT.md, wrong-example #3) — the source mirror doesn't
  filter generation failures out, and neither did we; they're a small
  fraction of 15,715 AI essays and arguably useful negative-space signal
  (a real detector will encounter broken AI output too), but readers
  should know they're in there.

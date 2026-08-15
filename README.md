# AI Essay Detector

Paste a college admissions essay and see which sentences look
statistically AI-typical, and why — not a single percentage.

## Why this shape

The verdict comes entirely from code we own: a small local language
model (GPT-2) supplies *signals* (per-token predictability), our own
hand-written code supplies *stylometric features* (burstiness, lexical
diversity, repetition, punctuation rhythm), and a logistic regression
we trained ourselves combines them into a probability per sentence. No
step in that path asks a chat model "is this AI-written?" — see
`backend/signal_layer.py` and `backend/analyze.py` for the actual code
path that produces a prediction.

## Architecture

```
backend/
  sentence_utils.py      shared rule-based sentence splitter
  signal_layer.py         GPT-2 per-token log-prob/rank -> per-sentence signals
  features.py              hand-engineered stylometric features (no model calls)
  feature_schema.py        canonical ordered feature list (signal + style)
  explain.py                turns a prediction's feature contributions into plain-language reasons
  analyze.py                ties it all together for one live request
  db.py                     SQLite: users, sessions, essay-analysis history
  app.py                    FastAPI app -- /api/analyze, /api/auth/*, /api/history/*, serves frontend/

  essay_prompts.py               prompt bank used only to build training data
  pipeline_01_fetch_data.py      downloads DAIGT + ELL datasets from HF
  pipeline_02_generate_ai_text.py locally generates AI/polished training essays
  pipeline_03_build_sentence_dataset.py  combines everything into sentence-level labels + train/test split
  pipeline_04_extract_features.py         runs signal+feature layers over every document
  pipeline_05_train_classifier.py         trains + evaluates the classifier, writes eval_report.json
  pipeline_06_write_honesty_report.py     renders documents/HONESTY_REPORT.md from eval_report.json

frontend/                 HTML/CSS/JS: paste box + highlighting, login/signup, essay history,
                           animated background, opening intro -- no framework, no build step
documents/
  DATASET.md              dataset composition, sourcing, known gaps
  HONESTY_REPORT.md        precision/recall/F1, confidently-wrong examples, fairness probe
```

## Running it

```
# one-time setup
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt

# one-time dataset + training pipeline (run in order; each stage caches
# its output as parquet, so re-running is cheap / resumable)
cd backend
python pipeline_01_fetch_data.py
python pipeline_02_generate_ai_text.py      # slow: local GPU generation
python pipeline_03_build_sentence_dataset.py
python pipeline_04_extract_features.py      # runs GPT-2 over every document
python pipeline_05_train_classifier.py      # trains classifier, writes eval data
python pipeline_06_write_honesty_report.py  # renders documents/HONESTY_REPORT.md

# serve the app
uvicorn app:app --reload
# open http://127.0.0.1:8000
```

## Status

Functional end-to-end, visually designed, and tested against a running
server (Playwright, real HTTP requests) rather than just read over.
Held-out logistic regression: 0.79 precision / 0.82 recall / 0.81 F1 /
0.89 ROC-AUC — see `documents/HONESTY_REPORT.md` for the full picture,
including a real bug found and fixed during testing (document-level
features were causing every sentence in an essay to get the same
verdict) and why the shipped numbers are lower than an earlier, rejected
version.

Accounts are lightweight by design: hashed+salted passwords, HttpOnly
session cookies, no rate limiting/CSRF token/HTTPS enforcement -- fine
for local use, not hardened for a public deployment as-is.

# AI Essay Detector — Full Project Explanation

This document explains the whole project from scratch, in plain English.
No prior knowledge of AI or machine learning is assumed. If you're
reading this on GitHub to understand what was built and how, start here.

---

## Table of Contents

1. [What is this project?](#1-what-is-this-project)
2. [The core rule this project follows](#2-the-core-rule-this-project-follows)
3. [How it works — the three-layer pipeline](#3-how-it-works--the-three-layer-pipeline)
4. [Where the training data came from](#4-where-the-training-data-came-from)
5. [Does it actually work? (the honest results)](#5-does-it-actually-work-the-honest-results)
6. [The web app — what you see and do](#6-the-web-app--what-you-see-and-do)
7. [The look and feel](#7-the-look-and-feel)
8. [Technology stack](#8-technology-stack)
9. [Project structure](#9-project-structure)
10. [How to run it yourself](#10-how-to-run-it-yourself)
11. [Known limitations — being honest](#11-known-limitations--being-honest)
12. [Credits and data sources](#12-credits-and-data-sources)

---

## 1. What is this project?

**AI Essay Detector** is a web app for one specific job: you paste a
college admissions essay into it, and it shows you **which sentences**
look statistically like they were written or heavily polished by AI —
and **why** it thinks so, in plain language.

It deliberately does **not** give you one big number like "73% AI." A
single percentage hides more than it tells you. Instead, every sentence
gets its own likelihood, highlighted directly in the text, with a short
list of reasons underneath the ones that got flagged. The idea is to
give an admissions reader *evidence they can actually evaluate*, not a
verdict they have to blindly trust.

It also supports creating an account, so if you're logged in, every
essay you analyze gets saved to your history and you can pull it back up
later.

---

## 2. The core rule this project follows

This is the single most important design decision in the whole project,
so it's worth explaining clearly.

**The easy way to build something like this would be to ask a chatbot
"does this essay look AI-written?" and show its answer.** That approach
was explicitly ruled out from day one, for three reasons:

1. **It's unreliable.** A chatbot's opinion on "is this AI" is just
   another guess — it has no real access to how the text was actually
   produced, and it can be wrong in ways that are hard to predict or
   reproduce.
2. **It can't explain itself honestly.** If you ask a chatbot *why* it
   thinks something is AI, it will generate a plausible-sounding
   explanation — but that explanation isn't necessarily the real reason
   behind its answer, because chatbots don't have introspective access
   to their own reasoning. The "why" would be fiction dressed up as fact.
3. **It's a black box you don't control.** You can't inspect it, fix a
   specific mistake it makes, or prove to someone else exactly why it
   flagged a sentence.

So instead, **every part of the verdict comes from code and math that
this project owns, wrote, and can fully explain.** A small language
model is used, but only to measure a plain statistical fact (see Layer
1 below) — it is never asked the question "is this AI?" and has no
channel to express an opinion. The actual decision is made by a
formula this project trained itself, on data this project collected
itself, and every number that formula uses can be traced back to
exact, auditable arithmetic over the essay's text.

---

## 3. How it works — the three-layer pipeline

Every essay you paste goes through three separate stages. Think of it
like a factory line: each stage does one clear job and hands its output
to the next stage.

### Layer 1: The Signal Layer — "how predictable is each word?"

This layer uses a small, older, publicly available AI language model
called **GPT-2** (from 2019 — much smaller and simpler than modern
chatbots like ChatGPT). GPT-2's only job here is to answer one narrow
question, over and over, for every single word in the essay:

> "Given everything written so far, how likely was this *specific* next
> word, out of every word GPT-2 knows?"

That's it. GPT-2 never sees the question "is this AI-written?" — it
literally has no way to answer that. It only ever computes probabilities
for individual words, which is the exact same thing it was built to do
in the first place.

From those word-by-word probabilities, the project calculates things
like:

- **How predictable was this sentence, on average?** (called
  "perplexity" — low perplexity means very predictable, high perplexity
  means the words were more surprising)
- **How often was the actual word GPT-2's #1 guess?** AI-generated text
  tends to pick the most statistically "safe" word very often. Human
  writing tends to have more quirky, less predictable word choices.
- **How much does the predictability *swing* within a sentence?** Human
  writing tends to mix very predictable stretches with sudden surprising
  word choices. AI writing tends to be more uniformly smooth.

### Layer 2: The Feature Layer — "counting things, no AI involved at all"

This layer is pure arithmetic — counting and measuring things directly
in the text, with zero machine learning. Examples:

- **Sentence-length variation ("burstiness")**: humans naturally mix
  short punchy sentences with longer winding ones. AI text is often
  more uniform in length.
- **Vocabulary variety**: how many different words are used, versus how
  repetitive the word choice is.
- **Repeated phrases**: does the same pair or triplet of words show up
  again elsewhere in the essay?
- **Punctuation and rhythm**: comma usage, use of contractions
  ("don't," "I'm" — AI text tends to under-use these), sentence-starter
  patterns.
- **"AI cliché" words**: certain words and phrases (like *delve*,
  *moreover*, *tapestry*, *in today's society*, *it is important to
  note*) show up disproportionately often in AI-generated writing. Their
  presence in a specific sentence is one small piece of evidence, not
  proof by itself.

### Layer 3: The Decision Layer — the actual verdict

This is the only layer that produces the final probability, and it's a
model this project **trained itself** — a **logistic regression**. In
plain terms: it's a formula that takes every number produced by Layers
1 and 2, multiplies each one by a weight the training process learned,
adds them up, and turns the result into a probability between 0% and
100% that a given sentence is AI-written.

A more powerful model (a gradient-boosted tree ensemble) was also
trained for comparison and actually scored higher. It was **not**
shipped, on purpose: logistic regression's math is simple enough that,
for any single sentence, you can look at exactly which features pushed
its score up or down and by how much. That's what generates the plain-
language "why" shown under every flagged sentence. A model that scores
better but can't explain itself would fail the whole point of the
project.

---

## 4. Where the training data came from

A model like this needs real examples to learn from — human-written
essays and AI-written essays, correctly labeled. Here's how that data
was put together.

- **Human essays**: sourced from a public dataset built from real
  6th–12th grade student essays (the **PERSUADE 2.0** corpus, accessed
  through a Hugging Face mirror of the well-known Kaggle "LLM – Detect
  AI Generated Text" competition data, called **DAIGT**).
- **AI essays**: partly from that same DAIGT collection (which already
  contains essays written by a mix of different AI models), and partly
  generated locally, on this project's own computer, using two small
  open AI models (**Qwen2.5-1.5B-Instruct** and **TinyLlama-1.1B-Chat**)
  — no external API calls, no cost, and full control over exactly what
  was generated.
- **The hardest, most realistic case — AI-polished paragraphs**: a
  fully-AI essay is actually the *easy* case to catch. The harder, more
  realistic scenario is a real student essay where only *some*
  paragraphs were rewritten or "polished" by AI. So a big chunk of the
  training data is exactly that: real human essays with a random subset
  of their paragraphs rewritten by the local AI models, with exact
  sentence-by-sentence records of which parts are human and which parts
  are AI. This is what makes sentence-level detection possible at all.
- **A fairness check group, kept completely separate**: 350 essays
  written by non-native English speakers (from the Kaggle "Feedback
  Prize – English Language Learning" competition data) were set aside
  and **never used to train the model** — they exist purely to test
  whether the finished detector unfairly flags non-native English
  writing patterns as AI (see the next section for the result).

All together, training used roughly **68,000 labeled sentences** across
about 3,750 documents.

---

## 5. Does it actually work? (the honest results)

This project includes an "honesty report" — a results writeup that
reports the real numbers, including a genuine mistake that was found and
fixed, rather than just the flattering parts. Full detail lives in
`documents/HONESTY_REPORT.md`; here's the plain-English summary.

### The headline numbers

Measured on essays the model never saw during training:

| Metric | What it means in plain English | Score |
|---|---|---|
| **Precision** | Of the sentences flagged as AI, how many really were AI | 79% |
| **Recall** | Of the sentences that really were AI, how many got caught | 82% |
| **F1** | A balance of the two above | 81% |
| **ROC-AUC** | Overall ability to rank AI sentences above human ones | 89% |

### A real bug that was found — and why the fix mattered more than the score

Early on, a version of the model scored noticeably *higher* than the
numbers above (93% F1). It looked great on paper. But when tested live
on a real essay that mixed genuine personal writing with one deliberately
AI-polished paragraph, it flagged **the entire essay as AI — including
the plainly human sentences.**

Digging into why: the model had found a shortcut. A handful of its
input numbers described the *whole document* rather than each
individual sentence (things like "how much does this whole essay repeat
phrases overall"). Since those numbers don't change from sentence to
sentence, the model learned to basically classify the *document* once
and repeat that same answer for every sentence in it. That trick still
scores well on the test data (because most training essays are either
entirely human or entirely AI, so guessing "the whole thing" usually
happens to be right) — but it completely defeats the actual purpose of
a sentence-by-sentence detector.

**The fix**: those whole-document shortcut features were removed from
the model, keeping only features that can genuinely vary from sentence
to sentence. This dropped the "official" score from 93% down to the 81%
F1 shown above. That was a deliberate trade: a model that scores lower
but actually points at the right sentences is more useful than one that
scores higher by quietly not doing its job. Re-tested afterward on the
same mixed essay: the AI-polished paragraph correctly scored 86–99%,
and the surrounding human writing correctly scored 7–49%.

### The fairness check: does it unfairly flag non-native English writing?

This was tested directly, since it's a well-known failure mode for AI
detectors in general — and it's specifically the kind of thing
evaluators care about catching.

**Finding: no, it does not over-flag non-native English writing.** If
anything, the opposite happened: essays from non-native English writers
were flagged as AI **10.3%** of the time, compared to **19.0%** for
native-English writers in the same test. That's the *opposite* direction
from the failure mode being tested for. The full report shows the
investigation into *why* (a mix of counteracting signals, not one single
cause) rather than just taking the number at face value.

### Honest weak spots

- About **1 in 5** genuinely human sentences still gets flagged
  (a 19% false-positive rate on native English writing). This is the
  direct cost of removing the "shortcut" features above — a model
  working sentence-by-sentence is noisier than one that can quietly
  fall back on judging the whole document. Individual flags should be
  read as "statistically AI-typical," not "proven AI."
- The detector struggles most with: AI text written in a deliberately
  casual, personal voice; AI text responding to source-based/analytical
  prompts (where human and AI writing naturally converge in structure);
  and any text with heavy non-standard spelling (real or AI-imitated),
  which reads as "unpredictable" to the signal layer regardless of who
  or what wrote it.

---

## 6. The web app — what you see and do

- **Paste box + Analyze button** on the home page.
- **Highlighted essay view**: every sentence is shown with a background
  tint whose intensity reflects its AI-likelihood — a continuous
  gradient, not a hard "flagged / not flagged" line.
- **Flagged sentence cards**: sentences above the threshold get their
  own card showing the exact percentage and a short bullet list of
  plain-language reasons (e.g. *"This sentence leans on longer, more
  elevated vocabulary than the rest of the essay"*).
- **Summary strip**: sentence count, flagged count, and mean
  AI-likelihood — shown as small supporting context, deliberately not
  presented as "the answer."
- **Accounts**: sign up / log in with a username and password. Session
  handled via a secure cookie.
- **History**: every essay you analyze while logged in is saved
  automatically. The History page lists them by date with a quick
  preview; clicking one reloads the exact same sentence-by-sentence
  breakdown.

---

## 7. The look and feel

The interface went through several rounds of visual iteration based on
direct feedback, ending on a dark, minimal, "quietly alive" design:

- A near-black background with a large field of drifting digits,
  letters, Greek letters, and math symbols at varying sizes — a nod to
  "text and data," which fits an AI-analysis tool. It reacts to your
  mouse: nearby characters drift away from the cursor and brighten.
- Glass-panel cards (translucent, blurred backgrounds) for the input box
  and every result card, with a subtle 3D tilt that follows your cursor
  on hover.
- A short opening animation on page load: a glowing icon assembles in
  the center, holds for two seconds, then cracks apart along its middle
  seam as two panels slide away to reveal the app underneath.
- "Elastic" motion on key interactions (buttons, card hover) — a slight
  springy overshoot rather than a flat linear animation.

---

## 8. Technology stack

**Backend (Python)**
- **FastAPI** — the web framework serving the API and the app
- **Uvicorn** — the server that runs FastAPI
- **PyTorch** + **Hugging Face Transformers** — run the GPT-2 signal
  model and the local text-generation models
- **scikit-learn** — trains and runs the logistic regression classifier
- **pandas** / **pyarrow** — data handling during dataset building
- **SQLite** (via Python's built-in `sqlite3`) — stores user accounts,
  sessions, and essay history
- **Passwords**: hashed and salted (PBKDF2), never stored in plain text

**Frontend**
- Plain **HTML, CSS, and JavaScript** — no framework (no React/Vue/etc.)
  and no build step, by choice, to keep the whole thing simple and
  transparent
- The animated background is drawn with the browser's **Canvas API**

**Models used**
- **GPT-2** (124M parameters) — the signal-layer language model
- **Qwen2.5-1.5B-Instruct** and **TinyLlama-1.1B-Chat** — used only to
  *generate training examples* locally, never part of the live verdict

**Testing approach**
- **Playwright** (browser automation) was used throughout development
  to actually load the running app in a real browser, click through it,
  and screenshot the result — catching real bugs (like a CSS layout bug
  and the document-level-feature bug above) that reading the code alone
  would have missed.

---

## 9. Project structure

```
backend/
  sentence_utils.py        rule-based sentence splitter (shared by every layer)
  signal_layer.py          GPT-2 word-predictability signals (Layer 1)
  features.py              hand-written stylometric features (Layer 2)
  feature_schema.py        the exact, ordered list of features fed to the model
  explain.py                turns a prediction into plain-language reasons
  analyze.py                combines everything for one live request
  db.py                     user accounts, sessions, essay history (SQLite)
  app.py                    the FastAPI app: all API endpoints + serves the frontend

  essay_prompts.py                        prompts used only to build training data
  pipeline_01_fetch_data.py               downloads the source datasets
  pipeline_02_generate_ai_text.py         locally generates AI / AI-polished training essays
  pipeline_03_build_sentence_dataset.py   builds the labeled sentence-level dataset
  pipeline_04_extract_features.py         runs Layers 1 & 2 over every training document
  pipeline_05_train_classifier.py         trains the classifier (Layer 3), evaluates it
  pipeline_06_write_honesty_report.py     writes documents/HONESTY_REPORT.md

  data/                     downloaded + built datasets, and the live SQLite database
  artifacts/                the trained model files

frontend/
  index.html / style.css / app.js    the whole UI — no framework, no build step

documents/
  DATASET.md                 exactly what data was used and its known gaps
  HONESTY_REPORT.md           full results, the bug story, and the fairness check
  PROJECT_OVERVIEW.md         this file
```

---

## 10. How to run it yourself

```bash
# one-time setup
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt

# one-time: build the dataset and train the model
# (each stage saves its own output, so re-running is cheap/safe)
cd backend
python pipeline_01_fetch_data.py
python pipeline_02_generate_ai_text.py      # slow — runs local AI models on your GPU/CPU
python pipeline_03_build_sentence_dataset.py
python pipeline_04_extract_features.py      # runs GPT-2 over every training document
python pipeline_05_train_classifier.py
python pipeline_06_write_honesty_report.py

# run the app
uvicorn app:app --reload
# then open http://127.0.0.1:8000
```

---

## 11. Known limitations — being honest

- **GPT-2 is a small, older model.** A larger signal model would likely
  sharpen the predictability-based features, at the cost of speed.
- **Sentence splitting is rule-based**, not AI-based, on purpose (for
  explainability) — so unusual punctuation can occasionally split a
  sentence in an odd place. This affects training and live use exactly
  the same way, so it shouldn't bias the verdict, just occasionally
  misplace a boundary.
- **~1 in 5 genuinely human sentences can still get flagged.** Read any
  single flag as "statistically AI-typical," not definitive proof.
- **Accounts are intentionally lightweight**: hashed passwords and
  secure session cookies, but no email verification, no rate limiting,
  no password reset flow, and no enforced HTTPS. Fine for local/personal
  use; not hardened for a public production deployment as-is.
- **The exact identity of which AI model produced each DAIGT essay is
  unknown** — the source data spans multiple AI models but doesn't
  label which one wrote which essay.

---

## 12. Credits and data sources

- **DAIGT / PERSUADE 2.0** — the human-essay and mixed AI-essay source
  data, via a Hugging Face mirror of the Kaggle "LLM – Detect AI
  Generated Text" competition data.
- **Kaggle "Feedback Prize – English Language Learning"** — the
  non-native-English essay set used for the fairness check.
- **GPT-2, Qwen2.5, TinyLlama** — open-source models from OpenAI,
  Alibaba/Qwen, and TinyLlama, used as described above.
- **Visual design inspiration**: the overall dark, minimal aesthetic and
  motion language were inspired by looking at
  [trionn.com](https://trionn.com), a design studio's portfolio site —
  used only as a *look-and-feel reference*, not as a source of any
  copied assets or content.

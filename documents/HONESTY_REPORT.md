# Honesty Report

Generated from `backend/data/processed/eval_report.json` (produced by `pipeline_05_train_classifier.py`).
Train set: 2999 documents / 54217 sentences.
Test set: 750 documents / 13762 sentences.
Split is document-grouped (see DATASET.md) so no sentence leaks between train and test.

## A finding that shaped the final feature set: document-level features broke sentence-level output

Worth stating up front because it explains why the metrics below are lower than an earlier version we tried and
discarded. The first trained model scored **0.929 precision / 0.936 recall / 0.932 F1** — better numbers than what
follows — but when tested live on a real mixed essay (a genuine personal-narrative paragraph plus one deliberately
AI-polished paragraph), it returned ~100% AI-likelihood for *every single sentence in the document, including the
plainly human ones*. Inspecting the logistic regression's per-feature contributions for individual sentences
showed why: whole-document aggregate features (`doc_bigram_repetition_rate`, `doc_trigram_repetition_rate`, then
after removing those, `doc_ai_tell_rate`) had by far the largest coefficients of any feature in the model, and are
*constant across every sentence in the same document* — so the model was effectively classifying the document and
broadcasting one verdict to every sentence, which happens to still score well on aggregate precision/recall
(most training documents are wholly-human or wholly-AI, so a document-level shortcut gets the sentence label right
most of the time) while completely failing the product's actual purpose: pointing at *which* sentences differ.

The fix was to exclude every `doc_*` broadcast feature from the classifier's input (see
`backend/feature_schema.py`), keeping only features that can vary sentence-to-sentence — each excluded feature has
a properly-scoped per-sentence analog that stays in (e.g. `doc_ai_tell_rate` → `sent_ai_tell_count`). This dropped
held-out F1 from 0.932 to 0.806, which we consider the right trade: a model that
scores worse in aggregate but actually does sentence-level attribution beats one that scores better by silently
not doing the thing the product is for. Re-tested on the same mixed essay afterward: the AI-polished paragraph
scored 0.86–0.99, the surrounding personal narrative scored 0.07–0.49 — real differentiation, not saturation.

## Held-out performance

Threshold: sentence flagged as AI-likely if P(AI) >= 0.5.

**Logistic regression (shipped model)** (n=13762 test sentences, 6479 true-AI)

| Metric | Value |
|---|---|
| Precision | 0.794 |
| Recall | 0.818 |
| F1 | 0.806 |
| ROC-AUC | 0.889 |

Confusion matrix (rows=true, cols=predicted, [human, AI]):
```
[5911, 1372]
[1179, 5300]
```


**Gradient-boosted trees (comparison only, not shipped)** (n=13762 test sentences, 6479 true-AI)

| Metric | Value |
|---|---|
| Precision | 0.843 |
| Recall | 0.845 |
| F1 | 0.844 |
| ROC-AUC | 0.932 |

Confusion matrix (rows=true, cols=predicted, [human, AI]):
```
[6265, 1018]
[1007, 5472]
```


Logistic regression ships in the live app instead of the (here, higher-F1)
gradient-boosted model because its coefficients give an exact, auditable per-feature contribution for every
prediction -- that's what `backend/explain.py` uses to generate the "why" for each flagged sentence. A classifier
we can't explain fails the product's hard constraint even if it scores higher.

## Three confidently-wrong examples

Restricted to documents that are unambiguously all-human or all-AI (excludes the polished-mixed bucket, where a
whole-document label doesn't mean anything -- those documents ARE partially AI by design). Ranked by how far the
mean predicted probability was from the true label.

### 1. `daigt_001681` (bucket: daigt_ai)

- True label: **AI** — Predicted: **human** (mean sentence-level P(AI) = 0.318)
- 27 sentences in this document

> Students shuld have da rite to grade their teechers. Teechers are sposed to be helping us learn and if we dont like da way they teech we shuld be able to say sumthin. Its only fair. First of all, teechers need to no wut theyre doing wrk. If a teecher is borin or not doing a good job, we shuld be able to tel them. Maybe dey wont get mad, but dey might try harder. Like, my LA teacher, Mr. Johnson, h...

**Hypothesis:** A false negative on an essay written with heavy deliberate phonetic misspelling ("shuld", "da rite", "teechers", "sposed", "wut theyre doing") -- whichever DAIGT generator produced this was evidently prompted to imitate a young or careless student's spelling. Our hypothesis: this breaks the same assumption as the non-native-English case (see the fairness probe below) from the opposite direction -- a model imitating non-standard orthography produces token sequences that are genuinely unusual/hard to predict for GPT-2, which our signal layer reads as human-typical unpredictability. Non-standard spelling of any origin, real or imitated, seems to be a blind spot.

### 2. `daigt_001744` (bucket: daigt_ai)

- True label: **AI** — Predicted: **human** (mean sentence-level P(AI) = 0.354)
- 15 sentences in this document

> The Enigma of the Face on Mars While the notion of an ancient civilization on Mars etched into its surface is an intriguing one, compelling evidence suggests the so-called "Face on Mars" is simply a naturally formed landform. Three main pieces of evidence from the article convince me that humans did not carve the formation. Firstly, higher resolution images from later Mars orbiter missions failed ...

**Hypothesis:** Another false negative, on a source-based analytical prompt ("The Enigma of the Face on Mars", citing "the article"). Our hypothesis: source-based/evidence-citing prompts constrain both human and AI writers toward similar structure (topic sentence, numbered evidence, citation phrasing), which compresses the stylistic gap our features look for -- burstiness and lexical diversity in particular are driven partly by the *task*, not just by who/what wrote it. This suggests the detector is weaker on structured/source-based writing tasks than on open-ended personal narrative, which matters since real admissions essays are closer to the latter -- but any supplemental essay asking students to respond to provided material would land in this weaker zone.

### 3. `daigt_004069` (bucket: daigt_ai)

- True label: **AI** — Predicted: **human** (mean sentence-level P(AI) = 0.367)
- 15 sentences in this document

> Employers are looking for specific qualities in potential candidates, and many people may not have experience or qualifications, but being hard-working and responsible can make a great candidate. I have experience volunteering at school and working with job owners as a helper. I have been a school bookkeeper and a helper in a library, and I have also volunteered at a two-company filer. I have mana...

**Hypothesis:** This is a false negative: true AI, predicted human with high confidence. The essay is written in a convincingly casual first-person voice ("I have experience volunteering at school...") describing concrete, specific personal history. Our signal and feature layers were built around a real but imperfect correlation -- AI text is *usually* more uniform/predictable and more formal -- and this essay breaks that correlation: whichever model in the DAIGT mix generated it produced something with human-like burstiness and informal register. This is precisely the kind of essay the product is weakest against: AI output deliberately or incidentally imitating an informal personal register.


## Does the detector over-flag non-native English writing?

We evaluated the trained detector on **5837 sentences from 350 essays**
written by non-native English speakers (8th-12th grade English Language Learners, from the Kaggle
"Feedback Prize - English Language Learning" competition data — genuinely human-written, never used in training),
and compared its false-positive rate to native-English human essays held out in the main test split.

| Population | Sentences | False-positive rate (flagged as AI) |
|---|---|---|
| Non-native English (ELL) | 5837 | 10.3% |
| Native English (test split) | 6781 | 19.0% |

**Finding: the detector does NOT over-flag (if anything, under-flags) non-native English writing, by 8.7 percentage points.**

This is the opposite of the failure mode we were asked to check for, and we looked for a reason before taking the
number at face value. Comparing mean feature values between the two groups, several *individual* signals actually
point the "expected" way -- ELL sentences have slightly lower GPT-2 rank (130 vs 155, i.e. more predictable) and
lower lexical diversity (MATTR 0.78 vs 0.81) than native-English test essays, both of which our features treat as
mildly AI-typical. But two things pull the other way and appear to dominate the trained model's combination: ELL
essays use contractions more often (1.15 vs 1.00 per 100 words -- AI text under-uses contractions, so this reads
strongly human), and ELL sentences run noticeably longer on average (25.9 vs 21.3 words) -- plausibly because our
rule-based sentence splitter (see DATASET.md) merges comma-spliced or run-on clauses, a punctuation pattern more
common in developing writers, into single "sentences," which shifts several length-based burstiness features.

We don't want to over-claim a single causal story here -- the net effect is a logistic regression combining ~30
features, not one feature -- but we also don't want to bury a real result: **on this test, the detector was not
biased against non-native English writing in the direction we were worried about.** The sample is one held-out
population of 350 essays from one competition; we'd want a second, independently-sourced non-native corpus before
calling this settled.


## Known limitations

- **Small signal model**: GPT-2 (124M) is not state-of-the-art; a larger observer model would likely sharpen the
  signal-layer features, at the cost of local inference speed.
- **Rule-based sentence splitter**: no ML sentence segmentation is used anywhere (deliberate, for
  explainability), so unusual punctuation can occasionally mis-split a sentence; this affects training and live
  inference identically, so it shouldn't bias the verdict, just occasionally misplace a boundary.
- **DAIGT source-model identity unknown**: see DATASET.md -- we can't attribute which specific LLM produced each
  DAIGT AI essay, only that the mirror we used spans multiple models.
- **Topic and register skew**: see DATASET.md -- PERSUADE-sourced human essays skew younger and more
  argumentative than a graduating senior's personal statement.
- **Real false-positive rate on native-English human writing is 19.0%** at the
  default 0.5 threshold -- roughly one in five genuinely human sentences in the held-out native-English test set
  gets flagged. This is the direct cost of removing the document-level features described above: a sentence-level
  classifier working from ~25 per-sentence signals alone is noisier than one that could quietly fall back on
  "this whole document reads human." We consider that trade worth it for a tool whose entire purpose is sentence
  attribution, but it means individual flags should be read as "statistically AI-typical," not "proven AI," and
  the UI is built around showing multiple sentences of evidence rather than trusting any single flag in isolation.

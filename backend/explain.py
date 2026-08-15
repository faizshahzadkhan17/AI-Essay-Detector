"""
Turns a trained logistic-regression prediction into a plain-language
"why" for a non-technical reader. Uses the model's own coefficients --
no separate heuristic guesses at what mattered. For a given sentence,
each feature's contribution to the AI-log-odds is
    contribution_i = standardized_value_i * coefficient_i
and the sentence's total log-odds is intercept + sum(contribution_i).
We surface the top few contributions in the SAME direction as the final
verdict, translated through a template that reads naturally whether the
standardized value was unusually high or unusually low.
"""

from __future__ import annotations

# Each entry: (high_z_text, low_z_text). {v} is filled with a rounded
# human-readable magnitude where useful.
_TEMPLATES: dict[str, tuple[str, str]] = {
    "sig_mean_nll": (
        "The words in this sentence were harder for a language model to predict than typical writing.",
        "GPT-2 found this sentence highly predictable -- it assigned high probability to almost every word choice, which is more typical of AI-generated text than human writing.",
    ),
    "sig_perplexity": (
        "This sentence has high 'perplexity' -- it surprised the language model more than average, a hallmark of human phrasing.",
        "This sentence has unusually low 'perplexity' -- a language model found it very easy to guess word-for-word, which leans AI-typical.",
    ),
    "sig_mean_rank": (
        "The words chosen here were often not the model's top-predicted choice, suggesting more idiosyncratic, human-style word choice.",
        "The words chosen here were almost always the model's single most-predicted next word -- an unusually 'safe' pattern typical of AI text.",
    ),
    "sig_median_rank": (
        "Word choices in this sentence frequently deviated from the most statistically expected option.",
        "Word choices in this sentence consistently matched the model's most statistically expected option.",
    ),
    "sig_frac_top1": (
        "Few of this sentence's words were the model's single most likely prediction.",
        "A large share of this sentence's words were exactly the model's single most likely prediction -- unusually uniform, AI-typical phrasing.",
    ),
    "sig_frac_top10": (
        "Word choices often fell outside the model's 10 most likely predictions, a sign of more distinctive phrasing.",
        "Nearly every word fell within the model's 10 most likely predictions -- very 'safe', predictable phrasing.",
    ),
    "sig_frac_top50": (
        "Several words were well outside the model's most-expected vocabulary for this context.",
        "Almost every word stayed within the model's narrow band of most-expected vocabulary.",
    ),
    "sig_surprisal_std": (
        "The predictability of words swings a lot within this sentence -- some very expected, some very surprising, which is typical of human writing.",
        "The predictability of every word in this sentence is oddly uniform, lacking the small surprises typical of human writing.",
    ),
    "sig_max_surprisal": (
        "This sentence contains at least one strikingly unpredictable word choice, typical of human writing.",
        "Even the least-predictable word in this sentence was still fairly expected -- nothing stands out the way an unusual human word choice would.",
    ),
    "sent_len_zscore_in_doc": (
        "This sentence is noticeably longer than the rest of the essay.",
        "This sentence is noticeably shorter than the rest of the essay.",
    ),
    "sent_len_abs_dev_ratio": (
        "This sentence's length breaks sharply from the essay's usual rhythm.",
        "This sentence's length closely matches the essay's average, contributing to an unusually uniform rhythm across the essay.",
    ),
    "sent_ai_tell_count": (
        "This exact sentence contains one or more words/phrases (e.g. 'delve', 'moreover', 'in today's society') that are disproportionately common in AI-generated writing.",
        "This sentence avoids the stock transition words/phrases common in AI-generated writing.",
    ),
    "sent_unique_word_ratio": (
        "This sentence uses an unusually wide range of distinct words for its length.",
        "This sentence repeats words within itself more than typical.",
    ),
    "sent_local_bigram_overlap": (
        "This sentence reuses word-pairs that also appear elsewhere in the essay.",
        "This sentence's phrasing is distinct from the rest of the essay.",
    ),
    "sent_comma_rate": (
        "This sentence is comma-heavy relative to its length.",
        "This sentence uses very few commas relative to its length.",
    ),
    "sent_semicolon_dash_rate": (
        "This sentence leans on semicolons or dashes more than typical.",
        "This sentence avoids semicolons and dashes, using simpler sentence construction.",
    ),
    "sent_punct_density": (
        "This sentence is unusually punctuation-dense for its length.",
        "This sentence has an unusually light punctuation for its length.",
    ),
    "sent_contraction_rate": (
        "This sentence uses contractions (like \"don't\", \"I'm\"), a casual pattern more typical of human writing.",
        "This sentence avoids contractions, a more formal pattern AI writing often defaults to.",
    ),
    "sent_avg_word_len": (
        "This sentence leans on longer, more elevated vocabulary than the rest of the essay.",
        "This sentence uses shorter, plainer words than the rest of the essay.",
    ),
}

_UNSCORED_FALLBACK = "This sentence's overall word-choice pattern statistically resembles {label} writing in our training data."


def explain_sentence(feature_values: dict, feature_names: list[str], coefficients: list[float],
                      scaler_mean: list[float], scaler_scale: list[float], predicted_ai: bool,
                      top_k: int = 3) -> list[str]:
    """Return up to top_k plain-language reasons, all pointing in the
    same direction as the sentence's final predicted label.
    """
    contributions = []
    for name, coef, mean, scale in zip(feature_names, coefficients, scaler_mean, scaler_scale):
        raw = feature_values.get(name, 0.0)
        z = (raw - mean) / scale if scale else 0.0
        contribution = z * coef
        contributions.append((name, contribution, z))

    wanted_sign = 1 if predicted_ai else -1
    aligned = [c for c in contributions if c[1] * wanted_sign > 0]
    aligned.sort(key=lambda c: abs(c[1]), reverse=True)

    reasons = []
    for name, contribution, z in aligned[:top_k]:
        template = _TEMPLATES.get(name)
        if not template:
            continue
        text = template[0] if z > 0 else template[1]
        reasons.append(text)

    if not reasons:
        reasons.append(_UNSCORED_FALLBACK.format(label="AI-generated" if predicted_ai else "human-written"))
    return reasons

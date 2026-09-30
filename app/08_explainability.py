"""
explainability.py
==================
SHAP-based explainability layer for the XGBoost classifier, plus the
score-formula and keyword-overlap explanations used by app.py.

Note: `shap` import is wrapped so a missing/broken shap install can't
crash the whole app — build_explainer() just returns None, and
explain_candidate() degrades gracefully (empty positive/negative lists)
when that happens.
"""

import numpy as np

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    shap = None
    _SHAP_AVAILABLE = False


def build_explainer(model):
    """
    Build SHAP explainer for the trained XGBoost model.
    Returns None if shap isn't installed — callers must handle that.
    """
    if not _SHAP_AVAILABLE:
        return None
    return shap.TreeExplainer(model)


def explain_xgboost_prediction(
    explainer,
    X_row,
    class_index,
    feature_names
):
    """
    Explain one XGBoost prediction.

    X_row MUST be the exact feature vector that was
    passed to XGBoost for prediction.
    """

    shap_output = explainer(
        X_row.reshape(1, -1)
    )

    values = shap_output.values

    if values.ndim == 3:
        shap_values = values[
            0,
            :,
            class_index
        ]
    else:
        shap_values = values[0]

    contributions = []

    for name, value in zip(
        feature_names,
        shap_values
    ):
        contributions.append({
            "feature": name,
            "shap_value": float(value)
        })

    contributions.sort(
        key=lambda x: abs(x["shap_value"]),
        reverse=True
    )

    positive = [
        x for x in contributions
        if x["shap_value"] > 0
    ]

    negative = [
        x for x in contributions
        if x["shap_value"] < 0
    ]

    return {
        "all_features": contributions,
        "positive": positive,
        "negative": negative
    }


def explain_candidate(
    explainer,
    model,
    le,
    tfidf,
    resume_vec,
    jd_vec,
    X_row,
    raw_label,
    final_label,
    confidence,
    bert_sim,
    cosine_sim,
    resume_text_raw=None,
    jd_text_raw=None,
):
    """
    Generate an explanation for one candidate.
    """

    # -----------------------------------------
    # 1. SHAP explanation (skipped gracefully if unavailable)
    # -----------------------------------------
    positive, negative = [], []

    if explainer is not None:
        try:

            class_index = int(
                le.transform([raw_label])[0]
            )

            feature_names = []

            tfidf_names = list(
                tfidf.get_feature_names_out()
            )

            feature_names.extend(
                ["resume_" + x for x in tfidf_names]
            )

            feature_names.extend(
                ["jd_" + x for x in tfidf_names]
            )

            

            feature_names.extend([
                f"resume_bert_{i}"
                for i in range(384)
            ])

            feature_names.extend([
                f"jd_bert_{i}"
                for i in range(384)
            ])

            feature_names.append(
                "bert_similarity"
            )

            X_row = np.asarray(X_row).reshape(-1)

            shap_result = explain_xgboost_prediction(
                explainer,
                X_row,
                class_index,
                feature_names
            )

            positive = shap_result["positive"][:10]
            negative = shap_result["negative"][:10]

        except Exception:
            positive, negative = [], []

    # -----------------------------------------
    # 2. Score explanation
    # -----------------------------------------
    # Kept for internal/report use — app.py no longer displays the raw
    # numbers or the override narrative to keep the HR-facing view to a
    # single clean category label.
    score_explanation = {
        "narrative": (
            f"The final score combines semantic similarity "
            f"({bert_sim:.3f}), TF-IDF similarity "
            f"({cosine_sim:.3f}), and model confidence "
            f"({confidence:.3f})."
        ),

        "override_applied": (
            raw_label != final_label
        ),

        "override_reason": (
            f"The model initially predicted '{raw_label}', "
            f"and the final label was adjusted to "
            f"'{final_label}' using the ranking rules."
            if raw_label != final_label
            else
            f"The final label remains '{final_label}' "
            f"because no label override was applied."
        )
    }

    # -----------------------------------------
    # 3. Keyword explanation
    # -----------------------------------------
    resume_terms = set(
        tfidf.inverse_transform(
            resume_vec.reshape(1, -1)
        )[0]
    )

    jd_terms = set(
        tfidf.inverse_transform(
            jd_vec.reshape(1, -1)
        )[0]
    )

    common_terms = sorted(
        resume_terms.intersection(jd_terms)
    )

    keyword_explanation = (
        common_terms[:20]
    )

    # -----------------------------------------
    # 4. Semantic explanation
    # -----------------------------------------
    if bert_sim >= 0.65:
        reading = "Strong semantic match"
    elif bert_sim >= 0.55:
        reading = "Moderate semantic match"
    else:
        reading = "Low semantic match"

    semantic_note = {
        "reading": reading,
        "bert_similarity": round(
            float(bert_sim),
            3
        )
    }

    # -----------------------------------------
    # 5. Model diagnostic
    # -----------------------------------------
    model_diagnostic = {
        "raw_model_prediction": raw_label,
        "final_label": final_label,
        "model_confidence": round(
            float(confidence),
            4
        ),
        "top_positive_features": positive,
        "top_negative_features": negative
    }

    # -----------------------------------------
    # 6. Return everything expected by app.py
    # -----------------------------------------
    return {
        "score_explanation": score_explanation,
        "keyword_explanation": keyword_explanation,
        "semantic_note": semantic_note,
        "model_diagnostic": model_diagnostic,
        "skill_gap": None,  # filled in by 06_ranking.py if skill_gap_fn is passed
    }
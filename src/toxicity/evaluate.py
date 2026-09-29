"""Measure each method against the hand labels, then estimate toxicity per channel.

Two independent prevalence estimates are reported:

1. Design-based: the weighted share of hand-labelled comments that are toxic.
   Needs no classifier, so it is the most trustworthy number, but it rests on ~400 labels.
2. Classifier-based: the best method applied to every comment, corrected for its
   measured error rates with the Rogan-Gladen estimator:
       true rate = (observed rate + specificity - 1) / (sensitivity + specificity - 1)
   Confidence intervals come from a bootstrap that resamples both the labelled
   sample (uncertainty in sensitivity and specificity) and whole videos within each
   channel (comments on the same video are not independent).
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

from .text import normalise

YES = {"1", "yes", "y", "true", "toxic", "t"}
NO = {"0", "no", "n", "false", "not toxic", "f", "clean"}
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


# ---------------------------------------------------------------- loading

def load_gold(labels_csv, key_csv, scored_csv) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.read_csv(labels_csv, dtype=str).fillna("")
    raw = labels["toxic"].str.strip().str.lower()
    bad = labels[~raw.isin(YES | NO) & (raw != "")]
    if len(bad):
        raise ValueError(f"Rows {list(bad['sample_id'][:5])} have a toxic value that isn't 1/0 or yes/no.")
    labels = labels[raw != ""].assign(y=raw[raw != ""].isin(YES).astype(int))
    key = pd.read_csv(key_csv)
    scored = pd.read_csv(scored_csv)
    gold = labels[["sample_id", "y", "category"]].merge(key, on="sample_id").merge(
        scored.drop(columns=["channel"]), on="comment_id", how="left")
    if gold["text"].isna().any():
        raise ValueError("Some labelled comments are missing from scored.csv. Re-run `score` before `evaluate`.")
    if len(gold) < 50:
        raise ValueError(f"Only {len(gold)} comments are labelled. Label at least 50, ideally all of them.")
    if gold["y"].sum() < 10:
        warnings.warn("Fewer than 10 comments are labelled toxic, so accuracy figures will be very uncertain.")
    return gold, scored


# ---------------------------------------------------------------- metrics

def weighted_metrics(y, pred, w) -> dict:
    y, pred, w = map(np.asarray, (y, pred, w))
    tp = w[(y == 1) & (pred == 1)].sum(); fp = w[(y == 0) & (pred == 1)].sum()
    fn = w[(y == 1) & (pred == 0)].sum(); tn = w[(y == 0) & (pred == 0)].sum()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * prec * sens / (prec + sens) if prec and sens and not np.isnan(prec + sens) else 0.0
    return {"precision": prec, "recall": sens, "specificity": spec, "f1": f1,
            "accuracy": (tp + tn) / w.sum()}


def _best_threshold(scores, y, w) -> float:
    f1s = [weighted_metrics(y, (scores >= t).astype(int), w)["f1"] for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.nanargmax(f1s))])


def char_model():
    # Character n-grams cope with the many spellings of romanised Hindi (bhai / bhaii / bhaiya).
    return make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
        LogisticRegression(class_weight="balanced", C=4.0, max_iter=3000),
    )


@dataclass
class MethodResult:
    name: str
    cv_pred: np.ndarray
    threshold: float | None
    metrics: dict
    ci: dict


def cross_validate(gold: pd.DataFrame, seed: int = 11, folds: int = 5) -> list[MethodResult]:
    """Out-of-fold predictions for each method, so no method is scored on data it was tuned on."""
    y, w = gold["y"].to_numpy(), gold["weight"].to_numpy()
    texts = gold["text"].map(normalise).to_numpy()
    k = max(2, min(folds, int(y.sum()), int((1 - y).sum())))
    skf = StratifiedKFold(k, shuffle=True, random_state=seed)

    lex = gold["lexicon_hit"].astype(int).to_numpy()
    model_pred, char_pred = np.zeros_like(y), np.zeros_like(y)
    for tr, te in skf.split(texts, y):
        t = _best_threshold(gold["model_score"].to_numpy()[tr], y[tr], w[tr])
        model_pred[te] = (gold["model_score"].to_numpy()[te] >= t).astype(int)
        m = char_model().fit(texts[tr], y[tr])
        char_pred[te] = (m.predict_proba(texts[te])[:, 1] >= 0.5).astype(int)

    full_t = _best_threshold(gold["model_score"].to_numpy(), y, w)
    results = []
    for name, pred, thr in (("Word list", lex, None), ("Detoxify multilingual", model_pred, full_t),
                            ("Char n-gram model (trained on your labels)", char_pred, 0.5)):
        results.append(MethodResult(name, pred, thr, weighted_metrics(y, pred, w), metric_ci(gold, pred)))
    return results


def _resample(gold: pd.DataFrame, rng) -> np.ndarray:
    """Bootstrap row indices, resampling within each sampling stratum."""
    idx = []
    for _, ix in gold.groupby("stratum").indices.items():
        idx.append(rng.choice(ix, size=len(ix), replace=True))
    return np.concatenate(idx)


def metric_ci(gold, pred, reps: int = 1000, seed: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    y, w = gold["y"].to_numpy(), gold["weight"].to_numpy()
    draws = []
    for _ in range(reps):
        i = _resample(gold, rng)
        draws.append(weighted_metrics(y[i], pred[i], w[i]))
    df = pd.DataFrame(draws)
    return {c: [float(np.nanpercentile(df[c], 2.5)), float(np.nanpercentile(df[c], 97.5))] for c in df}


# ---------------------------------------------------------------- prevalence

def design_prevalence(gold, draws: int = 20000, seed: int = 5) -> pd.DataFrame:
    """Weighted share of toxic labels per channel, with a stratified Jeffreys interval.

    Each sampling stratum gets a Beta(k + 0.5, n - k + 0.5) posterior for its toxic rate;
    channel-level draws combine the strata by their population sizes. Unlike a bootstrap,
    this stays honest when a stratum has zero toxic comments in the sample.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for ch, g in gold.groupby("channel"):
        total = np.zeros(draws)
        pop = 0.0
        for _, st in g.groupby("stratum"):
            n, k, size = len(st), int(st["y"].sum()), float(st["weight"].iloc[0]) * len(st)
            total += size * rng.beta(k + 0.5, n - k + 0.5, draws)
            pop += size
        rows.append({"channel": ch, "estimate": float(np.average(g["y"], weights=g["weight"])),
                     "low": float(np.percentile(total / pop, 2.5)), "high": float(np.percentile(total / pop, 97.5)),
                     "labelled": len(g)})
    return pd.DataFrame(rows)


def rogan_gladen(p_obs, sens, spec):
    denom = sens + spec - 1
    if denom <= 0.1:
        return np.nan
    return float(np.clip((p_obs + spec - 1) / denom, 0, 1))


def classifier_prevalence(best: MethodResult, gold, scored, reps: int = 1000, seed: int = 9) -> tuple[pd.DataFrame, pd.Series]:
    """Apply the best method to every comment and correct the observed rate for its error rates."""
    scored = scored.copy()
    if best.name.startswith("Word"):
        scored["pred"] = scored["lexicon_hit"].astype(int)
    elif best.name.startswith("Detoxify"):
        scored["pred"] = (scored["model_score"] >= best.threshold).astype(int)
    else:
        m = char_model().fit(gold["text"].map(normalise), gold["y"])
        scored["pred"] = (m.predict_proba(scored["text"].map(normalise))[:, 1] >= 0.5).astype(int)

    y, w = gold["y"].to_numpy(), gold["weight"].to_numpy()
    rng = np.random.default_rng(seed)
    rows = []
    for ch, g in scored.groupby("channel"):
        p_obs = g["pred"].mean()
        est = rogan_gladen(p_obs, best.metrics["recall"], best.metrics["specificity"])
        vids = g.groupby("video_id")["pred"].agg(["sum", "count"])
        boots = []
        for _ in range(reps):
            i = _resample(gold, rng)
            m = weighted_metrics(y[i], best.cv_pred[i], w[i])
            v = vids.iloc[rng.integers(0, len(vids), len(vids))]
            boots.append(rogan_gladen(v["sum"].sum() / v["count"].sum(), m["recall"], m["specificity"]))
        rows.append({"channel": ch, "observed": p_obs, "estimate": est,
                     "low": np.nanpercentile(boots, 2.5), "high": np.nanpercentile(boots, 97.5),
                     "comments": len(g), "videos": len(vids)})
    return pd.DataFrame(rows), scored["pred"]


# ---------------------------------------------------------------- entry point

def evaluate(labels_csv="data/annotation/labels.csv", key_csv="data/annotation/sample_key.csv",
             scored_csv="data/scored.csv", out_dir="results") -> dict:
    gold, scored = load_gold(labels_csv, key_csv, scored_csv)
    methods = cross_validate(gold)
    best = max(methods, key=lambda m: m.metrics["f1"])
    design = design_prevalence(gold)
    clf, preds = classifier_prevalence(best, gold, scored)
    scored = scored.assign(pred=preds.values)

    extras = {
        "reply_vs_top_level": scored.groupby("is_reply")["pred"].mean().rename({True: "replies", False: "top_level"}).to_dict(),
        "median_likes": scored.groupby("pred")["likes"].median().rename({1: "flagged", 0: "not_flagged"}).to_dict(),
        "latin_script_share": float(scored["latin_script"].mean()),
        "flagged_by_video": scored.groupby(["channel", "video_id"])["pred"].mean().rename("rate").reset_index().to_dict("records"),
    }
    out = {
        "labelled": int(len(gold)), "labelled_toxic": int(gold["y"].sum()), "comments": int(len(scored)),
        "methods": [{"name": m.name, "threshold": m.threshold, **{k: float(v) for k, v in m.metrics.items()}, "ci": m.ci} for m in methods],
        "best_method": best.name,
        "prevalence_design": design.to_dict("records"),
        "prevalence_classifier": clf.to_dict("records"),
        **extras,
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    return out

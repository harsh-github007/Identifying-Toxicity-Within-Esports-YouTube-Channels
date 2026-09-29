"""Draw the sample of comments you label by hand.

Toxic comments are rare, so a simple random sample would contain very few of them.
The sample is stratified by channel and by whether either automatic method flagged
the comment, with flagged comments over-sampled. Every labelled comment carries a
weight (stratum size / sample size) so the metrics and prevalence estimates stay unbiased.

The labelling sheet deliberately hides the model score and the word-list result,
so the labels can't be nudged by them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def flagged(df: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
    return (df["model_score"] >= threshold) | df["lexicon_hit"].astype(bool)


def draw_sample(scored: pd.DataFrame, n: int = 400, flagged_share: float = 0.5, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = scored.assign(stratum_flagged=flagged(scored))
    per_channel = n // df["channel"].nunique()
    parts = []
    for ch, g in df.groupby("channel"):
        want_flag = int(round(per_channel * flagged_share))
        fl, un = g[g.stratum_flagged], g[~g.stratum_flagged]
        take_fl = min(want_flag, len(fl))
        take_un = min(per_channel - take_fl, len(un))
        for sub, k, is_fl in ((fl, take_fl, True), (un, take_un, False)):
            if k == 0:
                continue
            pick = sub.iloc[rng.choice(len(sub), size=k, replace=False)].copy()
            pick["weight"] = len(sub) / k
            pick["stratum"] = f"{ch}|{'flagged' if is_fl else 'not flagged'}"
            parts.append(pick)
    out = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    out.insert(0, "sample_id", [f"s{i:04d}" for i in range(1, len(out) + 1)])
    return out


def write_sample(scored_csv="data/scored.csv", out_dir="data/annotation", n=400, seed=7) -> tuple[Path, Path]:
    scored = pd.read_csv(scored_csv)
    sample = draw_sample(scored, n=n, seed=seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sheet = out / "to_label.csv"
    key = out / "sample_key.csv"
    # What the annotator sees: text only, in shuffled order.
    sample[["sample_id", "text"]].assign(toxic="", category="", notes="").to_csv(sheet, index=False)
    sample[["sample_id", "comment_id", "channel", "stratum", "weight"]].to_csv(key, index=False)
    return sheet, key

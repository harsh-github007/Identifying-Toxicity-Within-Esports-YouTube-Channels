"""Charts and a written summary from results/metrics.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

INK, MUTED, ACCENT, SECOND = "#1B1B1F", "#6B6B73", "#E4572E", "#2E86AB"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#BBBBBB"); ax.spines["bottom"].set_color("#BBBBBB")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color="#EEEEEE", linewidth=0.8)
    ax.set_axisbelow(True)


def prevalence_chart(m: dict, path: Path):
    d = pd.DataFrame(m["prevalence_design"]).set_index("channel")
    c = pd.DataFrame(m["prevalence_classifier"]).set_index("channel")
    order = c["estimate"].fillna(d["estimate"]).sort_values().index
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(order) + 2.0), dpi=160)
    for off, df, colour, label in ((-0.14, d, SECOND, "Hand-labelled sample"), (0.14, c, ACCENT, f"All comments ({m['best_method']}, corrected)")):
        df = df.reindex(order)
        ys = [i + off for i in range(len(order))]
        # A percentile interval can sit wholly on one side of the estimate when events are rare.
        lo = ((df["estimate"] - df["low"]) * 100).clip(lower=0).fillna(0)
        hi = ((df["high"] - df["estimate"]) * 100).clip(lower=0).fillna(0)
        ax.errorbar(df["estimate"] * 100, ys, xerr=[lo, hi],
                    fmt="o", color=colour, ecolor=colour, elinewidth=1.4, capsize=3, markersize=5, label=label)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, color=INK)
    ax.set_xlabel("Toxic comments (%) with 95% confidence interval", color=MUTED, fontsize=9)
    ax.set_xlim(left=0)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.28), ncol=2)
    ax.set_title("Estimated share of toxic comments by channel", loc="left", color=INK, fontsize=11)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def methods_chart(m: dict, path: Path):
    rows = m["methods"]
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=160)
    metrics = ["precision", "recall", "f1"]
    colours = [SECOND, "#8FB8DE", ACCENT]
    h = 0.25
    for j, (key, col) in enumerate(zip(metrics, colours)):
        vals = [r[key] * 100 for r in rows]
        lo = [max(0.0, (r[key] - r["ci"][key][0]) * 100) for r in rows]
        hi = [max(0.0, (r["ci"][key][1] - r[key]) * 100) for r in rows]
        ys = [i + (j - 1) * h for i in range(len(rows))]
        ax.barh(ys, vals, height=h, color=col, label=key.upper() if key == "f1" else key.capitalize())
        ax.errorbar(vals, ys, xerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r["name"] for r in rows], color=INK, fontsize=9)
    ax.set_xlim(0, 100); ax.set_xlabel("Score against hand labels (%), out-of-fold", color=MUTED, fontsize=9)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.2))
    ax.set_title("How well each method detects toxicity", loc="left", color=INK, fontsize=11)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def pct(x):
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:.1f}%"


def _separation(design: pd.DataFrame) -> str:
    """States which channels differ, judged by whether their hand-labelled intervals overlap."""
    d = design.sort_values("estimate")
    apart = [(a.channel, b.channel) for a in d.itertuples() for b in d.itertuples()
             if a.estimate < b.estimate and a.high < b.low]
    if not apart:
        return "- Every channel's confidence interval overlaps the others, so none can be called more toxic than another at this sample size."
    pairs = "; ".join(f"{hi} above {lo}" for lo, hi in apart[:4])
    return f"- Intervals that do not overlap: {pairs}. Other differences are within the margin of error."


def write_summary(m: dict, path: Path):
    meth = pd.DataFrame(m["methods"])
    best = meth.loc[meth["f1"].idxmax()]
    design = pd.DataFrame(m["prevalence_design"])
    clf = pd.DataFrame(m["prevalence_classifier"])
    lines = [
        "# Results",
        "",
        f"{m['comments']:,} comments collected; {m['labelled']} labelled by hand, of which {m['labelled_toxic']} were toxic.",
        f"{pct(m['latin_script_share'])} of comments are written in Latin script (English or romanised Hindi).",
        "",
        "## Detection accuracy",
        "",
        "Scored against the hand labels with 5-fold cross-validation, weighted to reflect how the sample was drawn.",
        "",
        "| Method | Precision | Recall | Specificity | F1 (95% CI) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, r in meth.iterrows():
        lines.append(f"| {r['name']} | {pct(r['precision'])} | {pct(r['recall'])} | {pct(r['specificity'])} | "
                     f"{pct(r['f1'])} ({pct(r['ci']['f1'][0])}–{pct(r['ci']['f1'][1])}) |")
    lines += ["", f"Best method: **{best['name']}**.", "", "![Methods](figures/methods.png)", "",
              "## Toxicity by channel", "",
              "| Channel | Hand-labelled estimate | All-comments estimate | Comments | Videos |",
              "| --- | ---: | ---: | ---: | ---: |"]
    merged = design.merge(clf, on="channel", suffixes=("_d", "_c"))
    for _, r in merged.sort_values("estimate_c", ascending=False).iterrows():
        lines.append(f"| {r['channel']} | {pct(r['estimate_d'])} ({pct(r['low_d'])}–{pct(r['high_d'])}) | "
                     f"{pct(r['estimate_c'])} ({pct(r['low_c'])}–{pct(r['high_c'])}) | {int(r['comments']):,} | {int(r['videos'])} |")
    rv = m.get("reply_vs_top_level", {})
    lines += ["", "![Prevalence](figures/prevalence.png)", "", "## Other findings", "",
              f"- Flagged rate in replies: {pct(rv.get('replies'))}; in top-level comments: {pct(rv.get('top_level'))}.",
              f"- Median likes on flagged comments: {m['median_likes'].get('flagged', 'n/a')}; on the rest: {m['median_likes'].get('not_flagged', 'n/a')}.",
              "", "## Caveats", "",
              "- Labels come from one annotator following `docs/labelling-guide.md`. A second annotator on a subset would let us report inter-annotator agreement.",
              _separation(design),
              "- Recent uploads only: rates can shift with a controversial video or a live-stream spike."]
    if best["recall"] < 0.5:
        lines.append(f"- The best method still misses most toxic comments (recall {pct(best['recall'])}), so the all-comments estimates lean heavily on the error correction.")
    path.write_text("\n".join(lines) + "\n")


def report(metrics_json="results/metrics.json", out_dir="results"):
    m = json.loads(Path(metrics_json).read_text())
    figs = Path(out_dir) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    prevalence_chart(m, figs / "prevalence.png")
    methods_chart(m, figs / "methods.png")
    write_summary(m, Path(out_dir) / "results.md")

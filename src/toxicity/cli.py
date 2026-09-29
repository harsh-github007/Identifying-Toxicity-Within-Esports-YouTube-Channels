"""Command line: python -m toxicity <step>

Steps, in order:
  collect    download comments          (needs YOUTUBE_API_KEY)
  score      model + word-list scores   (downloads the Detoxify model once)
  sample     write the labelling sheet  -> data/annotation/to_label.csv
  evaluate   compare methods with your labels, estimate toxicity, draw charts
"""

import argparse
import os
import sys

from .collect import CollectError, collect


def main(argv=None):
    p = argparse.ArgumentParser(prog="toxicity", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="step", required=True)
    c = sub.add_parser("collect"); c.add_argument("--config", default="config/channels.yaml")
    sub.add_parser("score")
    s = sub.add_parser("sample"); s.add_argument("-n", type=int, default=400)
    e = sub.add_parser("evaluate"); e.add_argument("--labels", default="data/annotation/labels.csv")
    a = p.parse_args(argv)

    if a.step == "collect":
        key = os.environ.get("YOUTUBE_API_KEY")
        if not key:
            sys.exit("Set YOUTUBE_API_KEY first. See the README for how to get a free key.")
        try:
            df = collect(a.config, key)
        except CollectError as err:
            sys.exit(f"Stopped: {err}")
        print(f"Saved {len(df):,} comments to data/comments.csv")
    elif a.step == "score":
        from .score import score
        df = score()
        print(f"Scored {len(df):,} comments -> data/scored.csv")
    elif a.step == "sample":
        from .sample import write_sample
        sheet, _ = write_sample(n=a.n)
        print(f"Label the 'toxic' column (1 or 0) in {sheet}, then save it as data/annotation/labels.csv")
    elif a.step == "evaluate":
        from .evaluate import evaluate
        from .report import report
        m = evaluate(labels_csv=a.labels)
        report()
        print(f"Best method: {m['best_method']}. See results/results.md")


if __name__ == "__main__":
    main()

"""Score every comment with a pretrained multilingual toxicity model and the word list."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .lexicon import Lexicon
from .text import is_script_latin


class DetoxifyScorer:
    """Wraps Detoxify's multilingual XLM-RoBERTa model (trained on the Jigsaw multilingual data).

    It was not trained on romanised Hindi, which is exactly why the project measures
    its accuracy against hand labels instead of trusting it.
    """

    name = "detoxify-multilingual"

    def __init__(self, variant: str = "multilingual", device: str | None = None, batch_size: int = 64):
        from detoxify import Detoxify  # imported lazily: large dependency, not needed for tests

        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.model = Detoxify(variant, device=device)
        self.batch_size = batch_size

    def __call__(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t[:512] or "." for t in texts[i : i + self.batch_size]]
            out.extend(self.model.predict(batch)["toxicity"])
            if (i // self.batch_size) % 20 == 0:
                print(f"  scored {min(i + self.batch_size, len(texts)):,} / {len(texts):,}")
        return np.asarray(out, dtype=float)


def score(comments_csv: str | Path = "data/comments.csv", out_csv: str | Path = "data/scored.csv",
          lexicon_path: str | Path = "config/lexicon.csv", scorer=None) -> pd.DataFrame:
    df = pd.read_csv(comments_csv)
    scorer = scorer or DetoxifyScorer()
    df["latin_script"] = df["text"].map(is_script_latin)
    df = pd.concat([df.reset_index(drop=True), Lexicon.load(lexicon_path).score(df["text"])], axis=1)
    df["model_score"] = scorer(df["text"].astype(str).tolist())
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df

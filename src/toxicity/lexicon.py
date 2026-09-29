"""Word-list baseline for Hinglish and English abuse."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .text import normalise


class Lexicon:
    def __init__(self, terms: dict[str, str]):
        self.single = {t: c for t, c in terms.items() if " " not in t}
        self.phrases = {t: c for t, c in terms.items() if " " in t}

    @classmethod
    def load(cls, path: str | Path = "config/lexicon.csv") -> "Lexicon":
        df = pd.read_csv(path, comment="#")
        return cls(dict(zip(df["term"].str.strip().str.lower(), df["category"].str.strip())))

    def matches(self, text: str) -> list[tuple[str, str]]:
        """Returns (term, category) pairs found in the text."""
        norm = normalise(text)
        hits = []
        for tok in norm.split():
            single = re.sub(r"(.)\1", r"\1", tok)  # "chutiyaa" -> "chutiya"
            if tok in self.single:
                hits.append((tok, self.single[tok]))
            elif single in self.single:
                hits.append((single, self.single[single]))
            elif "*" in tok and len(tok) >= 4:  # ch*tiya, f**k
                pattern = re.compile("^" + re.escape(tok).replace(r"\*", "[a-z]{1,2}") + "$")
                hits += [(t, c) for t, c in self.single.items() if len(t) >= 4 and pattern.match(t)]
        padded = f" {norm} "
        hits += [(p, c) for p, c in self.phrases.items() if f" {p} " in padded]
        return hits

    def score(self, texts) -> pd.DataFrame:
        found = [self.matches(t) for t in texts]
        return pd.DataFrame({
            "lexicon_hit": [bool(f) for f in found],
            "lexicon_terms": ["|".join(sorted({t for t, _ in f})) for f in found],
            "lexicon_category": ["|".join(sorted({c for _, c in f})) for f in found],
        })

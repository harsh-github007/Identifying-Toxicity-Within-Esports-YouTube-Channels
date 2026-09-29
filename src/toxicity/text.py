"""Text normalisation shared by the lexicon, the model and the annotation sheet."""

import html
import re
import unicodedata

_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@[\w.-]+")
_REPEAT = re.compile(r"(.)\1{2,}")  # "noooob" -> "noob"
_SPACE = re.compile(r"\s+")

# Common character swaps people use to dodge filters.
_LEET = str.maketrans({"@": "a", "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "$": "s", "7": "t"})


def clean(text) -> str:
    """Readable version of a comment: HTML and links removed, whitespace tidied. Keeps case and emoji."""
    if not isinstance(text, str):
        return ""
    t = html.unescape(text)
    t = t.replace("<br />", " ").replace("<br>", " ")
    t = _TAG.sub(" ", t)
    t = _URL.sub(" ", t)
    return _SPACE.sub(" ", t).strip()


def normalise(text) -> str:
    """Aggressive form used for matching: lower case, accents folded, leetspeak undone, stretched letters squashed."""
    t = unicodedata.normalize("NFKD", clean(text)).encode("ascii", "ignore").decode()
    t = _MENTION.sub(" ", t.lower())
    t = t.translate(_LEET)
    t = _REPEAT.sub(r"\1\1", t)
    t = re.sub(r"[^a-z*\s]", " ", t)
    return _SPACE.sub(" ", t).strip()


def is_script_latin(text: str) -> bool:
    """True when most letters are Latin script (English or romanised Hindi)."""
    letters = [c for c in clean(text) if c.isalpha()]
    if not letters:
        return False
    latin = sum(1 for c in letters if "LATIN" in unicodedata.name(c, ""))
    return latin / len(letters) >= 0.6

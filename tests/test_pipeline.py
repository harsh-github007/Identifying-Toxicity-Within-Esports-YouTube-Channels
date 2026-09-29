import json

import numpy as np
import pandas as pd
import pytest

from toxicity.collect import CollectError, YouTube, check_distinct, collect
from toxicity.evaluate import cross_validate, design_prevalence, rogan_gladen, weighted_metrics
from toxicity.lexicon import Lexicon
from toxicity.sample import draw_sample
from toxicity.text import clean, is_script_latin, normalise


# ------------------------------------------------------------------ text + lexicon

def test_clean_and_normalise():
    assert clean("Hi<br />bro &amp; team https://x.com/y") == "Hi bro & team"
    assert normalise("NOOOOB!!! @user ch00tiya") == "noob chootiya"
    assert is_script_latin("bhai op gameplay") and not is_script_latin("भाई बहुत अच्छा")


@pytest.fixture
def lex():
    return Lexicon.load("config/lexicon.csv")


@pytest.mark.parametrize("text,term", [
    ("tu chutiya hai", "chutiya"),
    ("CHUTIYAAAA", "chutiya"),
    ("fuuuuck off", "fuck"),
    ("ch*tiya bhai", "chutiya"),
    ("f*ck this game", "fuck"),
    ("abey b$dk", "bsdk"),
    ("just go die already", "go die"),
])
def test_lexicon_catches_spelling_tricks(lex, text, term):
    assert term in {t for t, _ in lex.matches(text)}


@pytest.mark.parametrize("text", ["bhai op gameplay", "class is at 5", "scunthorpe united", "mcdonalds after stream"])
def test_lexicon_ignores_clean_text_and_substrings(lex, text):
    assert lex.matches(text) == []


# ------------------------------------------------------------------ collector (mocked API)

class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    """Minimal stand-in for the YouTube API: two channels, one video each, two pages of comments."""

    def __init__(self, comments_disabled=False):
        self.calls = []
        self.comments_disabled = comments_disabled

    def get(self, url, params, timeout):
        self.calls.append((url.rsplit("/", 1)[-1], params))
        ep = url.rsplit("/", 1)[-1]
        if ep == "channels":
            cid = params.get("id") or {"@good": "UCgood"}.get(params.get("forHandle"))
            if not cid:
                return FakeResponse(200, {"items": []})
            return FakeResponse(200, {"items": [{"id": cid, "snippet": {"title": cid},
                                                 "contentDetails": {"relatedPlaylists": {"uploads": "UU" + cid}}}]})
        if ep == "playlistItems":
            return FakeResponse(200, {"items": [{"contentDetails": {"videoId": "v_" + params["playlistId"], "videoPublishedAt": "2026-09-01T00:00:00Z"}}]})
        if ep == "commentThreads":
            if self.comments_disabled:
                return FakeResponse(403, {"error": {"errors": [{"reason": "commentsDisabled"}]}})
            vid, page = params["videoId"], params.get("pageToken", "p1")
            items = [{"snippet": {"topLevelComment": {"id": f"{vid}-{page}-{i}", "snippet": {
                "textOriginal": f"comment {i} on {vid} page {page} with enough words to count", "likeCount": i,
                "authorChannelId": {"value": f"user{i}"}, "publishedAt": "2026-09-02T00:00:00Z"}}},
                "replies": {"comments": [{"id": f"{vid}-{page}-{i}-r", "snippet": {"textOriginal": "reply text here", "likeCount": 0,
                                                                                  "authorChannelId": {"value": "u9"}, "publishedAt": "2026-09-02T01:00:00Z"}}]} if i == 0 else {}}
                     for i in range(3)]
            return FakeResponse(200, {"items": items, **({"nextPageToken": "p2"} if page == "p1" else {})})
        raise AssertionError(ep)


def write_config(tmp_path, channels):
    cfg = tmp_path / "channels.yaml"
    cfg.write_text(json.dumps({"videos_per_channel": 1, "comments_per_video": 100, "include_replies": True, "channels": channels}))
    return cfg


def test_collect_paginates_hashes_authors_and_saves(tmp_path):
    cfg = write_config(tmp_path, [{"name": "A", "id": "UCa"}, {"name": "B", "handle": "@good"}])
    df = collect(cfg, "k", out_dir=tmp_path, yt=YouTube("k", session=FakeSession(), pause=0))
    assert set(df["channel"]) == {"A", "B"}
    assert len(df) == 2 * (6 + 2)  # 2 pages x 3 threads + 2 replies, per channel
    assert df["author"].str.fullmatch(r"[0-9a-f]{12}").all()
    assert not df["author"].str.contains("user").any()
    assert (tmp_path / "comments.csv").exists() and (tmp_path / "collection.json").exists()


def test_collect_names_the_bad_handle(tmp_path):
    cfg = write_config(tmp_path, [{"name": "Missing", "handle": "@nope"}])
    with pytest.raises(CollectError, match="Missing"):
        collect(cfg, "k", out_dir=tmp_path, yt=YouTube("k", session=FakeSession(), pause=0))


def test_collect_skips_videos_with_comments_off(tmp_path):
    cfg = write_config(tmp_path, [{"name": "A", "id": "UCa"}])
    with pytest.raises(CollectError, match="No comments"):
        collect(cfg, "k", out_dir=tmp_path, yt=YouTube("k", session=FakeSession(comments_disabled=True), pause=0))


def test_overlap_guard_catches_the_2021_bug():
    # Same comments filed under two channel names, as in the original dataset.
    texts = [f"ajju bhai this is comment number {i} please reply" for i in range(50)]
    df = pd.DataFrame({"channel": ["Total Gaming"] * 50 + ["Techno Gamerz"] * 50,
                       "video_id": ["v1"] * 50 + ["v2"] * 50, "text": texts + texts})
    with pytest.raises(CollectError, match="share"):
        check_distinct(df)
    df2 = df.copy(); df2.loc[50:, "text"] = [f"different comment {i} about the new update" for i in range(50)]
    check_distinct(df2)  # distinct channels pass


# ------------------------------------------------------------------ sampling + evaluation

def make_scored(n_per_channel=600, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for ch, rate in (("A", 0.05), ("B", 0.15)):
        for i in range(n_per_channel):
            toxic = rng.random() < rate
            rows.append({"comment_id": f"{ch}{i}", "channel": ch, "video_id": f"{ch}v{i % 6}", "is_reply": i % 4 == 0,
                         "likes": int(rng.integers(0, 5)), "latin_script": True,
                         "text": f"tu chutiya hai {i}" if toxic and i % 2 else (f"you are trash {i}" if toxic else f"bhai op stream {i}"),
                         "truth": int(toxic)})
    df = pd.DataFrame(rows)
    df["lexicon_hit"] = df["text"].str.contains("chutiya|trash")
    df["model_score"] = np.clip(df["truth"] * 0.6 + rng.normal(0.2, 0.15, len(df)), 0, 1)
    return df


def test_sample_weights_recover_population_size():
    scored = make_scored()
    s = draw_sample(scored, n=200)
    assert len(s) == 200
    assert abs(s["weight"].sum() - len(scored)) < 1e-6
    assert s.groupby("stratum")["weight"].nunique().eq(1).all()


def test_weighted_metrics_and_rogan_gladen():
    m = weighted_metrics([1, 1, 0, 0], [1, 0, 0, 1], [1, 1, 1, 1])
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["specificity"] == 0.5
    assert rogan_gladen(0.2, 0.9, 0.95) == pytest.approx((0.2 + 0.95 - 1) / (0.9 + 0.95 - 1))
    assert np.isnan(rogan_gladen(0.2, 0.5, 0.55))  # correction refused when the classifier is near random


def test_evaluation_recovers_known_rates():
    scored = make_scored()
    gold = draw_sample(scored, n=300).assign(y=lambda d: d["truth"])
    methods = cross_validate(gold)
    assert {m.name for m in methods} >= {"Word list", "Detoxify multilingual"}
    lex = next(m for m in methods if m.name == "Word list")
    assert lex.metrics["f1"] > 0.9
    prev = design_prevalence(gold).set_index("channel")
    truth = scored.groupby("channel")["truth"].mean()
    for ch in ("A", "B"):
        assert prev.loc[ch, "low"] <= truth[ch] <= prev.loc[ch, "high"]

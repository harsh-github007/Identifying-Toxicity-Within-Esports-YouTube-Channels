"""Collect recent comments from YouTube channels with the YouTube Data API v3.

Quota cost is small: about 1 unit per channel lookup, 1 per 50 uploads listed,
and 1 per 100 comments. The default config uses a few hundred of the 10,000 free daily units.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yaml

from .text import clean, normalise

API = "https://www.googleapis.com/youtube/v3"


class CollectError(RuntimeError):
    pass


@dataclass
class YouTube:
    key: str
    session: requests.Session | None = None
    pause: float = 0.05  # be polite between calls

    def get(self, endpoint: str, **params):
        s = self.session or requests.Session()
        r = s.get(f"{API}/{endpoint}", params={**params, "key": self.key}, timeout=30)
        if r.status_code == 403:
            reason = _reason(r)
            if reason == "commentsDisabled":
                return None
            if reason in ("quotaExceeded", "dailyLimitExceeded"):
                raise CollectError("The API key has used today's quota. It resets at midnight Pacific time.")
            raise CollectError(f"YouTube refused the request ({reason}). Check that the YouTube Data API v3 is enabled for this key.")
        if r.status_code == 400 and _reason(r) in ("keyInvalid", "badRequest"):
            raise CollectError("YouTube rejected the API key. Copy it again from Google Cloud Console → Credentials.")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        time.sleep(self.pause)
        return r.json()


def _reason(response) -> str:
    try:
        return response.json()["error"]["errors"][0]["reason"]
    except Exception:
        return f"HTTP {response.status_code}"


def resolve_channel(yt: YouTube, entry: dict) -> dict:
    """Returns {id, title, uploads} for a config entry with `id` or `handle`."""
    params = {"part": "snippet,contentDetails"}
    if entry.get("id"):
        params["id"] = entry["id"]
    elif entry.get("handle"):
        params["forHandle"] = entry["handle"]
    else:
        raise CollectError(f"Channel '{entry.get('name')}' needs an `id` or a `handle` in config/channels.yaml.")
    data = yt.get("channels", **params)
    items = (data or {}).get("items") or []
    if not items:
        what = entry.get("id") or entry.get("handle")
        raise CollectError(f"No channel found for {entry.get('name')} ({what}). Open the channel page and copy its @handle or UC... ID into config/channels.yaml.")
    ch = items[0]
    return {"id": ch["id"], "title": ch["snippet"]["title"], "uploads": ch["contentDetails"]["relatedPlaylists"]["uploads"]}


def recent_videos(yt: YouTube, uploads_playlist: str, n: int) -> list[dict]:
    out, token = [], None
    while len(out) < n:
        data = yt.get("playlistItems", part="contentDetails", playlistId=uploads_playlist,
                      maxResults=min(50, n - len(out)), **({"pageToken": token} if token else {}))
        if not data:
            break
        for it in data.get("items", []):
            cd = it["contentDetails"]
            out.append({"video_id": cd["videoId"], "video_published_at": cd.get("videoPublishedAt")})
        token = data.get("nextPageToken")
        if not token:
            break
    return out[:n]


def _hash(author_channel_id: str | None) -> str:
    # Keep authors distinguishable (for repeat-offender analysis) without storing who they are.
    return hashlib.sha256((author_channel_id or "unknown").encode()).hexdigest()[:12]


def _row(snippet: dict, *, comment_id: str, parent_id: str | None, channel: str, video: dict) -> dict:
    return {
        "channel": channel,
        "video_id": video["video_id"],
        "video_published_at": video.get("video_published_at"),
        "comment_id": comment_id,
        "parent_id": parent_id,
        "is_reply": parent_id is not None,
        "author": _hash((snippet.get("authorChannelId") or {}).get("value")),
        "text": clean(snippet.get("textOriginal") or snippet.get("textDisplay") or ""),
        "likes": int(snippet.get("likeCount", 0)),
        "published_at": snippet.get("publishedAt"),
    }


def video_comments(yt: YouTube, video: dict, channel: str, limit: int, replies: bool) -> list[dict]:
    rows, token, threads = [], None, 0
    while threads < limit:
        data = yt.get("commentThreads", part="snippet,replies", videoId=video["video_id"], maxResults=100,
                      order="time", textFormat="plainText", **({"pageToken": token} if token else {}))
        if data is None:  # comments turned off
            break
        for th in data.get("items", []):
            top = th["snippet"]["topLevelComment"]
            rows.append(_row(top["snippet"], comment_id=top["id"], parent_id=None, channel=channel, video=video))
            threads += 1
            if replies:
                for rep in (th.get("replies") or {}).get("comments", []):
                    rows.append(_row(rep["snippet"], comment_id=rep["id"], parent_id=top["id"], channel=channel, video=video))
            if threads >= limit:
                break
        token = data.get("nextPageToken")
        if not token:
            break
    return rows


def check_distinct(df: pd.DataFrame, max_overlap: float = 0.15) -> None:
    """Stops if two channels share too many comments or any video.

    The 2021 version of this project saved one video's comments under six channel names;
    this guard makes that mistake impossible to repeat silently.
    """
    by_video = df.groupby("video_id")["channel"].nunique()
    shared = by_video[by_video > 1]
    if len(shared):
        raise CollectError(f"Video(s) {list(shared.index)[:3]} appear under more than one channel.")

    # Compare only substantive comments; "hi" and "op" are common everywhere.
    norm = df.assign(n=df["text"].map(normalise))
    norm = norm[norm["n"].str.len() >= 20]
    sets = {ch: set(g["n"]) for ch, g in norm.groupby("channel")}
    for a, b in itertools.combinations(sets, 2):
        union = sets[a] | sets[b]
        if union and len(sets[a] & sets[b]) / len(union) > max_overlap:
            raise CollectError(f"{a} and {b} share {len(sets[a] & sets[b]) / len(union):.0%} of their comments. "
                               "That points to a collection error, so nothing was saved.")


def collect(config_path: str | Path, api_key: str, out_dir: str | Path = "data", yt: YouTube | None = None) -> pd.DataFrame:
    cfg = yaml.safe_load(Path(config_path).read_text())
    yt = yt or YouTube(api_key)
    rows, channels = [], []
    for entry in cfg["channels"]:
        ch = resolve_channel(yt, entry)
        vids = recent_videos(yt, ch["uploads"], cfg.get("videos_per_channel", 10))
        n_before = len(rows)
        for v in vids:
            rows += video_comments(yt, v, entry["name"], cfg.get("comments_per_video", 500), cfg.get("include_replies", True))
        channels.append({**ch, "name": entry["name"], "videos": len(vids), "comments": len(rows) - n_before})
        print(f"  {entry['name']:<16} {len(vids):>3} videos  {len(rows) - n_before:>6} comments")

    df = pd.DataFrame(rows).drop_duplicates("comment_id")
    if df.empty:
        raise CollectError("No comments were collected. Check the channel list and that comments are enabled.")
    df = df[df["text"].str.len() > 0]
    check_distinct(df)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "comments.csv", index=False)
    meta = {"collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "channels": channels, "rows": len(df)}
    (out / "collection.json").write_text(json.dumps(meta, indent=2))
    return df

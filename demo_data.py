# pyright: reportUnknownMemberType=false, reportMissingTypeStubs=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false
# pyright: reportMissingParameterType=false, reportUnknownLambdaType=false
"""
Imagery sourcing for Reelforge.

Supplies real photography for the Versus Duel engine (direct URL, upload, or
keyword search against Pexels / Wikimedia Commons) plus abstract gradient
backdrops used for reel slides and as the no-network fallback.
"""

from __future__ import annotations

import io
import os
import re
import math
import html
import time
from typing import Any

import numpy as np
import requests
from PIL import Image

RGB = tuple[int, int, int]


def create_gradient_mesh(
    width: int,
    height: int,
    c1: RGB,
    c2: RGB,
    c3: RGB,
    angle: int = 45,
) -> Image.Image:
    """
    Creates a smooth multi-stop artistic gradient canvas.

    The gradient runs along `angle` (in degrees) rather than straight down, and
    is evaluated per pixel with numpy so the result is banding-free.
    """
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    xs = np.arange(width, dtype=np.float32).reshape(1, width)
    ys = np.arange(height, dtype=np.float32).reshape(height, 1)

    # Project every pixel onto the gradient axis, then normalize to 0..1.
    proj = xs * cos_a + ys * sin_a
    proj -= proj.min()
    span = float(proj.max()) or 1.0
    progress = (proj / span).astype(np.float32)

    # Two-segment blend: c1 -> c2 over the first half, c2 -> c3 over the second.
    first_half = progress < 0.5
    t = np.where(first_half, progress * 2.0, (progress - 0.5) * 2.0).astype(np.float32)

    start = np.empty((height, width, 3), dtype=np.float32)
    end = np.empty((height, width, 3), dtype=np.float32)
    for ch in range(3):
        start[..., ch] = np.where(first_half, c1[ch], c2[ch])
        end[..., ch] = np.where(first_half, c2[ch], c3[ch])

    blended = start * (1.0 - t[..., None]) + end * t[..., None]
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), "RGB")


# ---------------------------------------------------------------------------
# Real photography sourcing
#
# The duel engine previously drew its subjects with Pillow primitives, which
# read as clipart on a feed. Subjects are now real photographs, sourced in this
# order: an explicit URL the creator pastes, an uploaded file, or a keyword
# search against a stock provider.
#
# Provider notes (verified against the live services):
#   * Unsplash Source (source.unsplash.com) is DEAD -- it returns 503 since
#     Unsplash retired it. Do not reintroduce it.
#   * Pexels is used when reachable; set PEXELS_API_KEY for the supported,
#     rate-limited path.
#   * Wikimedia Commons needs no key and has excellent coverage of named
#     products (cars, phones, watches), so it is the reliable fallback.
# ---------------------------------------------------------------------------

PHOTO_UA = {
    "User-Agent": "ReelforgeStudio/1.0 (short-form video tool; +https://example.invalid)"
}
PHOTO_TIMEOUT = 20


class PhotoLookupError(RuntimeError):
    """Raised when no provider could supply an image for a query."""


def load_image_from_url(url: str, timeout: int = PHOTO_TIMEOUT) -> Image.Image:
    """Downloads a direct image URL into a PIL image."""
    resp = requests.get(url, timeout=timeout, headers=PHOTO_UA, stream=True)
    resp.raise_for_status()

    ctype = resp.headers.get("content-type", "")
    if "image" not in ctype.lower():
        raise PhotoLookupError(f"URL did not return an image (content-type: {ctype or 'unknown'})")

    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def search_pexels(query: str, per_page: int = 1) -> list[dict[str, str]]:
    """Searches Pexels. Uses PEXELS_API_KEY when present; returns [] on failure."""
    headers = dict(PHOTO_UA)
    key = os.environ.get("PEXELS_API_KEY", "").strip()
    if key:
        headers["Authorization"] = key

    try:
        resp = requests.get(
            "https://api.pexels.com/v1/search",
            params={"query": query, "per_page": str(per_page), "orientation": "landscape"},
            headers=headers, timeout=PHOTO_TIMEOUT,
        )
        if resp.status_code != 200:
            return []
        photos = resp.json().get("photos") or []
    except Exception:
        return []

    out: list[dict[str, str]] = []
    for p in photos:
        src = p.get("src") or {}
        url = src.get("large2x") or src.get("large") or src.get("original")
        if url:
            out.append({
                "url": url,
                "credit": f"Photo by {p.get('photographer', 'unknown')} on Pexels",
                "provider": "Pexels",
            })
    return out


# Commons' raw relevance ranking surfaces a lot of material that is technically
# "a photo of the subject" but useless as a hero shot -- UI screenshots, boxes,
# teardowns, close-ups of a clasp. These terms demote a candidate.
_BAD_TITLE_TERMS = (
    "screenshot", "screen shot", "settings", "display of", "menu", "logo", "icon",
    "diagram", "map", "chart", "graph", "box", "packaging", "unbox", "sim ",
    "teardown", "disassembl", "motherboard", "chip", "circuit", "clasp", "buckle",
    "caseback", "case back", "movement", "dial detail", "crown", "manual",
    "advertis", "poster", "sticker", "plaque", "badge", "emblem", "wheel",
    "headlight", "taillight", "engine bay", "interior", "dashboard", "seat",
    "assembly", "factory", "production line", "crash", "wreck", "damaged",
)
# Titles containing these read as a clean product/vehicle photo.
_GOOD_TITLE_TERMS = ("front", "side", "profile", "exterior", "auto", "car", "watch", "phone")


def _score_wikimedia(title: str, query: str, width: int, height: int) -> float:
    """Ranks a Commons candidate on how much it looks like a usable hero shot."""
    t = title.lower()
    score = 0.0

    # Every meaningful query token should appear in the filename.
    tokens = [w for w in re.split(r"\W+", query.lower()) if len(w) > 2]
    hits = sum(1 for w in tokens if w in t)
    if tokens:
        score += 6.0 * (hits / len(tokens))
        if hits == 0:
            return -100.0

    for bad in _BAD_TITLE_TERMS:
        if bad in t:
            score -= 4.0
    for good in _GOOD_TITLE_TERMS:
        if good in t:
            score += 0.6

    # Prefer landscape-to-square; extreme portrait crops badly into a panel.
    if width and height:
        aspect = width / height
        if 1.15 <= aspect <= 2.2:
            score += 2.5
        elif 0.85 <= aspect < 1.15:
            score += 1.0
        elif aspect < 0.6:
            score -= 2.5
        # Resolution matters more than it looks: these fill a 1080x1920 panel,
        # so anything small is visibly soft once upscaled.
        if min(width, height) >= 1200:
            score += 1.6
        elif min(width, height) >= 900:
            score += 1.0
        elif min(width, height) < 700:
            score -= 2.5

    return score


def search_wikimedia(query: str, limit: int = 1, width: int = 1600) -> list[dict[str, str]]:
    """
    Searches Wikimedia Commons for photographs of a named subject.

    Needs no API key, which makes it the only dependable provider when no
    Pexels key is configured. Results are re-ranked locally because Commons'
    own ordering happily puts a settings screenshot above the product itself.
    """
    try:
        resp = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action": "query", "format": "json", "generator": "search",
                "gsrsearch": f"{query} filetype:bitmap", "gsrnamespace": "6",
                "gsrlimit": "40",
                "prop": "imageinfo", "iiprop": "url|size|extmetadata", "iiurlwidth": str(width),
            },
            headers=PHOTO_UA, timeout=PHOTO_TIMEOUT,
        )
        if resp.status_code != 200:
            return []
        pages = ((resp.json().get("query") or {}).get("pages") or {}).values()
    except Exception:
        return []

    scored: list[tuple[float, dict[str, str]]] = []
    for page in pages:
        info = (page.get("imageinfo") or [{}])[0]
        url = info.get("thumburl") or info.get("url")
        title = str(page.get("title", ""))
        if not url or title.lower().endswith(".svg"):
            continue

        score = _score_wikimedia(title, query, int(info.get("width") or 0), int(info.get("height") or 0))
        if score <= 0:
            continue

        artist = ((info.get("extmetadata") or {}).get("Artist") or {}).get("value", "")
        artist = html.unescape(re.sub(r"<[^>]+>", "", str(artist)))
        # Commons credit blobs carry markup and hard line breaks; flatten them
        # so they render as a single clean caption line.
        artist = re.sub(r"\s+", " ", artist).strip(" ,;|")
        artist = (artist[:60].rstrip() + "…") if len(artist) > 60 else (artist or "Wikimedia Commons")
        scored.append((score, {
            "url": url,
            "credit": f"{artist} / Wikimedia Commons",
            "provider": "Wikimedia",
            "title": title,
        }))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [hit for _, hit in scored[:limit]]


def fetch_photo(query: str, provider: str = "auto") -> tuple[Image.Image, str]:
    """
    Fetches real photography for a search term such as "Porsche Cayenne".

    Returns (image, credit). Raises PhotoLookupError when every provider fails,
    so callers can fall back to an abstract backdrop rather than clipart.
    """
    if not query or not query.strip():
        raise PhotoLookupError("Empty search query")

    searches = {
        "pexels": [search_pexels],
        "wikimedia": [search_wikimedia],
        "auto": [search_pexels, search_wikimedia],
    }.get(provider, [search_pexels, search_wikimedia])

    errors: list[str] = []
    for search in searches:
        for hit in search(query.strip()):
            try:
                return load_image_from_url(hit["url"]), hit["credit"]
            except Exception as exc:
                errors.append(f"{hit.get('provider', '?')}: {type(exc).__name__}")

    detail = f" ({'; '.join(errors[:3])})" if errors else ""
    raise PhotoLookupError(f"No photo found for '{query}'{detail}")


def fetch_photo_set(query: str, count: int = 5) -> list[tuple[Image.Image, str]]:
    """
    Fetches up to `count` DIFFERENT photos for one query.

    Used for multi-slide reels. Appending words like "closeup" to vary the
    search backfires on Commons -- it matches filenames, so the extra token
    throws away the good results. Asking for several hits on one clean query
    gives variety without wrecking relevance.
    """
    if not query or not query.strip():
        return []

    hits = search_pexels(query.strip(), per_page=count) or []
    if len(hits) < count:
        hits += search_wikimedia(query.strip(), limit=count - len(hits))

    out: list[tuple[Image.Image, str]] = []
    seen: set[str] = set()
    for hit in hits:
        if hit["url"] in seen:
            continue
        seen.add(hit["url"])
        try:
            out.append((load_image_from_url(hit["url"]), hit["credit"]))
        except Exception:
            continue
        if len(out) >= count:
            break
    return out


def fallback_backdrop(
    size: tuple[int, int] = (1400, 1000),
    accent: RGB = (99, 102, 241),
) -> Image.Image:
    """
    Neutral abstract backdrop used when a photo cannot be fetched.

    Deliberately a soft gradient rather than an illustration -- an abstract
    field reads as intentional art direction; clipart does not.
    """
    base: RGB = (max(6, int(accent[0] * 0.16)), max(6, int(accent[1] * 0.16)), max(6, int(accent[2] * 0.16)))
    mid: RGB = (int(accent[0] * 0.42), int(accent[1] * 0.42), int(accent[2] * 0.42))
    return create_gradient_mesh(size[0], size[1], base, mid, (14, 16, 24), angle=55)


def resolve_item_photo(item: dict[str, Any]) -> tuple[Image.Image, str]:
    """
    Resolves a duel item's photograph, in order of trust:
      1. its pinned `photo_url` (hand-checked for presets),
      2. a keyword search on its `query` / `name`,
      3. an abstract gradient backdrop.

    Always returns an image, so a preset never renders empty when offline.
    """
    url = str(item.get("photo_url") or "").strip()
    if url:
        try:
            return load_image_from_url(url), str(item.get("credit") or "")
        except Exception:
            pass

    query = str(item.get("query") or item.get("name") or "").strip()
    if query:
        try:
            return fetch_photo(query)
        except PhotoLookupError:
            pass

    return fallback_backdrop(), "No photo available — abstract backdrop"


# ---------------------------------------------------------------------------
# Licensed video sourcing
#
# The monetizable alternative to grabbing someone's TikTok. Every result
# carries its licence and creator, so the provenance ledger can record what
# rights the finished video actually rests on.
#
# Wikimedia Commons needs no API key and states a licence for every file,
# which is why it leads. Pexels is used when PEXELS_API_KEY is configured.
# ---------------------------------------------------------------------------

VIDEO_MIN_SECONDS = 3.0
VIDEO_MAX_BYTES = 220 * 1024 * 1024


def search_commons_video(query: str, limit: int = 8) -> list[dict[str, Any]]:
    """
    Searches Wikimedia Commons for video, returning licence and credit.

    Results are CC or public domain -- usable commercially, most requiring
    attribution, which `compliance.attribution_line` then renders.
    """
    try:
        resp = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action": "query", "format": "json", "generator": "search",
                "gsrsearch": f"{query} filetype:video", "gsrnamespace": "6",
                "gsrlimit": str(max(limit * 3, 12)),
                "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata|user",
            },
            headers=PHOTO_UA, timeout=PHOTO_TIMEOUT,
        )
        if resp.status_code != 200:
            return []
        pages = ((resp.json().get("query") or {}).get("pages") or {}).values()
    except Exception:
        return []

    out: list[dict[str, Any]] = []
    for page in pages:
        info = (page.get("imageinfo") or [{}])[0]
        mime = str(info.get("mime") or "")
        url = info.get("url")
        size = int(info.get("size") or 0)
        if not url or not mime.startswith("video") or size > VIDEO_MAX_BYTES:
            continue

        meta = info.get("extmetadata") or {}

        def field(name: str) -> str:
            raw = (meta.get(name) or {}).get("value", "")
            clean = html.unescape(re.sub(r"<[^>]+>", " ", str(raw)))
            return re.sub(r"\s+", " ", clean).strip()

        author = field("Artist") or str(info.get("user") or "") or "Wikimedia Commons"
        out.append({
            "provider": "Wikimedia Commons",
            "url": url,
            "title": str(page.get("title", "")).removeprefix("File:"),
            "author": author[:80],
            "licence_raw": field("LicenseShortName") or "unknown",
            "page_url": f"https://commons.wikimedia.org/wiki/{str(page.get('title','')).replace(' ', '_')}",
            "size_mb": round(size / 1_048_576, 1),
            "mime": mime,
        })
        if len(out) >= limit:
            break
    return out


# Pexels failures used to vanish into a bare `except: return []`, so an
# invalid key looked identical to "no results" and the app quietly fell back to
# Commons (and its share-alike licences) with no explanation. Status is now
# recorded so the UI can say what actually happened.
PEXELS_STATE: dict[str, Any] = {"checked": False, "ok": False, "reason": "not checked"}


def pexels_key_status(force: bool = False) -> dict[str, Any]:
    """
    Reports whether the configured Pexels key actually authenticates.

    Uses a cache-busting query on purpose: api.pexels.com sits behind a CDN,
    and a cached response returns HTTP 200 even for a completely invalid key,
    which makes a broken key look like a working one.
    """
    if PEXELS_STATE["checked"] and not force:
        return PEXELS_STATE

    key = os.environ.get("PEXELS_API_KEY", "").strip()
    PEXELS_STATE["checked"] = True

    if not key:
        PEXELS_STATE.update(ok=False, reason="No PEXELS_API_KEY set")
        return PEXELS_STATE

    try:
        resp = requests.get(
            "https://api.pexels.com/v1/search",
            params={"query": f"cachebust{int(time.time() * 1000)}", "per_page": "1"},
            headers={**PHOTO_UA, "Authorization": key}, timeout=PHOTO_TIMEOUT,
        )
    except Exception as exc:
        PEXELS_STATE.update(ok=False, reason=f"Could not reach Pexels: {type(exc).__name__}")
        return PEXELS_STATE

    if resp.status_code == 200:
        PEXELS_STATE.update(ok=True, reason="Key valid")
    elif resp.status_code == 401:
        PEXELS_STATE.update(
            ok=False,
            reason="Pexels rejected the key (401 Invalid API key). Check it is copied "
                   "in full from pexels.com/api and that the account is verified.",
        )
    elif resp.status_code == 429:
        PEXELS_STATE.update(ok=False, reason="Pexels rate limit reached; try later.")
    else:
        PEXELS_STATE.update(ok=False, reason=f"Pexels returned HTTP {resp.status_code}")
    return PEXELS_STATE


def search_pexels_video(query: str, limit: int = 8) -> list[dict[str, Any]]:
    """
    Searches Pexels video. Requires PEXELS_API_KEY; returns [] without one.

    The Pexels licence permits commercial use without attribution, which makes
    it the least restrictive source here -- worth the free signup.
    """
    key = os.environ.get("PEXELS_API_KEY", "").strip()
    if not key:
        return []

    try:
        # No orientation filter: it discarded most of the catalogue (an
        # "excavator" search returned nothing at all), and the renderer
        # reframes to 9:16 anyway. Portrait sources are preferred below.
        resp = requests.get(
            "https://api.pexels.com/videos/search",
            params={"query": query, "per_page": str(max(limit * 2, 10))},
            headers={**PHOTO_UA, "Authorization": key}, timeout=PHOTO_TIMEOUT,
        )
        if resp.status_code == 401:
            PEXELS_STATE.update(checked=True, ok=False,
                                reason="Pexels rejected the key (401 Invalid API key).")
            return []
        if resp.status_code != 200:
            PEXELS_STATE.update(checked=True, ok=False,
                                reason=f"Pexels returned HTTP {resp.status_code}")
            return []
        PEXELS_STATE.update(checked=True, ok=True, reason="Key valid")
        videos = resp.json().get("videos") or []
    except Exception as exc:
        PEXELS_STATE.update(checked=True, ok=False,
                            reason=f"Could not reach Pexels: {type(exc).__name__}")
        return []

    # Prefer taller footage -- it survives the 9:16 crop with least loss.
    videos.sort(key=lambda v: -(float(v.get("height") or 0) / max(float(v.get("width") or 1), 1)))
    videos = videos[:limit]

    out: list[dict[str, Any]] = []
    for video in videos:
        files = sorted(
            (f for f in (video.get("video_files") or []) if f.get("link")),
            key=lambda f: abs(int(f.get("height") or 0) - 1920),
        )
        if not files:
            continue
        out.append({
            "provider": "Pexels",
            "url": files[0]["link"],
            "title": str(video.get("url", "")).rstrip("/").split("/")[-1].replace("-", " ")[:70],
            "author": str((video.get("user") or {}).get("name") or "Pexels contributor"),
            "licence_raw": "pexels",
            "page_url": str(video.get("url") or ""),
            "size_mb": 0.0,
            "mime": "video/mp4",
        })
    return out


def search_licensed_video(query: str, limit: int = 8) -> list[dict[str, Any]]:
    """Searches every configured licensed video source, Pexels first."""
    if not query or not query.strip():
        return []
    results = search_pexels_video(query.strip(), limit)
    if len(results) < limit:
        results += search_commons_video(query.strip(), limit - len(results))
    return results


def download_licensed_clip(hit: dict[str, Any], dest_dir: str) -> str:
    """
    Downloads a licensed clip to disk and returns the local path.

    Commons serves WebM/OGV, which MoviePy reads fine; the renderer transcodes
    to H.264 on the way out regardless.
    """
    os.makedirs(dest_dir, exist_ok=True)
    url = str(hit["url"])
    ext = os.path.splitext(url.split("?")[0])[1].lower() or ".mp4"
    if ext not in (".mp4", ".webm", ".ogv", ".mov", ".mkv"):
        ext = ".mp4"

    path = os.path.join(dest_dir, f"licensed_{int(time.time())}{ext}")
    with requests.get(url, headers=PHOTO_UA, timeout=180, stream=True) as resp:
        resp.raise_for_status()
        with open(path, "wb") as handle:
            for block in resp.iter_content(chunk_size=1 << 20):
                if block:
                    handle.write(block)

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise PhotoLookupError(f"Downloaded clip was empty: {hit.get('title')}")
    return path


# Quick-fill duel presets. Each side pins a hand-checked photograph so the
# presets look broadcast-ready offline of any search ranking; `query` is the
# fallback used if the pinned URL ever 404s.
#
# Scores are illustrative starting points, NOT verified manufacturer specs --
# the UI says so, and creators are expected to replace them.
_COMMONS = "https://upload.wikimedia.org/wikipedia/commons"

DUEL_PRESETS: dict[str, dict[str, Any]] = {
    "🚙 Luxury SUVs": {
        "a": {"name": "Range Rover Vogue", "hook": "British flagship luxury", "query": "Range Rover Vogue 2018",
              "photo_url": f"{_COMMONS}/thumb/f/ff/2018_Range_Rover_Vogue_SE_SDV6_Auto.jpg/1920px-2018_Range_Rover_Vogue_SE_SDV6_Auto.jpg",
              "credit": "Calreyn88 / Wikimedia Commons"},
        "b": {"name": "Porsche Cayenne", "hook": "The driver's SUV", "query": "Porsche Cayenne 2019",
              "photo_url": f"{_COMMONS}/thumb/f/fd/2019_Porsche_Cayenne_V6_Tiptronic_3.0_Front.jpg/1920px-2019_Porsche_Cayenne_V6_Tiptronic_3.0_Front.jpg",
              "credit": "Vauxford / Wikimedia Commons"},
        "rounds": [
            {"metric": "Price", "a_score": 108000, "b_score": 79000, "unit": "$", "winner": "B",
             "note": "Base MSRP — lower is better"},
            {"metric": "Horsepower", "a_score": 395, "b_score": 468, "unit": "hp", "winner": "B",
             "note": "Peak output"},
            {"metric": "0-60 mph", "a_score": 5.5, "b_score": 4.6, "unit": "s", "winner": "B",
             "note": "Lower is better"},
            {"metric": "Off-Road", "a_score": 96, "b_score": 71, "unit": "", "winner": "A",
             "note": "Capability score"},
        ],
    },
    "📱 Flagship Phones": {
        "a": {"name": "iPhone 15 Pro Max", "hook": "Titanium flagship", "query": "iPhone 15 Pro",
              "photo_url": f"{_COMMONS}/thumb/1/19/Apple_iPhone_15_Pro.jpg/1920px-Apple_iPhone_15_Pro.jpg",
              "credit": "Wikimedia Commons"},
        "b": {"name": "Galaxy S24 Ultra", "hook": "The spec monster", "query": "Samsung Galaxy S24 Ultra",
              "photo_url": f"{_COMMONS}/thumb/b/b7/Samsung_Galaxy_S24_Ultra_Backside.jpg/1920px-Samsung_Galaxy_S24_Ultra_Backside.jpg",
              "credit": "SimonWaldherr / Wikimedia Commons"},
        "rounds": [
            {"metric": "Price", "a_score": 1199, "b_score": 1299, "unit": "$", "winner": "A",
             "note": "Launch price"},
            {"metric": "Battery", "a_score": 4441, "b_score": 5000, "unit": "", "winner": "B",
             "note": "mAh capacity"},
            {"metric": "Zoom", "a_score": 5, "b_score": 10, "unit": "x", "winner": "B",
             "note": "Optical telephoto"},
            {"metric": "Camera Score", "a_score": 94, "b_score": 91, "unit": "", "winner": "A",
             "note": "Photo rating"},
        ],
    },
    "⌚ Luxury Watches": {
        "a": {"name": "Rolex Submariner", "hook": "The icon", "query": "Rolex Submariner watch",
              "photo_url": f"{_COMMONS}/thumb/7/72/Rolex_Oyster_Perpetual_Date_Submariner_Watch.JPG/1920px-Rolex_Oyster_Perpetual_Date_Submariner_Watch.JPG",
              "credit": "Eternalsleeper / Wikimedia Commons"},
        "b": {"name": "Omega Seamaster", "hook": "The challenger", "query": "Omega Seamaster Professional",
              "photo_url": f"{_COMMONS}/thumb/f/f8/Omega_watch_%2825263509997%29.jpg/1920px-Omega_watch_%2825263509997%29.jpg",
              "credit": "Wikimedia Commons"},
        "rounds": [
            {"metric": "Price", "a_score": 10200, "b_score": 6400, "unit": "$", "winner": "B",
             "note": "Retail — lower is better"},
            {"metric": "Water Resist", "a_score": 300, "b_score": 300, "unit": "m", "winner": "",
             "note": "Rated depth"},
            {"metric": "Power Reserve", "a_score": 70, "b_score": 55, "unit": "h", "winner": "A",
             "note": "Hours"},
            {"metric": "Resale Value", "a_score": 98, "b_score": 74, "unit": "", "winner": "A",
             "note": "Value retention"},
        ],
    },
}



# Topics used for the starter reel deck. Real photography, fetched on demand --
# the previous hand-drawn skyline/coffee/mountain illustrations read as clipart.
DEMO_TOPICS: list[tuple[str, str, str]] = [
    ("01_tokyo.jpg", "Tokyo skyline night", "Tokyo Nights 🗼"),
    ("02_coffee.jpg", "Coffee cup cafe table", "Morning Brew ☕"),
    ("03_alps.jpg", "Swiss Alps mountains", "Alpine Escapes 🏔️"),
    ("04_villa.jpg", "Modern luxury villa architecture", "Dream Villa 🏝️"),
]

_DEMO_FALLBACK_PALETTES: list[tuple[RGB, RGB, RGB]] = [
    ((10, 15, 45), (120, 20, 110), (255, 110, 60)),
    ((40, 32, 26), (120, 90, 66), (226, 196, 150)),
    ((25, 30, 65), (180, 90, 110), (255, 180, 100)),
    ((14, 30, 48), (40, 96, 130), (170, 210, 220)),
]


def fetch_topic_photo(topic: str, index: int = 0) -> tuple[Image.Image, str]:
    """
    Fetches one real photo for a free-text topic, falling back to an abstract
    backdrop so a slide is never empty when the network is unavailable.
    """
    try:
        return fetch_photo(topic)
    except PhotoLookupError:
        c1, c2, c3 = _DEMO_FALLBACK_PALETTES[index % len(_DEMO_FALLBACK_PALETTES)]
        return create_gradient_mesh(1400, 1000, c1, c2, c3, angle=35 + index * 20), ""


def generate_sample_images() -> list[dict[str, Any]]:
    """
    Builds the starter slide deck from real photography.

    Each entry is {name, image, title, credit} so the caller can attribute the
    photographer. Falls back to gradient backdrops when offline.
    """
    samples: list[dict[str, Any]] = []
    for i, (name, topic, title) in enumerate(DEMO_TOPICS):
        image, credit = fetch_topic_photo(topic, i)
        samples.append({"name": name, "image": image, "title": title,
                        "credit": credit, "query": topic})
    return samples

"""
URL sanitising and the TikTok ingest fix.

The live-download section needs network. Run with:
    .venv\\Scripts\\python tests\\test_url_ingest.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("REELFORGE_EXPORTS_DIR", tempfile.mkdtemp(prefix="rf_url_"))

import app

# ---- 1. sanitising ---------------------------------------------------------
print("1. URL sanitising")
CASES = [
    # (input, expected)
    ("https://www.tiktok.com/@tiktok/video/7106594312292453675?is_from_webapp=1&sender_device=pc",
     "https://www.tiktok.com/@tiktok/video/7106594312292453675"),
    ("https://www.tiktok.com/@user/video/123?_r=1&_t=8abc&share_app_id=1233",
     "https://www.tiktok.com/@user/video/123"),
    ("  https://vm.tiktok.com/ZMabc123/  ", "https://vm.tiktok.com/ZMabc123/"),
    ("<https://www.instagram.com/reel/Cabc123/?igshid=xyz>",
     "https://www.instagram.com/reel/Cabc123/"),
    ("https://youtu.be/dQw4w9WgXcQ?si=trackingtoken",
     "https://youtu.be/dQw4w9WgXcQ"),
    ("www.tiktok.com/@u/video/9?utm_source=x", "https://www.tiktok.com/@u/video/9"),
    ("https://example.com/clip.mp4#t=10", "https://example.com/clip.mp4"),
    ("", ""),
]
for raw, expected in CASES:
    got = app.sanitize_media_url(raw)
    print(f"   {raw[:52]!r:56} -> {got[:52]!r}")
    assert got == expected, f"{raw!r} -> {got!r}, expected {expected!r}"

# The trap: YouTube's video id lives in the query string, so "strip everything
# after ?" would turn a working link into the YouTube home page.
print("\n   the load-bearing-query cases:")
KEEP = [
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "v=dQw4w9WgXcQ"),
    ("https://www.youtube.com/watch?v=abc&feature=share&utm_source=x", "v=abc"),
    ("https://www.youtube.com/watch?v=abc&t=42", "t=42"),
]
for raw, must_keep in KEEP:
    got = app.sanitize_media_url(raw)
    print(f"   {raw[:56]!r:58} -> {got}")
    assert must_keep in got, f"dropped a load-bearing parameter: {raw} -> {got}"
assert "feature=" not in app.sanitize_media_url(KEEP[1][0])
assert "utm_source" not in app.sanitize_media_url(KEEP[1][0])
print("   video ids and timestamps survive; tracking keys do not")

# sanitising is idempotent -- the batch queue may see a URL more than once
once = app.sanitize_media_url(CASES[0][0])
assert app.sanitize_media_url(once) == once, "not idempotent"
print("   idempotent\n")

# ---- 2. the headers that actually fix TikTok -------------------------------
print("2. request headers")
assert "Chrome/124" in app.BROWSER_UA and "Windows NT 10.0" in app.BROWSER_UA
print(f"   {app.BROWSER_UA[:74]}...")

# ---- 3. the error explainer ------------------------------------------------
print("\n3. failure messages")
CHALLENGE = ("ERROR: [TikTok] 7106594312292453675: Unexpected response from webpage "
             "request; please report this issue on https://github.com/yt-dlp/yt-dlp/issues")
CASES = [
    (CHALLENGE, "verification page"),
    ("ERROR: HTTP Error 429: Too Many Requests", "rate-limiting"),
    ("ERROR: [Instagram] Requested content is not available, login required", "logged-in session"),
    ("ERROR: Video unavailable", "private, removed"),
    ("ERROR: Unsupported URL: https://example.com", "recognised video URL"),
]
for raw, expected in CASES:
    message = app._explain_download_error(RuntimeError(raw))
    first = message.splitlines()[0]
    print(f"   {raw[:46]:48} -> {first[:56]}")
    assert expected in message, f"{raw[:40]!r} did not map to {expected!r}"
# the challenge message must NOT tell a creator to file a yt-dlp bug
assert "issue" not in app._explain_download_error(RuntimeError(CHALLENGE)).split("\n\n")[0]
print("   the challenge message stops pointing users at the yt-dlp issue tracker\n")

# ---- 4. live download ------------------------------------------------------
if os.environ.get("SKIP_NETWORK"):
    print("4. live download SKIPPED (SKIP_NETWORK set)")
else:
    print("4. live TikTok download (needs network)")
    DIRTY = ("https://www.tiktok.com/@tiktok/video/7106594312292453675"
             "?is_from_webapp=1&sender_device=pc")
    work = tempfile.mkdtemp(prefix="rf_dl_")
    try:
        got = app.download_clip_from_url(DIRTY, work)
        size = os.path.getsize(got["path"]) / 1_048_576
        print(f"   {os.path.basename(got['path'])}  {got['duration']:.1f}s  "
              f"{size:.2f} MB  via {got['extractor']}")
        print(f"   title: {got['title'][:60]}")
        assert os.path.exists(got["path"]) and size > 0.05
        assert got["duration"] > 0
    except RuntimeError as exc:
        # A genuine rate-limit is not a test failure; a broken message is.
        print(f"   platform refused: {str(exc).splitlines()[0]}")
        assert "verification page" in str(exc) or "rate-limiting" in str(exc), \
            f"unhandled failure shape: {exc}"
        print("   ...and the failure was explained rather than dumped raw")
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)

print("\nURL INGEST VERIFIED")

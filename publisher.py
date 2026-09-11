# pyright: reportUnknownMemberType=false, reportMissingTypeStubs=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingImports=false
"""
Direct publishing to YouTube via the Data API v3.

Self-contained and additive: no other engine imports this, and nothing here
reaches into the render pipeline. It takes a finished file and some metadata.

Three things worth stating plainly, because they are what makes an uploader
either safe or a liability:

* **Credentials never leave the machine.** The OAuth client secret and the
  refresh token live under `.secrets/`, which is gitignored, written 0600 where
  the OS allows it, and never logged. Nothing in this module prints a token,
  and `account_status()` deliberately returns channel identity only.

* **Uploads resume.** A three-hour ambient render is 2-6GB. A non-resumable
  upload that dies at 94% on a flaky connection has to start again from zero,
  and it will die, because it is a single HTTP request held open for an hour.
  This uses the resumable protocol with explicit chunking, so a retry re-sends
  one chunk rather than the file.

* **Nothing is published by surprise.** The default privacy is `private`.
  Going public is a deliberate choice in the UI, and a scheduled publish is
  forced to `private` first, because that is what the API requires and getting
  it wrong publishes immediately instead of on the date you picked.
"""

from __future__ import annotations

import json
import os
import stat
import time
from typing import Any, Callable

from paths import PROJECT_ROOT, ensure_dir

ProgressFn = Callable[[float, str], None]

# Upload scope only. youtube.force-ssl or the full youtube scope would also
# grant read/write over comments, playlists and the channel itself; this
# module only ever inserts a video, so it asks for only that.
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

SECRETS_DIR = os.path.join(PROJECT_ROOT, ".secrets")
TOKEN_PATH = os.path.join(SECRETS_DIR, "youtube_token.json")
CLIENT_SECRETS_PATH = os.path.join(SECRETS_DIR, "client_secrets.json")

# 8MB chunks. Large enough that a multi-gigabyte file is not thousands of
# round trips, small enough that a dropped connection costs seconds to redo
# and that the progress bar moves often enough to look alive.
CHUNK_BYTES = 8 * 1024 * 1024
MAX_RETRIES = 5

# The two categories this studio's output falls into. The full list is long
# and mostly irrelevant here.
CATEGORIES: dict[str, str] = {
    "27": "Education",
    "10": "Music",
    "24": "Entertainment",
    "22": "People & Blogs",
    "28": "Science & Technology",
}
DEFAULT_CATEGORY = "27"

PRIVACY_STATUSES: dict[str, str] = {
    "private": "🔒 Private (only you)",
    "unlisted": "🔗 Unlisted (anyone with the link)",
    "public": "🌍 Public",
    "scheduled": "🗓️ Scheduled",
}
DEFAULT_PRIVACY = "private"

# YouTube's own limits. Enforced here so a 4,900-character description fails
# in the form rather than after a 4GB upload.
TITLE_LIMIT = 100
DESCRIPTION_LIMIT = 5000
TAGS_TOTAL_LIMIT = 500


class PublishError(RuntimeError):
    """Raised with a message meant for the person looking at the screen."""


# ---------------------------------------------------------------------------
# Credential storage
# ---------------------------------------------------------------------------

def _secure(path: str) -> None:
    """Best-effort 0600. Windows ignores the mode; the gitignore is the backstop."""
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def has_client_secrets() -> bool:
    return os.path.exists(CLIENT_SECRETS_PATH)


def save_client_secrets(raw: str | bytes) -> str:
    """
    Stores the OAuth client secret downloaded from Google Cloud Console.

    Validated before it is written: the commonest mistake is uploading the
    *service account* key instead, which cannot do an installed-app flow and
    fails much later with an opaque error.
    """
    ensure_dir(SECRETS_DIR)
    body = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise PublishError(f"That is not valid JSON: {exc}") from exc

    if not any(key in parsed for key in ("installed", "web")):
        kind = "service account" if parsed.get("type") == "service_account" else "unknown"
        raise PublishError(
            f"This looks like a {kind} key. YouTube uploads need an OAuth 2.0 "
            "Client ID of type 'Desktop app' — in Google Cloud Console, "
            "APIs & Services → Credentials → Create credentials → OAuth client ID."
        )

    with open(CLIENT_SECRETS_PATH, "w", encoding="utf-8") as handle:
        handle.write(body)
    _secure(CLIENT_SECRETS_PATH)
    return CLIENT_SECRETS_PATH


def has_token() -> bool:
    return os.path.exists(TOKEN_PATH)


def forget_channel() -> bool:
    """Deletes the stored token. The client secret is left in place."""
    if not os.path.exists(TOKEN_PATH):
        return False
    os.remove(TOKEN_PATH)
    return True


def _load_credentials() -> Any:
    """
    Returns stored credentials, refreshing them if they have expired.

    An access token lasts an hour; the refresh token does not expire unless the
    user revokes it or the app stays in testing mode, where Google expires it
    after seven days. That second case is the one that will actually happen, so
    it gets its own message.
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    if not os.path.exists(TOKEN_PATH):
        return None

    try:
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    except Exception as exc:
        raise PublishError(f"The stored token is unreadable ({exc}). "
                           "Re-authorize the channel.") from exc

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception as exc:
            raise PublishError(
                "Could not refresh the stored token. If your Google Cloud "
                "project is still in Testing, refresh tokens expire after 7 "
                f"days — re-authorize the channel. ({type(exc).__name__})"
            ) from exc
        _write_token(creds)
        return creds

    return None


def _write_token(creds: Any) -> None:
    ensure_dir(SECRETS_DIR)
    with open(TOKEN_PATH, "w", encoding="utf-8") as handle:
        handle.write(creds.to_json())
    _secure(TOKEN_PATH)


def authorize(port: int = 0, open_browser: bool = True) -> dict[str, Any]:
    """
    Runs the installed-app OAuth flow and stores the result.

    This opens a browser and blocks until the user finishes consenting, which
    means it cannot run on a headless server. That is called out in the UI
    rather than papered over: on a remote box the token has to be generated
    locally and the file copied across.
    """
    from google_auth_oauthlib.flow import InstalledAppFlow

    if not has_client_secrets():
        raise PublishError(
            "No OAuth client secret yet. Create one in Google Cloud Console "
            "(APIs & Services → Credentials → OAuth client ID → Desktop app), "
            "enable the YouTube Data API v3 for that project, then upload the "
            "downloaded JSON here."
        )

    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_PATH, SCOPES)
    try:
        creds = flow.run_local_server(
            port=port,
            open_browser=open_browser,
            authorization_prompt_message="",
            success_message="ReelForge is authorized. You can close this tab.",
        )
    except Exception as exc:
        raise PublishError(f"Authorization did not complete: {type(exc).__name__}: {exc}") from exc

    _write_token(creds)
    return {"authorized": True, "token_path": TOKEN_PATH, "scopes": list(SCOPES)}


def account_status() -> dict[str, Any]:
    """
    What the UI needs to show about the connection, and nothing more.

    Deliberately returns no token material. The channel title comes from the
    API only when the readonly scope was also granted; without it this reports
    connected-but-unnamed rather than failing, because the upload scope alone
    is enough to publish.
    """
    status: dict[str, Any] = {
        "client_secrets": has_client_secrets(),
        "token": has_token(),
        "connected": False,
        "channel": "",
        "channel_id": "",
        "error": "",
    }
    if not status["token"]:
        return status

    try:
        creds = _load_credentials()
    except PublishError as exc:
        status["error"] = str(exc)
        return status

    if creds is None:
        return status

    status["connected"] = True
    try:
        from googleapiclient.discovery import build

        youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)
        response = youtube.channels().list(part="snippet", mine=True).execute()
        items = response.get("items") or []
        if items:
            status["channel"] = str(items[0]["snippet"].get("title") or "")
            status["channel_id"] = str(items[0].get("id") or "")
    except Exception:
        # The upload scope does not include channels.list. Not an error.
        status["channel"] = "(name hidden — upload-only scope)"
    return status


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def validate_metadata(title: str, description: str, tags: list[str]) -> list[str]:
    """Returns the reasons YouTube would reject this, before the upload starts."""
    problems: list[str] = []

    clean_title = str(title or "").strip()
    if not clean_title:
        problems.append("A title is required.")
    elif len(clean_title) > TITLE_LIMIT:
        problems.append(f"The title is {len(clean_title)} characters; "
                        f"the limit is {TITLE_LIMIT}.")
    # These two are rejected outright by the API, and the error it returns does
    # not say which field was at fault.
    if "<" in clean_title or ">" in clean_title:
        problems.append("Angle brackets are not allowed in a title.")

    if len(str(description or "")) > DESCRIPTION_LIMIT:
        problems.append(f"The description is {len(description)} characters; "
                        f"the limit is {DESCRIPTION_LIMIT}.")

    total = sum(len(t) for t in tags) + max(0, len(tags) - 1)
    if total > TAGS_TOTAL_LIMIT:
        problems.append(f"Tags total {total} characters; the limit is "
                        f"{TAGS_TOTAL_LIMIT} across all of them.")

    return problems


def parse_tags(raw: str) -> list[str]:
    """Splits a comma-separated tag box, de-duplicated, order preserved."""
    seen: list[str] = []
    for chunk in str(raw or "").replace("\n", ",").split(","):
        tag = chunk.strip()
        if tag and tag.lower() not in {t.lower() for t in seen}:
            seen.append(tag)
    return seen


def build_body(
    title: str,
    description: str,
    tags: list[str],
    category_id: str = DEFAULT_CATEGORY,
    privacy: str = DEFAULT_PRIVACY,
    publish_at: str = "",
    made_for_kids: bool = False,
) -> dict[str, Any]:
    """
    Assembles the `videos.insert` request body.

    The scheduling rule is the one that bites: `publishAt` is only honoured
    when `privacyStatus` is `private`. Send it alongside `public` and YouTube
    ignores the date and publishes immediately — so a scheduled upload is
    forced private here rather than trusting the caller.
    """
    status: dict[str, Any] = {
        "privacyStatus": "private" if privacy == "scheduled" else privacy,
        "selfDeclaredMadeForKids": bool(made_for_kids),
    }
    if privacy == "scheduled":
        if not publish_at:
            raise PublishError("A scheduled upload needs a publish date and time.")
        status["publishAt"] = publish_at

    return {
        "snippet": {
            "title": str(title or "").strip()[:TITLE_LIMIT],
            "description": str(description or "")[:DESCRIPTION_LIMIT],
            "tags": list(tags or []),
            "categoryId": str(category_id or DEFAULT_CATEGORY),
        },
        "status": status,
    }


def to_rfc3339(when: Any) -> str:
    """
    Converts a local datetime to the UTC RFC-3339 string the API wants.

    A naive datetime is treated as local time and converted, which is what
    someone picking "8pm" in a date widget means. Sending the naive value with
    a Z suffix would schedule it at 8pm UTC instead.
    """
    import datetime as _dt

    if isinstance(when, str):
        return when
    if not isinstance(when, _dt.datetime):
        raise PublishError(f"Cannot schedule against {type(when).__name__}.")

    if when.tzinfo is None:
        when = when.astimezone()
    return when.astimezone(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Resumable upload
# ---------------------------------------------------------------------------

# Transport-level failures worth retrying. A 4xx other than 408 means the
# request itself is wrong and retrying just wastes the user's bandwidth.
_RETRY_STATUSES = (408, 500, 502, 503, 504)


def upload_video(
    file_path: str,
    title: str,
    description: str = "",
    tags: list[str] | None = None,
    category_id: str = DEFAULT_CATEGORY,
    privacy: str = DEFAULT_PRIVACY,
    publish_at: str = "",
    made_for_kids: bool = False,
    notify_subscribers: bool = True,
    progress: ProgressFn | None = None,
    chunk_bytes: int = CHUNK_BYTES,
) -> dict[str, Any]:
    """
    Uploads a file with the resumable protocol, reporting progress per chunk.

    Returns {"video_id", "url", "privacy", "bytes", "seconds"}.
    """
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload

    if not os.path.exists(file_path):
        raise PublishError(f"There is no file at {file_path}.")
    size = os.path.getsize(file_path)
    if size == 0:
        raise PublishError("That file is empty.")

    tags = list(tags or [])
    problems = validate_metadata(title, description, tags)
    if problems:
        raise PublishError(" ".join(problems))

    creds = _load_credentials()
    if creds is None:
        raise PublishError("No authorized channel. Connect one first.")

    body = build_body(title, description, tags, category_id, privacy,
                      publish_at, made_for_kids)
    youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)

    media = MediaFileUpload(file_path, chunksize=chunk_bytes, resumable=True,
                            mimetype="video/*")
    request = youtube.videos().insert(
        part="snippet,status", body=body, media_body=media,
        notifySubscribers=bool(notify_subscribers),
    )

    if progress:
        progress(0.0, f"Starting upload — {size / 1_048_576:.0f} MB in "
                      f"{max(1, size // chunk_bytes)} chunks...")

    started = time.time()
    response = None
    failures = 0

    while response is None:
        try:
            status, response = request.next_chunk()
        except HttpError as exc:
            code = int(getattr(exc.resp, "status", 0) or 0)
            if code in _RETRY_STATUSES and failures < MAX_RETRIES:
                failures += 1
                delay = 2 ** failures
                if progress:
                    progress(-1.0, f"YouTube returned {code}; retrying chunk in {delay}s "
                                   f"({failures}/{MAX_RETRIES})...")
                time.sleep(delay)
                continue
            raise PublishError(_explain_http_error(exc)) from exc
        except Exception as exc:
            # Connection reset mid-chunk. The whole point of resumable upload
            # is that this costs one chunk, not the file.
            if failures < MAX_RETRIES:
                failures += 1
                delay = 2 ** failures
                if progress:
                    progress(-1.0, f"{type(exc).__name__} mid-chunk; resuming in "
                                   f"{delay}s ({failures}/{MAX_RETRIES})...")
                time.sleep(delay)
                continue
            raise PublishError(
                f"Upload failed after {MAX_RETRIES} retries: {type(exc).__name__}: {exc}"
            ) from exc

        if status and progress:
            done = status.progress()
            sent = done * size
            rate = sent / max(0.1, time.time() - started)
            remaining = (size - sent) / max(1.0, rate)
            progress(done, f"Uploaded {sent / 1_048_576:.0f} of {size / 1_048_576:.0f} MB "
                           f"({rate / 1_048_576:.1f} MB/s, ~{remaining / 60:.0f} min left)")

    video_id = str((response or {}).get("id") or "")
    if not video_id:
        raise PublishError("YouTube accepted the upload but returned no video id.")

    if progress:
        progress(1.0, "Upload complete.")

    return {
        "video_id": video_id,
        "url": f"https://youtu.be/{video_id}",
        "studio_url": f"https://studio.youtube.com/video/{video_id}/edit",
        "privacy": body["status"]["privacyStatus"],
        "publish_at": body["status"].get("publishAt", ""),
        "bytes": size,
        "seconds": time.time() - started,
    }


def _explain_http_error(exc: Any) -> str:
    """Turns an HttpError into something a creator can act on."""
    code = int(getattr(exc.resp, "status", 0) or 0)
    detail = ""
    try:
        payload = json.loads(exc.content.decode("utf-8"))
        detail = str(payload.get("error", {}).get("message") or "")
        reasons = [e.get("reason", "") for e in payload.get("error", {}).get("errors", [])]
    except Exception:
        reasons = []

    if code == 401:
        return "YouTube rejected the credentials. Re-authorize the channel."
    if code == 403 and "quotaExceeded" in reasons:
        return ("The project's daily API quota is spent. A video upload costs "
                "1,600 units of the default 10,000/day, so this is roughly six "
                "uploads per day per project. It resets at midnight Pacific.")
    if code == 403 and "forbidden" in reasons:
        return ("The channel is not permitted to upload. Usually this means the "
                "account has not verified a phone number, which is required "
                "before uploads over 15 minutes are allowed — and every "
                "Atmosphere render is over 15 minutes.")
    if code == 400 and "invalidCategoryId" in reasons:
        return "That category id is not valid in this channel's region."
    if code == 400:
        return f"YouTube rejected the request: {detail or 'bad request'}"
    return f"YouTube returned {code}: {detail or exc}"


# ---------------------------------------------------------------------------
# SEO assistant
# ---------------------------------------------------------------------------

SEO_PROMPT = """You write titles and descriptions for a long-form ambient/sleep YouTube
channel. Return ONE JSON object and nothing else.

THE VIDEO: {soundscape}, running {hours}. {visual}

This audience does not browse — they search, at night, on a phone, for a
specific problem. Write for that search box.

"title": at most 95 characters. Lead with the literal search phrase someone
  would type ("Rain Sounds for Sleeping", "Deep Sleep Brown Noise",
  "Study Focus Rain"), then the duration, then one differentiator. State the
  runtime in hours as a number. No clickbait punctuation, no ALL CAPS words,
  no emoji in the title.

"description": 400-900 characters. First two lines are what shows above the
  fold, so put the search phrase and the runtime there. Then: what the
  soundscape actually contains, who it is for (insomnia, tinnitus masking,
  studying, a baby sleeping), and a line stating the audio is originally
  synthesized rather than a stock loop. End with 3-5 hashtags on their own line.

"tags": 12-18 tags. Mix exact search phrases ("rain sounds for sleeping"),
  problem terms ("insomnia relief", "can't sleep"), and format terms
  ("8 hours", "black screen"). Lowercase. Total under 450 characters.

"thumbnail_text": 2-4 words for an overlay, if one is wanted.

Do not claim medical benefits. "Helps you relax" is fine; "cures insomnia" is
not, and is the kind of claim that costs a channel its monetization.

JSON shape:
{{"title": "...", "description": "...", "tags": ["..."], "thumbnail_text": "..."}}"""


def generate_seo(
    soundscape: str,
    duration_seconds: float,
    visual_note: str = "",
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """
    Asks Gemini for a title, description and tags aimed at search intent.

    Imported lazily so this module stays usable with no Gemini key -- the
    uploader itself does not need one.
    """
    import gemini_engine as ge

    hours = duration_seconds / 3600.0
    if hours >= 1:
        runtime = f"{hours:.0f} hours" if abs(hours - round(hours)) < 0.05 else f"{hours:.1f} hours"
    else:
        runtime = f"{duration_seconds / 60:.0f} minutes"

    prompt = SEO_PROMPT.format(
        soundscape=soundscape, hours=runtime,
        visual=visual_note or "The visual is a still scene with a slow drift.",
    )

    client = ge.get_client()
    last: Exception | None = None

    for model in ge.MODEL_CANDIDATES:
        try:
            if progress:
                progress(f"Writing title and description with {model}...")
            response = ge.generate_with_retry(client, model, prompt)
            parsed = ge.parse_scene_response(getattr(response, "text", "") or "")

            title = str(parsed.get("title") or "").strip()
            if not title:
                continue

            tags = parsed.get("tags") or []
            if isinstance(tags, str):
                tags = parse_tags(tags)
            tags = [str(t).strip().lower() for t in tags if str(t).strip()]

            # Trim tags to the API limit here rather than letting the upload
            # fail on it later.
            kept: list[str] = []
            total = 0
            for tag in tags:
                cost = len(tag) + (1 if kept else 0)
                if total + cost > TAGS_TOTAL_LIMIT:
                    break
                kept.append(tag)
                total += cost

            return {
                "title": title[:TITLE_LIMIT],
                "description": str(parsed.get("description") or "")[:DESCRIPTION_LIMIT],
                "tags": kept,
                "thumbnail_text": str(parsed.get("thumbnail_text") or "").strip(),
                "model": model,
            }
        except Exception as exc:
            last = exc
            continue

    raise PublishError(
        f"Could not generate metadata: {type(last).__name__}: {last}" if last
        else "No model returned usable metadata."
    )


def fallback_seo(soundscape: str, duration_seconds: float) -> dict[str, Any]:
    """
    Metadata with no API call, so the publisher is never blocked on Gemini.

    Deliberately plain. It is a starting point to edit, not a substitute for
    the model, and it says so rather than pretending to be optimised.
    """
    hours = duration_seconds / 3600.0
    runtime = (f"{hours:.0f} Hours" if hours >= 1 and abs(hours - round(hours)) < 0.05
               else f"{hours:.1f} Hours" if hours >= 1
               else f"{duration_seconds / 60:.0f} Minutes")

    title = f"{soundscape} for Sleeping | {runtime} | Black Screen"
    description = (
        f"{soundscape} for {runtime.lower()} — for sleeping, studying, or masking "
        f"background noise.\n\n"
        f"Every layer in this recording was synthesized for this upload rather "
        f"than taken from a stock loop library, and the loop points are "
        f"crossfaded so there are no clicks or seams to wake you.\n\n"
        f"Play it quietly. Sweet dreams.\n\n"
        f"#sleepsounds #{soundscape.lower().replace(' ', '')} #whitenoise #relaxing"
    )
    tags = [
        soundscape.lower(), f"{soundscape.lower()} for sleeping", "sleep sounds",
        "white noise", "brown noise", "relaxing sounds", "insomnia relief",
        "study sounds", "focus", runtime.lower(), "black screen", "ambient",
    ]
    return {
        "title": title[:TITLE_LIMIT],
        "description": description[:DESCRIPTION_LIMIT],
        "tags": tags,
        "thumbnail_text": "",
        "model": "offline template",
    }

# ReelForge Studio

A private, multi-user short-form video workshop. Five generation engines behind
a login, each render sandboxed to the account that made it, and a compliance
gate that refuses to call something publish-ready when it is not.

| Engine | What it does |
|---|---|
| 🎙️ Commentary Machine | Gemini watches a clip and writes the commentary; TTS narrates it; kinetic captions burn in |
| ◼️ Minimalist Motion | Draws a 1080×1920 vector animation from code — no footage, no stock, no model-generated imagery |
| 📦 Batch Studio | Queues topics and renders them back to back |
| 🎬 Reel Studio | Slide-based reels from photos and licensed stock |
| ⚔️ Versus Duel | Split-screen comparison shorts with animated stat badges |

---

## Run it locally

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt     # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # Linux/macOS

copy .env.example .env                            # then fill in GEMINI_API_KEY
streamlit run app.py
```

Open http://localhost:8501. On first run an `admin` account is created and its
password is **printed to the terminal running Streamlit** — not to the browser,
because whoever can reach the login page is not yet known to be you.

To set the admin password yourself instead, put it in `.env` before the first
run:

```bash
python -c "import auth; print(auth.hash_password('your-password-here'))"
# paste the result into REELFORGE_ADMIN_PASSWORD_HASH in .env
```

---

## Accounts and roles

| Role | Engines | Can also |
|---|---|---|
| `admin` | all five, plus the Admin panel | manage users, set API keys, see every workspace |
| `creator` (`member`) | Commentary Machine, Minimalist Motion | nothing else — own files only |

Users live in `users.json` (gitignored) or in environment variables. Passwords
are stored as **PBKDF2-HMAC-SHA256**, salted per user, 240,000 iterations.
Plain SHA-256 digests are accepted on login for compatibility and are upgraded
to PBKDF2 automatically the first time that user signs in.

Every account renders into `exports/<username>/`, including its own
`provenance.json` ledger, so no account can see another's work through the app.

**Locked out?** Environment users override the file, so an admin is always
recoverable:

```bash
REELFORGE_ADMIN_USER=rescue \
REELFORGE_ADMIN_PASSWORD_HASH="$(python -c "import auth;print(auth.hash_password('temp-password'))")" \
streamlit run app.py
```

---

## Docker

```bash
docker build -t reelforge .
docker run -d --name reelforge -p 8501:8501 \
  --env-file .env \
  -v reelforge-exports:/app/exports \
  -v reelforge-users:/app/userdata \
  reelforge

docker logs -f reelforge          # the first-run admin password prints here
docker inspect --format '{{.State.Health.Status}}' reelforge
```

Or with compose:

```bash
docker compose up -d --build
docker compose logs -f
```

Notes on the image:

- `ffmpeg` comes from Debian rather than the bundled `imageio-ffmpeg` binary,
  because burning `.ass` captions needs libass and Debian's build has it.
  `IMAGEIO_FFMPEG_EXE` points moviepy at it.
- Fonts matter more than they look. PIL resolves a bare `arialbd.ttf` through
  the Windows font directory and nowhere else, so without real TTFs on the
  image every caption silently renders in an 11px bitmap face. DejaVu,
  Liberation and FreeFont are installed as hard requirements; Montserrat and
  Inter are attempted and allowed to fail, since not every mirror carries them.
  Drop your own `.ttf` in `assets/fonts/` and it wins over all of them.
- Runs as UID 10001, not root. `exports/` and `userdata/` are volumes so
  renders and accounts survive a rebuild.
- There is no GPU in the container, so renders use libx264. On a machine with
  NVENC the app detects and uses it automatically.

---

## Push to GitHub

```bash
git init
git add -A
git status                        # confirm .env and users.json are absent
git commit -m "ReelForge Studio"

gh repo create reelforge --private --source=. --remote=origin --push
# or, without the gh CLI:
git remote add origin git@github.com:<you>/reelforge.git
git branch -M main
git push -u origin main
```

`.gitignore` excludes `.env`, `users.json`, `exports/`, `*.mp4`, `__pycache__/`
and both `venv/` and `.venv/`. `.gitattributes` forces LF on the Dockerfile —
a CRLF checkout puts a carriage return after every line-continuation backslash
and the Linux build fails with "unknown instruction".

Verify before your first push:

```bash
git check-ignore -v .env users.json exports    # all three should print a rule
```

---

## Layout

```
app.py            Streamlit UI, login gate, all five studios
auth.py           password hashing, roles, user store, login rate limiting
paths.py          project-relative paths and cross-platform font resolution
motion_engine.py  the vector animation engine (Minimalist Motion)
video_engine.py   reframing, looping, captions, the duel engine
audio_engine.py   TTS, synthesized music beds, SFX, ducking
gemini_engine.py  scripts, scene specs, video grounding
compliance.py     licence model, provenance ledger, publish gate
```

---

## Security notes, stated plainly

- The login rate limiter is per-process and per-username. It makes guessing one
  account impractical; it does nothing about an attempt spread across many
  usernames or many replicas. Put real rate limiting at the reverse proxy.
- Sessions are Streamlit session state. Closing the tab ends the session; there
  is no "remember me" cookie and no idle timeout.
- Serve this over HTTPS. Streamlit sends credentials over the websocket, and
  without TLS they cross the network in the clear.
- The Admin API-key panel writes to the server's environment for the running
  process. Anything permanent belongs in `.env` or the container environment.

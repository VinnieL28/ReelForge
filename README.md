# ReelForge Studio

A private, multi-user short-form video workshop. Five generation engines behind
a login, each render sandboxed to the account that made it, and a compliance
gate that refuses to call something publish-ready when it is not.

| Engine | What it does |
|---|---|
| 🎙️ Commentary Machine | Gemini watches a clip and writes the commentary; TTS narrates it; kinetic captions burn in |
| 📖 Narrative Studio | Episodic storytelling: Gemini writes the script and casts the world, then a storyboard becomes Ken Burns shots cut to the voice |
| ◼️ Minimalist Motion | Draws a 1080×1920 vector animation from code — seven metaphor templates, no footage, no stock, no model-generated imagery |
| 📦 Batch Studio | Queues topics and renders them back to back |
| 🎬 Reel Studio | Slide-based reels from photos and licensed stock |
| ⚔️ Versus Duel | Split-screen comparison shorts with animated stat badges |

---

## Narrative Studio

A four-step wizard from a one-line premise to a finished episode.

1. **Concept** — topic, visual aesthetic (Desaturated Vector Comic / Cinematic
   Lofi Anime / Nordic Noir / Flat Graphic Minimalist), narrative tone (Quiet
   Self-Reflection / Stoic Motivation / Suspense Investigation), and format
   (60s 9:16, or 3–5 min in 9:16 or 16:9).
2. **Script & cast** — Gemini writes narration in beats and, in the same call,
   extracts the protagonist, the environments and the recurring objects.
3. **Storyboard** — beats become numbered segments with a word count, an
   estimated duration, a camera framing and entity tags.
4. **Assets & assembly** — narration, one still per segment, Ken Burns moves,
   dissolves, subtitles, an ambient bed, and a metadata pack.

**Why the entity block exists.** An image model asked for "the man" twelve
times draws twelve different men. Asked for the same forty-word description
twelve times, it draws something close enough to read as one person. That
description is written once and pasted verbatim into every image prompt.

**The voice decides the timing.** Estimated durations are always a little
wrong and the error accumulates — by segment thirty a picture can be two
seconds off the line being spoken. So narration is synthesized *first*, the
storyboard is re-cut against the real word timings, and only then are stills
generated. Providers with no word timings (Gemini TTS) get proportional
scaling instead.

**Visual sources**, tried in order and each a real fallback:

| Provider | Notes |
|---|---|
| `gemini` | The only one that can draw the same character twice. **Not on the API free tier** — without billing it returns a quota error and the chain falls through. |
| `pexels` | Real photography, CC0. Fetched as a set per environment so consecutive shots differ, and colour-graded to the chosen aesthetic. |
| `procedural` | Drawn locally from the aesthetic's palette. Always available, never blocks a render. |

Grading pushes stock toward the palette but cannot turn a photograph into an
illustration — for a look that is genuinely the aesthetic you asked for, the
Gemini provider is the one that does it.

---

## Minimalist Motion: the metaphor library

Gemini reads your concept and **chooses** the geometry that argues it, rather
than dropping every topic into the same shape. Ask for "why scrolling is hard
to stop" and it reaches for the funnel; ask about compounding and it reaches
for the jar. Pick a template by hand if you'd rather.

| Template | Geometry and motion |
|---|---|
| ① Steep vs. Shallow Path | Two routes race — the steep one dips hard and ends high, the flat one cruises and ends on spikes |
| ② Compounding Skill Jar | An outlined vessel filling on a real `1.01^n` curve with a live day counter and falling particles |
| ③ Exponential Staircase | A figure pushing up consistent steps; effort stays linear while reward goes `u³` |
| ④ Balance Scale | One pan loads instantly and stops, the other loads slowly and never stops; the beam flips on the beat and wobbles as it settles |
| ⑤ Gravity Funnel | A mote circling a throat, the orbit tightening and accelerating until it is gone in three frames |
| ⑥ Domino Chain | Six tiles, each 1.46× the last, falling faster as they go — the final one is 6.6× the first |
| ⑦ Comparison Split | Three tiers doing the same work at three rates — only the bottom one finishes |
| ⑧ The Steep Staircase | A figure walking up labelled stages, each lighting as it is passed |
| ⑨ The Delusion Mirror | A plain figure beside the glowing, flexing version it sees in the mirror |
| ⑩ The Chain & Anchor | A figure hauling named dead weight, until the chain lets go and it walks |
| ⑪ Growth & Consistency | A figure watering at the same rate while a seed becomes a canopy |
| ⑫ Dynamic AI Scene | Gemini writes the geometry from scratch: paths, followed dots, bars, text |

Templates ⑦–⑪ put a stick figure on screen, drawn by `character_rig.py`. The
model reaches for those when the concept is about a person doing something and
for the abstract ones when it is about a quantity or a shape of change —
verified live: 5/5 person-shaped topics chose a figure, 2/2 abstract chose a
shape.

Every scene runs the same three-beat structure, so the picture and the sound
design cannot drift apart:

1. **Draw** — the geometry writes itself on, a trim-path from 0% to 100%
2. **Travel** — the object moves along it under heavy cubic easing
3. **Impact** — the milestone lands, and the sub-bass drop is placed on that
   exact frame

Gemini supplies the beat boundaries as `animation_phases`, and the words
stamped onto the geometry as `labels` — so a video about instant versus
delayed reward is labelled "NOW / FOREVER", not "5 YEARS / 50 YEARS".

The glow is three Gaussian passes at different radii summed at half
resolution: a bright core within a few pixels of the stroke, a soft body out
to ~45px, and a faint atmosphere reaching 140px. One pass is either a tight rim
or a wide wash and cannot be both, which is what makes single-pass bloom look
like a filter. Behind it all sits a faint grid that pulses outward from the
impact, and slow motes drifting up through the frame.

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
narrative_engine.py  the episodic pipeline (Narrative Studio)
motion_engine.py  the vector animation engine (Minimalist Motion)
character_rig.py  the stick-figure rig: six poses, solved by forward kinematics
video_engine.py   reframing, looping, captions, the duel engine
audio_engine.py   TTS, synthesized music beds, SFX, ducking
gemini_engine.py  scripts, scene specs, video grounding
compliance.py     licence model, provenance ledger, publish gate
tests/            narrative unit tests and the ffmpeg pipeline suite
```

Run the tests:

```bash
.venv\Scripts\python tests\test_narrative_unit.py     # no network, ~2s
.venv\Scripts\python tests\test_narrative_ffmpeg.py   # needs ffmpeg, ~90s
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

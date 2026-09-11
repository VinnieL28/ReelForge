# ReelForge Studio

A private, multi-user video workshop. Seven generation engines behind a login,
each render sandboxed to the account that made it, and a compliance gate that
refuses to call something publish-ready when it is not.

| Engine | What it does |
|---|---|
| 🎙️ Commentary Machine | Gemini watches a clip and writes the commentary; TTS narrates it; kinetic captions burn in |
| 📖 Narrative Studio | Episodic storytelling: Gemini writes the script and casts the world, then a storyboard becomes Ken Burns shots cut to the voice |
| ◼️ Minimalist Motion | Draws a 1080×1920 vector animation from code — 15 metaphor templates, no footage, no stock, no model-generated imagery |
| 📦 Batch Studio | Queues topics and renders them back to back |
| 🎬 Reel Studio | Slide-based reels from photos and licensed stock, scripted from sourced facts |
| ⚔️ Versus Duel | Split-screen comparison shorts with animated stat badges |
| 🌙 Atmosphere Studio | 30-minute to 8-hour ambient/sleep video with a synthesized soundtrack, published straight to YouTube |
| 📁 Exports Library | Every render this account has made: filter, play, download, reveal, delete |

Every page opens with the same three things: a header whose subtitle names the
engine you are actually in, four live telemetry readings (renders, storage, the
encoder this machine will actually use, and whether the compliance gate is
holding anything back), and a card saying what the engine is for and roughly
how long a render takes. Then the work, and nothing after it — the creation
pages end at their own render button.

Everything you have made lives in **📁 Exports Library**, a mode of its own:
a grid of every render in `exports/<username>/` with its duration, size, date
and aspect tag, filterable by the engine that made it, with play, download,
reveal-in-folder and delete on each card. It used to be a four-item strip
pinned under every creation page, which meant every workflow ended in a row of
unrelated thumbnails and a video player that stayed open across mode switches.
It is a destination now, not a footer.

---

## The Viral Scorecard

Every mode scores its script 1-10 on the preview step, on three axes, before
you publish it.

| Axis | What it measures |
|---|---|
| **Hook intrigue (0-3s)** | Whether the first nine words open a loop or announce a topic |
| **Information density** | How much of the script is load-bearing -- figures, names, mechanisms |
| **Monetization safety** | Whether it survives YouTube's reused-content review and TikTok originality |

The offline card always runs; a button blends in a model read of the same three
axes. Under 8.0 you get a one-click **Rewrite for High Retention**, which
rewrites against the card's own diagnosis and then *re-scores the result* --
if the rewrite came out worse, it says so and keeps the original.

The density metric is worth one note, because getting it wrong is subtle. It
originally counted any number as a fact, which rated the old Reel Studio
listicle template 8.7 "facts per 100 words" on nothing but its own `Number 1 /
Number 2` scaffolding -- higher than a researched script. List ordinals are now
stripped before counting, and a second term measures how *evenly* the facts are
spread, because a script with one fact-stuffed sentence and five empty ones
loses the viewer in the five. Measured after the fix: every one of the five old
templates scores under 6.0, the offline fact bank 6.6-7.6, a researched script 8.7.

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
| ⑫ Sisyphus Boulder | A faceted stone up a 41° slope past checkpoints, losing ground between pushes |
| ⑬ The Discipline Iceberg | A small lit tip and the mass under the waterline, revealed downward |
| ⑭ The Divergent Path | One dark door, one lit; the unchosen door dims rather than vanishing |
| ⑮ Dynamic AI Scene | Gemini writes the geometry from scratch: paths, followed dots, bars, text |

Templates ⑦–⑭ put a stick figure on screen, drawn by `vector_rig.py`. The
model reaches for those when the concept is about a person doing something and
for the abstract ones when it is about a quantity or a shape of change —
verified live: 5/5 person-shaped topics chose a figure, 2/2 abstract chose a
shape.

It is also given explicit routing by premise: consistency and time to the jar, the plant or a staircase; overthinking versus doing to the tier split, the scale or the diverging paths; perseverance and hidden work to the boulder or the iceberg. Measured at 5/6 on those categories.

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

## Atmosphere Studio

Long-form ambient and sleep video — 16:9, 30 minutes to 8 hours — with an
originally synthesized soundtrack and a one-click upload to YouTube.

| Layer | Options |
|---|---|
| **Primary bed** | Rain on Window · Heavy Thunderstorm · Deep Brown Noise · Gentle Stream · Crackling Fireplace |
| **Secondary texture** | Distant Thunder · Soft Room Wind · Night Crickets · Binaural Drone 432Hz / 528Hz |
| **Canvas** | Five drawn presets, or your own 4K still or seamless video loop |
| **Runtime** | 60s test render · 30 min · 1 hour · 3 hours · 8 hours |

**Nothing is a stock loop.** Every layer is synthesized in numpy for each
render — filtered noise for the beds, scattered transients for droplets,
crackles and cricket chirps, and a genuinely stereo drone whose whole effect is
the few-Hz difference between the ears. That matters commercially, not just
aesthetically: this niche is where YouTube's Reused Content policy is enforced
hardest, and "the same purchased loop for eight hours" is the example it names.

**The seams are the product.** A loop that clicks once every two minutes is
worse than no loop at all, because the listener is asleep and the click wakes
them. Three different seeds are synthesized, each made continuous at its own
wrap point with an equal-power crossfade, then chained with `acrossfade` into a
two-minute super-loop and repeated with `aloop`. Measured on the finished AAC:
the step at the loop point is **0.95×** the size of an ordinary sample-to-sample
step — i.e. smaller than the signal around it, which is why it cannot be heard.

### Why an eight-hour render takes minutes

An 8-hour video at 24fps is 691,200 frames. Nothing can touch them all.

- **Audio is synthesized short and looped long.** Eight hours of rain costs the
  same as two minutes of it.
- **Video is rendered short and looped long.** The drift zoom follows a raised
  cosine — 1.00× at both ends, 1.05× in the middle — so a ten-minute segment
  joins to itself exactly and is stream-copied for the rest of the timeline.
  Verified geometrically: the last frame returns to **1.0007×**.
- **The mux is a stream copy.** Both tracks are already in their final codecs.

A true monotonic 1.00×–1.05× push over the whole runtime is offered as
*Continuous drift* and is honest about the cost — at eight hours it is about an
hour of encoding, for a zoom of 0.000002× per frame that nobody can see.

Two performance findings, both measured rather than assumed:

- **ffmpeg's `vignette` was 60% of the render** — 195 fps without it, 82 with,
  and `eval=init` changes nothing. It is now baked into the canvas once with
  PIL, which is visually identical at this zoom range.
- **zoompan, not NVENC, is the limit.** The encoder was never the bottleneck,
  so the NVENC profile is chosen for quality (`p4`, `-tune hq`, 4Mbps with an
  8Mbit buffer) rather than for speed. libx264 is the automatic fallback.

### Publishing to YouTube

One-time setup: in Google Cloud Console create a project, enable **YouTube Data
API v3**, create an **OAuth client ID** of type **Desktop app**, and upload the
JSON in Step 4.

```
.secrets/client_secrets.json   the OAuth client   (gitignored)
.secrets/youtube_token.json    the refresh token  (gitignored)
```

- **Upload-only scope.** `youtube.upload` and nothing else — no read or write
  access to comments, playlists or the channel itself.
- **Resumable, chunked upload.** A three-hour render is several gigabytes; a
  single held-open request will fail, and when it does a retry costs one 8MB
  chunk rather than the whole file.
- **Private by default.** Going public is a deliberate choice. A *scheduled*
  upload is forced to `private` first, because `publishAt` is only honoured on
  a private video — send it with `public` and YouTube ignores the date and
  publishes immediately.
- **✨ Auto-Generate** writes the title, description and tags with Gemini,
  aimed at what this audience actually types at 1am. There is an offline
  template when no key is configured.

Authorization opens a Google consent page in a browser **on the machine running
Streamlit**. Over a tunnel or on a headless box, generate the token locally and
copy `.secrets/youtube_token.json` across.

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

**Editing an engine means restarting the server.** Streamlit re-executes
`app.py` top to bottom on every interaction, which is why module-level state in
`app.py` never persists — but the engines it imports are cached in `sys.modules`
for the life of the process and are *not* re-imported. So a change to
`compliance.py`, `video_engine.py`, `duel_engine.py` and the rest has no effect
until you restart, and adding a new name to one of them produces a confusing
failure: the file on disk plainly defines it, and the app still raises

```
ImportError: cannot import name 'VIRAL_TARGET_SCORE' from 'compliance'
```

on every rerun, because `app.py`'s fresh import line is resolving against the
module object loaded when the server started. Ctrl-C and `streamlit run app.py`
again. Only `app.py` itself is hot.

---

## Accounts and roles

| Role | Engines | Can also |
|---|---|---|
| `admin` | all seven, plus the Exports Library and the Admin panel | manage users, set API keys, see every workspace |
| `creator` (`member`) | Commentary Machine, Minimalist Motion, Narrative Studio, and their own Exports Library | nothing else — own files only |

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
- There is no GPU in the container, so renders use `libx264 -preset veryfast`.
  That preset rather than `medium` because this is the path every containerised
  render takes: measured at 1080x1920 CRF 20, veryfast is 2.18x quicker and 12%
  smaller. On a machine with NVENC the app probes it with a real frame and uses
  it automatically, falling back to the same CPU profile if the driver refuses
  a frame partway through a render.

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
app.py            Streamlit UI, login gate, the command centre, all six studios
auth.py           password hashing, roles, user store, login rate limiting
paths.py          project-relative paths and cross-platform font resolution
narrative_engine.py  the episodic pipeline (Narrative Studio)
minimalist_engine.py  the vector animation engine (Minimalist Motion)
vector_rig.py     the stick-figure rig: ten poses, a two-segment torso,
                  solved by forward kinematics
duel_engine.py    the Versus Duel: split panels, stat cards, the winner reveal
ambient_engine.py the Atmosphere soundscape synthesizer and visual canvas
publisher.py      YouTube Data API v3: OAuth, resumable upload, SEO metadata
reel_engine.py    domain detection, the fact bank, fact-carrying scripts
video_engine.py   reframing, looping, captions, encoding, the Gemini pre-flight
audio_engine.py   TTS, synthesized music beds, SFX, ducking
gemini_engine.py  scripts, scene specs, video grounding, the viral scorecard
compliance.py     licence model, provenance ledger, publish gate, the scorecard
tests/            the pytest suite, plus the standalone `suite_*.py` scripts
```

Run the tests:

```bash
.venv\Scripts\python -m pytest tests/                  # ~2 min, 103 tests
.venv\Scripts\python -m pytest tests/ -m "not slow"    # ~15s, no ffmpeg
.venv\Scripts\python -m pytest tests/ --network        # also the live download
```

The `suite_*.py` files are scripts rather than pytest modules on purpose: they
print a readable account of what they measured, which is what you want while
working on the engine they cover. They are named `suite_` rather than `test_`
because pytest imports every `test_*.py` at collection time — as `test_*.py`
their whole bodies ran during `--collect-only`, which made collection take 30
seconds and fired a live TikTok download before a single test executed.
`test_suites.py` runs each of them as a subprocess instead.

```bash
.venv\Scripts\python tests\suite_narrative_unit.py    # no network, ~2s
.venv\Scripts\python tests\suite_narrative_ffmpeg.py  # needs ffmpeg, ~90s
.venv\Scripts\python tests\suite_vector_scenes.py     # no network, ~30s
.venv\Scripts\python tests\suite_url_ingest.py        # needs network, ~20s
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

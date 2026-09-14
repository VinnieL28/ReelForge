# pyright: reportUnknownMemberType=false, reportMissingTypeStubs=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false
# pyright: reportMissingParameterType=false, reportUnknownLambdaType=false
"""
Streamlit Web Application for the AI Reels & Shorts Studio.

Two production modes share one render pipeline:
  * Reel Studio      -- topic-to-reel scripting, Ken Burns slides, viral subtitles.
  * Versus Duel      -- split-screen A/B comparisons with animated stat badges
                        and a winner reveal, tuned for high-RPM comparison content.
"""

from __future__ import annotations

import os
import re
import sys
import copy
import glob
import time
import logging
import threading
import traceback
import tempfile
from typing import Any, Callable, Sequence

import streamlit as st
from PIL import Image

from demo_data import (
    generate_sample_images,
    create_gradient_mesh,
    resolve_item_photo,
    fetch_photo,
    fetch_photo_set,
    load_image_from_url,
    fallback_backdrop,
    search_licensed_video,
    download_licensed_clip,
    pexels_key_status,
    PhotoLookupError,
    DUEL_PRESETS,
)
from audio_engine import (
    generate_synth_music,
    save_wav_to_file,
    generate_slide_voiceovers,
    generate_voiceover,
    synthesize_with_word_timings,
    synthesize_narration,
    GEMINI_VOICES,
    build_ducked_bgm,
    BGM_DEFAULT_VOLUME,
    get_audio_duration,
    VIRAL_VOICES,
    DEFAULT_VOICE,
)
from compliance import (
    LICENCES,
    VIRAL_TARGET_SCORE,
    viral_scorecard,
    DEFAULT_LICENCE,
    TTS_PROVIDERS,
    SPEAKING_PROVIDERS,
    normalise_licence,
    append_ledger,
    find_entry,
    publish_readiness,
    build_publish_pack,
    attribution_line,
    summarise_ledger,
    AI_DISCLOSURE_LINE,
    PLATFORM_DISCLOSURE_STEPS,
    MONETIZATION_NOTES,
    TIKTOK_REWARDS_MIN_SECONDS,
)
import auth
import ambient_engine
import gemini_engine
import dashboard_view
import magic_studio
import niche_engine
import publisher
import reel_engine
from narrative_engine import (
    DEFAULT_AESTHETIC as NARRATIVE_DEFAULT_AESTHETIC,
    DEFAULT_FORMAT as NARRATIVE_DEFAULT_FORMAT,
    DEFAULT_TONE as NARRATIVE_DEFAULT_TONE,
    DURATION_FORMATS as NARRATIVE_FORMATS,
    IMAGE_PROVIDERS as NARRATIVE_IMAGE_PROVIDERS,
    NARRATIVE_TONES,
    SHORTS_MAX_SECONDS as NARRATIVE_SHORTS_MAX,
    SHORTS_MIN_SECONDS as NARRATIVE_SHORTS_MIN,
    VISUAL_AESTHETICS as NARRATIVE_AESTHETICS,
    generate_narrative,
    metadata_pack,
    produce_episode,
)
from paths import EXPORTS_ROOT, ensure_dir
from minimalist_engine import (
    AUTO_TEMPLATE,
    DEFAULT_TEMPLATE,
    METAPHOR_TYPES,
    TEMPLATES,
    resolve_template,
    active_face_name,
    build_minimalist_video,
    estimate_render_seconds,
    fallback_scene_spec,
    normalise_spec,
)
from video_engine import (
    ASPECT_RATIOS,
    DEFAULT_FIT,
    FIT_MODES,
    build_reel_video,
    format_stat,
    render_commentary_video,
    export_muted_video,
    write_ass_file,
    LOOP_MODES,
    FOCUS_POSITIONS,
    purge_scratch_renders,
    sweep_temp_renders,
    probe_stream_info,
    video_encoder,
    write_clip,
)
from gemini_engine import (
    SCENE_MAX_SECONDS,
    SCENE_MIN_SECONDS,
    SCENE_PRESETS,
    fallback_publish_meta,
    generate_scene_spec,
    generate_commentary,
    generate_commentary_angles,
    SCRIPT_ANGLES,
    ANGLE_ORDER,
    estimate_speech_seconds,
    DURATION_TARGETS,
    DEFAULT_TARGET,
    GeminiError,
    score_virality,
    rewrite_for_retention,
)

st.set_page_config(
    page_title="Reelforge Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design system
# ---------------------------------------------------------------------------

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --bg-base: #08090D;
    --bg-raise: #0E1016;
    --violet: #8B5CF6;
    --indigo: #6366F1;
    --amber: #FBBF24;
    --cyan: #22D3EE;
    --text-hi: #F2F4F8;
    --text-mid: #A8B0C0;
    --text-low: #6B7385;
    --edge: rgba(255, 255, 255, 0.08);
    --edge-hi: rgba(255, 255, 255, 0.14);
    --glass: rgba(255, 255, 255, 0.03);
    --radius: 12px;
}

html, body, .stApp, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-feature-settings: 'cv02', 'cv03', 'cv04', 'ss01';
}

.stApp {
    background:
        radial-gradient(1100px 620px at 12% -8%, rgba(139, 92, 246, 0.16), transparent 62%),
        radial-gradient(900px 520px at 92% 4%, rgba(34, 211, 238, 0.10), transparent 58%),
        radial-gradient(800px 620px at 50% 112%, rgba(251, 191, 36, 0.07), transparent 60%),
        var(--bg-base);
    color: var(--text-hi);
}

#MainMenu, footer, header [data-testid="stStatusWidget"] { visibility: hidden; }
.block-container { padding-top: 2.5rem !important; padding-bottom: 3rem; max-width: 1500px; }

h1, h2, h3, h4, h5 { font-family: 'Inter', sans-serif; color: var(--text-hi); letter-spacing: -0.022em; }
h1 { font-weight: 800; }
h2, h3 { font-weight: 700; }
p, span, label, li { color: var(--text-mid); }

/* ---------- Brand header ---------- */
.rf-brand {
    display: flex; align-items: center; gap: 14px; margin-bottom: 4px;
}
.rf-logo {
    width: 42px; height: 42px; border-radius: 12px; flex: 0 0 42px;
    background: linear-gradient(135deg, var(--violet), var(--indigo) 55%, var(--cyan));
    display: flex; align-items: center; justify-content: center;
    font-size: 21px; box-shadow: 0 8px 26px rgba(139, 92, 246, 0.42);
}
.rf-title {
    /* line-height was 1.1, which is the whole clipping bug: at 1.85rem that is
       a 32.6px line box around glyphs that need about 35.5px, and because the
       fill is background-clip:text there is no ink outside the box to spill --
       the ascenders were simply cut off. 1.3 plus an inline-block gives the
       gradient a box big enough to paint the whole letterform. */
    font-size: 1.85rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.3;
    display: inline-block; padding-block: 2px;
    background: linear-gradient(92deg, #FFFFFF 8%, #C4B5FD 48%, #FBBF24 96%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.rf-sub { color: var(--text-low); font-size: 0.94rem; margin: 2px 0 22px 0; font-weight: 450; }

/* ---------- Frosted glass containers ---------- */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > [data-testid="stVerticalBlock"]) {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: var(--radius);
    padding: 18px;
    box-shadow: 0 8px 26px rgba(0, 0, 0, 0.30);
    transition: border-color 160ms ease, box-shadow 160ms ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover { border-color: var(--edge-hi); }

/* ---------- Badges, tags, section labels ---------- */
.rf-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 11px; border-radius: 999px;
    font-size: 0.72rem; font-weight: 650; letter-spacing: 0.035em; text-transform: uppercase;
    border: 1px solid rgba(255, 255, 255, 0.12); background: rgba(255, 255, 255, 0.05);
    color: var(--text-mid); margin-right: 6px;
}
.rf-badge.violet { background: rgba(139, 92, 246, 0.16); border-color: rgba(139, 92, 246, 0.42); color: #C9B8FF; }
.rf-badge.cyan   { background: rgba(34, 211, 238, 0.14);  border-color: rgba(34, 211, 238, 0.40);  color: #9BE9F7; }
.rf-badge.amber  { background: rgba(251, 191, 36, 0.14);  border-color: rgba(251, 191, 36, 0.40);  color: #FBD98A; }
.rf-badge.green  { background: rgba(52, 211, 153, 0.14);  border-color: rgba(52, 211, 153, 0.38);  color: #86EFC5; }

.rf-tag-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0 14px 0; }
.rf-stat {
    flex: 1 1 120px; padding: 12px 14px; border-radius: 13px;
    background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
}
.rf-stat .k { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.07em; color: var(--text-low); font-weight: 650; }
.rf-stat .v { font-size: 1.32rem; font-weight: 750; color: var(--text-hi); margin-top: 3px; letter-spacing: -0.02em; }
.rf-stat.cyan .v  { color: var(--cyan); }
.rf-stat.amber .v { color: var(--amber); }

.rf-angle {
    font-size: 0.82rem; line-height: 1.45; color: var(--text-mid);
    background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 10px; padding: 10px 12px; margin: 8px 0 10px 0; min-height: 92px;
}
.rf-section { font-size: 0.74rem; font-weight: 700; letter-spacing: 0.10em; text-transform: uppercase;
              color: var(--text-low); margin: 6px 0 10px 0; }
.rf-hr { height: 1px; background: linear-gradient(90deg, transparent, var(--edge) 18%, var(--edge) 82%, transparent); margin: 22px 0; border: 0; }

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
    border-radius: 11px; font-weight: 600; font-size: 0.88rem; letter-spacing: -0.005em;
    border: 1px solid rgba(255, 255, 255, 0.12);
    background: rgba(255, 255, 255, 0.045); color: var(--text-hi);
    padding: 0.55rem 1.05rem; transition: all 150ms cubic-bezier(0.2, 0, 0.2, 1);
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: rgba(255, 255, 255, 0.09); border-color: rgba(255, 255, 255, 0.24);
    transform: translateY(-1px);
}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
    background: linear-gradient(96deg, var(--violet), var(--indigo) 62%, #4F46E5);
    border: 1px solid rgba(167, 139, 250, 0.60); color: #FFFFFF; font-weight: 650;
    box-shadow: 0 8px 26px rgba(99, 102, 241, 0.36);
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 12px 34px rgba(139, 92, 246, 0.52); transform: translateY(-1px);
    border-color: rgba(196, 181, 253, 0.85);
}
.stButton > button:focus-visible { outline: none; box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.42); }

/* ---------- Inputs ---------- */
.stTextInput input, .stTextArea textarea, .stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div, .stMultiSelect div[data-baseweb="select"] > div {
    background: rgba(255, 255, 255, 0.035) !important;
    border: 1px solid rgba(255, 255, 255, 0.10) !important;
    border-radius: 11px !important; color: var(--text-hi) !important;
    transition: border-color 150ms ease, box-shadow 150ms ease;
}
.stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {
    border-color: rgba(139, 92, 246, 0.75) !important;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.20) !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder { color: var(--text-low) !important; }
.stTextInput label, .stTextArea label, .stSelectbox label, .stSlider label,
.stNumberInput label, .stRadio label, .stCheckbox label, .stFileUploader label {
    font-size: 0.79rem !important; font-weight: 600 !important; color: var(--text-mid) !important;
    letter-spacing: 0.005em;
}

/* ---------- Pills / segmented control ---------- */
[data-baseweb="button-group"] button {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.10) !important;
    color: var(--text-mid) !important; border-radius: 999px !important;
    font-weight: 600 !important; font-size: 0.82rem !important; padding: 0.35rem 0.95rem !important;
    transition: all 140ms ease;
}
[data-baseweb="button-group"] button:hover {
    border-color: rgba(139, 92, 246, 0.45) !important; color: var(--text-hi) !important;
}
[data-baseweb="button-group"] button[aria-checked="true"],
[data-baseweb="button-group"] button[aria-pressed="true"] {
    background: linear-gradient(96deg, rgba(139, 92, 246, 0.30), rgba(99, 102, 241, 0.24)) !important;
    border-color: rgba(167, 139, 250, 0.70) !important; color: #FFFFFF !important;
    box-shadow: 0 0 0 1px rgba(139, 92, 246, 0.24), 0 6px 18px rgba(99, 102, 241, 0.24);
}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px; background: rgba(255, 255, 255, 0.03); padding: 6px;
    border-radius: 14px; border: 1px solid var(--edge); backdrop-filter: blur(12px);
}
.stTabs [data-baseweb="tab"] {
    height: auto; padding: 9px 17px; border-radius: 10px; background: transparent;
    color: var(--text-mid); font-weight: 600; font-size: 0.87rem; border: none;
    transition: all 150ms ease;
}
.stTabs [data-baseweb="tab"]:hover { background: rgba(255, 255, 255, 0.05); color: var(--text-hi); }
.stTabs [aria-selected="true"] {
    background: linear-gradient(96deg, rgba(139, 92, 246, 0.28), rgba(99, 102, 241, 0.20)) !important;
    color: #FFFFFF !important; box-shadow: inset 0 0 0 1px rgba(167, 139, 250, 0.45);
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* ---------- Sliders ---------- */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: linear-gradient(135deg, var(--violet), var(--indigo)) !important;
    border: 2px solid rgba(255, 255, 255, 0.85) !important;
    box-shadow: 0 3px 12px rgba(139, 92, 246, 0.55) !important;
}
.stSlider [data-testid="stSliderTickBarMin"], .stSlider [data-testid="stSliderTickBarMax"] {
    color: var(--text-low); font-size: 0.7rem;
}

/* ---------- Metrics ---------- */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.03); border: 1px solid var(--edge);
    border-radius: 14px; padding: 14px 16px; backdrop-filter: blur(12px);
}
[data-testid="stMetricLabel"] p {
    font-size: 0.70rem !important; text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--text-low) !important; font-weight: 650 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.5rem !important; font-weight: 750 !important; color: var(--text-hi) !important;
    letter-spacing: -0.025em;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(14, 16, 22, 0.96), rgba(8, 9, 13, 0.98));
    border-right: 1px solid var(--edge);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }

/* ---------- Expander, uploader, progress, alerts ---------- */
.stExpander, [data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.025); border: 1px solid var(--edge) !important;
    border-radius: 14px !important; backdrop-filter: blur(12px); overflow: hidden;
}
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255, 255, 255, 0.025); border: 1px dashed rgba(255, 255, 255, 0.16);
    border-radius: 14px; transition: border-color 150ms ease;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: rgba(139, 92, 246, 0.55); }
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--violet), var(--indigo) 55%, var(--cyan)) !important;
}
[data-testid="stAlert"] { border-radius: 13px; border: 1px solid var(--edge); backdrop-filter: blur(12px); }
[data-testid="stVideo"] video { border-radius: 16px; box-shadow: 0 18px 50px rgba(0, 0, 0, 0.55); }
[data-testid="stImage"] img { border-radius: 12px; }
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.12); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: rgba(139, 92, 246, 0.45); }
/* ---------- Command centre ---------- */
.rf-metric {
    border: 1px solid var(--edge); border-radius: var(--radius);
    background: linear-gradient(158deg, rgba(255, 255, 255, 0.055), rgba(255, 255, 255, 0.018));
    backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
    padding: 13px 15px 11px 15px; position: relative; overflow: hidden;
    transition: border-color 160ms ease, transform 160ms ease;
}
.rf-metric:hover { border-color: rgba(255, 255, 255, 0.15); transform: translateY(-1px); }
/* The accent rail is the only colour on the card: enough to group them at a
   glance without four saturated tiles fighting for attention. */
.rf-metric::before {
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    background: rgba(255, 255, 255, 0.16);
}
.rf-metric.violet::before { background: linear-gradient(180deg, var(--violet), var(--indigo)); }
.rf-metric.cyan::before   { background: linear-gradient(180deg, var(--cyan), #0EA5E9); }
.rf-metric.amber::before  { background: linear-gradient(180deg, var(--amber), #F59E0B); }
.rf-metric.green::before  { background: linear-gradient(180deg, #34D399, #10B981); }
.rf-metric .k {
    font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.085em;
    color: var(--text-low); font-weight: 700;
}
.rf-metric .v {
    font-size: 1.42rem; font-weight: 780; color: var(--text-hi);
    letter-spacing: -0.028em; line-height: 1.18; margin-top: 3px;
}
.rf-metric .d { font-size: 0.71rem; color: var(--text-low); margin-top: 3px; }

/* ---------- Mode guidance banner ---------- */
.rf-guide {
    border: 1px solid var(--edge); border-radius: var(--radius);
    background: linear-gradient(120deg, rgba(139, 92, 246, 0.10), rgba(34, 211, 238, 0.05) 58%, transparent);
    backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
    padding: 14px 16px; margin: 2px 0 18px 0; position: relative;
}
.rf-guide .rf-badge { margin-bottom: 8px; }
.rf-guide-body { font-size: 0.84rem; line-height: 1.5; color: var(--text-mid); }
.rf-guide-body b { color: var(--text-hi); font-weight: 650; }
.rf-guide-eta {
    position: absolute; top: 13px; right: 16px;
    font-size: 0.72rem; font-weight: 650; letter-spacing: 0.03em;
    color: var(--text-low); border: 1px solid var(--edge);
    border-radius: 999px; padding: 3px 10px; background: rgba(0, 0, 0, 0.22);
}

/* ---------- Recent exports drawer ---------- */
.rf-export-name {
    font-size: 0.78rem; font-weight: 650; color: var(--text-hi);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-top: 8px;
}
.rf-export-meta { font-size: 0.70rem; color: var(--text-low); margin: 2px 0 8px 0; }
.rf-thumb-blank {
    aspect-ratio: 9 / 16; max-height: 190px; border-radius: 10px;
    background: rgba(255, 255, 255, 0.04); border: 1px solid var(--edge);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem; color: var(--text-low);
}

/* ---------- Viral scorecard ---------- */
.rf-score { display: flex; align-items: baseline; gap: 6px; }
.rf-score .n { font-size: 2.6rem; font-weight: 800; letter-spacing: -0.04em; line-height: 1; }
.rf-score .d { font-size: 0.95rem; color: var(--text-low); font-weight: 600; }
.rf-score.green .n { color: #34D399; }
.rf-score.amber .n { color: var(--amber); }
.rf-score.red   .n { color: #F87171; }
.rf-score-verdict { font-size: 0.86rem; color: var(--text-mid); margin-top: 4px; }
.rf-score-src {
    font-size: 0.68rem; color: var(--text-low); margin-top: 2px;
    text-transform: lowercase; letter-spacing: 0.02em;
}
.rf-axis {
    border: 1px solid var(--edge); border-radius: 11px; padding: 11px 13px;
    background: rgba(255, 255, 255, 0.028); margin-bottom: 6px;
}
.rf-axis .k {
    font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.075em;
    color: var(--text-low); font-weight: 700;
}
.rf-axis .v { font-size: 1.30rem; font-weight: 760; color: var(--text-hi); letter-spacing: -0.025em; }
.rf-axis-bar {
    height: 4px; border-radius: 3px; background: rgba(255, 255, 255, 0.08);
    overflow: hidden; margin-top: 7px;
}
.rf-axis-bar i { display: block; height: 100%; border-radius: 3px; background: var(--text-low); }
.rf-axis.green .rf-axis-bar i { background: linear-gradient(90deg, #10B981, #34D399); }
.rf-axis.amber .rf-axis-bar i { background: linear-gradient(90deg, #F59E0B, var(--amber)); }
.rf-axis.red   .rf-axis-bar i { background: linear-gradient(90deg, #DC2626, #F87171); }
/* ---------- Dashboard landing ---------- */
.rf-hero {
    border: 1px solid var(--edge); border-radius: 18px; padding: 22px 26px;
    background:
        radial-gradient(720px 240px at 6% -40%, rgba(139, 92, 246, 0.20), transparent 70%),
        radial-gradient(620px 220px at 96% 140%, rgba(34, 211, 238, 0.13), transparent 68%),
        rgba(255, 255, 255, 0.028);
    backdrop-filter: blur(14px); margin-bottom: 18px;
}
.rf-hero-title {
    font-size: 1.42rem; font-weight: 800; color: var(--text-hi);
    letter-spacing: -0.028em; line-height: 1.3;
}
.rf-hero-lede {
    font-size: 0.90rem; color: var(--text-mid); margin-top: 6px;
    line-height: 1.55; max-width: 860px;
}
.rf-hero-badges { margin-top: 13px; }

/* Stepper */
.rf-step {
    border: 1px solid var(--edge); border-radius: 14px; padding: 15px 17px 13px 17px;
    background: rgba(255, 255, 255, 0.028); transition: border-color 140ms ease;
    /* Streamlit columns do not stretch their children, so the three cards are
       squared off against the tallest -- step two, which carries the two lane
       chip rows. Without it the cards end at three different heights and the
       buttons under them stagger. */
    min-height: 236px;
}
.rf-step:hover { border-color: var(--edge-hi); }
.rf-step-n {
    display: inline-flex; align-items: center; justify-content: center;
    width: 25px; height: 25px; border-radius: 8px; font-size: 0.80rem; font-weight: 800;
    color: #0B0C11; background: linear-gradient(135deg, var(--violet), var(--indigo));
    margin-bottom: 9px;
}
.rf-step-t {
    font-size: 0.99rem; font-weight: 720; color: var(--text-hi);
    letter-spacing: -0.018em; margin-bottom: 5px;
}
.rf-step-b { font-size: 0.805rem; line-height: 1.52; color: var(--text-mid); }
.rf-step-lane {
    font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.085em;
    color: var(--text-low); font-weight: 700; margin: 11px 0 5px 0;
}

/* Engine grid */
.rf-engine {
    border: 1px solid var(--edge); border-radius: 14px; padding: 15px 16px 11px 16px;
    background: rgba(255, 255, 255, 0.028); position: relative; overflow: hidden;
    transition: border-color 140ms ease, transform 140ms ease;
}
.rf-engine:hover { border-color: var(--edge-hi); transform: translateY(-1px); }
.rf-engine-head { display: flex; align-items: center; gap: 9px; margin-bottom: 9px; }
.rf-engine-icon {
    font-size: 1.16rem; line-height: 1; width: 32px; height: 32px; flex: 0 0 32px;
    border-radius: 10px; background: rgba(255, 255, 255, 0.05);
    border: 1px solid var(--edge); display: flex; align-items: center;
    justify-content: center;
}
.rf-engine-name {
    font-size: 0.93rem; font-weight: 720; color: var(--text-hi); letter-spacing: -0.018em;
}
.rf-engine-use {
    font-size: 0.785rem; line-height: 1.52; color: var(--text-mid); margin-top: 9px;
    min-height: 4.56em;
}

/* Connectivity panel */
.rf-api {
    border: 1px solid var(--edge); border-radius: 12px; padding: 11px 13px;
    background: rgba(255, 255, 255, 0.028); margin-bottom: 7px;
}
.rf-api-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.rf-api-name {
    font-size: 0.80rem; font-weight: 700; color: var(--text-hi); letter-spacing: -0.01em;
}
.rf-api-detail { font-size: 0.70rem; line-height: 1.45; color: var(--text-low); margin-top: 6px; }

/* Magic Studio hero */
.rf-magic { margin: 2px 0 12px 0; }
.rf-magic-title {
    font-size: 1.06rem; font-weight: 760; color: var(--text-hi);
    letter-spacing: -0.02em;
}
.rf-magic-sub { font-size: 0.83rem; color: var(--text-mid); margin-top: 3px; }

/* The retention sentence under the scorecard. One instruction, given room. */
.rf-tip {
    border: 1px solid rgba(251, 191, 36, 0.34); border-radius: 11px;
    background: rgba(251, 191, 36, 0.08); padding: 11px 13px; margin: 10px 0 6px 0;
    font-size: 0.83rem; line-height: 1.55; color: var(--text-mid);
}
.rf-tip b { color: var(--amber); font-weight: 700; }

/* The prompt box is the primary control on the landing page, so it is sized
   like one rather than like a filter field. */
.rf-magic + div [data-testid="stTextInput"] input {
    font-size: 1.0rem !important; padding: 13px 15px !important;
}

/* Sidebar navigation: full-width rows rather than centred pills, so the
   grouped menu reads as a list. */
[data-testid="stSidebar"] .stButton > button {
    justify-content: flex-start; text-align: left; font-size: 0.815rem;
    padding: 8px 12px; font-weight: 600;
}
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Export handling
#
# Renders go to a project-local exports/ folder with a unique timestamped name.
# The previous code wrote to tempfile.NamedTemporaryFile(...).name, which on
# Windows leaves an OPEN HANDLE holding a lock on the path (verified: renaming
# it raises WinError 32) -- so ffmpeg could intermittently fail to write it.
# %TEMP% is also swept by Windows cleanup, which made finished renders vanish.
# ---------------------------------------------------------------------------

def current_user() -> dict[str, Any]:
    """The signed-in user, or an empty record before the gate has run."""
    session = st.session_state.get("auth")
    return session if isinstance(session, dict) else {}


def user_exports() -> str:
    """
    The signed-in user's own exports folder, created on demand.

    Every render, ledger write and download resolves through here rather than
    through a module constant, because a module constant is process-wide and
    this process serves every user at once. `_unassigned` is unreachable from
    the UI -- the login gate stops before any panel renders -- and exists only
    so a stray call cannot land in someone else's directory.
    """
    return ensure_dir(auth.user_exports_dir(str(current_user().get("username") or "_unassigned")))


def new_export_path(prefix: str) -> str:
    """Returns a fresh, unique output path inside the current user's folder."""
    return os.path.join(user_exports(), f"{prefix}_{int(time.time())}.mp4")


# Intermediates a render leaves behind. Finished deliverables (commentary_*,
# reel_*, duel_*, muted_*, narration_*, source_*) are deliberately absent:
# those are the user's work product and are never touched.
SCRATCH_PATTERNS: tuple[str, ...] = (
    "bgm_*.wav",
    "captions_*.ass",
    "music_*.wav",
    "focus_*.png",
    "*__nosubs*.mp4",
)

SCRATCH_MAX_AGE = 3600.0        # one hour


def cleanup_exports(
    max_age_seconds: float = SCRATCH_MAX_AGE,
    directory: str = "",
    protect: Sequence[str] = (),
) -> dict[str, Any]:
    """
    Purges stale render intermediates from exports/.

    Only files matching SCRATCH_PATTERNS and older than `max_age_seconds` are
    removed, and anything still referenced by the live session is protected
    regardless of age -- a long editing session can easily outlive the cutoff
    while its .ass and BGM are still needed for the next render.
    """
    target = directory or user_exports()
    if not os.path.isdir(target):
        return {"removed": 0, "bytes": 0, "errors": 0}

    guarded = {os.path.abspath(p) for p in protect if p}
    cutoff = time.time() - max(0.0, float(max_age_seconds))
    removed = freed = errors = 0

    for pattern in SCRATCH_PATTERNS:
        for path in glob.glob(os.path.join(target, pattern)):
            try:
                if os.path.abspath(path) in guarded:
                    continue
                if os.path.getmtime(path) > cutoff:
                    continue
                size = os.path.getsize(path)
                os.remove(path)
                removed += 1
                freed += size
            except OSError:
                # A file being written by another process is simply skipped;
                # it will age out on the next sweep.
                errors += 1

    return {"removed": removed, "bytes": freed, "errors": errors}


def _session_protected_paths() -> list[str]:
    """Paths the current session still depends on, immune from the sweep."""
    keep: list[str] = []
    commentary = st.session_state.get("commentary") or {}
    for key in ("ass_path", "audio_path", "source_path"):
        value = commentary.get(key)
        if value:
            keep.append(str(value))
    for scope in ("commentary", "reel", "duel"):
        value = st.session_state.get(f"{scope}_video_path")
        if value:
            keep.append(str(value))
    return keep


def sweep_scratch_files(force: bool = False) -> dict[str, Any]:
    """
    Runs the sweep at most once per session unless `force` is set.

    Called on first load and again after each render, which is when new
    intermediates appear.
    """
    if not force and st.session_state.get("_scratch_swept"):
        return {"removed": 0, "bytes": 0, "errors": 0}
    st.session_state["_scratch_swept"] = True
    result = cleanup_exports(protect=_session_protected_paths())
    # The temp directory too: a render that crashed before its purge ran would
    # otherwise leave a source-clip-sized file there permanently.
    stale = sweep_temp_renders()
    result["removed"] += stale["removed"]
    result["bytes"] += stale["bytes"]
    return result


def clear_rendered_video(scope: str) -> None:
    """
    Drops a previously rendered video from session state.

    Called whenever a new script or duel is built, so the player can never show
    a stale render from the last build.
    """
    for key in (f"{scope}_video_path", f"{scope}_video_bytes", f"{scope}_video_name"):
        st.session_state.pop(key, None)


def show_rendered_video(scope: str, label: str, slot: str) -> None:
    """
    Renders the player + download button for a finished video.

    The download button is handed the raw bytes captured at render time with a
    filename unique to that render, so a browser can never serve a cached copy
    of an earlier export.

    `slot` identifies the call site. The same video is offered in more than one
    tab, and Streamlit rejects two widgets sharing a key -- so the key must mix
    in where it is being drawn, not just which video it is.
    """
    path = st.session_state.get(f"{scope}_video_path")
    data = st.session_state.get(f"{scope}_video_bytes")
    if not path or not data:
        return

    with st.container(border=True):
        st.markdown(f"#### 🎉 {label}")
        st.markdown(
            badge(os.path.basename(path), "violet") + badge(f"{len(data) / 1_048_576:.1f} MB", "cyan"),
            unsafe_allow_html=True,
        )
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.video(data)
        st.download_button(
            "📥 Download MP4",
            data=data,
            file_name=st.session_state.get(f"{scope}_video_name", os.path.basename(path)),
            mime="video/mp4",
            width="stretch",
            key=f"dl_{slot}_{scope}_{os.path.basename(path)}",
        )
        st.caption(f"Saved to `{path}`")


def badge(text: str, tone: str = "") -> str:
    """Inline pill badge markup."""
    return f'<span class="rf-badge {tone}">{text}</span>'


def section(label: str) -> None:
    """Small uppercase section label."""
    st.markdown(f'<div class="rf-section">{label}</div>', unsafe_allow_html=True)


def stat_row(items: Sequence[tuple[str, str, str]]) -> None:
    """Compact metric tags: a sequence of (label, value, tone)."""
    cells = "".join(
        f'<div class="rf-stat {tone}"><div class="k">{k}</div><div class="v">{v}</div></div>'
        for k, v, tone in items
    )
    st.markdown(f'<div class="rf-tag-row">{cells}</div>', unsafe_allow_html=True)


def divider() -> None:
    st.markdown('<hr class="rf-hr">', unsafe_allow_html=True)


def pick(
    label: str,
    options: Sequence[Any],
    default: Any,
    key: str,
    format_func: Callable[[Any], str] = str,
    help: str | None = None,
) -> Any:
    """
    Pill selector that never returns None.

    st.pills allows deselection, which would otherwise hand downstream code a
    None where it expects a real option.
    """
    chosen = st.pills(
        label, list(options), default=default, key=key,
        format_func=format_func, help=help,
    )
    return default if chosen is None else chosen


# ---------------------------------------------------------------------------
# Command centre
#
# Four live readings across the top of every page. Each one answers a question
# that otherwise needs a file browser or a terminal: how much have I made, how
# much disk is it holding, is this machine going to render fast, and is the
# compliance gate armed.
# ---------------------------------------------------------------------------

def _human_bytes(size: float) -> str:
    """1024-based, two significant-ish figures: '812 KB', '1.4 GB'."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:,.0f} {unit}" if unit in ("B", "KB") else f"{size:,.1f} {unit}"
        size /= 1024.0
    return f"{size:,.1f} TB"


def workspace_stats() -> dict[str, Any]:
    """
    Walks the signed-in user's export folder once and totals it.

    Scratch is counted separately from renders because the purge button must
    never be able to delete a finished video: `scratch_bytes` is what it will
    free, and it excludes every *.mp4 that is not a known intermediate.
    """
    root = user_exports()
    renders = 0
    render_bytes = 0
    scratch_bytes = 0
    scratch_files = 0
    newest = 0.0

    for entry in os.scandir(root) if os.path.isdir(root) else ():
        if not entry.is_file():
            continue
        try:
            stat = entry.stat()
        except OSError:
            continue
        name = entry.name.lower()

        if name.endswith(".mp4") and not name.startswith(("source_", "muted_", "reframed_")):
            renders += 1
            render_bytes += stat.st_size
            newest = max(newest, stat.st_mtime)
        elif name.startswith(("source_", "muted_", "reframed_")) or name.endswith(
                (".wav", ".mp3", ".webm", "_gemini.mp4")):
            scratch_bytes += stat.st_size
            scratch_files += 1

    return {
        "renders": renders,
        "render_bytes": render_bytes,
        "scratch_bytes": scratch_bytes,
        "scratch_files": scratch_files,
        "total_bytes": render_bytes + scratch_bytes,
        "newest": newest,
        "root": root,
    }


def hardware_label() -> tuple[str, str]:
    """
    (headline, detail) for the encoder this machine will actually use.

    video_encoder() probes NVENC by encoding a real frame rather than reading
    the -encoders list, so this reflects what the driver will accept, not what
    the binary was built with.
    """
    try:
        enc = video_encoder()
    except Exception:
        return "CPU", "libx264"

    if enc.get("gpu"):
        return "NVENC GPU", f"{enc['codec']} · preset {enc['preset']}"
    return "Multi-thread CPU", f"{enc['codec']} · preset {enc['preset']} · {os.cpu_count() or '?'} threads"


def render_command_center() -> None:
    """The four telemetry badges, plus the scratch purge."""
    stats = workspace_stats()
    ledger = summarise_ledger(stats["root"])
    hw_head, hw_detail = hardware_label()

    gate_strict = bool(ledger["total"]) and ledger["blocked"] > 0
    gate_head = "Strict" if gate_strict else "Active"
    gate_detail = (f"{ledger['blocked']} of {ledger['total']} blocked"
                   if ledger["total"] else "nothing logged yet")

    cols = st.columns([1, 1.15, 1.15, 1.1])
    cards = (
        ("Total renders", f"{stats['renders']}",
         f"newest {time.strftime('%d %b %H:%M', time.localtime(stats['newest']))}"
         if stats["newest"] else "no renders yet", "violet"),
        ("Storage in use", _human_bytes(stats["total_bytes"]),
         f"{_human_bytes(stats['scratch_bytes'])} of it scratch", "cyan"),
        ("Hardware engine", hw_head, hw_detail, "green" if hw_head.startswith("NVENC") else ""),
        ("Monetization gate", gate_head, gate_detail, "amber" if gate_strict else "green"),
    )

    for col, (label, value, detail, tone) in zip(cols, cards):
        with col:
            st.markdown(
                f'<div class="rf-metric {tone}"><div class="k">{label}</div>'
                f'<div class="v">{value}</div><div class="d">{detail}</div></div>',
                unsafe_allow_html=True,
            )

    with cols[1]:
        if st.button(f"🧹 Purge scratch files ({stats['scratch_files']})",
                     key="purge_scratch", width="stretch",
                     disabled=stats["scratch_files"] == 0,
                     help="Deletes downloaded sources, muted intermediates, Gemini "
                          "pre-flight transcodes and narration WAVs. Finished renders "
                          "are never touched."):
            freed = sweep_scratch_files(force=True)
            st.toast(f"Freed {_human_bytes(freed.get('bytes', 0))} "
                     f"across {freed.get('removed', 0)} files.", icon="🧹")
            st.rerun()


# ---------------------------------------------------------------------------
# Mode guidance
# ---------------------------------------------------------------------------

MODE_GUIDES: dict[str, dict[str, Any]] = {
    "commentary": {
        "badges": ["9:16 Vertical", "Kinetic Captions", "Fair-Use Gated"],
        "niche": "Reaction and breakdown channels — sports, drama, tech unboxings.",
        "eta": "90-180s",
        "note": "Gemini watches the clip before it writes, so the commentary is about "
                "what actually happens rather than about the title.",
    },
    "minimalist": {
        "badges": ["9:16 Vertical", "Zero Copyright Risk", "No Footage Needed"],
        "niche": "Self-improvement, finance psychology, stoicism — the @SimplyAnimated lane.",
        "eta": "60-120s",
        "note": "Every frame is drawn from code. There is nothing to license and "
                "nothing to claim.",
    },
    "narrative": {
        "badges": ["9:16 or 16:9", "Episodic", "Consistent Cast"],
        "niche": "Story channels: true crime, folklore, quiet reflective essays.",
        "eta": "3-8 min",
        "note": "Narration is synthesized first and the storyboard is re-cut against "
                "the real word timings, so picture and voice cannot drift.",
    },
    "batch": {
        "badges": ["9:16 Vertical", "Queued", "Unattended"],
        "niche": "Volume posting — one topic list becomes a week of uploads.",
        "eta": "~90s per topic",
        "note": "Runs the full pipeline per topic back to back. Leave it going.",
    },
    "reel": {
        "badges": ["9:16 or 1:1", "Licensed Stock", "Sourced Facts"],
        "niche": "Listicle and explainer pages that need a figure in every line.",
        "eta": "45-90s",
        "note": "Beats come from researched facts with sources attached, not from a "
                "template with the nouns swapped.",
    },
    "atmosphere": {
        "badges": ["16:9 Horizontal", "30 min – 8 hours", "Original Soundscape"],
        "niche": "Sleep, study and focus channels — rain, storms, brown noise, fire.",
        "eta": "1-4 min",
        "note": "Every audio layer is synthesized rather than licensed, and the loop "
                "points are crossfaded, so there is nothing to claim and nothing to "
                "click. Publishes straight to YouTube.",
    },
    "duel": {
        "badges": ["9:16 Vertical", "Split Screen", "High RPM"],
        "niche": "Comparison content — cars, phones, watches. Strong comment sections.",
        "eta": "60-120s",
        "note": "Stat cards count up on the same clock the SFX are scheduled against, "
                "so every hit lands on the frame its card appears.",
    },
}


def render_mode_guide(mode: str) -> None:
    """The glass info banner under a mode header."""
    guide = MODE_GUIDES.get(mode)
    if not guide:
        return

    badges = "".join(badge(b, tone) for b, tone in
                     zip(guide["badges"], ("violet", "cyan", "amber", "green")))
    st.markdown(
        f'<div class="rf-guide">{badges}'
        f'<div class="rf-guide-body"><b>Best for:</b> {guide["niche"]}<br>{guide["note"]}</div>'
        f'<div class="rf-guide-eta">~{guide["eta"]} per render</div></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Recent exports drawer
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Exports Library
#
# Everything the signed-in account has rendered, in one place. It used to be a
# four-item strip pinned under every creation page, which meant every workflow
# ended in a row of unrelated thumbnails and a video player that stayed open
# across mode switches. It is a destination now, not a footer.
#
# Scoped to user_exports() and nowhere else: one account cannot see, play or
# delete another's work through this page any more than through any other.
# ---------------------------------------------------------------------------

# Filename prefix -> what made it. The render paths write these, so the prefix
# is a reliable tag without opening the file.
EXPORT_KINDS: dict[str, str] = {
    "commentary_": "🎙️ Commentary",
    "minimalist_": "◼️ Minimalist",
    "narrative_": "📖 Narrative",
    "batch_": "📦 Batch",
    "reel_": "🎬 Reel",
    "duel_": "⚔️ Duel",
    "atmosphere_": "🌙 Atmosphere",
    "licensed_": "🎞️ Licensed clip",
}

# Intermediates. They live in the same folder and are not finished work, so
# they are hidden here and swept by the command centre's purge button instead.
SCRATCH_PREFIXES = ("source_", "muted_", "reframed_", "narration_", "music_", "preview_")

SORT_MODES: dict[str, str] = {
    "newest": "🕑 Newest first",
    "oldest": "🕐 Oldest first",
    "largest": "💾 Largest first",
    "name": "🔤 Name",
}

# st.download_button reads the whole file into memory and pushes it through the
# websocket. An eight-hour Atmosphere render is several gigabytes; offering a
# browser download for that would hang the tab and can take the server with it.
DOWNLOAD_LIMIT_BYTES = 400 * 1024 * 1024


def export_kind(name: str) -> str:
    """The engine that produced a file, from its filename prefix."""
    lowered = name.lower()
    for prefix, label in EXPORT_KINDS.items():
        if lowered.startswith(prefix):
            return label
    return "📄 Other"


def aspect_tag(width: int, height: int) -> str:
    """'9:16', '16:9', '1:1' or the raw ratio when it is none of those."""
    if width <= 0 or height <= 0:
        return ""
    ratio = width / height
    for label, value in (("9:16", 9 / 16), ("16:9", 16 / 9), ("1:1", 1.0), ("4:5", 4 / 5)):
        if abs(ratio - value) < 0.02:
            return label
    return f"{ratio:.2f}:1"


def list_exports(sort: str = "newest", kind: str = "all",
                 include_scratch: bool = False) -> list[dict[str, Any]]:
    """Every finished render for the signed-in user, filtered and sorted."""
    root = user_exports()
    found: list[dict[str, Any]] = []

    for entry in os.scandir(root) if os.path.isdir(root) else ():
        name = entry.name
        lowered = name.lower()
        if not entry.is_file() or not lowered.endswith((".mp4", ".webm", ".mov", ".mkv")):
            continue
        if not include_scratch and (lowered.startswith(SCRATCH_PREFIXES)
                                    or lowered.endswith("_gemini.mp4")):
            continue
        try:
            stat = entry.stat()
        except OSError:
            continue

        label = export_kind(name)
        if kind != "all" and label != kind:
            continue

        found.append({"path": entry.path, "name": name, "kind": label,
                      "bytes": stat.st_size, "mtime": stat.st_mtime})

    keys = {
        "newest": (lambda item: item["mtime"], True),
        "oldest": (lambda item: item["mtime"], False),
        "largest": (lambda item: item["bytes"], True),
        "name": (lambda item: item["name"].lower(), False),
    }
    key, reverse = keys.get(sort, keys["newest"])
    found.sort(key=key, reverse=reverse)
    return found


@st.cache_data(show_spinner=False, max_entries=64)
def export_details(path: str, mtime: float, size: int) -> dict[str, Any]:
    """
    Duration, resolution and a poster frame for one file.

    Keyed on (path, mtime, size) so a re-render to the same filename produces a
    new cache entry rather than showing the previous video's thumbnail.
    """
    info: dict[str, Any] = {"duration": 0.0, "width": 0, "height": 0, "thumb": None}
    try:
        probe = probe_stream_info(path)
        info["duration"] = float(probe.get("duration") or 0.0)
        info["width"] = int(probe.get("width") or 0)
        info["height"] = int(probe.get("height") or 0)
    except Exception:
        pass

    try:
        from moviepy import VideoFileClip

        with VideoFileClip(path) as clip:
            if not info["duration"]:
                info["duration"] = float(clip.duration or 0.0)
            if not info["width"]:
                info["width"], info["height"] = int(clip.w), int(clip.h)
            # 12% in rather than frame zero -- a duel opens on a half-drawn VS
            # medallion and a reel on a fade from black. Capped at 30s because
            # seeking an hour into a multi-gigabyte file to make a thumbnail is
            # not worth the wait.
            at = min(max(0.3, clip.duration * 0.12), 30.0, max(0.0, clip.duration - 0.1))
            frame = clip.get_frame(at)
        thumb = Image.fromarray(frame)
        thumb.thumbnail((480, 480))
        info["thumb"] = thumb
    except Exception:
        pass

    return info


def reveal_in_file_manager(path: str) -> str:
    """
    Opens the folder containing `path` in the OS file manager.

    Returns "" on success or a reason it could not. Worth being explicit about
    what this does: it opens a window on the machine running Streamlit, which
    is the server. Over a tunnel, or in Docker, that is not the machine looking
    at the page -- so the UI offers the path to copy as well.
    """
    import subprocess

    folder = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(folder):
        return f"{folder} does not exist"

    try:
        if sys.platform == "win32":
            # /select, highlights the file rather than just opening the folder.
            subprocess.Popen(["explorer", "/select,", os.path.abspath(path)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", os.path.abspath(path)])
        else:
            subprocess.Popen(["xdg-open", folder])
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return ""


def delete_export(path: str) -> str:
    """
    Removes one render. Returns "" on success or the reason it failed.

    The provenance ledger entry is deliberately left alone: it is the record of
    what was made from what, and it is the thing that answers a copyright claim
    months later. Deleting a video does not un-make it.
    """
    target = os.path.abspath(path)
    root = os.path.abspath(user_exports())

    # Never delete outside the signed-in account's own folder, whatever a
    # crafted path says.
    if os.path.commonpath([target, root]) != root:
        return "That file is outside your workspace."
    if not os.path.isfile(target):
        return "That file is already gone."

    try:
        os.remove(target)
    except OSError as exc:
        return f"{type(exc).__name__}: {exc}"
    return ""


def render_exports_library() -> None:
    """The gallery: everything this account has rendered."""
    st.markdown("### 📁 Exports Library")

    root = user_exports()
    everything = list_exports()
    kinds = sorted({item["kind"] for item in everything})

    if not everything:
        st.info(
            "Nothing rendered yet. Anything you make in the production modes lands "
            f"here, in `{root}`.", icon="📭")
        return

    # --- filters ----------------------------------------------------------
    with st.container(border=True):
        f1, f2, f3 = st.columns([1.4, 1.4, 1])
        with f1:
            sort = pick("Sort", list(SORT_MODES), "newest", "lib_sort",
                        format_func=lambda k: SORT_MODES[k])
        with f2:
            kind = pick("Made by", ["all"] + kinds, "all", "lib_kind",
                        format_func=lambda k: "All modes" if k == "all" else k)
        with f3:
            columns = st.selectbox("Per row", [2, 3, 4], index=1, key="lib_cols")

        items = list_exports(sort=sort, kind=kind)
        total = sum(item["bytes"] for item in everything)
        shown = sum(item["bytes"] for item in items)
        stat_row([
            ("Videos", f"{len(items)}" + (f" of {len(everything)}"
                                          if len(items) != len(everything) else ""), "violet"),
            ("Shown", _human_bytes(shown), "cyan"),
            ("Library total", _human_bytes(total), "amber"),
            ("Folder", os.path.basename(root) or root, ""),
        ])

    if not items:
        st.caption("Nothing matches that filter.")
        return

    # --- the expanded player, above the grid -------------------------------
    playing = str(st.session_state.get("library_playing") or "")
    if playing and os.path.exists(playing):
        with st.container(border=True):
            st.markdown(f"#### ▶ {os.path.basename(playing)}")
            st.video(playing)
            if st.button("Close player", key="lib_close_player"):
                st.session_state.pop("library_playing", None)
                st.rerun()
    elif playing:
        # The file was deleted while it was open.
        st.session_state.pop("library_playing", None)

    # --- confirmation, which has to be resolved before anything else --------
    pending = str(st.session_state.get("library_pending_delete") or "")
    if pending:
        with st.container(border=True):
            st.warning(
                f"Delete **{os.path.basename(pending)}** permanently? "
                f"This frees "
                f"{_human_bytes(os.path.getsize(pending) if os.path.exists(pending) else 0)} "
                f"and cannot be undone. The provenance entry is kept.",
                icon="⚠️")
            yes, no = st.columns(2)
            with yes:
                if st.button("Delete it", key="lib_confirm_delete", type="primary",
                             width="stretch"):
                    problem = delete_export(pending)
                    st.session_state.pop("library_pending_delete", None)
                    if str(st.session_state.get("library_playing") or "") == pending:
                        st.session_state.pop("library_playing", None)
                    if problem:
                        st.error(problem)
                    else:
                        st.toast(f"Deleted {os.path.basename(pending)}", icon="🗑️")
                        st.rerun()
            with no:
                if st.button("Keep it", key="lib_cancel_delete", width="stretch"):
                    st.session_state.pop("library_pending_delete", None)
                    st.rerun()

    # --- the grid ----------------------------------------------------------
    for row_start in range(0, len(items), columns):
        row = items[row_start:row_start + columns]
        cols = st.columns(columns)
        for col, item in zip(cols, row):
            with col, st.container(border=True):
                _render_export_tile(item)


def _render_export_tile(item: dict[str, Any]) -> None:
    """One card: poster, metadata, and the four actions."""
    details = export_details(item["path"], item["mtime"], item["bytes"])
    key = item["name"]

    if details["thumb"] is not None:
        st.image(details["thumb"], width="stretch")
    else:
        st.markdown('<div class="rf-thumb-blank">▶</div>', unsafe_allow_html=True)

    ratio = aspect_tag(details["width"], details["height"])
    st.markdown(
        badge(item["kind"], "violet")
        + (badge(ratio, "cyan") if ratio else "")
        + (badge(f"{details['width']}×{details['height']}", "")
           if details["width"] else ""),
        unsafe_allow_html=True,
    )

    duration = details["duration"]
    length = (f"{duration / 3600:.1f} h" if duration >= 3600
              else f"{duration / 60:.0f} min" if duration >= 120
              else f"{duration:.1f}s")
    st.markdown(
        f'<div class="rf-export-name" title="{item["name"]}">{item["name"]}</div>'
        f'<div class="rf-export-meta">{length} · {_human_bytes(item["bytes"])} · '
        f'{time.strftime("%d %b %Y, %H:%M", time.localtime(item["mtime"]))}</div>',
        unsafe_allow_html=True,
    )

    play, download = st.columns(2)
    with play:
        if st.button("▶ Play", key=f"lib_play_{key}", width="stretch"):
            st.session_state["library_playing"] = item["path"]
            st.rerun()
    with download:
        if item["bytes"] <= DOWNLOAD_LIMIT_BYTES:
            with open(item["path"], "rb") as handle:
                st.download_button("⬇ Download", data=handle.read(),
                                   file_name=item["name"], mime="video/mp4",
                                   width="stretch", key=f"lib_dl_{key}")
        else:
            st.button("⬇ Download", key=f"lib_dl_{key}", width="stretch", disabled=True,
                      help=f"{_human_bytes(item['bytes'])} is too large to push through "
                           f"the browser — copy it from disk instead.")

    reveal, remove = st.columns(2)
    with reveal:
        if st.button("📂 Reveal", key=f"lib_open_{key}", width="stretch",
                     help="Opens the folder in the file manager on the machine "
                          "running Streamlit — which is the server, not "
                          "necessarily the device you are reading this on."):
            problem = reveal_in_file_manager(item["path"])
            if problem:
                st.error(f"Could not open the folder: {problem}")
            else:
                st.toast("Opened on the server's desktop.", icon="📂")
    with remove:
        if st.button("🗑 Delete", key=f"lib_del_{key}", width="stretch"):
            st.session_state["library_pending_delete"] = item["path"]
            st.rerun()

    st.caption(f"`{item['path']}`")


# ---------------------------------------------------------------------------
# Progress with an ETA, and the finish chime
# ---------------------------------------------------------------------------

class StageProgress:
    """
    A progress bar that says how long is left, not just how far along it is.

    The estimate is elapsed/fraction rather than a fixed per-stage budget,
    because render time is dominated by clip length and encoder, both of which
    vary by an order of magnitude between a 10s minimalist scene and a 5-minute
    narrative episode. It is deliberately not shown until 8% of the way in --
    an ETA extrapolated from the first half second is noise.
    """

    def __init__(self, slot: Any = None, label: str = "") -> None:
        self.slot = slot if slot is not None else st.empty()
        self.bar = self.slot.progress(0.0, text=label or "Starting...")
        self.started = time.time()
        self.label = label

    def update(self, fraction: float, message: str) -> None:
        fraction = max(0.0, min(1.0, float(fraction)))
        elapsed = time.time() - self.started

        suffix = ""
        if fraction > 0.08:
            remaining = elapsed * (1.0 - fraction) / fraction
            suffix = f"  ·  {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining"
        elif elapsed > 2:
            suffix = f"  ·  {elapsed:.0f}s elapsed"

        try:
            self.bar.progress(fraction, text=f"{message}{suffix}")
        except Exception:
            pass

    def step(self, current: int, total: int, message: str) -> None:
        """Adapter for the (current, total, message) callbacks the engines use."""
        self.update(current / max(1, total), message)

    def finish(self, message: str = "Done.") -> float:
        elapsed = time.time() - self.started
        try:
            self.bar.progress(1.0, text=f"{message}  ·  {elapsed:.0f}s total")
        except Exception:
            pass
        return elapsed

    def empty(self) -> None:
        """Clears the bar, for the failure paths that used to call st.empty()."""
        try:
            self.slot.empty()
        except Exception:
            pass


# Fired once a render finishes. The chime is synthesized in the browser rather
# than shipped as an asset -- an <audio> element needs a file the artifact
# sandbox would have to serve, and a two-oscillator arpeggio is 20 lines.
#
# Both the sound and the notification are best-effort by design: autoplay
# policy blocks audio until the tab has been interacted with, and notification
# permission may be denied outright. Neither failure is worth reporting.
_COMPLETION_JS = """
<script>
(function () {
  try {
    var Ctx = window.AudioContext || window.webkitAudioContext;
    var ctx = new Ctx();
    // A major triad arpeggio: reads as "finished" rather than as an alert.
    [[523.25, 0.00], [659.25, 0.09], [783.99, 0.18], [1046.5, 0.27]].forEach(function (n) {
      var osc = ctx.createOscillator(), gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.value = n[0];
      var at = ctx.currentTime + n[1];
      gain.gain.setValueAtTime(0.0001, at);
      gain.gain.exponentialRampToValueAtTime(0.22, at + 0.015);
      gain.gain.exponentialRampToValueAtTime(0.0001, at + 0.42);
      osc.connect(gain); gain.connect(ctx.destination);
      osc.start(at); osc.stop(at + 0.45);
    });
  } catch (e) { /* autoplay policy; nothing to do */ }

  try {
    var title = %TITLE%, body = %BODY%;
    var fire = function () { new Notification(title, { body: body, icon: "" }); };
    if (!("Notification" in window)) { return; }
    if (Notification.permission === "granted") { fire(); }
    else if (Notification.permission !== "denied") {
      Notification.requestPermission().then(function (p) { if (p === "granted") { fire(); } });
    }
  } catch (e) { /* denied or unsupported */ }
})();
</script>
"""


def notify_complete(title: str, body: str) -> None:
    """Plays the finish chime and raises a native notification, if allowed."""
    import html
    import json as _json

    import streamlit.components.v1 as components

    script = (_COMPLETION_JS
              .replace("%TITLE%", _json.dumps(str(title)))
              .replace("%BODY%", _json.dumps(str(body))))
    try:
        components.html(script, height=0, width=0)
    except Exception:
        # Never let a notification failure take down a finished render.
        _ = html


# ---------------------------------------------------------------------------
# Viral scorecard card
# ---------------------------------------------------------------------------

def _score_tone(score: float) -> str:
    if score >= 8.0:
        return "green"
    if score >= 6.0:
        return "amber"
    return "red"


def render_viral_scorecard(script: str, scope: str, entry: dict[str, Any] | None = None,
                           on_rewrite: Callable[[str], None] | None = None) -> None:
    """
    The 1-10 scorecard, with a one-click rewrite when it lands under target.

    Scored by the model, once per distinct script. The offline heuristic used
    to be what showed by default and the model read was a button press away,
    which meant the number on screen was almost always the heuristic's -- and
    the heuristic has no opinion about whether a sentence says anything, so
    unremarkable scripts all landed in a flat band around five. Caching on the
    script body rather than on the render means this is one API call per
    script, not one per rerun, which is what made the opt-in necessary.
    """
    body = str(script or "").strip()
    if len(body.split()) < 6:
        return

    key = f"viral_card_{scope}"
    card = st.session_state.get(key)
    if not card or card.get("_for") != body:
        with st.spinner("Scoring the script..."):
            try:
                card = dict(score_virality(body, entry or {}))
            except Exception:
                # Never block the page on the scorer. The offline card is a
                # worse answer, not no answer.
                card = dict(viral_scorecard(body, entry or {}))
        card["_for"] = body
        st.session_state[key] = card

    with st.container(border=True):
        st.markdown("#### 📈 Viral Scorecard")

        head, actions = st.columns([2.4, 1])
        with head:
            tone = _score_tone(card["overall"])
            st.markdown(
                f'<div class="rf-score {tone}"><span class="n">{card["overall"]:.1f}</span>'
                f'<span class="d">/ 10</span></div>'
                f'<div class="rf-score-verdict">{card["verdict"]}</div>'
                f'<div class="rf-score-src">{card.get("source", "heuristic")}</div>',
                unsafe_allow_html=True,
            )
        with actions:
            if st.button("🔄 Re-score", key=f"aiscore_{scope}", width="stretch",
                         help="Scores this script again. Costs one API call."):
                with st.spinner("Scoring..."):
                    try:
                        fresh = dict(score_virality(body, entry or {}))
                        fresh["_for"] = body
                        st.session_state[key] = fresh
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Scoring failed: {type(exc).__name__}: {exc}")

        axes = st.columns(3)
        for col, (axis, label) in zip(axes, (
            ("hook", "Hook intrigue (0-3s)"),
            # The keys stay as they are -- they are written into every ledger
            # entry -- but the labels say what the axes actually measure.
            ("density", "Domain specificity"),
            ("monetization", "Originality & reuse safety"),
        )):
            with col:
                part = card[axis]
                st.markdown(
                    f'<div class="rf-axis {_score_tone(part["score"])}">'
                    f'<div class="k">{label}</div>'
                    f'<div class="v">{part["score"]:.1f}</div>'
                    f'<div class="rf-axis-bar"><i style="width:{part["score"] * 10:.0f}%"></i></div>'
                    f'</div>', unsafe_allow_html=True,
                )
                for note in part.get("notes", [])[:3]:
                    st.caption(note)

        tip = str(card.get("retention_tip") or "").strip()
        if tip:
            st.markdown(
                f'<div class="rf-tip"><b>To hold the first three seconds:</b> '
                f'{tip}</div>', unsafe_allow_html=True)

        for fix in card.get("fixes", [])[:4]:
            st.markdown(f"- {fix}")

        if card.get("needs_rewrite"):
            st.warning(
                f"Under the {VIRAL_TARGET_SCORE:.1f} target — {card['verdict'].lower()}",
                icon="⚠️",
            )
            if st.button("✍️ Rewrite for High Retention", key=f"rewrite_{scope}",
                         type="primary", width="stretch"):
                with st.spinner("Rewriting..."):
                    try:
                        result = rewrite_for_retention(body, card)
                    except Exception as exc:
                        st.error(f"Rewrite failed: {type(exc).__name__}: {exc}")
                        return

                st.session_state[f"rewrite_{scope}_result"] = result
                st.rerun()

        result = st.session_state.get(f"rewrite_{scope}_result")
        if result:
            delta = result["after"] - result["before"]
            st.markdown(
                badge(f"{result['before']:.1f} → {result['after']:.1f}",
                      "green" if delta > 0 else "amber")
                + badge(result["model"], ""), unsafe_allow_html=True,
            )
            if not result["improved"]:
                # Reporting this honestly matters more than the button looking
                # like it always works.
                st.caption("The rewrite did not score better than the original. "
                           "Keep the original unless you prefer how this reads.")
            st.text_area("Rewritten script", value=result["script"], height=180,
                         key=f"rewritten_{scope}")
            apply_col, drop_col = st.columns(2)
            with apply_col:
                if st.button("Use this script", key=f"userewrite_{scope}", width="stretch"):
                    if on_rewrite:
                        on_rewrite(result["script"])
                    st.session_state.pop(f"rewrite_{scope}_result", None)
                    st.session_state.pop(key, None)
                    st.rerun()
            with drop_col:
                if st.button("Discard", key=f"droprewrite_{scope}", width="stretch"):
                    st.session_state.pop(f"rewrite_{scope}_result", None)
                    st.rerun()


# ---------------------------------------------------------------------------
# Reel scripting: Hook -> Fact Beats -> Call To Action
#
# The copy itself lives in reel_engine, which researches real figures for the
# topic instead of filling a template. What stays here is the slide assembly:
# photography, motion and caption styling.
# ---------------------------------------------------------------------------

_AUTO_PALETTES = [
    ((10, 15, 45), (120, 20, 110), (255, 110, 60)),
    ((5, 30, 50), (10, 90, 120), (120, 220, 200)),
    ((40, 10, 60), (130, 30, 120), (255, 140, 180)),
    ((15, 20, 25), (70, 60, 40), (240, 200, 120)),
    ((8, 12, 40), (60, 40, 130), (150, 120, 255)),
]


def generate_viral_script(
    topic: str,
    num_points: int = 3,
    size: tuple[int, int] = (1080, 1350),
    use_photos: bool = True,
    use_ai: bool = True,
    seconds: int = 30,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Builds a complete Hook -> Fact Beats -> CTA slide deck from a single topic.

    Returns (slides, script). The script carries `source`, `needs_editing` and
    the per-beat sources, which the UI shows -- an unsourced line has to be
    visible as unsourced, not quietly rendered as though it were researched.
    """
    script = reel_engine.build_fact_script(
        topic, count=num_points, seconds=seconds, use_ai=use_ai, progress=progress)
    lines = reel_engine.script_lines(script)
    subject = reel_engine.classify_topic(topic)
    motions = ["zoom_in", "pan_right", "zoom_out", "pan_left", "breathe"]

    # One clean query, several distinct results -- one per slide.
    photo_set: list[tuple[Any, str]] = []
    if use_photos:
        try:
            photo_set = _cached_photo_set(reel_engine._subject(topic), len(lines))
        except Exception:
            photo_set = []

    beats = list(script.get("beats") or [])
    slides: list[dict[str, Any]] = []
    beat_index = 0

    for idx, (role, line) in enumerate(lines):
        credit = ""
        image = None

        if photo_set:
            image, credit = photo_set[idx % len(photo_set)]

        if image is None:
            c1, c2, c3 = _AUTO_PALETTES[idx % len(_AUTO_PALETTES)]
            image = create_gradient_mesh(size[0], size[1], c1, c2, c3, angle=30 + idx * 15)

        source = ""
        if role == "point" and beat_index < len(beats):
            source = str(beats[beat_index].get("source") or "")
            beat_index += 1

        slides.append({
            "kind": "image",
            "image": image,
            "credit": credit,
            "query": subject,
            "title": f"{role.upper()} — {topic.strip()}",
            "caption": reel_engine.short_caption(line),
            "voiceover": line,
            # Long fact lines need longer on screen than a template slogan did.
            "duration": max(3.5, min(7.0, len(line.split()) / 2.75 + 0.6)),
            "motion": motions[idx % len(motions)],
            "caption_pos": "center",
            "caption_style": "viral",
            "role": role,
            "fact_source": source,
        })

    return slides, script


# ---------------------------------------------------------------------------
# Versus Duel: narration + slide assembly
# ---------------------------------------------------------------------------

# Spoken forms of stat units -- "23h" must be narrated as "23 hours".
_UNIT_SPEECH = {
    "$": "dollars", "h": "hours", "hp": "horsepower", "s": "seconds",
    "%": "percent", "": "", "★": "stars", "mi": "miles", "kg": "kilograms",
}


def speak_stat(value: float, unit: str) -> str:
    """Renders a score the way a narrator should say it."""
    body = f"{value:,.0f}" if float(value).is_integer() else f"{value:,.1f}"
    word = _UNIT_SPEECH.get(unit, unit)
    return f"{body} {word}".strip()


def duel_tally(rounds: Sequence[dict[str, Any]]) -> tuple[int, int]:
    """Counts rounds won by A and by B."""
    a = sum(1 for r in rounds if str(r.get("winner", "")).upper() == "A")
    b = sum(1 for r in rounds if str(r.get("winner", "")).upper() == "B")
    return a, b


def build_duel_slides(duel: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Assembles a full duel reel: intro hook, one slide per metric round, then the
    winner reveal -- each with narration written to read smoothly aloud.

    Rounds carry their own pacing overrides so the winner reveal always lands
    after the narrator has finished the round.
    """
    item_a, item_b = duel["a"], duel["b"]
    rounds = [r for r in duel["rounds"] if str(r.get("metric", "")).strip()]
    name_a, name_b = item_a.get("name", "Item A"), item_b.get("name", "Item B")
    layout = duel.get("layout", "stacked")

    slides: list[dict[str, Any]] = [{
        "kind": "duel_intro",
        "item_a": item_a,
        "item_b": item_b,
        "layout": layout,
        "caption": duel.get("headline") or f"{name_a} vs {name_b}",
        # Narration is deliberately terse: spoken numbers eat runtime fast, and
        # duels only monetize if the whole reel stays inside a short's window.
        "voiceover": f"{name_a} versus {name_b}. Who wins?",
        "duration": 4.0,
        "lead_in": 0.30, "tail_out": 0.80, "min_duration": 3.2,
        "role": "duel_intro",
    }]

    for i, rnd in enumerate(rounds, start=1):
        unit = str(rnd.get("unit", "") or "")
        a_val = float(rnd.get("a_score", 0) or 0)
        b_val = float(rnd.get("b_score", 0) or 0)
        winner = str(rnd.get("winner", "")).upper()
        metric = str(rnd.get("metric", f"Round {i}"))

        if winner == "A":
            verdict = f" {name_a} wins."
        elif winner == "B":
            verdict = f" {name_b} wins."
        else:
            verdict = " Dead even."

        # The second score drops the unit -- the narrator already established it,
        # and every repeated word costs a second of a very short runtime.
        slides.append({
            "kind": "duel_round",
            "item_a": item_a,
            "item_b": item_b,
            "round": rnd,
            "layout": layout,
            "caption": f"{metric}: {format_stat(a_val, unit)} vs {format_stat(b_val, unit)}",
            "voiceover": (
                f"{metric}. {name_a}, {speak_stat(a_val, unit)}. "
                f"{name_b}, {speak_stat(b_val, '')}.{verdict}"
            ),
            "duration": 5.5,
            # Tail room so the winner glow (at 72% of the round) reads after the line.
            "lead_in": 0.40, "tail_out": 0.90, "min_duration": 4.5,
            "role": "duel_round",
        })

    ta, tb = duel_tally(rounds)
    a_wins = ta >= tb
    winner_item = item_a if a_wins else item_b
    winner_name = name_a if a_wins else name_b
    cta = duel.get("cta") or "Which one would you pick?"

    if ta == tb:
        verdict_line = f"Dead heat, {ta} all. You decide."
    else:
        verdict_line = f"{winner_name} takes it, {max(ta, tb)} to {min(ta, tb)}."

    slides.append({
        "kind": "duel_winner",
        "winner_item": winner_item,
        "winner_is_a": a_wins,
        "tally": (max(ta, tb), min(ta, tb)),
        "layout": layout,
        "caption": cta,
        "voiceover": f"{verdict_line} {cta} Comment below.",
        "duration": 4.5,
        "lead_in": 0.40, "tail_out": 1.00, "min_duration": 4.0,
        "role": "duel_winner",
    })

    return slides


@st.cache_data(show_spinner=False, max_entries=48)
def _cached_search(query: str) -> tuple[Any, str]:
    """Caches a stock-photo lookup so re-runs don't re-download on every keystroke."""
    return fetch_photo(query)


@st.cache_data(show_spinner=False, max_entries=24)
def _cached_photo_set(query: str, count: int) -> list[tuple[Any, str]]:
    """Caches a multi-photo lookup used to give each reel slide its own image."""
    return fetch_photo_set(query, count)


@st.cache_data(show_spinner=False, max_entries=48)
def _cached_url(url: str) -> Any:
    """Caches a direct-URL image fetch across Streamlit re-runs."""
    return load_image_from_url(url)


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_preset_photo(preset_name: str, side: str) -> tuple[Any, str]:
    """Resolves (and caches) a preset side's pinned photograph."""
    return resolve_item_photo(DUEL_PRESETS[preset_name][side])


# Every widget key in the duel studio whose value belongs to one matchup.
#
# These have to be cleared when a preset is loaded, and the reason is a
# Streamlit rule that is easy to forget: a widget whose `key` already exists in
# session state ignores its `value=` argument entirely and returns the stored
# value instead. Since the studio writes those return values straight back into
# the duel dict, loading "Luxury SUVs" over "Flagship Phones" produced a Range
# Rover photograph labelled "iPhone 15 Pro Max" -- and then re-fetched the
# photo from the stale query, which is where the residual phone images came
# from. Verified with AppTest before and after.
def reset_slides() -> None:
    """
    Empties the shared slide list.

    `st.session_state.slides` is shared by Reel Studio and Versus Duel, and it
    holds PIL images. Leaving the previous build in it is how one mode's
    pictures end up under another mode's captions.
    """
    st.session_state.slides = []


_DUEL_SIDE_KEYS = ("duel_name_{s}", "duel_hook_{s}", "duel_src_{s}",
                   "duel_q_{s}", "duel_url_{s}", "duel_img_{s}")
_DUEL_ROUND_KEYS = ("rm_{i}", "ru_{i}", "ra_{i}", "rb_{i}", "rw_{i}", "rn_{i}")
_DUEL_GLOBAL_KEYS = ("duel_layout", "duel_headline", "duel_cta", "duel_rounds_n")


def clear_duel_widget_state() -> None:
    """Drops every duel widget key so a freshly loaded preset is what shows."""
    stale = [key.format(s=side) for side in ("a", "b") for key in _DUEL_SIDE_KEYS]
    stale += [key.format(i=i) for i in range(8) for key in _DUEL_ROUND_KEYS]
    stale += list(_DUEL_GLOBAL_KEYS)
    for key in stale:
        st.session_state.pop(key, None)

    # Everything downstream of the old matchup goes too. `slides` is the one
    # that actually leaked: it is shared with Reel Studio, it holds the PIL
    # images the previous duel was built from, and nothing was clearing it -- so
    # loading Luxury SUVs and going straight to Preview or Export rendered the
    # *previous* pairing's photographs under the new names.
    st.session_state.pop("library_playing", None)
    st.session_state.pop("duel_built", None)
    clear_rendered_video("duel")
    reset_slides()

    # The free-text photo cache is keyed on the query string alone, so a
    # repeated search term would hand back the image fetched for the matchup
    # before this one. The preset cache is keyed on (preset, side) and is left
    # alone: it is correct by construction and expensive to refill.
    try:
        _cached_search.clear()
    except Exception:
        pass


def duel_from_preset(preset_name: str) -> dict[str, Any]:
    """Materializes a quick-fill preset, resolving each side's real photograph."""
    preset = copy.deepcopy(DUEL_PRESETS[preset_name])
    duel: dict[str, Any] = {"preset": preset_name, "layout": "stacked"}

    for side in ("a", "b"):
        item = dict(preset[side])
        try:
            item["image"], item["credit"] = _cached_preset_photo(preset_name, side)
        except Exception as exc:
            item["image"] = fallback_backdrop()
            item["credit"] = f"Photo unavailable ({type(exc).__name__})"
        duel[side] = item

    duel["rounds"] = [dict(r) for r in preset["rounds"]]
    duel["headline"] = f"{duel['a']['name']} vs {duel['b']['name']}"
    duel["cta"] = "Which one would you pick?"

    clear_duel_widget_state()
    return duel


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "slides" not in st.session_state:
    demo_samples = generate_sample_images()
    captions_map = {
        "01_Tokyo_Skyline.jpg": "Tokyo Nights & Cyber Glow 🗼✨",
        "02_Aesthetic_Coffee.jpg": "Morning Brew & Artisan Coffee ☕",
        "03_Swiss_Alps.jpg": "Alpine Escapes & Sunset Peaks 🏔️",
        "04_Luxury_Villa.jpg": "Modern Architecture & Dream Living 🏡",
    }
    st.session_state.slides = [{
        "kind": "image",
        "image": s["image"],
        "title": s["title"],
        "caption": captions_map.get(s["name"], s["title"]),
        "voiceover": "",
        "duration": 3.5,
        "motion": "zoom_in",
        "caption_pos": "center",
        "caption_style": "viral",
    } for s in demo_samples]

if "audio_settings" not in st.session_state:
    st.session_state.audio_settings = {
        "source": "Procedural AI Synth Music",
        "style": "lofi",
        "volume": 0.85,
        "custom_path": None,
    }

if "voice_settings" not in st.session_state:
    st.session_state.voice_settings = {
        "enabled": False,
        "voice": DEFAULT_VOICE,
        "rate_pct": 12,
        "pitch_hz": 0,
        "volume": 1.0,
        "music_duck": 0.28,
        "synced": False,
        "sfx_enabled": True,
        "sfx_volume": 0.55,
    }

if "commentary" not in st.session_state:
    st.session_state.commentary = {
        "source_path": None,     # the raw clip saved to disk
        "source_name": "",
        "source_origin": "",     # "upload" or the URL it was pulled from
        "model": "",
        "audio_path": None,      # synthesized narration
        "audio_bytes": None,
        "words": [],             # word boundaries for kinetic captions
        "ass_path": "",          # generated .ass subtitle file
        # Provenance -- what rights the finished video rests on.
        "licence": DEFAULT_LICENCE,
        "licence_reference": "",
        "source_author": "",
        "source_title": "",
        "source_url": "",
        "source_provider": "",
        "tts_provider": "gemini",
        "ai_disclosed": False,
        "voice": "en-US-ChristopherNeural",
    }

# The script lives in its own top-level key, and the Step-3 text area is bound
# to it BY KEY rather than by `value=`. A keyed widget reads its content from
# session state, so passing a fresh `value=` after generation is not guaranteed
# to win -- writing this entry is what makes new text appear in the box.
st.session_state.setdefault("commentary_script", "")

if "duel" not in st.session_state:
    st.session_state.duel = duel_from_preset(next(iter(DUEL_PRESETS)))

# Minimalist Motion keeps its selections here rather than in its widget keys.
# Streamlit discards the state of widgets it stops drawing, so a trip to
# another Production Mode would otherwise reset the whole panel; the widgets
# seed themselves from this dict on the way back in.
# Narrative Studio keeps its wizard state here for the same reason Minimalist
# Motion does: Streamlit drops the state of widgets it stops drawing, and an
# episode is far too expensive to lose to a mode switch.
# Atmosphere Studio keeps its own state for the same reason the other wizards
# do: Streamlit drops the state of widgets it stops drawing, and a multi-hour
# render is not something to lose to a mode switch.
# Niche Scout's research is expensive to regenerate -- three model calls and a
# YouTube quota spend -- so it survives a mode switch like the other wizards.
if "scout" not in st.session_state:
    st.session_state.scout = {
        "flow": "validate",
        "topic": "",
        "ideas": None,
        "assessment": None,
        "recon": None,
        "buckets": None,
        "handoff": "",
    }

if "atmosphere" not in st.session_state:
    st.session_state.atmosphere = {
        "bed": next(iter(ambient_engine.PRIMARY_BEDS)),
        "texture": "none",
        "bed_volume": 1.0,
        "texture_volume": 0.35,
        "duration_key": ambient_engine.DEFAULT_DURATION,
        "source_mode": "preset",
        "preset": ambient_engine.DEFAULT_CANVAS,
        "visual_source": "",
        "drift": "cycle",
        "grain": 6.0,
        "vignette": True,
        "result": None,
        "render_path": "",
        "meta": None,
        "uploaded": None,
    }

if "narrative" not in st.session_state:
    st.session_state.narrative = {
        "topic": "",
        "aesthetic": NARRATIVE_DEFAULT_AESTHETIC,
        "tone": NARRATIVE_DEFAULT_TONE,
        "format": NARRATIVE_DEFAULT_FORMAT,
        "voice_provider": "gemini",
        "voice": "Charon",
        "transition": "dissolve",
        "subtitles": True,
        "ambient": True,
        "image_providers": list(NARRATIVE_IMAGE_PROVIDERS),
        "ai_disclosed": False,
        "episode": None,     # script, cast and storyboard
        "result": None,      # what production returned
        "pack": "",          # the metadata pack
        "entry_name": "",
    }

if "minimal" not in st.session_state:
    st.session_state.minimal = {
        "preset": None,
        "concept": "",
        "template": AUTO_TEMPLATE,
        "duration": 18,
        "bgm": True,
        "bgm_volume": 0.30,
        "sfx": True,
        "narrate": False,
        "voice": "Charon",
        "ai_disclosed": False,
        "spec": None,        # the scene the last render used
        "publish": None,     # title / description / hashtags
        "result": None,      # runtime, frames, encoder
        "entry_name": "",
    }

# Rendered videos are tracked per scope ("reel" / "duel") as
# <scope>_video_path / _video_bytes / _video_name, so the two modes never
# overwrite each other's output and a player can't show a stale render.


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def render_auto_creator(aspect_name: str) -> None:
    """Topic -> full viral script in one click."""
    with st.container(border=True):
        st.markdown("#### 🤖 Topic → Reel in One Click")
        st.caption("A topic becomes a Hook → Value Points → CTA script with slides, subtitles and narration.")

        topic = st.text_input(
            "Your Topic",
            value="5 Mind-Blowing Facts About Space",
            placeholder="e.g. Luxury Habits, 5 Mind-Blowing Facts About Space",
        )

        domain = reel_engine.classify_topic(topic)
        spec = reel_engine.domain_spec(domain)
        st.markdown(
            badge(f"📚 {spec['label']}", "cyan")
            + badge(f"argues in {spec['metrics'][0]}", "")
            + (badge(f"{len(reel_engine.FACT_BANK.get(domain, ()))} banked facts", "green")
               if reel_engine.FACT_BANK.get(domain) else badge("no offline facts", "amber")),
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            num_points = st.slider("Fact beats", 2, 6,
                                   reel_engine.count_from_topic(topic, 3))
        with c2:
            auto_voice = st.selectbox(
                "Narrator Voice", list(VIRAL_VOICES.keys()),
                format_func=lambda v: VIRAL_VOICES[v],
                index=list(VIRAL_VOICES.keys()).index(st.session_state.voice_settings["voice"])
                if st.session_state.voice_settings["voice"] in VIRAL_VOICES else 0,
                key="auto_voice_select",
            )

        opt_a, opt_b, opt_c = st.columns(3)
        with opt_a:
            auto_narrate = st.checkbox(
                "Narrate + auto-sync durations", value=True,
                help="Synthesizes narration and stretches each slide to fit its spoken line.",
            )
        with opt_b:
            use_photos = st.checkbox(
                "Fetch real photos for slides", value=True,
                help="Sources real photography for the topic instead of abstract backdrops.",
            )
        with opt_c:
            use_ai = st.checkbox(
                "Research the facts with AI", value=True, key="auto_research",
                help="Asks Gemini for figures, named entities and mechanisms for this "
                     "exact topic. Off, or with no API key, the offline fact bank is "
                     "used instead — true, but general to the subject rather than "
                     "specific to your angle.",
            )

        if use_photos and not os.environ.get("PEXELS_API_KEY"):
            st.caption(
                "📷 Using Wikimedia Commons (no key needed). It is excellent for named "
                "products but returns encyclopedic diagrams for abstract topics — set "
                "`PEXELS_API_KEY` for curated lifestyle photography, or set each slide's "
                "image by URL/upload in Slide Studio."
            )

        if st.button("⚡ Generate Full Reel Script", type="primary", width="stretch"):
            if not topic.strip():
                st.error("Please enter a topic first.")
            else:
                # A new script invalidates any previously rendered reel.
                clear_rendered_video("reel")
                status = st.empty()
                try:
                    with st.spinner("Researching facts and sourcing photography..."):
                        slides, script = generate_viral_script(
                            topic, num_points,
                            size=ASPECT_RATIOS[aspect_name],
                            use_photos=use_photos,
                            use_ai=use_ai,
                            progress=lambda msg: status.caption(msg),
                        )
                        st.session_state.slides = slides
                        st.session_state["reel_script_meta"] = {
                            k: v for k, v in script.items() if k != "beats"
                        }
                        # Kept whole so the sources can be pasted into the
                        # video description, which is where a viewer can
                        # actually check them.
                        st.session_state["reel_sources"] = \
                            reel_engine.attribution_block(script)
                except Exception as exc:
                    st.error(f"**Script generation failed:** `{type(exc).__name__}: {exc}`")
                    st.code(traceback.format_exc())
                    st.stop()
                status.empty()
                st.session_state.voice_settings["voice"] = auto_voice
                st.session_state.voice_settings["synced"] = False

                if auto_narrate:
                    _synthesize(st.session_state.slides, auto_voice)
                else:
                    st.success(f"Generated {len(st.session_state.slides)} slides.")
                st.rerun()

    if any(s.get("role") for s in st.session_state.slides):
        with st.container(border=True):
            st.markdown("#### 📝 Generated Script")

            meta = st.session_state.get("reel_script_meta") or {}
            if meta:
                origin = str(meta.get("source") or "")
                st.markdown(
                    badge(f"researched by {origin}" if origin not in ("fact bank", "skeleton")
                          else f"source: {origin}",
                          "green" if origin not in ("fact bank", "skeleton") else "amber")
                    + badge(str(meta.get("domain_label") or ""), "cyan")
                    + badge(f"confidence: {meta.get('confidence', '?')}", ""),
                    unsafe_allow_html=True,
                )
                if meta.get("needs_editing"):
                    st.warning(
                        "These beats are true but **general to the subject**, not "
                        "researched for your exact angle. Check every figure and swap "
                        "in your own before publishing."
                        if meta.get("source") == "fact bank" else
                        "No facts were found for this topic — the beats are a skeleton "
                        "with the real figures still missing. Fill them in before "
                        "publishing.",
                        icon="✏️",
                    )

            icons = {"hook": ("🪝 HOOK", "violet"), "point": ("💎 FACT", "cyan"), "cta": ("📣 CTA", "amber")}
            for i, s in enumerate(st.session_state.slides):
                role = s.get("role")
                if role not in icons:
                    continue
                text, tone = icons[role]
                voiced = badge("🔊 voiced", "green") if s.get("voice_path") else badge("🔇 silent")
                st.markdown(
                    badge(text, tone) + badge(f"slide {i+1}") + badge(f"{s.get('duration', 0):.1f}s") + voiced,
                    unsafe_allow_html=True,
                )
                st.markdown(f"**On-screen:** {s.get('caption', '')}  \n*Spoken:* {s.get('voiceover', '')}")
                # An unsourced claim has to look unsourced. Rendering it the
                # same as a cited one is how a made-up statistic ends up in a
                # published video.
                if role == "point":
                    source = str(s.get("fact_source") or "").strip()
                    st.caption(f"📎 {source}" if source
                               else "⚠️ No source — verify this before publishing.")

            sources = str(st.session_state.get("reel_sources") or "").strip()
            if sources:
                with st.expander("📎 Fact sources — paste into the description"):
                    st.code(sources, language="text")

            render_viral_scorecard(
                " ".join(str(s.get("voiceover") or "") for s in st.session_state.slides),
                scope="reel",
            )


# A current desktop Chrome string.
#
# This is the whole fix for the TikTok failure, and it is worth being precise
# about why. With yt-dlp's own User-Agent, TikTok answers the webpage request
# with a challenge page instead of the post, and the extractor reports
# "Unexpected response from webpage request" -- which reads like a broken
# extractor and sends you to the yt-dlp issue tracker. Measured on
# tiktok.com/@tiktok/video/7106594312292453675: default UA fails, this UA
# succeeds, on the identical URL. It is not a version problem.
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

# Query parameters that only identify where a share link came from.
#
# Note what is NOT here: `v`, `list`, `t` and `start` are load-bearing.
# youtube.com/watch?v=ABC123 IS the video id, so stripping everything after
# "?" would turn a working YouTube link into the YouTube home page. Only known
# tracking keys are dropped; anything unrecognised is kept.
TRACKING_PARAMS: frozenset[str] = frozenset({
    "is_from_webapp", "sender_device", "sender_web_id", "web_id", "_r", "_t",
    "refer", "referer", "share_app_id", "share_item_id", "share_link_id",
    "shareid", "timestamp", "u_code", "tt_from", "source", "enter_from",
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "igshid", "igsh", "si", "feature", "app", "pp",
    "ref_src", "ref_url", "s", "spm_id_from",
})


def sanitize_media_url(url: str) -> str:
    """
    Strips share-tracking parameters from a pasted link.

    TikTok share links arrive as `.../video/12345?is_from_webapp=1&sender_device=pc`
    and the tail is noise. It is dropped here so the extractor, the ledger and
    the batch queue all see one canonical URL for the same post rather than
    three variants of it.

    Whitespace and surrounding angle brackets go too -- pasting from a chat
    client routinely brings both.
    """
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    cleaned = str(url or "").strip().strip("<>").strip()
    if not cleaned:
        return ""
    if not cleaned.lower().startswith(("http://", "https://")):
        cleaned = "https://" + cleaned.lstrip("/")

    try:
        parts = urlsplit(cleaned)
    except ValueError:
        return cleaned

    kept = [(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if key.lower() not in TRACKING_PARAMS]
    # The fragment is never meaningful for a video post and often carries
    # analytics of its own.
    return urlunsplit((parts.scheme, parts.netloc, parts.path,
                       urlencode(kept), ""))


def download_clip_from_url(url: str, dest_dir: str) -> dict[str, Any]:
    """
    Pulls a TikTok / Instagram Reel / YouTube Short down to a local MP4.

    Returns {"path", "title", "duration", "extractor"}. Raises RuntimeError with
    a readable reason so the UI can surface exactly why a link failed.
    """
    import yt_dlp
    import imageio_ffmpeg

    url = sanitize_media_url(url)
    os.makedirs(dest_dir, exist_ok=True)
    stem = os.path.join(dest_dir, f"source_{int(time.time())}")

    options: dict[str, Any] = {
        "outtmpl": f"{stem}.%(ext)s",
        # Prefer a ready-made MP4; fall back to muxing best video+audio.
        "format": "b[ext=mp4]/bv*[ext=mp4]+ba[ext=m4a]/bv*+ba/b",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        # Present as a desktop browser. Without this TikTok serves a challenge
        # page and the extractor fails with "Unexpected response from webpage
        # request" -- see the note on BROWSER_UA.
        "http_headers": {
            "User-Agent": BROWSER_UA,
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.tiktok.com/",
        },
        # A share link that redirects (vm.tiktok.com, youtu.be) needs to be
        # followed, and a single transient 5xx should not lose the clip.
        "retries": 3,
        "fragment_retries": 3,
        "extractor_retries": 2,
        "socket_timeout": 30,
        # yt-dlp needs ffmpeg to mux separate streams; use the bundled binary
        # rather than assuming one is on PATH.
        "ffmpeg_location": os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe()),
    }

    try:
        with yt_dlp.YoutubeDL(options) as ydl:  # type: ignore[arg-type]
            info = ydl.extract_info(url, download=True)
            produced = ydl.prepare_filename(info)
    except Exception as exc:
        raise RuntimeError(_explain_download_error(exc)) from exc

    # After a merge the real file may carry a different extension.
    candidates = [produced, f"{stem}.mp4", f"{stem}.webm", f"{stem}.mkv"]
    path = next((c for c in candidates if c and os.path.exists(c) and os.path.getsize(c) > 0), None)
    if path is None:
        found = glob.glob(f"{stem}.*")
        path = found[0] if found else None
    if path is None:
        raise RuntimeError("yt-dlp reported success but no video file was written.")

    return {
        "path": path,
        "title": str((info or {}).get("title") or os.path.basename(path)),
        "duration": float((info or {}).get("duration") or 0.0),
        "extractor": str((info or {}).get("extractor_key") or ""),
    }


def _explain_download_error(exc: Exception) -> str:
    """Turns a raw yt-dlp failure into something a creator can act on."""
    raw = str(exc)
    low = raw.lower()

    # A platform challenge. TikTok answers an unrecognised client with an
    # interstitial instead of the post, and yt-dlp surfaces that as a parse
    # failure pointing at its own issue tracker -- which sends people to file
    # bugs about a working extractor.
    if ("unexpected response from webpage" in low
            or "unable to extract webpage video data" in low
            or "captcha" in low
            or ("verify" in low and "human" in low)):
        return ("The platform served a verification page instead of the post.\n\n"
                "That usually means it is rate-limiting this network, or the post "
                "is region-locked. Try again in a few minutes, use a different "
                "link, or download it manually and use the file uploader.\n\n"
                + raw[:300])

    if ("429" in raw or "too many requests" in low or "rate limit" in low
            or "rate-limit" in low):
        return ("The platform is rate-limiting this network — too many downloads "
                "in a short window.\n\n"
                "Wait a few minutes before trying again. Pasting five links at "
                "once makes this more likely, not less.\n\n" + raw[:300])

    if "login" in low or "cookies" in low:
        return ("This post needs a logged-in session (Instagram does this for many Reels). "
                "Download it manually and use the file uploader instead.\n\n" + raw[:300])
    if "private" in low or "unavailable" in low or "removed" in low:
        return "That post is private, removed, or unavailable in this region.\n\n" + raw[:300]
    if "unsupported url" in low:
        return "That link isn't a recognised video URL. Paste the direct post link.\n\n" + raw[:300]
    if "ffmpeg" in low:
        return "The video and audio streams could not be merged (ffmpeg issue).\n\n" + raw[:300]
    return f"{type(exc).__name__}: {raw[:400]}"


def _probe_duration(video_path: str) -> float:
    """Reads a clip's duration without decoding it, for the loop-warning hint."""
    try:
        from moviepy import VideoFileClip

        clip = VideoFileClip(video_path)
        try:
            return float(clip.duration or 0.0)
        finally:
            clip.close()
    except Exception:
        return 0.0


def _reset_commentary_chain(cm: dict[str, Any]) -> None:
    """Clears everything downstream of the source clip when a new one arrives."""
    cm.update({"model": "", "audio_path": None, "audio_bytes": None,
               "words": [], "ass_path": ""})
    st.session_state["commentary_script"] = ""
    _clear_muted_asset()
    clear_rendered_video("commentary")


def _clear_muted_asset() -> None:
    """Drops the cached muted export -- it is cut to a specific narration length."""
    for key in ("commentary_muted_bytes", "commentary_muted_name"):
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# Multi-URL batch for the Commentary Machine
#
# Paste several links, get several finished shorts. Reuses the same stages the
# single-clip flow uses, so behaviour cannot drift between the two paths.
# ---------------------------------------------------------------------------

MAX_URL_BATCH = 5


def _url_job(url: str) -> dict[str, Any]:
    return {
        "url": url, "status": "queued", "stage": "", "error": "",
        "title": "", "video_name": "", "video_path": "", "duration": 0.0,
        "script": "", "ready": False, "blockers": [],
    }


def run_url_batch(urls: Sequence[str], fair_use: bool = False,
                  ai_disclosed: bool = False) -> None:
    """
    Downloads, scripts, voices and renders each URL in turn.

    A failure on one link is recorded and the queue continues -- losing four
    finished videos because the third link was private would be indefensible.

    `fair_use` records the uploader's assertion that their commentary makes the
    result transformative. It is stored as a dated claim, never as a licence.
    """
    cm = st.session_state.commentary
    vs = st.session_state.voice_settings
    jobs = [_url_job(u) for u in urls]
    st.session_state["url_batch"] = jobs

    target_size = ASPECT_RATIOS[st.session_state.get("render_aspect", next(iter(ASPECT_RATIOS)))]
    fps = int(st.session_state.get("render_fps", 24))
    fit_mode = str(st.session_state.get("render_fit", DEFAULT_FIT))
    watermark = str(st.session_state.get("render_watermark", ""))
    duration_target = str(st.session_state.get("cm_target") or DEFAULT_TARGET)
    provider = str(cm.get("tts_provider") or "gemini")
    voice = str(cm.get("voice") or ("Charon" if provider == "gemini" else DEFAULT_VOICE))
    angle_pref = str(cm.get("chosen_angle") or "suspense")

    tracker = StageProgress(label=f"Queued {len(jobs)} clip(s)...")
    line = st.empty()
    started = time.time()
    stages = ("download", "script", "voice", "render")

    for index, job in enumerate(jobs):
        job["status"] = "running"

        def mark(stage: str, _i: int = index) -> None:
            job["stage"] = stage
            done = stages.index(stage) / len(stages)
            tracker.update(min(1.0, (_i + done) / len(jobs)),
                           f"Clip {_i + 1}/{len(jobs)} — {stage}...")
            line.markdown(f"**Clip {_i + 1}/{len(jobs)} — {stage}...**")

        try:
            mark("download")
            got = download_clip_from_url(job["url"], user_exports())
            job["title"] = str(got["title"])[:70]

            mark("script")
            angles = generate_commentary_angles(got["path"], duration_target=duration_target)
            key = angle_pref if angle_pref in angles["angles"] else next(iter(angles["angles"]))
            script = str(angles["angles"][key])
            job["script"] = script

            mark("voice")
            narration = synthesize_narration(
                script, provider=provider, voice=voice,
                output_path=os.path.join(user_exports(), f"narration_{int(time.time())}"
                                         + (".wav" if provider == "gemini" else ".mp3")),
                rate=f"{int(st.session_state.get('cm_rate', 12)):+d}%",
                style=str(st.session_state.get("cm_gstyle") or "Punchy viral narrator, fast pace"),
            )
            speech = float(narration["duration"])

            mark("render")
            ass_path = None
            if narration["words"]:
                ass_path = write_ass_file(
                    list(narration["words"]),
                    os.path.join(user_exports(), f"captions_{int(time.time())}.ass"),
                    size=target_size,
                    position=str(st.session_state.get("cm_cappos") or "bottom"),
                    highlight=str(st.session_state.get("cm_kin_colour") or "yellow"),
                )

            bgm_path = None
            if bool(st.session_state.get("cm_bgm", True)):
                bgm_path = build_ducked_bgm(
                    speech + 0.4, str(narration["path"]),
                    os.path.join(user_exports(), f"bgm_{int(time.time())}.wav"),
                    volume=float(st.session_state.get("cm_bgmvol") or BGM_DEFAULT_VOLUME),
                )

            use_focus = bool(st.session_state.get("cm_focus", False))
            focus_spec = None
            if use_focus:
                focus_spec = {
                    "position": str(st.session_state.get("cm_focus_pos") or "center"),
                    "start": 0.0, "end": float(st.session_state.get("cm_focus_secs") or 2.0),
                    "pulse": True, "diameter_px": int(target_size[0] * 0.40),
                }

            out_path = new_export_path("commentary")
            result = render_commentary_video(
                source_video=got["path"], narration_path=str(narration["path"]),
                script=script, output_path=out_path, target_size=target_size, fps=fps,
                original_volume=float(st.session_state.get("cm_origvol") or 0.15),
                loop_mode=str(st.session_state.get("cm_loop") or "boomerang"),
                fit=fit_mode,
                burn_captions=False, ass_path=ass_path, bgm_path=bgm_path,
                focus=focus_spec,
                # The audio anchor is tied to the visual one: same hook, same frame.
                hook_sfx=use_focus,
                watermark_text=watermark,
            )

            entry = append_ledger(user_exports(), {
                "video_name": os.path.basename(out_path), "video_path": out_path,
                "duration": float(result["duration"]),
                # A link off a feed is someone else's work unless the uploader
                # asserts fair use, which is logged as a dated claim.
                "licence": "fair_use" if fair_use else DEFAULT_LICENCE,
                "fair_use_asserted_at": (time.strftime("%Y-%m-%d %H:%M:%S")
                                         if fair_use else ""),
                "licence_reference": "",
                "source_title": job["title"], "source_author": "",
                "source_url": job["url"], "source_provider": "",
                "tts_provider": str(narration.get("provider") or ""),
                "voice": str(narration.get("voice") or ""),
                "script_model": str(angles.get("model") or ""),
                "ai_disclosed": bool(ai_disclosed or cm.get("ai_disclosed")),
                "script": script, "batch_url": job["url"],
            })
            verdict = publish_readiness(entry)

            job.update({
                "status": "done", "stage": "",
                "licence": str(entry.get("licence") or ""),
                "video_name": entry["video_name"], "video_path": out_path,
                "duration": float(result["duration"]),
                "ready": bool(verdict["ready"]), "blockers": verdict["blockers"],
            })
        except Exception as exc:
            job.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})

        tracker.update((index + 1) / len(jobs), f"Clip {index + 1}/{len(jobs)} done.")

    sweep_scratch_files(force=True)
    ok = sum(1 for j in jobs if j["status"] == "done")
    line.markdown(f"**Finished — {ok}/{len(jobs)} rendered in "
                  f"{(time.time() - started) / 60:.1f} min.**")


def render_url_batch_results() -> None:
    """Queue table for the multi-URL run, with a download per finished short."""
    jobs = st.session_state.get("url_batch") or []
    if not jobs:
        return

    with st.container(border=True):
        st.markdown("#### 📦 Batch queue")
        ok = sum(1 for j in jobs if j["status"] == "done")
        ready = sum(1 for j in jobs if j.get("ready"))
        stat_row([
            ("Rendered", f"{ok}/{len(jobs)}", "cyan"),
            ("Publish-ready", str(ready), "green" if ready and ready == ok else "amber"),
            ("Failed", str(sum(1 for j in jobs if j["status"] == "failed")), ""),
        ])

        claimed = any(str(j.get("licence")) == "fair_use" for j in jobs)
        if ok and not ready and claimed:
            # The footage side passed; something else is holding them back.
            reasons = {b for j in jobs for b in (j.get("blockers") or [])}
            st.warning(
                "Fair use was recorded for these, but they are still not cleared:\n\n"
                + "\n".join(f"- {r}" for r in sorted(reasons)),
                icon="🚫",
            )
        elif ok and not ready:
            st.warning(
                "These came from feed links, so they are logged as unverified footage and "
                "are not cleared for a monetized upload. Tick the fair-use box before "
                "running, record the rights you hold, or source from the licensed library.",
                icon="🚫",
            )
        elif ready and claimed:
            st.info(
                "Cleared on your fair-use assertion, which is recorded with a timestamp in "
                "the ledger and printed in each publish pack. That is a claim you are making, "
                "not a licence — a claim can still be filed, so keep the commentary "
                "substantial and credit the source in your description.",
                icon="⚖️",
            )

        for index, job in enumerate(jobs):
            icon = {"done": "✅", "failed": "❌", "running": "⏳"}.get(str(job["status"]), "•")
            head = job["title"] or str(job["url"])[:52]
            with st.expander(
                f"{icon} {head}" + (f" · {job['duration']:.0f}s" if job["duration"] else ""),
                expanded=job["status"] == "failed",
            ):
                st.caption(str(job["url"])[:100])

                if job["status"] == "failed":
                    st.error(f"**Failed at '{job['stage'] or 'start'}':** {job['error']}")
                    continue
                if job["status"] != "done":
                    st.info(f"Status: {job['status']}")
                    continue

                st.markdown(
                    badge(f"{job['duration']:.0f}s", "cyan")
                    + (badge("publish-ready", "green") if job["ready"] else badge("not cleared", "amber")),
                    unsafe_allow_html=True,
                )
                for blocker in job.get("blockers", [])[:2]:
                    st.markdown(f"- 🚫 {blocker}")
                st.markdown(f"*{str(job['script'])[:200]}...*")

                path = str(job["video_path"])
                if os.path.exists(path):
                    prev, dl = st.columns([1, 1])
                    with prev:
                        st.video(path)
                    with dl:
                        with open(path, "rb") as handle:
                            st.download_button(
                                "📥 Download", data=handle.read(),
                                file_name=str(job["video_name"]), mime="video/mp4",
                                width="stretch", key=f"url_dl_{index}_{job['video_name']}",
                            )


def render_commentary_studio() -> None:
    """
    The AI Faceless Video Commentary Machine: a clean three-step flow from raw
    clip to narrated, captioned vertical short.

      1. Drop a raw viral clip.
      2. Gemini watches it and writes the commentary script.
      3. edge-tts speaks the (editable) script; export audio or a full video.
    """
    cm = st.session_state.commentary

    done_upload = bool(cm.get("source_path") and os.path.exists(str(cm.get("source_path"))))
    done_script = bool(str(st.session_state.get("commentary_script") or "").strip())
    done_audio = bool(cm.get("audio_path") and os.path.exists(str(cm.get("audio_path"))))

    st.markdown(
        badge("① Upload clip", "green" if done_upload else "violet")
        + badge("② Gemini script", "green" if done_script else ("violet" if done_upload else ""))
        + badge("③ Voice & export", "green" if done_audio else ("violet" if done_script else "")),
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------------- STEP 1
    with st.container(border=True):
        st.markdown("#### ① Drop your raw clip")
        st.caption("Footage you can monetize. The licensed library is the safe default; "
                   "uploads and links require you to state what rights you hold.")

        # --- licensed library: the monetizable default ----------------------
        with st.expander("🎬 Licensed library — cleared for commercial use", expanded=not done_upload):
            st.caption("Creative Commons and public-domain footage with the licence and "
                       "credit recorded automatically.")

            pex = pexels_key_status()
            if pex["ok"]:
                st.markdown(
                    badge("Pexels active — no attribution required", "green"),
                    unsafe_allow_html=True,
                )
            elif os.environ.get("PEXELS_API_KEY"):
                st.warning(
                    f"**Pexels key not working** - {pex['reason']}\n\n"
                    "Falling back to Wikimedia Commons, which is mostly CC BY-SA: "
                    "usable commercially, but it requires credit and makes your finished "
                    "video share-alike.",
                    icon="🔑",
                )
            else:
                st.caption("Set `PEXELS_API_KEY` in .env for footage that needs no attribution.")

            lib_q = st.text_input(
                "Search licensed footage", key="cm_lib_q",
                placeholder="e.g. excavator digging, ocean waves, city traffic night",
            )
            if st.button("🔎 Search library", width="stretch", key="cm_lib_search"):
                with st.spinner(f"Searching licensed sources for {lib_q}..."):
                    try:
                        st.session_state["cm_lib_hits"] = search_licensed_video(lib_q, limit=6)
                    except Exception as exc:
                        st.session_state["cm_lib_hits"] = []
                        st.error(f"**Library search failed:** `{type(exc).__name__}: {exc}`")

            hits = st.session_state.get("cm_lib_hits") or []
            if hits:
                for index, hit in enumerate(hits):
                    licence_key = normalise_licence(str(hit.get("licence_raw")))
                    licence = LICENCES[licence_key]
                    tone = "green" if licence.commercial else ""
                    with st.container(border=True):
                        st.markdown(
                            badge(hit["provider"], "violet")
                            + badge(licence.label, tone)
                            + (badge("credit required", "amber") if licence.attribution else ""),
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"**{hit['title'][:70]}**")
                        st.caption(f"by {hit['author'][:60]} · {hit['size_mb']} MB · {hit['mime']}")
                        if st.button("Use this clip", key=f"cm_lib_use_{index}", width="stretch"):
                            try:
                                with st.spinner("Downloading licensed clip..."):
                                    path = download_licensed_clip(hit, user_exports())
                                cm.update({
                                    "source_path": path,
                                    "source_name": hit["title"][:60],
                                    "source_origin": hit.get("page_url") or hit["url"],
                                    "licence": licence_key,
                                    "licence_reference": "",
                                    "source_author": hit["author"],
                                    "source_title": hit["title"],
                                    "source_url": hit.get("page_url") or hit["url"],
                                    "source_provider": hit["provider"],
                                })
                                _reset_commentary_chain(cm)
                                st.success(f"Ingested under {licence.label}.")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"**Could not fetch that clip:** `{type(exc).__name__}: {exc}`")
            elif st.session_state.get("cm_lib_hits") == []:
                st.info("No licensed clips matched. Try broader wording.", icon="🔍")

        st.markdown("---")
        st.caption("**Or bring your own** — you must hold the rights.")

        up_col, url_col = st.columns(2)

        with up_col:
            st.markdown("**Upload a file**")
            upload = st.file_uploader(
                "Raw video clip", type=["mp4", "mov", "webm", "m4v"], key="cm_upload",
                label_visibility="collapsed",
            )

            if upload is not None and upload.name != cm.get("source_name"):
                try:
                    os.makedirs(user_exports(), exist_ok=True)
                    ext = os.path.splitext(upload.name)[1] or ".mp4"
                    dest = os.path.join(user_exports(), f"source_{int(time.time())}{ext}")
                    with open(dest, "wb") as fh:
                        fh.write(upload.getbuffer())

                    cm.update({"source_path": dest, "source_name": upload.name,
                               "source_origin": "upload", "source_provider": "",
                               "licence": DEFAULT_LICENCE})
                    _reset_commentary_chain(cm)
                    st.rerun()
                except Exception as exc:
                    st.error(f"**Could not save the upload:** `{type(exc).__name__}: {exc}`")
                    st.code(traceback.format_exc())

        with url_col:
            st.markdown("**Or paste link(s)**")
            url_text = st.text_area(
                "Or paste TikTok / Instagram Reel URLs",
                key="cm_url", label_visibility="collapsed", height=110,
                placeholder=("https://www.tiktok.com/@user/video/…\n"
                             "one per line — up to 5"),
            )
            # Cleaned here rather than inside the downloader, so that the
            # canonical URL is what reaches session state, the batch queue and
            # the provenance ledger. Sanitising only at download time still
            # left `?is_from_webapp=1&sender_device=pc` in the publish pack,
            # and made one post pasted twice with different share tokens look
            # like two different sources.
            typed = [u for u in (sanitize_media_url(line) for line in url_text.splitlines()) if u]
            urls: list[str] = []
            for candidate in typed:
                if candidate not in urls:
                    urls.append(candidate)
            duplicates = len(typed) - len(urls)
            urls = urls[:MAX_URL_BATCH]
            extra = len(typed) - duplicates - len(urls)

            if duplicates > 0:
                st.caption(f"↔️ {duplicates} duplicate link{'s' if duplicates > 1 else ''} "
                           "ignored (same post, different share tracking).")
            if extra > 0:
                st.caption(f"⚠️ Only the first {MAX_URL_BATCH} links will be used ({extra} ignored).")

            fair_use = st.checkbox(
                "☑️ Transformative Commentary / Fair Use (Clear for publish)",
                value=False, key="cm_fair_use",
                help="Records your assertion that your commentary makes this transformative. "
                     "It is a claim you are making, not a licence — a copyright claim can "
                     "still be filed, and it is logged as an assertion in the ledger.",
            )

            # Fair use clears the footage. The AI label is a separate legal
            # requirement, and the publish check blocks on it independently --
            # so ask for it here, or the batch silently stays 'not cleared'.
            disclose = bool(cm.get("ai_disclosed"))
            if fair_use:
                disclose = st.checkbox(
                    "🤖 I will label these as AI-generated when I upload",
                    value=disclose, key="cm_batch_disclose",
                    help="Both YouTube and TikTok require synthetic media to be labelled. "
                         "This is the same declaration as the box in step 3.",
                )
                if not disclose:
                    st.caption(
                        "⚠️ Fair use clears the footage rights only. Without the AI "
                        "label these still come out **not cleared**."
                    )

            single = len(urls) <= 1
            label = "⚡ Download & Ingest" if single else f"⚡ Render {len(urls)} shorts"

            if st.button(label, width="stretch", key="cm_ingest", disabled=not urls):
                if single:
                    # Unchanged single-clip path: ingest, then the user drives
                    # steps 2 and 3 by hand.
                    try:
                        with st.spinner("Downloading clip with yt-dlp..."):
                            got = download_clip_from_url(urls[0], user_exports())

                        cm.update({
                            "source_path": got["path"],
                            "source_name": got["title"][:60] or os.path.basename(got["path"]),
                            "source_origin": urls[0],
                            "source_provider": "",
                            "source_url": urls[0],
                            # A downloaded feed post is someone else's work until
                            # the creator says otherwise.
                            "licence": DEFAULT_LICENCE,
                        })
                        _reset_commentary_chain(cm)
                        source = got["extractor"] or "source"
                        length = f" — {got['duration']:.0f}s" if got["duration"] else ""
                        st.success(f"Ingested from {source}{length}")
                        st.rerun()
                    except RuntimeError as exc:
                        st.error(f"**Download failed:**\n\n{exc}")
                    except Exception as exc:
                        st.error(f"**Unexpected download error:** `{type(exc).__name__}: {exc}`")
                        st.code(traceback.format_exc())
                else:
                    run_url_batch(urls, fair_use=bool(fair_use),
                                  ai_disclosed=bool(disclose))
                    st.rerun()

            st.caption("TikTok, YouTube Shorts and public Reels. Private posts need the uploader. "
                       f"Paste up to {MAX_URL_BATCH} links to render them back to back.")

        # --- rights declaration for anything not from the library -----------
        if done_upload and str(cm.get("source_provider") or "") == "":
            st.markdown("---")
            st.markdown("**What rights do you hold for this clip?**")
            st.caption("Downloading someone's post and republishing it is not cleared for "
                       "monetization, and it breaches most platforms' terms. Declare it honestly — "
                       "this is what the publish check reads.")

            keys = list(LICENCES.keys())
            current = str(cm.get("licence") or DEFAULT_LICENCE)
            chosen = st.selectbox(
                "Licence / rights", keys,
                index=keys.index(current) if current in keys else keys.index(DEFAULT_LICENCE),
                format_func=lambda k: LICENCES[k].label,
                key="cm_licence",
            )
            cm["licence"] = chosen
            st.caption(LICENCES[chosen].note)

            if LICENCES[chosen].needs_evidence:
                cm["licence_reference"] = st.text_input(
                    "Licence reference (order id, or where the permission message is saved)",
                    value=str(cm.get("licence_reference") or ""), key="cm_licref",
                )
            if LICENCES[chosen].attribution:
                cred_a, cred_b = st.columns(2)
                with cred_a:
                    cm["source_author"] = st.text_input(
                        "Creator to credit", value=str(cm.get("source_author") or ""), key="cm_author")
                with cred_b:
                    cm["source_url"] = st.text_input(
                        "Source URL", value=str(cm.get("source_url") or cm.get("source_origin") or ""),
                        key="cm_srcurl")

        if done_upload:
            st.markdown("---")
            info_col, prev_col = st.columns([2, 1])
            with info_col:
                size_mb = os.path.getsize(str(cm["source_path"])) / 1_048_576
                stat_row([
                    ("Clip", str(cm["source_name"])[:22], "cyan"),
                    ("Size", f"{size_mb:.1f} MB", ""),
                ])
                origin = str(cm.get("source_origin") or "")
                if origin and origin != "upload":
                    st.caption(f"🔗 {origin[:70]}")
            with prev_col:
                st.video(str(cm["source_path"]))
        else:
            st.info("Upload a clip or paste a link to begin.", icon="🎬")

    # ---------------------------------------------------------------- STEP 2
    with st.container(border=True):
        st.markdown("#### ② Gemini watches it and writes the script")
        st.caption("The clip is uploaded to Gemini, processed, then analysed frame-by-frame for a viral commentary.")

        target = pick(
            "Script duration target",
            list(DURATION_TARGETS.keys()), DEFAULT_TARGET, "cm_target",
            format_func=lambda k: str(DURATION_TARGETS[k]["label"]),
            help="Gemini is given a matching word budget so the narration lands inside this window.",
        )
        spec = DURATION_TARGETS[target]
        st.caption(
            f"Target ≈ {int(spec['low'] * 2.75)}–{int(spec['high'] * 2.75)} words "
            f"({spec['low']}–{spec['high']}s spoken)."
        )

        if st.button("🧠 Generate 3 Viral Angles", type="primary",
                     width="stretch", disabled=not done_upload):
            clear_rendered_video("commentary")
            cm["audio_path"] = None
            cm["audio_bytes"] = None
            cm["angles"] = {}

            status = st.empty()
            spin = StageProgress(label="Uploading to Gemini...")
            steps = {"n": 0}

            def note(msg: str) -> None:
                steps["n"] += 1
                spin.update(min(0.9, 0.15 * steps["n"]),
                            f"Stage 1/3 · Gemini grounding — {msg}")
                status.markdown(f"**Stage 1/3 · Gemini grounding — {msg}**")

            try:
                result = generate_commentary_angles(
                    str(cm["source_path"]), progress=note, duration_target=target,
                )
                cm["target"] = target
                cm["angles"] = result["angles"]
                cm["model"] = result["model"]

                # Pre-load the first angle so the box is never empty; the cards
                # below swap it with one tap.
                first = next((k for k in ANGLE_ORDER if k in result["angles"]), None)
                if first:
                    st.session_state["commentary_script"] = result["angles"][first]
                    cm["chosen_angle"] = first
                spin.progress(1.0)
                status.markdown(f"✅ **{len(result['angles'])} angles ready.**")
                st.rerun()
            except GeminiError as exc:
                spin.empty()
                status.empty()
                st.error(f"**Gemini failed:**\n\n{exc}")
            except Exception as exc:
                spin.empty()
                status.empty()
                st.error(f"**Unexpected error during analysis:** `{type(exc).__name__}: {exc}`")
                st.code(traceback.format_exc())

        if cm.get("model"):
            st.markdown(badge(f"model: {cm['model']}", "violet"), unsafe_allow_html=True)

        # --- angle picker: one tap loads a script into the editor ----------
        angles = cm.get("angles") or {}
        if angles:
            st.markdown("---")
            st.markdown("##### Pick your angle")
            st.caption("Same footage, three framings. Post one, or publish all three and let the feed decide.")

            chosen = str(cm.get("chosen_angle") or "")
            cards = st.columns(len(ANGLE_ORDER))
            for col, key in zip(cards, ANGLE_ORDER):
                script_text = str(angles.get(key) or "")
                if not script_text:
                    continue
                meta = SCRIPT_ANGLES[key]
                spoken = estimate_speech_seconds(script_text)
                with col, st.container(border=True):
                    tone = "green" if key == chosen else "violet"
                    st.markdown(
                        badge(meta["label"], tone)
                        + badge(f"{len(script_text.split())}w · {spoken:.0f}s"),
                        unsafe_allow_html=True,
                    )
                    st.caption(meta["blurb"])
                    preview = script_text if len(script_text) <= 165 else script_text[:162] + "..."
                    st.markdown(f"<div class='rf-angle'>{preview}</div>", unsafe_allow_html=True)
                    if st.button(
                        "✓ Selected" if key == chosen else "Use this angle",
                        key=f"cm_angle_{key}", width="stretch",
                        disabled=key == chosen,
                    ):
                        # Written before Step 3's text area is created this run,
                        # so the editor picks it up immediately.
                        st.session_state["commentary_script"] = script_text
                        cm["chosen_angle"] = key
                        cm["audio_path"] = None
                        cm["audio_bytes"] = None
                        _clear_muted_asset()
                        clear_rendered_video("commentary")
                        st.rerun()

    # ---------------------------------------------------------------- STEP 3
    with st.container(border=True):
        st.markdown("#### ③ Edit, voice and export")

        # Bound by key, not by `value=`: a keyed widget takes its contents from
        # session state, so the state entry is the single source of truth and
        # generated text shows up immediately.
        st.text_area(
            "Commentary script (edit freely before voicing)",
            height=190, key="commentary_script",
            placeholder="Generate a script above, or paste your own.",
        )

        script_now = str(st.session_state.get("commentary_script") or "").strip()
        if script_now:
            est = estimate_speech_seconds(script_now)
            tgt = DURATION_TARGETS.get(str(cm.get("target") or DEFAULT_TARGET), DURATION_TARGETS[DEFAULT_TARGET])
            lo, hi = float(tgt["low"]), float(tgt["high"])
            in_range = lo - 3 <= est <= hi + 3

            stat_row([
                ("Words", str(len(script_now.split())), ""),
                ("Est. spoken", f"{est:.0f}s", "cyan" if in_range else "amber"),
                ("Target", f"{tgt['low']}–{tgt['high']}s", ""),
            ])
            if est > hi + 3:
                st.caption(f"⏱️ Runs past the {tgt['low']}–{tgt['high']}s target — trim a sentence for retention.")
            elif est < lo - 3:
                st.caption(f"⏱️ Shorter than the {tgt['low']}–{tgt['high']}s target — add a beat of detail.")

        provider = pick(
            "Narration engine", list(SPEAKING_PROVIDERS),
            str(cm.get("tts_provider") or "gemini"), "cm_tts_provider",
            format_func=lambda k: str(TTS_PROVIDERS[k]["label"]).split(" (")[0]
                                  + (" ✅" if TTS_PROVIDERS[k]["commercial"] else " ⚠️ draft"),
        )
        cm["tts_provider"] = provider
        st.caption(str(TTS_PROVIDERS[provider]["note"]))
        if not TTS_PROVIDERS[provider]["commercial"]:
            st.warning(
                "edge-tts is not licensed for monetized publishing. Use it to draft and to "
                "check caption timing, then re-voice with Gemini before you upload.",
                icon="⚠️",
            )

        v1, v2 = st.columns([2, 1])
        if provider == "gemini":
            with v1:
                g_voices = list(GEMINI_VOICES.keys())
                cm["voice"] = st.selectbox(
                    "Narrator voice", g_voices,
                    format_func=lambda v: GEMINI_VOICES[v],
                    index=g_voices.index(cm["voice"]) if cm.get("voice") in g_voices else 0,
                    key="cm_gvoice",
                )
            with v2:
                # Gemini takes delivery notes in the prompt rather than a rate knob.
                style = st.selectbox(
                    "Delivery", ["Punchy viral narrator, fast pace", "Calm documentary narration",
                                 "High-energy hype", "Suspenseful and tense"],
                    key="cm_gstyle",
                )
            rate = 0
        else:
            with v1:
                voices = list(VIRAL_VOICES.keys())
                cm["voice"] = st.selectbox(
                    "Narrator voice", voices,
                    format_func=lambda v: VIRAL_VOICES[v],
                    index=voices.index(cm["voice"]) if cm.get("voice") in voices else 0,
                    key="cm_voice",
                )
            with v2:
                rate = st.slider("Speed", -20, 40, 12, 2, format="%+d%%", key="cm_rate")
            style = ""

        if st.button("🎙️ Synthesize Narration", type="primary",
                     width="stretch", disabled=not script_now):
            try:
                with st.spinner("Stage 2/3 · Synthesizing narration with edge-tts..."):
                    os.makedirs(user_exports(), exist_ok=True)
                    mp3 = os.path.join(user_exports(), f"narration_{int(time.time())}.mp3")
                    ext = ".wav" if provider == "gemini" else ".mp3"
                    mp3 = os.path.splitext(mp3)[0] + ext
                    spoken = synthesize_narration(
                        script_now, provider=provider, voice=str(cm["voice"]),
                        output_path=mp3, rate=f"{int(rate):+d}%", style=style,
                    )
                    cm["timings_exact"] = bool(spoken.get("timings_exact"))
                    with open(mp3, "rb") as fh:
                        cm["audio_bytes"] = fh.read()
                    cm["audio_path"] = mp3
                    cm["words"] = spoken["words"]

                    # Kinetic captions are built straight from the word
                    # boundaries, so they stay locked to the audio no matter
                    # how the clip is looped underneath.
                    ass_path = ""
                    if spoken["words"]:
                        ass_path = write_ass_file(
                            spoken["words"],
                            os.path.join(user_exports(), f"captions_{int(time.time())}.ass"),
                            size=ASPECT_RATIOS[st.session_state.get(
                                "render_aspect", next(iter(ASPECT_RATIOS)))],
                            position=str(st.session_state.get("cm_cappos", "bottom")),
                            highlight=str(st.session_state.get("cm_kin_colour", "yellow")),
                        )
                    cm["ass_path"] = ass_path
                # The muted export is cut to the previous narration's length.
                _clear_muted_asset()
                clear_rendered_video("commentary")
                st.success(f"Narration ready — {get_audio_duration(mp3):.1f}s")
                st.rerun()
            except Exception as exc:
                st.error(
                    f"**Voiceover failed:** `{type(exc).__name__}: {exc}`\n\n"
                    "edge-tts streams from Microsoft's servers, so this needs an internet connection."
                )
                st.code(traceback.format_exc())

        if done_audio and cm.get("audio_bytes"):
            # Gemini returns WAV, edge-tts returns MP3 -- label whichever it is.
            audio_ext = os.path.splitext(str(cm["audio_path"]))[1].lower() or ".mp3"
            audio_mime = "audio/wav" if audio_ext == ".wav" else "audio/mp3"
            st.audio(cm["audio_bytes"], format=audio_mime)

            st.markdown("##### 📥 Standalone assets for CapCut")
            st.caption("Drop these straight onto a timeline — the muted video is already cut to the narration length.")

            speech_len = get_audio_duration(str(cm["audio_path"]))
            stamp = os.path.splitext(os.path.basename(str(cm["audio_path"])))[0]

            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.download_button(
                    f"🔊 Download Voiceover ({audio_ext})", data=cm["audio_bytes"],
                    file_name=f"{stamp}{audio_ext}",
                    mime="audio/wav" if audio_ext == ".wav" else "audio/mpeg",
                    width="stretch", key=f"cm_dl_audio_{stamp}",
                )
            with d2:
                muted_bytes = st.session_state.get("commentary_muted_bytes")
                muted_name = st.session_state.get("commentary_muted_name", f"{stamp}_muted.mp4")
                if muted_bytes:
                    st.download_button(
                        "🎞️ Download Muted Video (.mp4)", data=muted_bytes,
                        file_name=str(muted_name), mime="video/mp4",
                        width="stretch", key=f"cm_dl_muted_{muted_name}",
                    )
                elif st.button("🎞️ Build Muted Video (.mp4)", width="stretch", key="cm_build_muted"):
                    mbar = StageProgress(label="Muxing...")
                    mstatus = st.empty()

                    def muted_progress(step: int, total: int, msg: str) -> None:
                        mbar.update(step / max(total, 1), str(msg))
                        mstatus.markdown(f"**{msg}**")

                    try:
                        mpath = os.path.join(user_exports(), f"muted_{int(time.time())}.mp4")
                        export_muted_video(
                            source_video=str(cm["source_path"]),
                            output_path=mpath,
                            duration=speech_len + 0.4,
                            script=script_now,
                            target_size=ASPECT_RATIOS[st.session_state.get(
                                "render_aspect", next(iter(ASPECT_RATIOS)))],
                            fps=int(st.session_state.get("render_fps", 30)),
                            burn_captions=False,
                            watermark_text="",
                            loop_mode=str(st.session_state.get("cm_loop", "boomerang")),
                            fit=str(st.session_state.get("render_fit", DEFAULT_FIT)),
                            progress_callback=muted_progress,
                        )
                        with open(mpath, "rb") as fh:
                            st.session_state["commentary_muted_bytes"] = fh.read()
                        st.session_state["commentary_muted_name"] = os.path.basename(mpath)
                        st.rerun()
                    except Exception as exc:
                        mbar.empty()
                        mstatus.empty()
                        st.error(f"**Muted export failed:** `{type(exc).__name__}: {exc}`")
                        st.code(traceback.format_exc())
            with d3:
                st.download_button(
                    "📝 Script (.txt)", data=script_now.encode("utf-8"),
                    file_name=f"{stamp}.txt", mime="text/plain",
                    width="stretch", key=f"cm_dl_txt_{stamp}",
                )
            with d4:
                ass_file = str(cm.get("ass_path") or "")
                if ass_file and os.path.exists(ass_file):
                    with open(ass_file, "rb") as fh:
                        ass_data = fh.read()
                    st.download_button(
                        "🎬 Captions (.ass)", data=ass_data,
                        file_name=f"{stamp}.ass", mime="text/plain",
                        width="stretch", key=f"cm_dl_ass_{stamp}",
                        help="Word-timed subtitles - import straight into CapCut or Premiere.",
                    )
                else:
                    st.button("🎬 Captions (.ass)", disabled=True, width="stretch",
                              key=f"cm_dl_ass_none_{stamp}",
                              help="Re-synthesize the narration to capture word timings.")

            st.markdown("##### 🎬 Render Quick Video")

            clip_len = _probe_duration(str(cm["source_path"]))
            if clip_len and speech_len > clip_len + 0.5:
                st.info(
                    f"Clip is {clip_len:.1f}s but the narration runs {speech_len:.1f}s — "
                    f"the footage will be looped ×{speech_len / clip_len:.1f} to cover it.",
                    icon="🔁",
                )

            r1, r2 = st.columns(2)
            with r1:
                loop_mode = pick(
                    "If the clip is shorter than the voiceover",
                    list(LOOP_MODES.keys()), "boomerang", "cm_loop",
                    format_func=lambda m: {
                        "boomerang": "🔁 Boomerang", "loop": "↩️ Hard loop", "hold": "⏸️ Freeze",
                    }[m],
                    help=" · ".join(LOOP_MODES.values()),
                )
            with r2:
                cap_pos = pick("Caption position", ["bottom", "center", "top"], "bottom", "cm_cappos",
                               format_func=lambda p: {"bottom": "Lower third",
                                                      "center": "Centre",
                                                      "top": "Upper third"}[p])

            has_words = bool(cm.get("words"))
            r3, r4 = st.columns(2)
            with r3:
                cap_style = pick(
                    "Caption style",
                    ["kinetic", "chunked", "none"],
                    "kinetic" if has_words else "chunked", "cm_capstyle",
                    format_func=lambda m: {
                        "kinetic": "⚡ Kinetic (word-by-word)",
                        "chunked": "Phrase blocks",
                        "none": "No captions",
                    }[m],
                    help="Kinetic uses the exact word timings edge-tts returned.",
                )
            with r4:
                kin_colour = pick(
                    "Highlight colour", ["yellow", "lime"], "yellow", "cm_kin_colour",
                    format_func=lambda c: {"yellow": "🟡 Yellow", "lime": "🟢 Lime"}[c],
                )

            if cap_style == "kinetic" and not has_words:
                st.warning(
                    "No word timings on this narration - re-synthesize it to enable "
                    "kinetic captions. Falling back to phrase blocks.",
                    icon="⚠️",
                )

            cm["ai_disclosed"] = st.checkbox(
                "I will label this as AI-generated when I upload",
                value=bool(cm.get("ai_disclosed")), key="cm_ai_disclosed",
                help="Both YouTube and TikTok require synthetic media to be labelled. "
                     "The publish pack below spells out where the toggle lives.",
            )

            st.markdown("**Hook anchor**")
            f1, f2, f3 = st.columns([1, 1, 1])
            with f1:
                use_focus = st.checkbox("⭕ Add Focus Circle on Hook", value=False, key="cm_focus")
            with f2:
                focus_pos = pick(
                    "Position", list(FOCUS_POSITIONS.keys()), "center", "cm_focus_pos",
                    format_func=lambda k: FOCUS_POSITIONS[k],
                )
            with f3:
                focus_secs = st.slider("Hold for (s)", 1.5, 3.0, 2.0, 0.1, key="cm_focus_secs",
                                       disabled=not use_focus)

            st.markdown("**Audio mix**")
            a1, a2, a3 = st.columns(3)
            with a1:
                orig_vol = st.slider("Original clip", 0.0, 0.5, 0.15, 0.01, key="cm_origvol",
                                     help="How much of the clip's own sound stays under the narration.")
            with a2:
                use_bgm = st.checkbox("🎵 Add Suspense BGM", value=True, key="cm_bgm")
            with a3:
                bgm_vol = st.slider("BGM level", 0.0, 0.40, BGM_DEFAULT_VOLUME, 0.01,
                                    key="cm_bgmvol", disabled=not use_bgm,
                                    help="Auto-ducked: the bed drops while the narrator speaks.")

            if st.button("🚀 Render Quick Video", type="primary", width="stretch"):
                clear_rendered_video("commentary")
                tracker = StageProgress(label="Rendering...")
                status = st.empty()

                def prog(step: int, total: int, msg: str) -> None:
                    tracker.step(step, total, msg)
                    status.markdown(f"**{msg}**")

                try:
                    out_path = new_export_path("commentary")
                    target = ASPECT_RATIOS[st.session_state.get(
                        "render_aspect", next(iter(ASPECT_RATIOS)))]

                    # Kinetic captions need word timings; rebuild the .ass here
                    # so a change of position or colour is picked up without
                    # re-running the text-to-speech.
                    use_kinetic = cap_style == "kinetic" and has_words
                    ass_for_render = None
                    if use_kinetic:
                        prog(1, 4, "Stage 1/4 - Building kinetic caption track...")
                        ass_for_render = write_ass_file(
                            list(cm["words"]),
                            os.path.join(user_exports(), f"captions_{int(time.time())}.ass"),
                            size=target,
                            position=str(cap_pos),
                            highlight=str(kin_colour),
                        )
                        cm["ass_path"] = ass_for_render

                    bgm_track = None
                    if use_bgm and bgm_vol > 0:
                        prog(1, 4, "Stage 1/4 - Synthesizing ducked suspense bed...")
                        bgm_track = build_ducked_bgm(
                            duration=speech_len + 0.4,
                            narration_path=str(cm["audio_path"]),
                            output_path=os.path.join(user_exports(), f"bgm_{int(time.time())}.wav"),
                            volume=float(bgm_vol),
                        )

                    focus_spec = None
                    if use_focus:
                        focus_spec = {
                            "position": str(focus_pos),
                            "start": 0.0,
                            "end": float(focus_secs),
                            "pulse": True,
                            # Scale the ring with the frame so it looks the
                            # same on any output size.
                            "diameter_px": int(target[0] * 0.40),
                        }

                    render_commentary_video(
                        source_video=str(cm["source_path"]),
                        narration_path=str(cm["audio_path"]),
                        script=script_now,
                        output_path=out_path,
                        target_size=target,
                        fps=int(st.session_state.get("render_fps", 30)),
                        original_volume=float(orig_vol),
                        caption_position=str(cap_pos),
                        burn_captions=(cap_style == "chunked" or (cap_style == "kinetic" and not has_words)),
                        loop_mode=str(loop_mode),
                        fit=str(st.session_state.get("render_fit", DEFAULT_FIT)),
                        bgm_path=bgm_track,
                        ass_path=ass_for_render,
                        focus=focus_spec,
                        hook_sfx=bool(use_focus),
                        watermark_text=str(st.session_state.get("render_watermark", "")),
                        progress_callback=prog,
                    )

                    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
                        raise FileNotFoundError(f"Renderer finished but {out_path} is missing or empty")

                    with open(out_path, "rb") as fh:
                        data = fh.read()
                    st.session_state["commentary_video_path"] = out_path
                    st.session_state["commentary_video_bytes"] = data
                    st.session_state["commentary_video_name"] = os.path.basename(out_path)

                    # Provenance is written at render time, while every fact
                    # about the render is still to hand.
                    append_ledger(user_exports(), {
                        "video_name": os.path.basename(out_path),
                        "video_path": out_path,
                        "duration": float(speech_len + 0.4),
                        "licence": str(cm.get("licence") or DEFAULT_LICENCE),
                        "licence_reference": str(cm.get("licence_reference") or ""),
                        "source_title": str(cm.get("source_title") or cm.get("source_name") or ""),
                        "source_author": str(cm.get("source_author") or ""),
                        "source_url": str(cm.get("source_url") or cm.get("source_origin") or ""),
                        "source_provider": str(cm.get("source_provider") or ""),
                        "tts_provider": str(cm.get("tts_provider") or "edge"),
                        "voice": str(cm.get("voice") or ""),
                        "script_model": str(cm.get("model") or ""),
                        "ai_disclosed": bool(cm.get("ai_disclosed")),
                        "script": script_now,
                    })

                    elapsed = tracker.finish("Render complete.")
                    status.markdown("✅ **Render complete.**")
                    sweep_scratch_files(force=True)
                    notify_complete("Commentary rendered",
                                    f"{os.path.basename(out_path)} in {elapsed:.0f}s")
                    st.balloons()
                except Exception as exc:
                    tracker.empty()
                    status.empty()
                    st.error(f"**Video render failed:** `{type(exc).__name__}: {exc}`")
                    st.code(traceback.format_exc())

    show_rendered_video("commentary", "Your Commentary Short", slot="commentary")
    render_publish_gate()
    render_url_batch_results()


def render_publish_gate() -> None:
    """
    The monetization checkpoint: is this render actually safe to upload?

    Reads the provenance the render wrote, then either hands over the publish
    pack or spells out exactly what is blocking it.
    """
    name = st.session_state.get("commentary_video_name")
    if not name:
        return

    entry = find_entry(user_exports(), str(name))
    if not entry:
        return

    # The scorecard sits above the publish gate deliberately: the gate answers
    # "may I upload this", the scorecard answers "is it worth uploading".
    def _apply(rewritten: str) -> None:
        st.session_state["commentary_script"] = rewritten

    render_viral_scorecard(str(entry.get("script") or ""), scope="commentary",
                           entry=entry, on_rewrite=_apply)

    verdict = publish_readiness(entry)

    with st.container(border=True):
        st.markdown("#### ✅ Publish check")

        if verdict["ready"]:
            st.success("Cleared for a monetized upload. Grab the publish pack below.", icon="✅")
        else:
            st.error(
                f"Not cleared yet — {len(verdict['blockers'])} thing"
                f"{'s' if len(verdict['blockers']) != 1 else ''} to fix.",
                icon="🚫",
            )
            for item in verdict["blockers"]:
                st.markdown(f"- 🚫 {item}")

        if verdict["passed"]:
            with st.expander(f"What passed ({len(verdict['passed'])})", expanded=verdict["ready"]):
                for item in verdict["passed"]:
                    st.markdown(f"- ✅ {item}")

        if verdict["warnings"]:
            with st.expander(f"Worth knowing ({len(verdict['warnings'])})"):
                for item in verdict["warnings"]:
                    st.markdown(f"- ⚠️ {item}")

        credit = attribution_line(entry)
        if credit:
            st.markdown("**Credit line — this must appear in your description:**")
            st.code(credit, language=None)

        pack = build_publish_pack(entry, str(entry.get("script") or ""))
        st.download_button(
            "📋 Download publish pack (.txt)", data=pack.encode("utf-8"),
            file_name=f"{os.path.splitext(str(name))[0]}_publish.txt",
            mime="text/plain", width="stretch", key=f"cm_pack_{name}",
            help="Description, credits, AI-disclosure wording, the per-platform label steps, "
                 "and the full licence trail.",
        )

        with st.expander("Upload steps and disclosure wording"):
            st.markdown(f"**Disclosure line:** {AI_DISCLOSURE_LINE}")
            for platform, step in PLATFORM_DISCLOSURE_STEPS.items():
                st.markdown(f"- **{platform}** — {step}")
            st.markdown("---")
            for platform, note in MONETIZATION_NOTES.items():
                st.markdown(f"- **{platform}** — {note}")
            st.caption("Platform rules change; verify before relying on these.")

        stats = summarise_ledger(user_exports())
        stat_row([
            ("Renders logged", str(stats["total"]), ""),
            ("Publish-ready", str(stats["ready"]), "cyan"),
            ("Blocked", str(stats["blocked"]), "amber" if stats["blocked"] else ""),
        ])


def render_duel_studio() -> None:
    """Dual-item setup, metric rounds, and one-click duel assembly."""
    duel = st.session_state.duel

    with st.container(border=True):
        st.markdown("#### ⚡ Quick-Fill Presets")
        st.caption("Start from a ready-made matchup, then edit any field.")
        cols = st.columns(len(DUEL_PRESETS))
        for col, preset_name in zip(cols, DUEL_PRESETS.keys()):
            with col:
                if st.button(preset_name, key=f"preset_{preset_name}", width="stretch"):
                    st.session_state.duel = duel_from_preset(preset_name)
                    st.rerun()

    c_a, c_b = st.columns(2)
    for col, side, tone, accent in ((c_a, "a", "cyan", "Cyan"), (c_b, "b", "amber", "Amber")):
        with col, st.container(border=True):
            st.markdown(badge(f"Item {side.upper()} · {accent}", tone), unsafe_allow_html=True)
            item = duel[side]

            if isinstance(item.get("image"), Image.Image):
                st.image(item["image"], width="stretch")
                credit = str(item.get("credit") or "")
                if credit:
                    st.caption(f"📷 {credit}")

            item["name"] = st.text_input("Name", value=item.get("name", ""), key=f"duel_name_{side}")
            item["hook"] = st.text_input("Subtitle hook", value=item.get("hook", ""), key=f"duel_hook_{side}")

            src_mode = pick(
                "Image source", ["search", "url", "upload"], "search", f"duel_src_{side}",
                format_func=lambda m: {"search": "🔎 Search", "url": "🔗 URL", "upload": "⬆️ Upload"}[m],
            )

            if src_mode == "search":
                q = st.text_input(
                    "Search real photos",
                    value=str(item.get("query") or item.get("name") or ""),
                    key=f"duel_q_{side}",
                    placeholder="e.g. Porsche Cayenne, Rolex Submariner",
                    help="Searches Pexels (needs PEXELS_API_KEY) then Wikimedia Commons.",
                )
                if st.button("Fetch photo", key=f"duel_fetch_{side}", width="stretch"):
                    with st.spinner(f"Finding a photo of {q}..."):
                        try:
                            img, credit = _cached_search(q)
                            item["image"], item["credit"], item["query"] = img, credit, q
                            st.rerun()
                        except PhotoLookupError as exc:
                            st.error(str(exc))

            elif src_mode == "url":
                url = st.text_input(
                    "Direct image URL", value="", key=f"duel_url_{side}",
                    placeholder="https://…/photo.jpg",
                )
                if st.button("Load URL", key=f"duel_loadurl_{side}", width="stretch"):
                    if not url.strip():
                        st.error("Paste an image URL first.")
                    else:
                        try:
                            item["image"] = _cached_url(url.strip())
                            item["credit"] = "Custom URL"
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Could not load that URL: {exc}")

            else:
                up = st.file_uploader(
                    "Upload photo", type=["jpg", "jpeg", "png", "webp"], key=f"duel_img_{side}",
                )
                if up is not None:
                    item["image"] = Image.open(up).convert("RGB")
                    item["credit"] = f"Uploaded — {up.name}"

    with st.container(border=True):
        st.markdown("#### 🥊 Metric Rounds")
        st.caption("Three to five head-to-head categories. Each round animates its scores, then reveals a winner.")

        n_rounds = st.slider("Number of rounds", 3, 5, min(5, max(3, len(duel["rounds"]))),
                             key="duel_rounds_n")
        rounds: list[dict[str, Any]] = []
        units = ["", "$", "h", "hp", "s", "%", "★", "mi", "kg"]

        for i in range(n_rounds):
            existing = duel["rounds"][i] if i < len(duel["rounds"]) else {
                "metric": f"Round {i + 1}", "a_score": 0.0, "b_score": 0.0,
                "unit": "", "winner": "A", "note": "",
            }
            with st.expander(
                f"Round {i + 1} — {existing.get('metric', '')}", expanded=(i < 3)
            ):
                r1, r2 = st.columns([3, 1])
                metric = r1.text_input("Category", value=str(existing.get("metric", "")), key=f"rm_{i}")
                unit_val = str(existing.get("unit", ""))
                unit = r2.selectbox(
                    "Unit", units,
                    index=units.index(unit_val) if unit_val in units else 0,
                    key=f"ru_{i}",
                )

                s1, s2, s3 = st.columns([1, 1, 1.4])
                a_score = s1.number_input(
                    f"{duel['a'].get('name', 'A')} score",
                    value=float(existing.get("a_score", 0) or 0), step=1.0, key=f"ra_{i}",
                )
                b_score = s2.number_input(
                    f"{duel['b'].get('name', 'B')} score",
                    value=float(existing.get("b_score", 0) or 0), step=1.0, key=f"rb_{i}",
                )
                with s3:
                    win = pick(
                        "Round winner", ["A", "B", "Tie"],
                        str(existing.get("winner", "A")).upper() if str(existing.get("winner", "A")).upper() in ("A", "B") else "Tie",
                        key=f"rw_{i}",
                        format_func=lambda w: {"A": "◀ A wins", "B": "B wins ▶", "Tie": "Tie"}[w],
                    )

                note = st.text_input(
                    "Note (shown under the category)", value=str(existing.get("note", "")), key=f"rn_{i}",
                )

            rounds.append({
                "metric": metric, "unit": unit, "a_score": a_score, "b_score": b_score,
                "winner": "" if win == "Tie" else win, "note": note,
            })

        duel["rounds"] = rounds

        ta, tb = duel_tally(rounds)
        stat_row([
            (duel["a"].get("name", "A"), f"{ta} won", "cyan"),
            (duel["b"].get("name", "B"), f"{tb} won", "amber"),
            ("Rounds", str(len(rounds)), ""),
        ])

    with st.container(border=True):
        st.markdown("#### 🎬 Build The Duel")
        b1, b2 = st.columns(2)
        with b1:
            duel["layout"] = pick(
                "Split layout", ["stacked", "side_by_side"], duel.get("layout", "stacked"), "duel_layout",
                format_func=lambda l: {"stacked": "▤ Top vs Bottom", "side_by_side": "▥ Side by Side"}[l],
            )
        with b2:
            duel_voice = st.selectbox(
                "Narrator Voice", list(VIRAL_VOICES.keys()),
                format_func=lambda v: VIRAL_VOICES[v],
                index=list(VIRAL_VOICES.keys()).index(st.session_state.voice_settings["voice"])
                if st.session_state.voice_settings["voice"] in VIRAL_VOICES else 0,
                key="duel_voice_select",
            )

        duel["headline"] = st.text_input("Opening hook", value=duel.get("headline", ""),
                                         key="duel_headline")
        duel["cta"] = st.text_input("Closing call to action", value=duel.get("cta", ""),
                                    key="duel_cta")

        opt1, opt2 = st.columns(2)
        with opt1:
            narrate = st.checkbox("Narrate + auto-sync durations", value=True, key="duel_narrate")
        with opt2:
            auto_render = st.checkbox(
                "Render the video immediately", value=True, key="duel_autorender",
                help="Runs the full pipeline: narration → frames → SFX mix → MP4.",
            )

        if st.button("⚔️ Build Versus Duel Reel", type="primary", width="stretch"):
            if not [r for r in duel["rounds"] if str(r.get("metric", "")).strip()]:
                st.error("Add at least one round with a category name.")
            else:
                # A new build must never leave the previous render on screen.
                clear_rendered_video("duel")

                try:
                    with st.spinner("Assembling duel timeline..."):
                        st.session_state.slides = build_duel_slides(duel)
                except Exception as exc:
                    st.error(f"**Could not assemble the duel:** `{type(exc).__name__}: {exc}`")
                    st.code(traceback.format_exc())
                    st.stop()

                st.session_state.voice_settings["voice"] = duel_voice
                if narrate:
                    _synthesize(st.session_state.slides, duel_voice)

                if auto_render:
                    ok = run_render_pipeline(
                        st.session_state.slides, "duel",
                        st.session_state.get("render_aspect", next(iter(ASPECT_RATIOS))),
                        st.session_state.get("render_fit", "blur_pad"),
                        st.session_state.get("render_transition", "crossfade"),
                        float(st.session_state.get("render_transition_dur", 0.5)),
                        int(st.session_state.get("render_fps", 24)),
                        str(st.session_state.get("render_watermark", "@viral_reels")),
                    )
                    if ok:
                        st.balloons()
                else:
                    st.success(f"Duel built — {len(st.session_state.slides)} slides. Render it in Export.")
                    st.rerun()

    # The duel's script is the narration across its slides. Worth scoring for
    # the same reason as any other: the opening hook decides the scroll.
    spoken = " ".join(str(slide.get("voiceover") or "")
                      for slide in st.session_state.get("slides") or []
                      if str(slide.get("role") or "").startswith("duel"))
    if spoken.strip():
        entry = find_entry(user_exports(), str(st.session_state.get("duel_video_name") or ""))
        render_viral_scorecard(spoken, scope="duel", entry=entry or {})

    show_rendered_video("duel", "Your Versus Duel", slot="duelstudio")


def run_render_pipeline(
    slides: list[dict[str, Any]],
    scope: str,
    aspect_name: str,
    fit_mode: str,
    transition_type: str,
    transition_dur: float,
    fps: int,
    watermark_text: str,
) -> bool:
    """
    Runs the complete render for `slides` and stores the result in session state.

    Four visible stages: (1) audio synthesis, (2) visual rendering,
    (3) audio/SFX multiplexing, (4) final export. Every failure path surfaces
    the real exception in the UI rather than dying quietly.

    Returns True on success.
    """
    vs = st.session_state.voice_settings
    audio_cfg = st.session_state.audio_settings

    if not slides:
        st.error("Nothing to render — add at least one slide first.")
        return False

    tracker = StageProgress(label="Preparing...")
    status = st.empty()

    def stage(fraction: float, message: str) -> None:
        tracker.update(fraction, message)
        status.markdown(f"**{message}**")

    # ---- Stage 1: audio synthesis ----------------------------------------
    audio_p = audio_cfg.get("custom_path")
    total_runtime = sum(s.get("duration", 3.5) for s in slides)

    try:
        if audio_cfg["source"] == "Procedural AI Synth Music" and not audio_p:
            stage(0.04, "Stage 1/4 · Audio synthesis — generating music bed...")
            stereo, sr = generate_synth_music(style=audio_cfg["style"], duration=total_runtime + 2.0)
            music_path = os.path.join(user_exports(), f"music_{int(time.time())}.wav")
            os.makedirs(user_exports(), exist_ok=True)
            save_wav_to_file(stereo, sr, music_path)
            audio_p = music_path
    except Exception as exc:
        st.error(f"**Stage 1 — music synthesis failed:** `{type(exc).__name__}: {exc}`")
        st.code(traceback.format_exc())
        return False

    use_audio = audio_p if audio_cfg["source"] != "No Audio (Mute)" else None

    # Honour the voiceover toggle without discarding synthesized files.
    if vs["enabled"]:
        render_slides = slides
    else:
        render_slides = [{k: v for k, v in s.items() if k != "voice_path"} for s in slides]

    # ---- Stages 2-4: handled inside build_reel_video, reported via callback
    out_path = new_export_path(scope)

    def progress(step: int, total: int, message: str) -> None:
        # Reserve the first 8% for stage 1, the rest for the engine's stages.
        stage(0.08 + 0.92 * (step / max(total, 1)), message)

    try:
        result = build_reel_video(
            slides=render_slides,
            aspect_ratio_name=aspect_name,
            fit_mode=fit_mode,
            transition_type=transition_type,
            transition_dur=transition_dur,
            audio_path=use_audio,
            audio_volume=audio_cfg.get("volume", 0.8),
            watermark_text=watermark_text,
            fps=fps,
            output_path=out_path,
            progress_callback=progress,
            voiceover_volume=vs["volume"],
            music_duck=vs["music_duck"],
            sfx_enabled=vs.get("sfx_enabled", True),
            sfx_volume=vs.get("sfx_volume", 0.55),
        )
    except Exception as exc:
        st.error(
            f"**Render failed during stage 2-4:** `{type(exc).__name__}: {exc}`\n\n"
            "Common causes: FFmpeg missing from the environment, a corrupt source "
            "image, or a slide with zero duration."
        )
        st.code(traceback.format_exc())
        return False

    # ---- Read the bytes back for a cache-proof download --------------------
    try:
        if not os.path.exists(out_path):
            raise FileNotFoundError(f"Renderer reported success but {out_path} is missing")
        with open(out_path, "rb") as fh:
            data = fh.read()
        if not data:
            raise ValueError("Rendered file is empty (0 bytes)")
    except Exception as exc:
        st.error(f"**Stage 4 — could not read the exported file:** `{type(exc).__name__}: {exc}`")
        st.code(traceback.format_exc())
        return False

    st.session_state[f"{scope}_video_path"] = out_path
    st.session_state[f"{scope}_video_bytes"] = data
    st.session_state[f"{scope}_video_name"] = os.path.basename(out_path)

    elapsed = tracker.finish("Render complete.")
    status.markdown("✅ **Render complete.**")
    notify_complete("Render finished",
                    f"{os.path.basename(out_path)} — {result['duration']:.0f}s in {elapsed:.0f}s")
    st.success(
        f"Exported {os.path.basename(out_path)} — {result['duration']:.1f}s, "
        f"{result['num_slides']} slides, {result.get('num_voiceovers', 0)} voiceovers, "
        f"{len(data) / 1_048_576:.1f} MB"
    )
    return True


def _synthesize(slides: list[dict[str, Any]], voice: str) -> None:
    """Runs voiceover synthesis with a progress bar and reports the outcome."""
    vs = st.session_state.voice_settings
    tracker = StageProgress(label="Synthesizing narration...")
    status = st.empty()

    def prog(step: int, total: int, msg: str) -> None:
        tracker.step(step, total, f"Stage 1/4 · Audio synthesis — {msg}")
        status.markdown(f"**Stage 1/4 · Audio synthesis — {msg}**")

    try:
        summary = generate_slide_voiceovers(
            slides, voice=voice,
            rate=f"{int(vs['rate_pct']):+d}%",
            pitch=f"{int(vs['pitch_hz']):+d}Hz",
            progress_callback=prog,
        )
    except Exception as exc:
        st.error(
            f"**Stage 1 — voiceover synthesis failed:** `{type(exc).__name__}: {exc}`\n\n"
            "edge-tts streams from Microsoft's servers, so this needs an internet connection."
        )
        st.code(traceback.format_exc())
        return

    if summary["errors"]:
        st.warning("Some voiceovers failed:\n\n" + "\n\n".join(summary["errors"]))
    if summary["voiced_slides"]:
        vs["enabled"] = True
        vs["synced"] = True
        st.success(
            f"✅ {summary['voiced_slides']} voiceovers synthesized "
            f"({summary['total_spoken']:.1f}s spoken) — runtime synced to {summary['total_runtime']:.1f}s."
        )
    else:
        st.error("No voiceovers were produced. Check that slides have script text.")


def render_slide_studio() -> None:
    """Per-slide editing for image slides; duel slides are edited in their own tab."""
    with st.container(border=True):
        h1, h2 = st.columns([3, 1])
        with h1:
            st.markdown("#### Slide Sequence")
            st.caption("Captions, narration, motion and timing for every slide.")
        with h2:
            if st.button("🔄 Reset Demo Slides", width="stretch"):
                st.session_state.slides = [{
                    "kind": "image", "image": s["image"], "title": s["title"],
                    "caption": s["title"], "voiceover": "", "duration": 3.5,
                    "motion": "zoom_in", "caption_pos": "center", "caption_style": "viral",
                } for s in generate_sample_images()]
                st.rerun()

        uploaded = st.file_uploader(
            "Add images", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True,
        )
        if uploaded:
            for uf in uploaded:
                st.session_state.slides.append({
                    "kind": "image",
                    "image": Image.open(uf).convert("RGB"),
                    "title": uf.name,
                    "caption": f"Highlight: {os.path.splitext(uf.name)[0]} ✨",
                    "voiceover": "",
                    "duration": 4.0, "motion": "zoom_in",
                    "caption_pos": "center", "caption_style": "viral",
                })
            st.success(f"Added {len(uploaded)} image(s).")
            st.rerun()

    if not st.session_state.slides:
        st.info("No slides yet — use the AI Auto-Creator, the Duel Engine, or upload images.")
        return

    to_delete: list[int] = []
    for idx, slide in enumerate(st.session_state.slides):
        kind = slide.get("kind", "image")
        label = str({
            "duel_intro": "⚔️ Duel Intro", "duel_round": "🥊 Duel Round",
            "duel_winner": "🏆 Winner Reveal",
        }.get(kind) or slide.get("title") or "Slide")

        with st.expander(f"Slide {idx + 1} · {label} · {slide.get('duration', 0):.1f}s", expanded=(idx == 0)):
            if kind != "image":
                st.markdown(badge(label, "violet"), unsafe_allow_html=True)
                slide["voiceover"] = st.text_area(
                    "🎙️ Narration", value=slide.get("voiceover", ""), height=90, key=f"dvo_{idx}",
                )
                st.caption("Duel visuals are configured in the Versus Duel tab.")
                if st.button("🗑️ Delete", key=f"ddel_{idx}"):
                    to_delete.append(idx)
                continue

            cols = st.columns([1, 2])
            with cols[0]:
                img = slide.get("image")
                if isinstance(img, Image.Image):
                    st.image(img, width="stretch")
                elif img:
                    st.image(str(img), width="stretch")
                if slide.get("credit"):
                    st.caption(f"📷 {slide['credit']}")

                slide_src = pick(
                    "Image", ["search", "url", "upload"], "search", f"slide_src_{idx}",
                    format_func=lambda m: {"search": "🔎", "url": "🔗", "upload": "⬆️"}[m],
                )

                if slide_src == "search":
                    sq = st.text_input(
                        "Photo search", value=str(slide.get("query") or ""),
                        key=f"slide_q_{idx}", placeholder="e.g. Human Brain",
                    )
                    if st.button("Fetch", key=f"slide_fetch_{idx}", width="stretch"):
                        try:
                            slide["image"], slide["credit"] = _cached_search(sq)
                            slide["query"] = sq
                            st.rerun()
                        except PhotoLookupError as exc:
                            st.error(str(exc))
                elif slide_src == "url":
                    su = st.text_input("Image URL", value="", key=f"slide_url_{idx}",
                                       placeholder="https://…/photo.jpg")
                    if st.button("Load", key=f"slide_load_{idx}", width="stretch"):
                        try:
                            slide["image"] = _cached_url(su.strip())
                            slide["credit"] = "Custom URL"
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Could not load: {exc}")
                else:
                    upl = st.file_uploader(
                        "Upload", type=["jpg", "jpeg", "png", "webp"], key=f"slide_up_{idx}",
                    )
                    if upl is not None:
                        slide["image"] = Image.open(upl).convert("RGB")
                        slide["credit"] = f"Uploaded — {upl.name}"
            with cols[1]:
                slide["caption"] = st.text_input(
                    "On-screen subtitle", value=slide.get("caption", ""), key=f"cap_{idx}",
                )
                slide["voiceover"] = st.text_area(
                    "🎙️ Voiceover script", value=slide.get("voiceover", ""), height=70, key=f"vo_{idx}",
                    help="Spoken over this slide. Blank reuses the subtitle text.",
                )
                if slide.get("voice_path"):
                    st.markdown(
                        badge(f"🔊 {slide.get('voice_duration', 0):.1f}s spoken", "green")
                        + badge(f"slide {slide.get('duration', 0):.1f}s"),
                        unsafe_allow_html=True,
                    )

                s1, s2 = st.columns(2)
                with s1:
                    slide["duration"] = st.slider(
                        "Duration (s)", 1.0, 12.0, float(slide.get("duration", 3.5)), 0.5, key=f"dur_{idx}",
                    )
                with s2:
                    motions = ["zoom_in", "zoom_out", "pan_right", "pan_left", "breathe", "static"]
                    slide["motion"] = st.selectbox(
                        "Motion", motions,
                        index=motions.index(slide.get("motion", "zoom_in"))
                        if slide.get("motion") in motions else 0,
                        key=f"mot_{idx}",
                    )

                styles = ["viral", "glass", "neon", "dark", "minimal"]
                slide["caption_style"] = pick(
                    "Caption style", styles,
                    slide.get("caption_style", "viral") if slide.get("caption_style") in styles else "viral",
                    key=f"sty_{idx}",
                    format_func=lambda s: {
                        "viral": "🔥 Viral", "glass": "Glass", "neon": "Neon",
                        "dark": "Dark", "minimal": "Minimal",
                    }[s],
                )

                if st.button("🗑️ Delete", key=f"del_{idx}"):
                    to_delete.append(idx)

    if to_delete:
        for idx in sorted(to_delete, reverse=True):
            st.session_state.slides.pop(idx)
        st.rerun()


def render_audio_studio() -> None:
    """Voiceover synthesis plus the music bed."""
    vs = st.session_state.voice_settings

    with st.container(border=True):
        st.markdown("#### 🎙️ AI Voiceover")
        st.caption("Ultra-realistic neural narration via edge-tts — free, no API key.")

        v1, v2 = st.columns([2, 1])
        with v1:
            vs["voice"] = st.selectbox(
                "Narrator voice", list(VIRAL_VOICES.keys()),
                format_func=lambda v: VIRAL_VOICES[v],
                index=list(VIRAL_VOICES.keys()).index(vs["voice"]) if vs["voice"] in VIRAL_VOICES else 0,
                key="studio_voice_select",
            )
        with v2:
            vs["enabled"] = st.checkbox("Enable in render", value=vs["enabled"])

        c1, c2, c3 = st.columns(3)
        with c1:
            vs["rate_pct"] = st.slider("Speed", -30, 50, int(vs["rate_pct"]), 2, format="%+d%%")
        with c2:
            vs["pitch_hz"] = st.slider("Pitch", -30, 30, int(vs["pitch_hz"]), 2, format="%+dHz")
        with c3:
            vs["volume"] = st.slider("Narration vol", 0.3, 1.5, float(vs["volume"]), 0.05)

        vs["music_duck"] = st.slider(
            "Music ducking under narration", 0.0, 1.0, float(vs["music_duck"]), 0.02,
            help="How far the music drops while the narrator speaks. Lower = clearer speech.",
        )

        g1, g2 = st.columns(2)
        with g1:
            do_sync = st.button("🎬 Generate Voiceovers + Sync", type="primary", width="stretch")
        with g2:
            if st.button("🔇 Clear Voiceovers", width="stretch"):
                for s in st.session_state.slides:
                    for k in ("voice_path", "voice_duration", "voice_offset"):
                        s.pop(k, None)
                vs["enabled"] = False
                vs["synced"] = False
                st.rerun()

        if do_sync:
            if not st.session_state.slides:
                st.error("Add slides first.")
            else:
                _synthesize(st.session_state.slides, vs["voice"])
                first = next((s for s in st.session_state.slides if s.get("voice_path")), None)
                if first:
                    st.caption("Preview — first narration line:")
                    st.audio(first["voice_path"])

    with st.container(border=True):
        st.markdown("#### 💥 Kinetic Sound Design")
        st.caption("Synthesized risers, impacts and chimes cut to the on-screen animation.")

        s1, s2 = st.columns([1, 2])
        with s1:
            vs["sfx_enabled"] = st.checkbox("Enable SFX", value=vs.get("sfx_enabled", True))
        with s2:
            vs["sfx_volume"] = st.slider(
                "SFX level", 0.0, 1.2, float(vs.get("sfx_volume", 0.55)), 0.05,
                help="Risers land on each transition; impacts fire as stat cards appear; "
                     "a chime marks every round winner.",
            )

        if st.button("🔊 Preview SFX", width="stretch"):
            from audio_engine import SFX_WHOOSH, SFX_IMPACT, SFX_CHIME, get_sfx_path
            p1, p2, p3 = st.columns(3)
            for col, name, label in (
                (p1, SFX_WHOOSH, "Riser"), (p2, SFX_IMPACT, "Impact"), (p3, SFX_CHIME, "Chime"),
            ):
                with col:
                    st.caption(label)
                    st.audio(get_sfx_path(name))

    with st.container(border=True):
        st.markdown("#### 🎵 Background Music")
        options = ["Procedural AI Synth Music", "Upload Custom Audio File", "No Audio (Mute)"]
        current = st.session_state.audio_settings.get("source")
        audio_src = pick(
            "Source", options, current if current in options else options[0], "audio_src",
        )
        st.session_state.audio_settings["source"] = audio_src

        if audio_src == "Procedural AI Synth Music":
            m1, m2 = st.columns(2)
            with m1:
                styles = ["lofi", "ambient", "upbeat"]
                st.session_state.audio_settings["style"] = pick(
                    "Vibe", styles, st.session_state.audio_settings.get("style", "lofi"), "music_style",
                    format_func=lambda x: {
                        "lofi": "☕ Lofi Chill", "ambient": "🌌 Ambient", "upbeat": "⚡ Upbeat",
                    }[x],
                )
            with m2:
                st.session_state.audio_settings["volume"] = st.slider(
                    "Music volume", 0.1, 1.0, float(st.session_state.audio_settings["volume"]), 0.05,
                )

            if st.button("🎵 Generate & Preview Track", width="stretch"):
                with st.spinner("Synthesizing procedural track..."):
                    total = sum(s.get("duration", 3.5) for s in st.session_state.slides) + 2.0
                    stereo, sr = generate_synth_music(
                        style=st.session_state.audio_settings["style"], duration=min(total, 45.0),
                    )
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    save_wav_to_file(stereo, sr, tmp.name)
                    st.session_state.audio_settings["custom_path"] = tmp.name
                st.success("Track generated.")
                st.audio(tmp.name)

        elif audio_src == "Upload Custom Audio File":
            up = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"])
            if up:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1])
                tmp.write(up.read())
                tmp.close()
                st.session_state.audio_settings["custom_path"] = tmp.name
                st.success("Audio uploaded.")
                st.audio(tmp.name)
            st.session_state.audio_settings["volume"] = st.slider(
                "Music volume", 0.1, 1.0, float(st.session_state.audio_settings["volume"]), 0.05,
            )
        else:
            st.session_state.audio_settings["custom_path"] = None
            st.info("Exporting without a music bed.")


def render_preview(aspect_name: str, fps: int) -> None:
    """Project summary and slide gallery."""
    slides = st.session_state.slides
    total = sum(s.get("duration", 3.5) for s in slides)
    voiced = sum(1 for s in slides if s.get("voice_path"))
    w, h = ASPECT_RATIOS[aspect_name]

    stat_row([
        ("Slides", str(len(slides)), ""),
        ("Runtime", f"{total:.1f}s", "cyan"),
        ("Resolution", f"{w}×{h}", ""),
        ("FPS", str(fps), ""),
        ("Voiced", f"{voiced}/{len(slides)}", "amber"),
    ])

    with st.container(border=True):
        st.markdown("#### Slide Gallery")
        if not slides:
            st.info("No slides yet.")
            return
        cols = st.columns(min(len(slides), 4))
        for idx, slide in enumerate(slides):
            with cols[idx % len(cols)]:
                kind = slide.get("kind", "image")
                st.markdown(
                    badge(f"#{idx + 1}", "violet") + badge(f"{slide.get('duration', 0):.1f}s"),
                    unsafe_allow_html=True,
                )
                img = slide.get("image")
                if isinstance(img, Image.Image):
                    st.image(img, width="stretch")
                elif kind == "duel_winner":
                    win = slide.get("winner_item", {})
                    if isinstance(win.get("image"), Image.Image):
                        st.image(win["image"], width="stretch")
                elif kind in ("duel_intro", "duel_round"):
                    a_img = slide.get("item_a", {}).get("image")
                    if isinstance(a_img, Image.Image):
                        st.image(a_img, width="stretch")
                st.caption(slide.get("caption", ""))


def render_export(
    aspect_name: str, fit_mode: str, transition_type: str,
    transition_dur: float, fps: int, watermark_text: str,
) -> None:
    """Render controls, progress, and the finished vertical preview."""
    slides = st.session_state.slides
    vs = st.session_state.voice_settings
    n_voiced = sum(1 for s in slides if s.get("voice_path"))
    total = sum(s.get("duration", 3.5) for s in slides)

    with st.container(border=True):
        st.markdown("#### 🚀 Render")
        stat_row([
            ("Slides", str(len(slides)), ""),
            ("Runtime", f"{total:.1f}s", "cyan" if total <= 60 else "amber"),
            ("Voiceovers", str(n_voiced) if vs["enabled"] else "off", "amber"),
        ])
        if n_voiced and not vs["enabled"]:
            st.info("Voiceovers are synthesized but disabled — enable them in Audio Studio.", icon="🔇")
        if total > 60:
            st.warning(
                f"This reel runs {total:.0f}s. Shorts and Reels reward sub-60s — "
                "trim a round, shorten the narration, or raise voice speed in Audio Studio.",
                icon="⏱️",
            )

        scope = "duel" if any(str(s.get("kind", "")).startswith("duel") for s in slides) else "reel"

        if st.button("🚀 Generate Video Now", type="primary", width="stretch"):
            clear_rendered_video(scope)
            if run_render_pipeline(
                slides, scope, aspect_name, fit_mode, transition_type,
                transition_dur, fps, watermark_text,
            ):
                st.balloons()

    show_rendered_video("reel", "Your Reel", slot="export")
    show_rendered_video("duel", "Your Versus Duel", slot="export")


# ---------------------------------------------------------------------------
# Batch mode
#
# One topic per line in, finished vertical videos out. Each job runs the same
# pipeline the single-clip studio does -- licensed footage, Gemini script,
# licensed narration, kinetic captions, ducked bed, provenance -- and a failure
# in one job never stops the queue.
# ---------------------------------------------------------------------------

BATCH_STAGES = ("footage", "script", "voice", "render")


def _batch_job(topic: str) -> dict[str, Any]:
    """A fresh queue entry."""
    return {
        "topic": topic.strip(),
        "status": "queued",       # queued | running | done | failed | skipped
        "stage": "",
        "error": "",
        "video_name": "",
        "video_path": "",
        "duration": 0.0,
        "licence": "",
        "credit": "",
        "script": "",
        "ready": False,
    }


def run_batch_job(
    job: dict[str, Any],
    settings: dict[str, Any],
    on_stage: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """
    Takes one topic all the way to a rendered, ledgered video.

    Raises nothing: any failure is recorded on the job so the queue can carry
    on. A batch that dies on job three of twenty is worse than useless.
    """
    def stage(name: str) -> None:
        job["stage"] = name
        if on_stage:
            on_stage(name)

    topic = str(job["topic"])
    target = ASPECT_RATIOS[str(settings.get("aspect") or next(iter(ASPECT_RATIOS)))]

    # Resolved once, from settings when the caller supplied it. Batch Studio
    # does not, and gets the signed-in user's folder exactly as before; the
    # one-click flow does, because it runs on a worker thread where
    # st.session_state -- and therefore user_exports() -- is unavailable and
    # would quietly resolve to the wrong account's directory.
    exports = str(settings.get("exports") or "") or user_exports()
    os.makedirs(exports, exist_ok=True)

    # ---- 1. licensed footage ------------------------------------------
    stage("footage")
    hits = search_licensed_video(topic, limit=4)
    if not hits:
        raise RuntimeError(f"No licensed footage found for '{topic}'. Try broader wording.")

    hit = hits[0]
    licence_key = normalise_licence(str(hit.get("licence_raw")))
    if not LICENCES[licence_key].commercial:
        raise RuntimeError(f"Best match for '{topic}' is not cleared for commercial use.")

    clip_path = download_licensed_clip(hit, exports)
    job["licence"] = licence_key

    # ---- 2. script -----------------------------------------------------
    stage("script")
    angles = generate_commentary_angles(
        clip_path, duration_target=str(settings.get("target") or DEFAULT_TARGET),
    )
    preferred = str(settings.get("angle") or "suspense")
    key = preferred if preferred in angles["angles"] else next(iter(angles["angles"]))
    script = str(angles["angles"][key])
    job["script"] = script

    # ---- 3. narration ---------------------------------------------------
    stage("voice")
    narration = synthesize_narration(
        script,
        provider=str(settings.get("tts_provider") or "gemini"),
        voice=str(settings.get("voice") or "Charon"),
        output_path=os.path.join(exports, f"narration_{int(time.time())}.wav"),
        style=str(settings.get("style") or "Punchy viral narrator, fast pace"),
    )
    speech = float(narration["duration"])

    # ---- 4. render -------------------------------------------------------
    stage("render")
    ass_path = None
    if settings.get("kinetic", True) and narration["words"]:
        ass_path = write_ass_file(
            list(narration["words"]),
            os.path.join(exports, f"captions_{int(time.time())}.ass"),
            size=target, position=str(settings.get("caption_position") or "bottom"),
            highlight=str(settings.get("highlight") or "yellow"),
        )

    bgm_path = None
    if settings.get("bgm", True):
        bgm_path = build_ducked_bgm(
            speech + 0.4, narration["path"],
            os.path.join(exports, f"bgm_{int(time.time())}.wav"),
            volume=float(settings.get("bgm_volume") or BGM_DEFAULT_VOLUME),
        )

    out_path = os.path.join(exports, f"batch_{int(time.time())}.mp4")
    result = render_commentary_video(
        source_video=clip_path,
        narration_path=str(narration["path"]),
        script=script,
        output_path=out_path,
        target_size=target,
        fps=int(settings.get("fps") or 24),
        original_volume=float(settings.get("original_volume") or 0.15),
        loop_mode=str(settings.get("loop_mode") or "boomerang"),
        fit=str(settings.get("fit") or DEFAULT_FIT),
        burn_captions=False,
        ass_path=ass_path,
        bgm_path=bgm_path,
        watermark_text=str(settings.get("watermark") or ""),
    )

    # ---- 5. provenance ---------------------------------------------------
    entry = append_ledger(exports, {
        "video_name": os.path.basename(out_path),
        "video_path": out_path,
        "duration": float(result["duration"]),
        "licence": licence_key,
        "licence_reference": "",
        "source_title": str(hit.get("title") or ""),
        "source_author": str(hit.get("author") or ""),
        "source_url": str(hit.get("page_url") or hit.get("url") or ""),
        "source_provider": str(hit.get("provider") or ""),
        "tts_provider": str(narration.get("provider") or ""),
        "voice": str(narration.get("voice") or ""),
        "script_model": str(angles.get("model") or ""),
        "ai_disclosed": bool(settings.get("ai_disclosed")),
        "script": script,
        "batch_topic": topic,
    })

    verdict = publish_readiness(entry)
    job.update({
        "status": "done",
        "stage": "",
        "video_name": entry["video_name"],
        "video_path": out_path,
        "duration": float(result["duration"]),
        "credit": attribution_line(entry),
        "ready": bool(verdict["ready"]),
        "blockers": verdict["blockers"],
    })
    return job


def render_batch_studio() -> None:
    """Queue several topics and render them one after another."""
    queue: list[dict[str, Any]] = st.session_state.setdefault("batch_queue", [])

    with st.container(border=True):
        st.markdown("#### ① Queue your topics")
        st.caption("One per line. Each becomes a search against the licensed library, "
                   "then a full render.")

        raw = st.text_area(
            "Topics", key="batch_topics", height=140,
            placeholder="ocean waves\ncity traffic at night\nexcavator digging\nvolcano eruption",
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            target = pick(
                "Length", list(DURATION_TARGETS.keys()), "rewards", "batch_target",
                format_func=lambda k: str(DURATION_TARGETS[k]["label"]).split(" · ")[0],
            )
        with c2:
            angle = pick(
                "Angle", list(ANGLE_ORDER), "suspense", "batch_angle",
                format_func=lambda k: str(SCRIPT_ANGLES[k]["label"]),
            )
        with c3:
            voice = st.selectbox(
                "Voice", list(GEMINI_VOICES.keys()),
                format_func=lambda v: GEMINI_VOICES[v].split(" — ")[0],
                key="batch_voice",
            )

        d1, d2, d3 = st.columns(3)
        with d1:
            kinetic = st.checkbox("Kinetic captions", value=True, key="batch_kinetic")
        with d2:
            bgm = st.checkbox("Suspense BGM", value=True, key="batch_bgm")
        with d3:
            disclosed = st.checkbox("Label as AI on upload", value=True, key="batch_disclosed")

        topics = [t.strip() for t in raw.splitlines() if t.strip()]
        est = len(topics) * (int(DURATION_TARGETS[target]["high"]) * 1.4 + 60)
        if topics:
            stat_row([
                ("Topics", str(len(topics)), "cyan"),
                ("Est. total", f"~{est / 60:.0f} min", "amber" if est > 1800 else ""),
                ("Per video", f"~{est / max(len(topics), 1) / 60:.1f} min", ""),
            ])

        b1, b2 = st.columns([3, 1])
        with b1:
            start = st.button("🚀 Run batch", type="primary", width="stretch",
                              disabled=not topics)
        with b2:
            if st.button("Clear queue", width="stretch"):
                st.session_state["batch_queue"] = []
                st.rerun()

    if start:
        st.session_state["batch_queue"] = [_batch_job(t) for t in topics]
        queue = st.session_state["batch_queue"]

        settings = {
            "aspect": st.session_state.get("render_aspect", next(iter(ASPECT_RATIOS))),
            "fit": st.session_state.get("render_fit", DEFAULT_FIT),
            "fps": st.session_state.get("render_fps", 24),
            "watermark": st.session_state.get("render_watermark", ""),
            "target": target, "angle": angle, "voice": voice,
            "tts_provider": "gemini", "kinetic": kinetic, "bgm": bgm,
            "ai_disclosed": disclosed,
            "caption_position": "bottom", "highlight": "yellow",
            "loop_mode": "boomerang", "original_volume": 0.15,
            "bgm_volume": BGM_DEFAULT_VOLUME,
            "style": "Punchy viral narrator, fast pace",
        }

        tracker = StageProgress(label=f"Queued {len(queue)} topic(s)...")
        line = st.empty()
        started = time.time()

        for index, job in enumerate(queue):
            job["status"] = "running"

            def on_stage(name: str, _i: int = index, _t: str = job["topic"]) -> None:
                done = BATCH_STAGES.index(name) / len(BATCH_STAGES)
                tracker.update(min(1.0, (_i + done) / len(queue)),
                               f"Job {_i + 1}/{len(queue)} · {_t} — {name}...")
                line.markdown(f"**Job {_i + 1}/{len(queue)} · {_t} — {name}...**")

            try:
                run_batch_job(job, settings, on_stage)
            except Exception as exc:
                # One bad topic must not take the rest of the queue with it.
                job.update({"status": "failed", "stage": "",
                            "error": f"{type(exc).__name__}: {exc}"})

            tracker.update((index + 1) / len(queue), f"Job {index + 1}/{len(queue)} done.")

        sweep_scratch_files(force=True)
        done = sum(1 for j in queue if j["status"] == "done")
        minutes = (time.time() - started) / 60
        tracker.finish(f"{done}/{len(queue)} rendered.")
        line.markdown(f"**Finished — {done}/{len(queue)} rendered in {minutes:.1f} min.**")
        if done:
            notify_complete("Batch finished",
                            f"{done} of {len(queue)} topics rendered in {minutes:.1f} min")
            st.balloons()

    if queue:
        with st.container(border=True):
            st.markdown("#### ② Results")
            ok = sum(1 for j in queue if j["status"] == "done")
            ready = sum(1 for j in queue if j.get("ready"))
            stat_row([
                ("Rendered", f"{ok}/{len(queue)}", "cyan"),
                ("Publish-ready", str(ready), "green" if ready == ok and ok else "amber"),
                ("Failed", str(sum(1 for j in queue if j["status"] == "failed")), ""),
            ])

            for index, job in enumerate(queue):
                icon = {"done": "✅", "failed": "❌", "running": "⏳"}.get(str(job["status"]), "•")
                with st.expander(
                    f"{icon} {job['topic']} "
                    f"{'· ' + str(job['duration']) [:4] + 's' if job['duration'] else ''}",
                    expanded=job["status"] == "failed",
                ):
                    if job["status"] == "failed":
                        st.error(f"**Failed at '{job['stage'] or 'start'}':** {job['error']}")
                        continue
                    if job["status"] != "done":
                        st.info(f"Status: {job['status']}")
                        continue

                    st.markdown(
                        badge(LICENCES[str(job["licence"])].label, "green")
                        + badge(f"{job['duration']:.0f}s")
                        + (badge("publish-ready", "green") if job["ready"] else badge("blocked", "amber")),
                        unsafe_allow_html=True,
                    )
                    if job.get("credit"):
                        st.caption(f"Credit required: {job['credit']}")
                    if not job["ready"]:
                        for blocker in job.get("blockers", []):
                            st.markdown(f"- 🚫 {blocker}")

                    st.markdown(f"*{str(job['script'])[:220]}...*")

                    # Each queued topic gets its own reading. A batch is where a
                    # weak template does the most damage, because it ships ten
                    # times before anyone reads one of them.
                    render_viral_scorecard(
                        str(job["script"]), scope=f"batch{index}",
                        entry={"licence": str(job["licence"]),
                               "duration": float(job["duration"] or 0.0),
                               "tts_provider": str(job.get("tts_provider") or "edge"),
                               "ai_disclosed": bool(job.get("ai_disclosed"))})

                    path = str(job["video_path"])
                    if os.path.exists(path):
                        prev, dl = st.columns([1, 1])
                        with prev:
                            st.video(path)
                        with dl:
                            with open(path, "rb") as handle:
                                st.download_button(
                                    "📥 Download", data=handle.read(),
                                    file_name=str(job["video_name"]), mime="video/mp4",
                                    width="stretch", key=f"batch_dl_{index}_{job['video_name']}",
                                )


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Minimalist Motion studio
#
# The only mode that needs no footage at all: the picture is drawn from code,
# so the render is the user's own work outright and clears the publish gate
# without a licence question anywhere in the flow.
# ---------------------------------------------------------------------------

MM_VOICES = ("Charon", "Kore", "Puck", "Fenrir", "Aoede")


def _mm() -> dict[str, Any]:
    """
    The mode's own state, kept outside the widget keys.

    Streamlit drops the state of widgets that stop being drawn, so switching to
    another Production Mode would wipe every selection here. Mirroring each
    choice into this dict and seeding the widgets from it is what makes the
    mode survive a round trip.
    """
    return st.session_state.minimal


def mm_publish_text(spec: dict[str, Any], publish: dict[str, Any]) -> str:
    """The publish pack for an animation: no licence trail, because there is none."""
    tags = " ".join(publish.get("hashtags") or [])
    lines = [
        "REELFORGE - MINIMALIST MOTION PUBLISH PACK",
        "=" * 52,
        "",
        "TITLE",
        str(publish.get("title") or ""),
        "",
        "DESCRIPTION",
        str(publish.get("description") or ""),
        "",
        tags,
        "",
        "-" * 52,
        "FOOTAGE",
        "Vector animation generated procedurally by ReelForge Studio.",
        "No stock footage, no third-party assets, no model-generated imagery.",
        "You hold the rights outright.",
        "",
        f"Template       : {TEMPLATES[str(spec.get('template'))]['label']}",
        f"Runtime        : {float(spec.get('duration') or 0):.1f}s",
        f"On-screen copy : {spec.get('title')} / {spec.get('subtitle')}",
        f"Closing line   : {spec.get('payoff')}",
    ]
    if spec.get("thesis"):
        lines += ["", "NARRATION SCRIPT", str(spec.get("thesis"))]
    lines += ["", "-" * 52, AI_DISCLOSURE_LINE, ""]
    for platform, step in PLATFORM_DISCLOSURE_STEPS.items():
        lines.append(f"{platform}: {step}")
    return "\n".join(lines)


def run_minimalist_render(spec: dict[str, Any]) -> bool:
    """Renders the animation and files it in the ledger. Returns success."""
    state = _mm()
    tracker = StageProgress(label="Drawing the scene...")
    status = st.empty()

    def progress(step: int, total: int, message: str) -> None:
        tracker.step(step, total, message)
        status.markdown(f"**{message}**")

    try:
        out_path = os.path.join(user_exports(), f"minimalist_{int(time.time())}.mp4")
        os.makedirs(user_exports(), exist_ok=True)

        narrate = bool(state.get("narrate"))
        result = build_minimalist_video(
            spec, out_path,
            fps=int(st.session_state.get("render_fps", 30)),
            bgm=bool(state.get("bgm", True)),
            bgm_volume=float(state.get("bgm_volume", 0.30)),
            sfx=bool(state.get("sfx", True)),
            narrate=narrate,
            voice=str(state.get("voice") or "Charon"),
            progress_callback=progress,
        )

        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise FileNotFoundError(f"Renderer finished but {out_path} is missing or empty")

        with open(out_path, "rb") as handle:
            data = handle.read()

        publish = dict(result["publish"])
        entry = append_ledger(user_exports(), {
            "video_name": os.path.basename(out_path), "video_path": out_path,
            "duration": float(result["duration"]),
            # Drawn from code in this process. Nothing was licensed because
            # nothing was borrowed.
            "licence": "own", "licence_reference": "",
            "source_title": "Procedural vector animation",
            "source_author": "", "source_url": "", "source_provider": "reelforge",
            "tts_provider": str(result["tts_provider"]),
            "voice": str(result.get("voice") or ""),
            "script_model": str(spec.get("source") or "preset"),
            "ai_disclosed": bool(state.get("ai_disclosed")),
            "script": str(spec.get("thesis") or ""),
            "template": str(spec.get("template")),
        })

        state.update({
            "spec": result["spec"], "publish": publish,
            "result": {k: result[k] for k in
                       ("duration", "fps", "frames", "template", "climax",
                        "narration", "narration_trimmed", "bgm", "sfx", "gpu")},
            "entry_name": entry["video_name"],
        })
        st.session_state["minimal_video_path"] = out_path
        st.session_state["minimal_video_bytes"] = data
        st.session_state["minimal_video_name"] = os.path.basename(out_path)

        # Beds, mixes and voice takes all went to the temp directory; this is
        # the one call that clears them.
        purge_scratch_renders()
        elapsed = tracker.finish("Animation complete.")
        status.markdown("**Done.**")
        notify_complete("Animation rendered",
                        f"{os.path.basename(out_path)} — {result['duration']:.0f}s "
                        f"in {elapsed:.0f}s")
        return True

    except Exception as exc:
        tracker.empty()
        status.empty()
        st.error(f"**Render failed:** `{type(exc).__name__}: {exc}`")
        st.code(traceback.format_exc())
        purge_scratch_renders()
        return False


def render_minimalist_studio() -> None:
    """Concept in, finished 1080x1920 vector animation out. No footage anywhere."""
    state = _mm()
    has_spec = bool(state.get("spec"))

    st.markdown(
        badge("① Concept", "green" if state.get("concept") else "violet")
        + badge("② Scene", "green" if has_spec else "violet")
        + badge("③ Render", "green" if st.session_state.get("minimal_video_path") else "violet"),
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------------- STEP 1
    with st.container(border=True):
        st.markdown("#### ① Metaphor")
        st.caption("Everything on screen is drawn from code — white vector on pure "
                   "black. There is no footage to license, so this mode clears the "
                   "publish gate on its own.")

        preset_keys = list(SCENE_PRESETS.keys())
        chosen = st.pills(
            "Quick-select", preset_keys, default=state.get("preset"),
            format_func=lambda k: SCENE_PRESETS[k]["label"], key="mm_preset",
        )
        if chosen and chosen != state.get("preset"):
            state["preset"] = chosen
            state["concept"] = SCENE_PRESETS[chosen]["concept"]
            state["template"] = SCENE_PRESETS[chosen]["template"]
            st.rerun()

        concept = st.text_area(
            "Metaphor Concept or Topic", value=str(state.get("concept") or ""),
            key="mm_concept", height=96,
            placeholder="Why the people who start ugly finish first",
            help="One idea, stated plainly. The model turns it into geometry, copy "
                 "and timing — it never draws anything itself.",
        )
        state["concept"] = concept

        c1, c2 = st.columns([2, 1])
        with c1:
            choices = [AUTO_TEMPLATE] + list(TEMPLATES.keys())
            # Resolve through the alias table: a template name saved before the
            # rename would otherwise not be found in `choices`, and the box
            # would quietly reset to Auto -- discarding a choice the user or a
            # preset had just made.
            stored = str(state.get("template") or AUTO_TEMPLATE)
            current = stored if stored == AUTO_TEMPLATE else resolve_template(stored)
            template = st.selectbox(
                "Scene template", choices,
                index=choices.index(current) if current in choices else 0,
                format_func=lambda k: ("✨ Auto — let Gemini pick the metaphor"
                                       if k == AUTO_TEMPLATE else str(TEMPLATES[k]["label"])),
                key="mm_template",
                help="Auto is the interesting one: the model reads your concept and "
                     "reaches for the geometry that argues it, rather than dropping "
                     "every topic into the same shape.",
            )
            state["template"] = template
            st.caption("Gemini chooses from the six metaphors below based on your concept."
                       if template == AUTO_TEMPLATE else str(TEMPLATES[template]["blurb"]))
        with c2:
            duration = st.slider(
                "Length (s)", int(SCENE_MIN_SECONDS), int(SCENE_MAX_SECONDS),
                int(state.get("duration", 18)), 1, key="mm_duration",
                help="15–25s is the retention sweet spot for Shorts: long enough to "
                     "land an idea, short enough to loop.",
            )
            state["duration"] = duration

        estimate = estimate_render_seconds(float(duration), int(st.session_state.get("render_fps", 30)))
        st.caption(f"≈ {estimate:.0f}s to draw at {int(st.session_state.get('render_fps', 30))}fps · "
                   f"font in use: `{active_face_name()}`")

    # ---------------------------------------------------------------- STEP 2
    with st.container(border=True):
        st.markdown("#### ② Sound")
        a1, a2, a3 = st.columns([1, 1, 1])
        with a1:
            state["bgm"] = st.checkbox("🎵 Atmospheric bed", value=bool(state.get("bgm", True)),
                                       key="mm_bgm",
                                       help="Slowed, reverbed phonk, synthesized to the exact "
                                            "length of the animation so there is no loop seam.")
            state["bgm_volume"] = st.slider("Bed volume", 0.0, 1.0,
                                            float(state.get("bgm_volume", 0.30)), 0.05,
                                            key="mm_bgmvol", disabled=not state["bgm"])
        with a2:
            state["sfx"] = st.checkbox("💥 Sub drop on the climax",
                                       value=bool(state.get("sfx", True)), key="mm_sfx",
                                       help="A riser into a sub-bass hit, placed so the impact "
                                            "lands on the frame the object clears the obstacle.")
        with a3:
            state["narrate"] = st.checkbox("🎙️ Narrate the thesis",
                                           value=bool(state.get("narrate", False)), key="mm_narrate",
                                           help="Gemini TTS reads the one-line thesis. Leading "
                                                "silence is trimmed so the voice starts at 0.0s.")
            state["voice"] = pick("Voice", MM_VOICES, str(state.get("voice") or "Charon"),
                                  "mm_voice") if state["narrate"] else state.get("voice", "Charon")

        if state["narrate"]:
            state["ai_disclosed"] = st.checkbox(
                "I will label this as AI-generated when I upload",
                value=bool(state.get("ai_disclosed")), key="mm_disclose",
                help="Required once there is a synthetic voice on the track. A silent "
                     "cut has no synthetic media in it and needs no label.",
            )
        else:
            state["ai_disclosed"] = bool(state.get("ai_disclosed"))
            st.caption("Silent cut — no synthetic voice, so no synthetic-media label is required.")

    # ---------------------------------------------------------------- STEP 3
    with st.container(border=True):
        st.markdown("#### ③ Generate & animate")
        b1, b2 = st.columns([2, 1])

        with b1:
            if st.button("✨ Generate Metaphor & Animate", width="stretch",
                         key="mm_go", disabled=not str(state.get("concept") or "").strip()):
                spec: dict[str, Any] | None = None
                with st.spinner("Gemini is writing the scene..."):
                    try:
                        raw = generate_scene_spec(
                            str(state["concept"]), str(state["template"]), float(state["duration"]),
                        )
                        spec = normalise_spec(raw)
                    except Exception as exc:
                        # A dead key or a 503 must not cost the user the video.
                        st.warning(
                            f"Gemini could not write the scene ({exc}). Animating with the "
                            "template's own copy instead — edit it below and re-render.",
                            icon="⚠️",
                        )
                        spec = fallback_scene_spec(
                            str(state["concept"]), str(state["template"]), float(state["duration"]),
                        )
                if spec is not None:
                    if not spec.get("publish"):
                        spec["publish"] = fallback_publish_meta(
                            str(state["concept"]), str(spec.get("title") or ""),
                            str(spec.get("payoff") or ""),
                        )
                    if run_minimalist_render(spec):
                        st.rerun()

        with b2:
            if st.button("🎬 Animate without AI", width="stretch", key="mm_local"):
                # No model to choose with, so "auto" picks deterministically
                # from the concept: the same words always give the same
                # metaphor, and different words give different ones.
                picked = str(state["template"])
                if picked == AUTO_TEMPLATE:
                    concept = str(state.get("concept") or "")
                    picked = METAPHOR_TYPES[sum(ord(c) for c in concept) % len(METAPHOR_TYPES)]
                spec = fallback_scene_spec(
                    str(state.get("concept") or ""), picked, float(state["duration"]),
                )
                spec["publish"] = fallback_publish_meta(
                    str(state.get("concept") or ""), str(spec.get("title") or ""),
                    str(spec.get("payoff") or ""),
                )
                if run_minimalist_render(spec):
                    st.rerun()

        if has_spec:
            spec = dict(state["spec"])
            with st.expander("✏️ Edit the copy and re-render"):
                spec["title"] = st.text_input("Title", value=str(spec.get("title") or ""),
                                              key="mm_title")
                spec["subtitle"] = st.text_input("Subtitle", value=str(spec.get("subtitle") or ""),
                                                 key="mm_subtitle")
                spec["payoff"] = st.text_input("Closing line", value=str(spec.get("payoff") or ""),
                                               key="mm_payoff")
                spec["thesis"] = st.text_area("Narration thesis",
                                              value=str(spec.get("thesis") or ""),
                                              key="mm_thesis", height=88)
                if st.button("🔄 Re-render with these edits", width="stretch", key="mm_rerender"):
                    state["spec"] = spec
                    if run_minimalist_render(spec):
                        st.rerun()

    # ---------------------------------------------------------------- OUTPUT
    result = state.get("result") or {}
    if result:
        chosen = resolve_template(result.get("template"))
        stat_row([
            ("Metaphor", str(TEMPLATES[chosen]["label"]).split(" ", 1)[-1], "violet"),
            ("Runtime", f"{float(result.get('duration', 0)):.1f}s", "cyan"),
            ("Frames", str(result.get("frames", 0)), ""),
            ("Climax", f"{float(result.get('climax', 0)):.1f}s", ""),
        ])
        if result.get("narration") and float(result.get("narration_trimmed", 0)) > 0:
            st.caption(f"Trimmed {float(result['narration_trimmed']):.2f}s of dead air "
                       "off the front of the voice track.")

    show_rendered_video("minimal", "Your Minimalist Motion short", slot="minimal")
    render_minimalist_publish()


def render_minimalist_publish() -> None:
    """Publish pack for an animation: the check, the metadata, the download."""
    state = _mm()
    name = st.session_state.get("minimal_video_name")
    publish = state.get("publish") or {}
    spec = state.get("spec") or {}
    if not name or not publish:
        return

    entry = find_entry(user_exports(), str(name))
    verdict = publish_readiness(entry) if entry else None

    # The animation's spoken script is its thesis; there is no other narration.
    render_viral_scorecard(str(spec.get("thesis") or ""),
                           scope="minimalist", entry=entry or {})

    with st.container(border=True):
        st.markdown("#### 📤 Publish pack")

        if verdict and verdict["ready"]:
            st.success("Cleared for a monetized upload — the animation is your own work "
                       "outright, with nothing borrowed to claim.", icon="✅")
        elif verdict:
            st.error(f"Not cleared yet — {len(verdict['blockers'])} to fix.", icon="🚫")
            for blocker in verdict["blockers"]:
                st.markdown(f"- 🚫 {blocker}")

        st.markdown("**High-CTR title**")
        st.code(str(publish.get("title") or ""), language=None)
        st.markdown("**Description**")
        st.code(str(publish.get("description") or ""), language=None)
        st.markdown("**Hashtags**")
        st.code(" ".join(publish.get("hashtags") or []), language=None)

        pack = mm_publish_text(spec, publish)
        st.download_button(
            "📋 Download publish pack (.txt)", data=pack.encode("utf-8"),
            file_name=f"{os.path.splitext(str(name))[0]}_publish.txt",
            mime="text/plain", width="stretch", key=f"mm_pack_{name}",
        )

        if verdict and verdict["warnings"]:
            with st.expander(f"Worth knowing ({len(verdict['warnings'])})"):
                for item in verdict["warnings"]:
                    st.markdown(f"- ⚠️ {item}")



# ---------------------------------------------------------------------------
# Authentication gate
#
# Nothing else in this file renders until `require_login()` returns True. The
# gate is the first statement in main(), so an unauthenticated visitor sees the
# login card and nothing else -- no sidebar, no mode selector, no filenames.
# ---------------------------------------------------------------------------

LOGIN_CSS = """
<style>
.rf-login-wrap { max-width: 400px; margin: 6vh auto 0 auto; }
.rf-login-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015));
    border: 1px solid var(--edge);
    border-radius: 18px;
    padding: 34px 30px 26px 30px;
    box-shadow: 0 24px 70px rgba(0,0,0,0.55);
}
.rf-login-mark {
    font-size: 30px; font-weight: 800; letter-spacing: -0.02em;
    text-align: center; color: var(--text-hi); margin-bottom: 4px;
}
.rf-login-sub {
    text-align: center; color: var(--text-low); font-size: 13px;
    margin-bottom: 22px; letter-spacing: 0.02em;
}
.rf-login-rule { height: 1px; background: var(--edge); margin: 20px 0 14px 0; }
.rf-login-foot { text-align: center; color: var(--text-low); font-size: 11.5px; }
</style>
"""

# Rate limiting lives in auth.py, not here: Streamlit re-executes this script
# on every interaction, so a counter defined at this level would be reset by
# the very click it is counting. See the note there.
LOGIN_MAX_ATTEMPTS = auth.LOGIN_MAX_ATTEMPTS
LOGIN_LOCKOUT_SECONDS = auth.LOGIN_LOCKOUT_SECONDS
login_locked_for = auth.login_locked_for
note_login_failure = auth.note_login_failure
clear_login_failures = auth.clear_login_failures


def _boot_admin_once() -> None:
    """
    Makes sure an admin exists, printing a generated password to the console.

    To the console and never to the page: whoever can reach the login screen is
    not yet known to be the operator.
    """
    if st.session_state.get("_auth_bootstrapped"):
        return
    st.session_state["_auth_bootstrapped"] = True

    result = auth.bootstrap()
    if result["created"]:
        banner = (
            "\n" + "=" * 66 +
            "\n  ReelForge Studio - first run: an admin account was created."
            f"\n    username: {result['username']}"
            f"\n    password: {result['password']}"
            "\n  This is shown once, here in the server console only."
            "\n  Sign in and change it under Admin > Users."
            "\n" + "=" * 66 + "\n"
        )
        # Both stdout and the logger. print() alone is not reliable enough for
        # the one credential the operator cannot recover any other way: it can
        # sit in a buffer behind a redirect, and `docker logs` and a Windows
        # service capture stderr more consistently than stdout.
        print(banner, flush=True)
        logging.getLogger("reelforge.auth").warning(banner)
        st.session_state["_auth_first_run"] = True


def render_login() -> None:
    """The whole page when nobody is signed in."""
    st.markdown(LOGIN_CSS, unsafe_allow_html=True)
    _, middle, _ = st.columns([1, 1.15, 1])

    with middle:
        st.markdown('<div class="rf-login-wrap">', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="rf-login-mark">🎬 ReelForge Studio</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="rf-login-sub">Private workspace · sign in to continue</div>',
                        unsafe_allow_html=True)

            waiting = float(st.session_state.get("_auth_waiting", 0.0))

            with st.form("rf_login", clear_on_submit=False):
                username = st.text_input("Username", key="login_user",
                                         autocomplete="username")
                password = st.text_input("Password", type="password", key="login_pass",
                                         autocomplete="current-password")
                submitted = st.form_submit_button(
                    "Sign in" if waiting <= 0 else f"Locked · {waiting:.0f}s",
                    width="stretch", disabled=waiting > 0,
                )

            if waiting > 0:
                # Say why. A button that has silently turned into "Locked" with
                # no explanation reads as a broken app, not as a cooldown.
                st.warning(
                    f"Too many attempts on that account. Try again in {waiting:.0f} seconds.",
                    icon="⏳",
                )

            if submitted:
                # Checked here as well as on the button: a disabled button is
                # only a hint to the browser, and the cooldown has to hold
                # against a client that ignores it.
                remaining = login_locked_for(username)
                if remaining > 0:
                    st.session_state["_auth_waiting"] = remaining
                    st.warning(f"Too many attempts. Try again in {remaining:.0f} seconds.",
                               icon="⏳")
                else:
                    who = auth.authenticate(username, password)
                    if who:
                        clear_login_failures(who["username"])
                        st.session_state["auth"] = {
                            "username": who["username"],
                            "role": who["role"],
                            "signed_in_at": time.strftime("%Y-%m-%d %H:%M"),
                        }
                        st.session_state.pop("_auth_waiting", None)
                        ensure_dir(auth.user_exports_dir(who["username"]))
                        st.rerun()
                    else:
                        cooldown = note_login_failure(username)
                        st.session_state["_auth_waiting"] = cooldown
                        # One message for both failure modes: saying "no such
                        # user" would let anyone test which names exist.
                        st.error("Incorrect username or password.", icon="🚫")
                        if cooldown > 0:
                            st.warning(
                                f"Too many attempts. Try again in {cooldown:.0f} seconds.",
                                icon="⏳",
                            )

            if st.session_state.get("_auth_first_run"):
                st.info(
                    "First run — an admin account was created and its password was "
                    "printed to the server console. Check the terminal running "
                    "Streamlit, or `docker logs` for the container.",
                    icon="🔑",
                )

            st.markdown('<div class="rf-login-rule"></div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="rf-login-foot">Accounts are issued by an administrator.</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


def require_login() -> bool:
    """True once signed in; otherwise draws the login card and returns False."""
    _boot_admin_once()

    # Refresh the countdown from the authoritative table so it ticks down and
    # the form unlocks itself once the cooldown has passed.
    typed = str(st.session_state.get("login_user") or "")
    if typed:
        st.session_state["_auth_waiting"] = login_locked_for(typed)

    session = st.session_state.get("auth")
    if isinstance(session, dict) and session.get("username"):
        # Re-read the role every run so a change made in Admin takes effect on
        # the next click rather than at the user's next sign-in.
        record = auth.load_users().get(str(session["username"]))
        if record is None:
            st.session_state.pop("auth", None)      # account deleted mid-session
        else:
            session["role"] = auth.normalise_role(record.get("role"))
            return True

    render_login()
    return False


def render_identity_bar() -> None:
    """Who you are and the way out, at the top of the sidebar."""
    user = current_user()
    role = auth.normalise_role(user.get("role"))
    spec = auth.ROLES[role]

    st.markdown(
        badge(f"👤 {user.get('username', '')}", "violet")
        + badge(spec["label"], "cyan" if role == auth.ADMIN else ""),
        unsafe_allow_html=True,
    )
    if st.button("Log out", width="stretch", key="rf_logout"):
        # Clear the whole session, not just the auth key: session state holds
        # this user's rendered video bytes and scripts, and the next person at
        # this browser must not inherit them.
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    st.caption(f"Files: `exports/{auth.safe_slug(str(user.get('username') or ''))}/`")


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------

def render_api_panel() -> None:
    """Connectivity, for whoever has to fix it."""
    section("API connectivity")
    rows = dashboard_view.api_status()
    for column, row in zip(st.columns(len(rows)), rows):
        with column:
            st.markdown(
                f'<div class="rf-api"><div class="rf-api-head">'
                f'<div class="rf-api-name">{row["name"]}</div>'
                + badge(row["state"], row["tone"])
                + f'</div><div class="rf-api-detail">{row["detail"]}</div></div>',
                unsafe_allow_html=True,
            )


def render_admin_studio() -> None:
    """User management, API keys and a workspace overview. Admin only."""
    user = current_user()
    if not auth.can(str(user.get("role")), "manage_users"):
        st.error("Administrators only.", icon="🚫")
        return

    st.markdown(badge("🛠️ Administration", "violet"), unsafe_allow_html=True)

    # ------------------------------------------------------------ TELEMETRY
    # These four readings and the connectivity panel used to sit on top of
    # every page, including the one a customer lands on. Encoder presets and
    # scratch byte counts answer "is this deployment healthy", which is an
    # administrator's question, so this is where they live.
    render_command_center()
    divider()
    render_api_panel()
    divider()

    # ---------------------------------------------------------------- USERS
    with st.container(border=True):
        st.markdown("#### 👥 Users")
        users = auth.load_users()
        admins = sum(1 for r in users.values() if auth.normalise_role(r.get("role")) == auth.ADMIN)
        stat_row([
            ("Accounts", str(len(users)), "cyan"),
            ("Admins", str(admins), ""),
            ("Workspaces", str(len(auth.list_user_dirs())), ""),
        ])

        for name, record in sorted(users.items()):
            role = auth.normalise_role(record.get("role"))
            from_env = record.get("source") == "env"
            with st.expander(
                f"{'🛡️' if role == auth.ADMIN else '🎬'} {name}"
                + (" · from environment" if from_env else "")
            ):
                st.markdown(
                    badge(auth.ROLES[role]["label"], "cyan" if role == auth.ADMIN else "")
                    + badge(f"created {record.get('created_at') or 'unknown'}", "")
                    + badge(f"last login {record.get('last_login') or 'never'}", ""),
                    unsafe_allow_html=True,
                )
                if from_env:
                    st.caption("Declared by REELFORGE_USERS / REELFORGE_ADMIN_USER. "
                               "Change the environment variable and restart to edit it.")
                    continue

                c1, c2 = st.columns([1, 1])
                with c1:
                    wanted = st.selectbox(
                        "Role", list(auth.ROLES.keys()),
                        index=list(auth.ROLES.keys()).index(role),
                        format_func=lambda k: str(auth.ROLES[k]["label"]),
                        key=f"adm_role_{name}",
                    )
                    if wanted != role and st.button("Apply role", key=f"adm_setrole_{name}"):
                        try:
                            auth.set_role(name, wanted)
                            st.success(f"{name} is now {auth.ROLES[wanted]['label']}.")
                            st.rerun()
                        except ValueError as exc:
                            st.error(str(exc))
                with c2:
                    fresh = st.text_input("New password", type="password",
                                          key=f"adm_pw_{name}")
                    if st.button("Reset password", key=f"adm_setpw_{name}", disabled=not fresh):
                        try:
                            auth.set_password(name, fresh)
                            st.success(f"Password reset for {name}.")
                        except ValueError as exc:
                            st.error(str(exc))

                if name != str(user.get("username")):
                    if st.button(f"Delete {name}", key=f"adm_del_{name}"):
                        try:
                            auth.delete_user(name)
                            st.success(f"Deleted {name}. Their files were left in place.")
                            st.rerun()
                        except ValueError as exc:
                            st.error(str(exc))
                else:
                    st.caption("This is you — delete from another admin account.")

        st.markdown("---")
        st.markdown("**Add a user**")
        n1, n2, n3 = st.columns([1.2, 1.2, 0.9])
        with n1:
            new_name = st.text_input("Username", key="adm_new_name",
                                     placeholder="ana.k")
        with n2:
            new_pass = st.text_input("Password", type="password", key="adm_new_pass",
                                     placeholder="at least 8 characters")
        with n3:
            new_role = st.selectbox("Role", list(auth.ROLES.keys()), index=1,
                                    format_func=lambda k: str(auth.ROLES[k]["label"]),
                                    key="adm_new_role")
        if st.button("➕ Create account", width="stretch", key="adm_create",
                     disabled=not (new_name and new_pass)):
            try:
                made = auth.add_user(new_name, new_pass, new_role)
                st.success(f"Created {made['username']} ({auth.ROLES[made['role']]['label']}).")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    # ------------------------------------------------------------- API KEYS
    with st.container(border=True):
        st.markdown("#### 🔑 API configuration")
        st.caption("Keys apply to the whole server, so only administrators can see or "
                   "change them. A key set here lasts until the process restarts — put "
                   "it in `.env` or the container environment to make it permanent.")

        gem = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        pex = pexels_key_status()
        stat_row([
            ("Gemini", f"set ({len(gem)} chars)" if gem else "missing",
             "green" if gem else "amber"),
            ("Pexels", "active" if pex["ok"] else ("set, failing" if os.environ.get("PEXELS_API_KEY") else "missing"),
             "green" if pex["ok"] else "amber"),
        ])
        if not pex["ok"] and os.environ.get("PEXELS_API_KEY"):
            st.caption(f"Pexels says: {pex.get('reason') or 'no detail'}")

        k1, k2 = st.columns([1, 1])
        with k1:
            gem_new = st.text_input("GEMINI_API_KEY", type="password", key="adm_gem")
            if st.button("Set Gemini key", key="adm_set_gem", disabled=not gem_new):
                os.environ["GEMINI_API_KEY"] = gem_new.strip()
                st.success("Set for this server process.")
                st.rerun()
        with k2:
            pex_new = st.text_input("PEXELS_API_KEY", type="password", key="adm_pex")
            if st.button("Set Pexels key", key="adm_set_pex", disabled=not pex_new):
                os.environ["PEXELS_API_KEY"] = pex_new.strip()
                # The probe caches its verdict, and the cached one is about the
                # old key. force=True re-checks against the live endpoint.
                checked = pexels_key_status(force=True)
                if checked["ok"]:
                    st.success("Set and verified against the Pexels API.")
                else:
                    st.warning(f"Set, but Pexels rejected it: {checked['reason']}", icon="⚠️")
                st.rerun()

    # ------------------------------------------------------------ WORKSPACES
    with st.container(border=True):
        st.markdown("#### 📁 Workspaces")
        st.caption("Each account renders into its own folder and only ever sees its own. "
                   "This is the operator's view across all of them.")

        rows: list[tuple[str, str, str]] = []
        for folder in auth.list_user_dirs():
            path = str(EXPORTS_ROOT / folder)
            videos = [f for f in os.listdir(path) if f.lower().endswith(".mp4")] \
                if os.path.isdir(path) else []
            size = sum(os.path.getsize(os.path.join(path, f)) for f in videos) / 1_048_576
            summary = summarise_ledger(path)
            rows.append((folder, f"{len(videos)} clips · {size:.0f} MB",
                         f"{summary['ready']}/{summary['total']} publish-ready"))

        if not rows:
            st.caption("No renders yet.")
        for folder, files, ready in rows:
            st.markdown(f"- **{folder}** — {files} · {ready}")



# ---------------------------------------------------------------------------
# Narrative Studio
#
# A four-step wizard, and the steps are genuinely sequential: the storyboard
# cannot exist before the script, and the stills are not worth generating
# before the voice has decided how long each segment actually is.
# ---------------------------------------------------------------------------

NARRATIVE_VOICES = {
    "gemini": ("Charon", "Kore", "Puck", "Fenrir", "Aoede"),
    "edge": ("en-US-ChristopherNeural", "en-US-GuyNeural", "en-GB-RyanNeural",
             "en-US-JennyNeural", "en-GB-SoniaNeural"),
}


def _nv() -> dict[str, Any]:
    """Narrative Studio's own state, kept out of the widget keys."""
    return st.session_state.narrative


def _narrative_status(slot: Any) -> Callable[[str], None]:
    """A progress callback that returns None -- st.markdown returns a container."""
    def report(message: str) -> None:
        slot.markdown(f"**{message}**")
    return report


def render_narrative_studio() -> None:
    """Premise in, episode out, in four reviewable steps."""
    state = _nv()
    episode = state.get("episode")
    result = state.get("result")

    st.markdown(
        badge("① Concept", "green" if state.get("topic") else "violet")
        + badge("② Script & cast", "green" if episode else "violet")
        + badge("③ Storyboard", "green" if episode else "violet")
        + badge("④ Produce", "green" if result else "violet"),
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------------- STEP 1
    with st.container(border=True):
        st.markdown("#### ① Channel style & concept")

        state["topic"] = st.text_area(
            "Topic", value=str(state.get("topic") or ""), key="nv_topic", height=90,
            placeholder="POV you lived with your parents during your 20s — now you're wealthy",
            help="One premise. The model turns it into a script, a cast and a world, "
                 "then keeps that world consistent across every shot.",
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            state["aesthetic"] = pick(
                "Visual aesthetic", list(NARRATIVE_AESTHETICS.keys()),
                str(state.get("aesthetic") or NARRATIVE_DEFAULT_AESTHETIC), "nv_look",
                format_func=lambda k: str(NARRATIVE_AESTHETICS[k]["label"]),
            )
            st.caption(str(NARRATIVE_AESTHETICS[state["aesthetic"]]["prompt"])[:120] + "...")
        with c2:
            state["tone"] = pick(
                "Narrative tone", list(NARRATIVE_TONES.keys()),
                str(state.get("tone") or NARRATIVE_DEFAULT_TONE), "nv_tone",
                format_func=lambda k: str(NARRATIVE_TONES[k]["label"]),
            )
            st.caption(str(NARRATIVE_TONES[state["tone"]]["prompt"])[:120] + "...")

        keys = list(NARRATIVE_FORMATS.keys())
        state["format"] = st.selectbox(
            "Duration & framing", keys,
            index=keys.index(str(state.get("format") or NARRATIVE_DEFAULT_FORMAT))
            if state.get("format") in keys else 0,
            format_func=lambda k: str(NARRATIVE_FORMATS[k]["label"]), key="nv_format",
        )
        fmt = NARRATIVE_FORMATS[state["format"]]
        st.caption(f"{fmt['note']} · {fmt['segments'][0]}–{fmt['segments'][1]} segments · "
                   f"{fmt['aspect'][0]}×{fmt['aspect'][1]}")

        if st.button("✍️ Write the script & cast the world", width="stretch",
                     key="nv_write", disabled=not str(state.get("topic") or "").strip()):
            status = st.empty()
            with st.spinner("Gemini is writing..."):
                try:
                    state["episode"] = generate_narrative(
                        str(state["topic"]), str(state["aesthetic"]), str(state["tone"]),
                        str(state["format"]),
                        progress=_narrative_status(status),
                    )
                    state["result"] = None
                except Exception as exc:
                    st.error(f"**Could not write the episode:** `{type(exc).__name__}: {exc}`")
                    st.code(traceback.format_exc())
            status.empty()
            st.rerun()

    if not episode:
        st.info("Write the script to unlock the storyboard and production steps.", icon="✍️")
        return

    # ---------------------------------------------------------------- STEP 2
    with st.container(border=True):
        st.markdown("#### ② Script & entity extraction")
        if str(episode.get("source")) == "fallback":
            st.warning("Gemini was unavailable, so this is the skeleton draft. Edit the "
                       "premise and write again for a real script.", icon="⚠️")

        st.markdown(f"**{episode['title']}**")
        st.caption(episode.get("logline") or "")
        stat_row([
            ("Segments", str(len(episode["segments"])), "cyan"),
            ("Estimated", f"{float(episode['runtime']):.0f}s", ""),
            ("Words", str(len(str(episode["script"]).split())), ""),
            ("Model", str(episode.get("source") or ""), "violet"),
        ])

        ents = episode["entities"]
        p = ents["protagonist"]
        with st.expander(f"🎭 Protagonist — {p['name']}, {p['age']}", expanded=True):
            st.markdown(f"**Appearance** · {p['appearance']}")
            st.markdown(f"**Clothing** · {p['clothing']}")
            st.markdown(f"**Demeanour** · {p['demeanour']}")
            st.caption("This exact text is pasted into every image prompt. It is what "
                       "keeps the same person on screen from shot to shot.")

        with st.expander(f"🏙️ Environments ({len(ents['environments'])}) "
                         f"and objects ({len(ents['objects'])})"):
            for env in ents["environments"]:
                st.markdown(f"**{env['name']}** `{env['tag']}` — {env['description']}")
            for obj in ents["objects"]:
                st.markdown(f"**{obj['name']}** `{obj['tag']}` — {obj['description']}")

        with st.expander("📄 Full narration"):
            st.write(episode["script"])

    # ---------------------------------------------------------------- STEP 3
    with st.container(border=True):
        st.markdown("#### ③ Storyboard")
        st.caption("Durations here are estimated from the word count. They are re-cut "
                   "against the real voice before a single still is generated.")

        rows = [{
            "#": s["index"],
            "Start": f"{float(s['start']):.1f}s",
            "Secs": round(float(s["duration"]), 1),
            "Words": s["words"],
            "Beat": s["beat_type"],
            "Environment": s["environment"],
            "Voiceover": s["line"],
            "Camera": s["framing"],
        } for s in episode["segments"]]
        st.dataframe(rows, width="stretch", hide_index=True,
                     column_config={"Voiceover": st.column_config.TextColumn(width="large"),
                                    "Camera": st.column_config.TextColumn(width="large")})

    # ---------------------------------------------------------------- STEP 4
    with st.container(border=True):
        st.markdown("#### ④ Assets & assembly")

        a1, a2, a3 = st.columns([1, 1, 1])
        with a1:
            state["voice_provider"] = pick(
                "Narration engine", list(NARRATIVE_VOICES.keys()),
                str(state.get("voice_provider") or "gemini"), "nv_provider",
                format_func=lambda k: "Gemini TTS ✅" if k == "gemini" else "edge-tts ⚠️ draft",
            )
            voices = NARRATIVE_VOICES[state["voice_provider"]]
            current = str(state.get("voice") or voices[0])
            state["voice"] = st.selectbox(
                "Voice", list(voices), index=list(voices).index(current) if current in voices else 0,
                key=f"nv_voice_{state['voice_provider']}",
            )
            if state["voice_provider"] == "edge":
                st.caption("edge-tts gives exact word timings, so the cuts land on the "
                           "word. It is not licensed for monetized publishing.")
        with a2:
            state["transition"] = pick(
                "Between shots", ["dissolve", "cut"], str(state.get("transition") or "dissolve"),
                "nv_transition",
                format_func=lambda k: "Soft dissolve" if k == "dissolve" else "Hard cut",
            )
            state["subtitles"] = st.checkbox("Burn subtitles",
                                             value=bool(state.get("subtitles", True)),
                                             key="nv_subs")
            state["ambient"] = st.checkbox("Ambient bed under the voice",
                                           value=bool(state.get("ambient", True)),
                                           key="nv_ambient")
        with a3:
            state["image_providers"] = st.multiselect(
                "Visual sources, in order", list(NARRATIVE_IMAGE_PROVIDERS),
                default=list(state.get("image_providers") or NARRATIVE_IMAGE_PROVIDERS),
                key="nv_images",
                help="Tried in order; the first that delivers wins. Gemini is the only "
                     "one that can draw the same character twice.",
            )

        if "gemini" in (state["image_providers"] or []):
            st.caption("ℹ️ Gemini image generation is not on the API free tier. Without "
                       "billing enabled it returns a quota error and the chain falls "
                       "through to stock photography, graded to your aesthetic.")

        # Runtime is decided by the narration, so it can only be capped after
        # the voice has been synthesized -- this is a hard limit on the finished
        # file, not a hint to the model.
        state["shorts_mode"] = pick(
            "Duration", [True, False], bool(state.get("shorts_mode", False)), "nv_shorts",
            format_func=lambda on: (
                f"📱 Shorts mode ({NARRATIVE_SHORTS_MIN:.0f}-{NARRATIVE_SHORTS_MAX:.0f}s max)"
                if on else "🎞️ Long-form story (3-5 min)"),
            help=f"Shorts mode compresses the finished board to at most "
                 f"{NARRATIVE_SHORTS_MAX:.0f}s so the episode stays eligible for the "
                 f"vertical Shorts shelf. It is applied after narration, because that "
                 f"is what actually decides the runtime. If compression alone would "
                 f"push shots under {1.6:.1f}s, beats are dropped from the end and you "
                 f"are told.",
        )
        if state["shorts_mode"] and str(state.get("format") or "").startswith("longform"):
            st.caption(f"⚠️ A 3-5 minute script capped at {NARRATIVE_SHORTS_MAX:.0f}s will "
                       "lose most of its beats. Pick the 60s format above instead.")

        state["ai_disclosed"] = st.checkbox(
            "I will label this as AI-generated when I upload",
            value=bool(state.get("ai_disclosed")), key="nv_disclose",
        )

        estimate = len(episode["segments"]) * 7 + float(episode["runtime"]) * 0.6
        st.caption(f"≈ {estimate / 60:.0f}–{estimate / 30:.0f} min to produce "
                   f"{len(episode['segments'])} segments.")

        if st.button("🎬 Produce the episode", width="stretch", key="nv_produce",
                     disabled=not state.get("image_providers")):
            if run_narrative_production(episode):
                st.rerun()

    if result:
        stat_row([
            ("Runtime", f"{float(result.get('duration') or 0):.1f}s", "cyan"),
            ("Segments", str(len(result.get("segments") or [])), ""),
            ("Stills", ", ".join(result.get("image_providers") or []), "violet"),
            ("Cuts", "on the word" if result.get("timings_exact") else "scaled", ""),
        ])
        for note in (result.get("provider_notes") or [])[:1]:
            st.caption(f"⚠️ {note[:180]}")

    show_rendered_video("narrative", "Your episode", slot="narrative")
    render_narrative_pack()


def run_narrative_production(episode: dict[str, Any]) -> bool:
    """Runs stages 3 and 4 and files the result. Returns success."""
    state = _nv()
    tracker = StageProgress(label="Starting production...")
    status = st.empty()
    total = max(1, len(episode["segments"]) * 2 + 6)
    done = {"n": 0}

    def progress(message: str) -> None:
        done["n"] += 1
        tracker.update(min(0.98, done["n"] / total), message)
        status.markdown(f"**{message}**")

    workspace = ""
    try:
        stamp = int(time.time())
        out_path = os.path.join(user_exports(), f"narrative_{stamp}.mp4")
        workspace = os.path.join(user_exports(), f"narrative_{stamp}_work")

        result = produce_episode(
            episode, out_path, workspace=workspace,
            fps=int(st.session_state.get("render_fps", 30)),
            voice_provider=str(state.get("voice_provider") or "gemini"),
            voice=str(state.get("voice") or "Charon"),
            transition=str(state.get("transition") or "dissolve"),
            subtitles=bool(state.get("subtitles", True)),
            ambient=bool(state.get("ambient", True)),
            image_providers=tuple(state.get("image_providers") or NARRATIVE_IMAGE_PROVIDERS),
            shorts_mode=bool(state.get("shorts_mode")),
            progress=progress,
        )

        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise FileNotFoundError(f"Producer finished but {out_path} is missing or empty")

        with open(out_path, "rb") as handle:
            data = handle.read()

        # Stock photography carries the Pexels licence; anything drawn by the
        # model or by us is the user's own work. Mixed sources take the
        # stricter of the two.
        providers = set(result.get("image_providers") or [])
        licence = "pexels" if "pexels" in providers else "own"

        entry = append_ledger(user_exports(), {
            "video_name": os.path.basename(out_path), "video_path": out_path,
            "duration": float(result["duration"]),
            "licence": licence, "licence_reference": "",
            "source_title": str(episode.get("title") or ""),
            "source_author": "", "source_url": "",
            "source_provider": ", ".join(sorted(providers)) or "reelforge",
            "tts_provider": str(result["tts_provider"]),
            "voice": str(result.get("voice") or ""),
            "script_model": str(episode.get("source") or ""),
            "ai_disclosed": bool(state.get("ai_disclosed")),
            "script": str(episode.get("script") or ""),
            "template": "narrative",
        })

        state["result"] = result
        state["entry_name"] = entry["video_name"]
        state["pack"] = metadata_pack(episode, result)
        st.session_state["narrative_video_path"] = out_path
        st.session_state["narrative_video_bytes"] = data
        st.session_state["narrative_video_name"] = os.path.basename(out_path)

        purge_scratch_renders()
        elapsed = tracker.finish("Episode complete.")
        status.markdown("**Done.**")
        if result.get("shorts_trimmed"):
            st.warning(
                "Shorts mode had to drop beats from the end to fit "
                f"{NARRATIVE_SHORTS_MAX:.0f}s. The script is longer than the format "
                "allows -- shorten it rather than letting the landing be cut.",
                icon="✂️",
            )
        notify_complete("Episode rendered",
                        f"{os.path.basename(out_path)} — {result['duration']:.0f}s "
                        f"in {elapsed:.0f}s")
        return True

    except Exception as exc:
        tracker.finish("Failed.")
        status.empty()
        st.error(f"**Production failed:** `{type(exc).__name__}: {exc}`")
        st.code(traceback.format_exc())
        purge_scratch_renders()
        return False
    finally:
        # The workspace holds the stills and the narration, which are worth
        # keeping; only the assembly intermediates are removed, and
        # produce_episode has already done that.
        if workspace and os.path.isdir(workspace):
            try:
                remaining = os.listdir(workspace)
                if not remaining:
                    os.rmdir(workspace)
            except OSError:
                pass


def render_narrative_pack() -> None:
    """The publish check and the episode pack."""
    state = _nv()
    name = st.session_state.get("narrative_video_name")
    if not name or not state.get("pack"):
        return

    entry = find_entry(user_exports(), str(name))
    verdict = publish_readiness(entry) if entry else None

    render_viral_scorecard(str((state.get("episode") or {}).get("script")
                               or (entry or {}).get("script") or ""),
                           scope="narrative", entry=entry or {})

    with st.container(border=True):
        st.markdown("#### 📤 Episode pack")

        if verdict and verdict["ready"]:
            st.success("Cleared for a monetized upload.", icon="✅")
        elif verdict:
            st.error(f"Not cleared yet — {len(verdict['blockers'])} to fix.", icon="🚫")
            for blocker in verdict["blockers"]:
                st.markdown(f"- 🚫 {blocker}")

        credit = attribution_line(entry) if entry else ""
        if credit:
            st.markdown("**Credit line — this must appear in your description:**")
            st.code(credit, language=None)

        st.download_button(
            "📋 Download episode pack (.txt)",
            data=str(state["pack"]).encode("utf-8"),
            file_name=f"{os.path.splitext(str(name))[0]}_pack.txt",
            mime="text/plain", width="stretch", key=f"nv_pack_{name}",
            help="Script, cast, environments, the full storyboard with timings, and "
                 "how every still was sourced.",
        )

        with st.expander("Storyboard as produced"):
            for segment in (state.get("result") or {}).get("segments") or []:
                st.markdown(
                    f"**{segment['index']}.** `{segment['start']:.1f}–{segment['end']:.1f}s` "
                    f"· {segment.get('image_provider', '')} · {segment['line']}"
                )

# ---------------------------------------------------------------------------
# Atmosphere Studio
#
# Long-form ambient and sleep video, plus direct publishing to YouTube.
# Entirely additive: the engine is ambient_engine, the uploader is publisher,
# and neither is touched by any other mode.
# ---------------------------------------------------------------------------

def _at() -> dict[str, Any]:
    """Atmosphere Studio's own state, kept out of the widget keys."""
    return st.session_state.atmosphere


def atmosphere_soundscape_name(state: dict[str, Any]) -> str:
    """A human phrase for the current mix, used in filenames and SEO prompts."""
    bed = ambient_engine.PRIMARY_BEDS.get(
        str(state.get("bed")), {}).get("label", "Ambient")
    # Strip the leading emoji the picker labels carry.
    bed = bed.split(" ", 1)[-1] if " " in bed else bed

    texture = str(state.get("texture") or "none")
    if texture == "none":
        return bed
    tex = ambient_engine.SECONDARY_TEXTURES.get(texture, {}).get("label", "")
    tex = tex.split(" ", 1)[-1] if " " in tex else tex
    return f"{bed} with {tex}"


def render_atmosphere_monetization() -> None:
    """The guidance banner: what gets an ambient channel demonetized."""
    with st.container(border=True):
        st.markdown("#### 💰 Before you publish relaxation content")
        st.caption("YouTube's Reused Content policy is enforced harder in this niche "
                   "than almost any other. These are the specifics.")
        for note in ambient_engine.MONETIZATION_NOTES:
            st.markdown(f"- {note}")


def render_atmosphere_studio() -> None:
    """Soundscape → canvas → render → publish."""
    state = _at()

    # ---- Step 1: the soundscape -------------------------------------------
    with st.container(border=True):
        st.markdown("#### ① Soundscape")
        st.caption("Every layer is synthesized here and now — nothing is a licensed "
                   "loop, which is what keeps the audio your own work.")

        beds = list(ambient_engine.PRIMARY_BEDS)
        state["bed"] = pick(
            "Primary bed", beds, state.get("bed", beds[0]), "at_bed",
            format_func=lambda k: ambient_engine.PRIMARY_BEDS[k]["label"],
        )
        st.caption(ambient_engine.PRIMARY_BEDS[state["bed"]]["blurb"])

        textures = list(ambient_engine.SECONDARY_TEXTURES)
        state["texture"] = pick(
            "Secondary texture", textures, state.get("texture", "none"), "at_texture",
            format_func=lambda k: ambient_engine.SECONDARY_TEXTURES[k]["label"],
        )
        st.caption(ambient_engine.SECONDARY_TEXTURES[state["texture"]]["blurb"])

        c1, c2 = st.columns(2)
        with c1:
            state["bed_volume"] = st.slider(
                "Bed level", 0.2, 1.0, float(state.get("bed_volume", 1.0)), 0.05,
                key="at_bedvol")
        with c2:
            state["texture_volume"] = st.slider(
                "Texture level", 0.0, 1.0, float(state.get("texture_volume", 0.35)), 0.05,
                key="at_texvol",
                disabled=state["texture"] == "none",
                help="Kept well under the bed by default. A texture you notice is a "
                     "texture that wakes people up.")

        durations = list(ambient_engine.DURATIONS)
        state["duration_key"] = pick(
            "Runtime", durations, state.get("duration_key", ambient_engine.DEFAULT_DURATION),
            "at_duration",
            format_func=lambda k: ambient_engine.DURATIONS[k]["label"],
        )
        st.caption(ambient_engine.DURATIONS[state["duration_key"]]["note"])

        if st.button("🔊 Preview 20 seconds of this mix", key="at_preview",
                     width="stretch"):
            with st.spinner("Synthesizing a preview..."):
                try:
                    work = os.path.join(user_exports(), "atmosphere_preview")
                    ensure_dir(work)
                    state["preview_path"] = ambient_engine.render_preview(
                        state["bed"], state["texture"], work,
                        float(state["bed_volume"]), float(state["texture_volume"]))
                except Exception as exc:
                    st.error(f"Preview failed: `{type(exc).__name__}: {exc}`")

        if state.get("preview_path") and os.path.exists(state["preview_path"]):
            st.audio(state["preview_path"])
            if state["texture"].startswith("drone_"):
                st.caption("🎧 The binaural offset only exists between the two ears — "
                           "on a speaker it collapses to one tone.")

    # ---- Step 2: the canvas -----------------------------------------------
    with st.container(border=True):
        st.markdown("#### ② Visual canvas")
        st.caption("16:9, 1920×1080. Slow drift and grain are added at render time. "
                   "The sidebar's aspect, fit, FPS and transition controls belong to "
                   "the short-form pipeline and are not read here — this mode fixes "
                   "its own format.")

        source_mode = pick(
            "Background", ["preset", "upload"], state.get("source_mode", "preset"),
            "at_srcmode",
            format_func=lambda k: {"preset": "🎨 Drawn preset",
                                   "upload": "⬆️ Upload image or loop"}[k],
        )
        state["source_mode"] = source_mode

        if source_mode == "preset":
            presets = list(ambient_engine.CANVAS_PRESETS)
            state["preset"] = pick(
                "Preset", presets, state.get("preset", ambient_engine.DEFAULT_CANVAS),
                "at_preset",
                format_func=lambda k: ambient_engine.CANVAS_PRESETS[k]["label"],
            )
            st.caption(ambient_engine.CANVAS_PRESETS[state["preset"]]["blurb"])
            try:
                work = os.path.join(user_exports(), "atmosphere_preview")
                ensure_dir(work)
                path = ambient_engine.build_canvas_preset(state["preset"], work)
                state["visual_source"] = path
                st.image(path, width="stretch")
            except Exception as exc:
                st.error(f"Could not draw that preset: `{type(exc).__name__}: {exc}`")
        else:
            upload = st.file_uploader(
                "Background image or seamless video loop",
                type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "mkv", "webm"],
                key="at_upload",
                help="A 4K still is ideal — the zoom resamples from it, so more "
                     "pixels than the output is exactly the headroom it wants.",
            )
            if upload is not None:
                work = os.path.join(user_exports(), "atmosphere_preview")
                ensure_dir(work)
                path = os.path.join(work, f"upload_{upload.name}")
                with open(path, "wb") as handle:
                    handle.write(upload.getbuffer())
                state["visual_source"] = path
                if not ambient_engine.looks_like_video(path):
                    st.image(path, width="stretch")
                else:
                    st.video(path)
                    st.caption("A video loop keeps its own motion — the drift zoom is "
                               "skipped, because zooming a moving plate reads as a mistake.")

        m1, m2, m3 = st.columns(3)
        with m1:
            state["drift"] = pick(
                "Drift", list(ambient_engine.DRIFT_MODES),
                state.get("drift", "cycle"), "at_drift",
                format_func=lambda k: ambient_engine.DRIFT_MODES[k],
            )
        with m2:
            state["grain"] = st.slider("Grain", 0.0, 16.0,
                                       float(state.get("grain", 6.0)), 1.0, key="at_grain")
        with m3:
            state["vignette"] = st.checkbox("Vignette",
                                            value=bool(state.get("vignette", True)),
                                            key="at_vignette")

        seconds = float(ambient_engine.DURATIONS[state["duration_key"]]["seconds"])
        if state["drift"] == "continuous" and seconds > 3600:
            st.warning(
                f"Continuous drift renders every frame of "
                f"{seconds / 3600:.0f} hours — about "
                f"{ambient_engine.estimate_render_seconds(seconds, 'continuous') / 3600:.1f} "
                f"hours of encoding. The looping drift gives the same 1.00×–1.05× move "
                f"in about a minute, and at this runtime the difference is a zoom of "
                f"0.000002× per frame. Pick it unless you have a reason not to.",
                icon="⏳",
            )

    # ---- Step 3: render ----------------------------------------------------
    with st.container(border=True):
        st.markdown("#### ③ Render")

        seconds = float(ambient_engine.DURATIONS[state["duration_key"]]["seconds"])
        estimate = ambient_engine.estimate_render_seconds(seconds, state["drift"])
        gpu = ambient_engine.has_nvenc()
        stat_row([
            ("Runtime", ambient_engine.DURATIONS[state["duration_key"]]["label"], "cyan"),
            ("Est. render", f"{estimate / 60:.0f} min" if estimate > 90
             else f"{estimate:.0f}s", "amber"),
            ("Encoder", "NVENC" if gpu else "libx264 (CPU)", "green" if gpu else ""),
            ("Est. size", f"{seconds * 4.3 / 8 / 1024:.1f} GB"
             if seconds * 4.3 / 8 / 1024 >= 1 else f"{seconds * 4.3 / 8:.0f} MB", ""),
        ])

        if not gpu:
            st.caption("⚠️ No usable NVENC encoder was detected, so this will fall back "
                       "to libx264. It still works; it is several times slower.")

        if st.button("🌙 Render Atmosphere", type="primary", width="stretch",
                     disabled=not state.get("visual_source")):
            if not state.get("visual_source"):
                st.error("Pick a background first.")
            else:
                tracker = StageProgress(label="Starting...")
                status = st.empty()

                def on_progress(fraction: float, message: str) -> None:
                    tracker.update(fraction, message)
                    status.markdown(f"**{message}**")

                try:
                    stamp = int(time.time())
                    out_path = os.path.join(user_exports(), f"atmosphere_{stamp}.mp4")
                    work = os.path.join(user_exports(), f"atmosphere_{stamp}_work")
                    result = ambient_engine.render_atmosphere(
                        bed=state["bed"], texture=state["texture"],
                        visual_source=str(state["visual_source"]),
                        duration_key=state["duration_key"],
                        output_path=out_path, workspace=work,
                        bed_volume=float(state["bed_volume"]),
                        texture_volume=float(state["texture_volume"]),
                        fps=ambient_engine.DEFAULT_FPS,
                        drift=state["drift"], grain=float(state["grain"]),
                        vignette=bool(state["vignette"]),
                        progress=on_progress,
                    )
                except Exception as exc:
                    tracker.empty()
                    status.empty()
                    st.error(f"**Render failed:** `{type(exc).__name__}: {exc}`")
                    st.code(traceback.format_exc())
                    st.stop()

                elapsed = tracker.finish("Atmosphere complete.")
                status.empty()

                # The render is the user's own work outright: synthesized audio
                # and either a drawn preset or their own upload.
                try:
                    entry = append_ledger(user_exports(), {
                        "video_name": os.path.basename(out_path),
                        "video_path": out_path,
                        "duration": float(result["duration"]),
                        "licence": "own" if state["source_mode"] == "preset" else "unverified",
                        "licence_reference": "",
                        "source_title": atmosphere_soundscape_name(state),
                        "source_author": "", "source_url": "",
                        "source_provider": "reelforge-ambient",
                        "tts_provider": "none", "voice": "",
                        "script_model": "", "ai_disclosed": True,
                        "script": "", "template": "atmosphere",
                    })
                    state["entry_name"] = entry["video_name"]
                except Exception:
                    state["entry_name"] = os.path.basename(out_path)

                state["result"] = result
                state["render_path"] = out_path
                notify_complete(
                    "Atmosphere rendered",
                    f"{result['duration_label']} — {result['size_bytes'] / 1_073_741_824:.2f} GB "
                    f"in {elapsed / 60:.0f} min")
                st.balloons()
                st.rerun()

        result = state.get("result")
        if result and os.path.exists(str(state.get("render_path") or "")):
            st.success(
                f"{os.path.basename(state['render_path'])} — "
                f"{result['duration'] / 3600:.2f} h, "
                f"{result['size_bytes'] / 1_073_741_824:.2f} GB, "
                f"{result['encoder']}, rendered in {result['render_seconds'] / 60:.1f} min",
                icon="✅")
            stat_row([
                ("Audio loop", f"{result['loop_seconds']:.0f}s", "cyan"),
                ("Seeds", str(result["variants"]), ""),
                ("Loops", str(result["audio_loops"]), ""),
                ("Drift", ambient_engine.DRIFT_MODES.get(result["drift"], "—"), "amber"),
            ])
            # A multi-gigabyte file is not something to push through the
            # browser -- the path and the publisher are the way out.
            st.caption(f"Saved to `{state['render_path']}`. Files this size are not "
                       f"offered as a browser download; publish it below or copy it "
                       f"from disk.")

    # ---- Step 4: publish ---------------------------------------------------
    render_atmosphere_publisher()
    render_atmosphere_monetization()


def render_atmosphere_publisher() -> None:
    """Authorize a channel, write the metadata, upload."""
    state = _at()
    path = str(state.get("render_path") or "")

    with st.expander("④ 📺 Publish to YouTube", expanded=bool(path)):
        status = publisher.account_status()

        # --- connection ----------------------------------------------------
        if status["connected"]:
            st.markdown(
                badge("channel connected", "green")
                + badge(status["channel"] or "authorized", "cyan"),
                unsafe_allow_html=True)
            if st.button("Disconnect this channel", key="at_disconnect"):
                publisher.forget_channel()
                st.rerun()
        else:
            st.markdown(badge("not connected", "amber"), unsafe_allow_html=True)
            if status["error"]:
                st.error(status["error"])

            st.caption(
                "One-time setup: in Google Cloud Console create a project, enable "
                "**YouTube Data API v3**, then create an **OAuth client ID** of type "
                "**Desktop app** and upload the JSON it gives you."
            )
            secret = st.file_uploader("client_secrets.json", type=["json"],
                                      key="at_secret")
            if secret is not None:
                try:
                    publisher.save_client_secrets(secret.getvalue())
                    st.success("Client secret stored in `.secrets/` (gitignored).")
                except publisher.PublishError as exc:
                    st.error(str(exc))

            if publisher.has_client_secrets():
                st.caption("Authorizing opens a Google consent page in a browser on "
                           "**this machine**. Over a tunnel or on a headless server "
                           "that will not work — generate the token locally and copy "
                           "`.secrets/youtube_token.json` across.")
                if st.button("🔐 Authorize channel", key="at_authorize", type="primary"):
                    with st.spinner("Waiting for Google consent in your browser..."):
                        try:
                            publisher.authorize()
                            st.success("Channel authorized.")
                            st.rerun()
                        except publisher.PublishError as exc:
                            st.error(str(exc))

        if not path:
            st.info("Render something first and the uploader will open here.", icon="🎬")
            return

        divider()

        # --- metadata -------------------------------------------------------
        soundscape = atmosphere_soundscape_name(state)
        seconds = float((state.get("result") or {}).get("duration") or 0.0)

        if st.button("✨ Auto-Generate High-CTR Title & Description",
                     key="at_seo", width="stretch"):
            with st.spinner("Writing for search intent..."):
                try:
                    meta = publisher.generate_seo(soundscape, seconds)
                except Exception as exc:
                    st.warning(f"Gemini unavailable ({type(exc).__name__}); using the "
                               f"offline template instead.")
                    meta = publisher.fallback_seo(soundscape, seconds)
                state["meta"] = meta
                st.rerun()

        meta = state.get("meta") or publisher.fallback_seo(soundscape, seconds)
        if state.get("meta"):
            st.caption(f"Written by {meta.get('model', '?')}")

        title = st.text_input("Title", value=meta["title"], key="at_title",
                              max_chars=publisher.TITLE_LIMIT)
        description = st.text_area("Description", value=meta["description"],
                                   height=200, key="at_desc")
        tags_raw = st.text_area("Tags (comma separated)",
                                value=", ".join(meta["tags"]), height=80, key="at_tags")
        tags = publisher.parse_tags(tags_raw)

        c1, c2 = st.columns(2)
        with c1:
            category = st.selectbox(
                "Category", list(publisher.CATEGORIES),
                index=list(publisher.CATEGORIES).index(publisher.DEFAULT_CATEGORY),
                format_func=lambda k: f"{k} · {publisher.CATEGORIES[k]}",
                key="at_category")
        with c2:
            privacy = st.selectbox(
                "Privacy", list(publisher.PRIVACY_STATUSES),
                index=list(publisher.PRIVACY_STATUSES).index(publisher.DEFAULT_PRIVACY),
                format_func=lambda k: publisher.PRIVACY_STATUSES[k],
                key="at_privacy")

        publish_at = ""
        if privacy == "scheduled":
            import datetime as _dt

            d1, d2 = st.columns(2)
            with d1:
                when_date = st.date_input(
                    "Publish date",
                    value=_dt.date.today() + _dt.timedelta(days=1), key="at_date")
            with d2:
                when_time = st.time_input("Publish time (your local time)",
                                          value=_dt.time(20, 0), key="at_time")
            publish_at = publisher.to_rfc3339(
                _dt.datetime.combine(when_date, when_time))
            st.caption(f"Sent to YouTube as `{publish_at}` — it stays private until then.")

        notify = st.checkbox("Notify subscribers", value=True, key="at_notify")

        problems = publisher.validate_metadata(title, description, tags)
        counts = (f"title {len(title)}/{publisher.TITLE_LIMIT} · "
                  f"description {len(description)}/{publisher.DESCRIPTION_LIMIT} · "
                  f"tags {sum(len(t) for t in tags)}/{publisher.TAGS_TOTAL_LIMIT}")
        st.caption(counts)
        for problem in problems:
            st.error(problem, icon="🚫")

        size_gb = os.path.getsize(path) / 1_073_741_824 if os.path.exists(path) else 0.0
        ready = status["connected"] and not problems and os.path.exists(path)

        if st.button(f"🚀 Upload {size_gb:.2f} GB to YouTube", key="at_upload_go",
                     type="primary", width="stretch", disabled=not ready):
            tracker = StageProgress(label="Preparing upload...")
            line = st.empty()

            def on_progress(fraction: float, message: str) -> None:
                # A negative fraction is a retry notice, not progress.
                if fraction >= 0:
                    tracker.update(fraction, message)
                line.markdown(f"**{message}**")

            try:
                uploaded = publisher.upload_video(
                    path, title=title, description=description, tags=tags,
                    category_id=category, privacy=privacy, publish_at=publish_at,
                    notify_subscribers=notify, progress=on_progress,
                )
            except publisher.PublishError as exc:
                tracker.empty()
                line.empty()
                st.error(str(exc), icon="🚫")
                return
            except Exception as exc:
                tracker.empty()
                line.empty()
                st.error(f"**Upload failed:** `{type(exc).__name__}: {exc}`")
                st.code(traceback.format_exc())
                return

            tracker.finish("Uploaded.")
            line.empty()
            state["uploaded"] = uploaded
            notify_complete("Upload complete", f"{title[:60]} is on YouTube")
            st.rerun()

        uploaded = state.get("uploaded")
        if uploaded:
            st.success(
                f"Uploaded as **{uploaded['privacy']}** in "
                f"{uploaded['seconds'] / 60:.1f} min — {uploaded['url']}", icon="✅")
            st.markdown(f"[Open the video]({uploaded['url']}) · "
                        f"[Edit in Studio]({uploaded['studio_url']})")
            if uploaded.get("publish_at"):
                st.caption(f"Goes public at {uploaded['publish_at']}.")

# ---------------------------------------------------------------------------
# Niche Scout
#
# Research and strategy, and a feeder into the production modes. It renders
# nothing and imports no rendering engine; the handoff buttons write plain
# dicts into the target mode's session state.
# ---------------------------------------------------------------------------

def _ns() -> dict[str, Any]:
    """Niche Scout's own state, kept out of the widget keys."""
    return st.session_state.scout


def scout_send_to_batch(topic: dict[str, Any]) -> str:
    """Queues an episode in Batch Studio. Returns the line that was queued."""
    line = niche_engine.batch_handoff(topic)
    queue: list[dict[str, Any]] = st.session_state.setdefault("batch_queue", [])
    if not any(str(job.get("topic")) == line for job in queue):
        queue.append(_batch_job(line))
    return line


def scout_send_to_narrative(topic: dict[str, Any], band: str) -> None:
    """
    Pre-fills Narrative Studio with an episode.

    The widget keys have to go, not just the state: a Streamlit widget whose
    key already exists ignores its `value=` argument, so writing the topic into
    `state` alone would leave the old premise on screen. This is the same trap
    the duel presets hit.
    """
    payload = niche_engine.narrative_handoff(topic, band)
    state = st.session_state.narrative
    state.update(payload)
    # A new premise invalidates the script and storyboard built from the old one.
    state.update({"episode": None, "result": None, "pack": "", "entry_name": ""})
    for key in ("nv_topic", "nv_look", "nv_tone", "nv_format"):
        st.session_state.pop(key, None)


def _scout_topic_actions(topic: dict[str, Any], band: str, key: str,
                         allowed: Sequence[str]) -> None:
    """The three handoff buttons under one episode concept."""
    can_batch = "batch" in allowed
    can_narrative = "narrative" in allowed

    left, right = st.columns(2)
    with left:
        if st.button("⚡ Send to Batch Studio", key=f"ns_batch_{key}", width="stretch",
                     disabled=not can_batch,
                     help=None if can_batch else "Your role does not have Batch Studio."):
            line = scout_send_to_batch(topic)
            st.toast(f"Queued: {line[:48]}", icon="⚡")
    with right:
        if st.button("🎬 Produce in Narrative Studio", key=f"ns_narr_{key}",
                     width="stretch", disabled=not can_narrative,
                     help=None if can_narrative
                     else "Your role does not have Narrative Studio."):
            scout_send_to_narrative(topic, band)
            _ns()["handoff"] = topic.get("hook_title", "")
            st.toast("Narrative Studio is pre-filled — switch to it in the sidebar.",
                     icon="🎬")

    with st.expander("📋 Copy full script brief & SEO"):
        # st.code carries Streamlit's own copy button, which is more reliable
        # than a clipboard shim and works over the tunnel.
        st.code(niche_engine.seo_block(topic), language="text")
        chapters = niche_engine.chapters_of(str(topic.get("description") or ""))
        if chapters:
            st.caption(f"{len(chapters)} chapters, first at {chapters[0][0]} — "
                       f"YouTube only builds a clickable list when each timestamp "
                       f"starts its own line and the first is 0:00.")


def render_scout_scorecard(assessment: dict[str, Any]) -> None:
    """The opportunity card: CPM band, saturation meter, verdict."""
    cpm = assessment["cpm"]
    saturation = assessment["saturation"]

    with st.container(border=True):
        st.markdown("#### ② Opportunity scorecard")

        head, meter = st.columns([1.1, 1.4])
        with head:
            tone = ("green" if assessment["overall"] >= 7.5
                    else "amber" if assessment["overall"] >= 5.0 else "red")
            st.markdown(
                f'<div class="rf-score {tone}"><span class="n">{assessment["overall"]:.1f}</span>'
                f'<span class="d">/ 10</span></div>'
                f'<div class="rf-score-verdict">{assessment["verdict"]}</div>'
                f'<div class="rf-score-src">{assessment.get("model", "")}</div>',
                unsafe_allow_html=True)
        with meter:
            measured = saturation["measured"]
            st.markdown(
                f'<div class="rf-axis {"green" if saturation["score"] <= 35 else "amber" if saturation["score"] <= 60 else "red"}">'
                f'<div class="k">Saturation {"(measured)" if measured else "(no data)"}</div>'
                f'<div class="v">{saturation["score"]}<span style="font-size:.6em">/100</span></div>'
                f'<div class="rf-axis-bar"><i style="width:{saturation["score"]}%"></i></div>'
                f'</div>', unsafe_allow_html=True)
            st.caption(saturation["verdict"])
            if measured:
                signals = saturation["signals"]
                st.caption(
                    f"From {signals['channels_sampled']} channels: "
                    f"{signals['small_channel_share']:.0%} under 50k subs, "
                    f"median video reaches {signals['median_view_to_sub']:.1f}× the "
                    f"subscriber count, leader is {signals['leader_concentration']:.1f}× "
                    f"the rest.")
            else:
                st.caption("Run Competitor Recon below to measure this. Without it "
                           "the score is a placeholder, not an estimate.")

        stat_row([
            ("CPM band", f"${cpm['low']:.1f}–${cpm['high']:.0f}", "violet"),
            ("Category", cpm["label"], "cyan"),
            ("Evergreen", f"{assessment['evergreen']['score']:.0f}/10", ""),
            ("Faceless fit", f"{assessment['faceless_fit']['score']:.0f}/10", ""),
        ])
        st.caption(f"💰 {cpm['why']}")
        st.caption(f"⚠️ {cpm['caveat']}")

        if assessment.get("summary"):
            st.markdown(assessment["summary"])

        a, b = st.columns(2)
        with a:
            st.markdown("**Risks**")
            for risk in assessment.get("risks", []):
                st.markdown(f"- 🚩 {risk}")
        with b:
            st.markdown("**Angles that would differentiate you**")
            for angle in assessment.get("angles", []):
                st.markdown(f"- 🎯 {angle}")


def render_scout_recon(state: dict[str, Any]) -> None:
    """Real channels working the niche, and what is working for them."""
    with st.container(border=True):
        st.markdown("#### 🔭 Competitor recon")
        st.caption(f"Real channels ranking for this niche in the last "
                   f"{niche_engine.RECON_WINDOW_DAYS} days, from the YouTube Data API.")

        if not niche_engine.has_youtube_key():
            st.info(
                "Needs a **YouTube Data API v3** key, which is not the Gemini key — "
                "an AI Studio key returns 401 here. In Google Cloud Console: enable "
                "YouTube Data API v3, create an API key, then put it in `.env` as "
                "`YOUTUBE_API_KEY`. Everything else in Niche Scout works without it; "
                "only the saturation score goes unmeasured.",
                icon="🔑")
            return

        if st.button("🔭 Run recon", key="ns_recon", width="stretch"):
            note = st.empty()
            try:
                with st.spinner("Reading YouTube..."):
                    state["recon"] = niche_engine.competitor_recon(
                        state["topic"], progress=lambda m: note.caption(m))
            except niche_engine.NicheError as exc:
                note.empty()
                st.error(str(exc), icon="🚫")
                return
            note.empty()
            # A fresh recon means the saturation score can now be measured.
            if state.get("assessment"):
                state["assessment"]["saturation"] = niche_engine.saturation_from_recon(
                    state["recon"])
            st.rerun()

        recon = state.get("recon") or {}
        channels = recon.get("channels") or []
        if not channels:
            return

        st.caption(f"{len(channels)} channels · {recon.get('quota_units', 0)} quota units "
                   f"spent of the 10,000/day default")

        loose = list(recon.get("loose_match") or ())
        if loose:
            st.caption(
                f"⚠️ {', '.join(loose[:3])} ranked once for this query and "
                f"{'write' if len(loose) > 1 else 'writes'} mostly about something "
                f"else. Shown rather than dropped — but discount "
                f"{'them' if len(loose) > 1 else 'it'} when reading the saturation "
                f"score, because a large off-niche channel inflates the leader "
                f"concentration signal.")

        for channel in channels:
            with st.container(border=True):
                subs = ("hidden" if channel["hidden_subs"]
                        else f"{channel['subscribers']:,} subs")
                st.markdown(
                    f"**[{channel['name']}]({channel['url']})** &nbsp; "
                    + badge(subs, "cyan")
                    + badge(f"median {channel['median_views']:,} views", "violet")
                    + badge(f"{channel['view_to_sub']:.1f}× subs", "green"
                            if channel["view_to_sub"] >= 1 else "")
                    + badge(f"{channel['videos_in_window']} uploads", "")
                    # Flagged rather than filtered. A big general-interest
                    # channel that ranked once on one video would otherwise sit
                    # here looking like a competitor and drag the leader
                    # concentration signal with it.
                    + (badge("loose match", "amber")
                       if (channel.get("search_hits", 0) < 2
                           and channel.get("topical_share", 0)
                           < niche_engine.RECON_TOPICAL_MIN) else ""),
                    unsafe_allow_html=True)
                for video in channel["top_videos"]:
                    st.markdown(
                        f"- [{video['title']}](https://youtu.be/{video['video_id']}) — "
                        f"{video['views']:,} views, {video['published']}")


def render_niche_scout() -> None:
    """Discover or validate a niche, then plan and hand off episodes."""
    state = _ns()
    role = auth.normalise_role(current_user().get("role"))
    allowed = auth.allowed_modes(role)

    if state.get("handoff"):
        st.success(f"Narrative Studio is pre-filled with **{state['handoff']}** — "
                   f"switch to it in the sidebar.", icon="🎬")
        state["handoff"] = ""

    # ---- Step 1: the entry flow -------------------------------------------
    with st.container(border=True):
        st.markdown("#### ① What are you looking for?")
        state["flow"] = pick(
            "Mode", ["validate", "ideas"], state.get("flow", "validate"), "ns_flow",
            format_func=lambda k: {"validate": "🔍 Validate my niche",
                                   "ideas": "💡 I need ideas"}[k])

        if state["flow"] == "validate":
            topic = st.text_input(
                "Your niche", value=str(state.get("topic") or ""), key="ns_topic",
                placeholder="e.g. Ancient mysteries and lost cities",
                help="A format, not just a subject. 'Declassified military logistics, "
                     "10-minute archival documentaries' beats 'history'.")
            if topic.strip():
                band = niche_engine.classify_band(topic)
                preview = niche_engine.cpm_estimate(topic)
                st.markdown(
                    badge(preview["label"], "cyan")
                    + badge(f"${preview['low']:.1f}–${preview['high']:.0f} CPM band", "violet"),
                    unsafe_allow_html=True)

            if st.button("🔍 Validate this niche", type="primary", width="stretch",
                         key="ns_validate", disabled=not topic.strip()):
                note = st.empty()
                try:
                    with st.spinner("Assessing..."):
                        state["topic"] = topic.strip()
                        state["assessment"] = niche_engine.validate_niche(
                            topic, state.get("recon"), progress=lambda m: note.caption(m))
                        state["buckets"] = None
                except niche_engine.NicheError as exc:
                    note.empty()
                    st.error(str(exc), icon="🚫")
                    st.stop()
                note.empty()
                st.rerun()
        else:
            st.caption("Twelve niches that work without a face on camera, each with "
                       "its honest downsides — the cons are the useful half.")
            if st.button("💡 Suggest 12 niches", type="primary", width="stretch",
                         key="ns_ideas"):
                note = st.empty()
                try:
                    with st.spinner("Researching..."):
                        state["ideas"] = niche_engine.suggest_niches(
                            progress=lambda m: note.caption(m))
                except niche_engine.NicheError as exc:
                    note.empty()
                    st.error(str(exc), icon="🚫")
                    st.stop()
                note.empty()
                st.rerun()

    # ---- the idea list ----------------------------------------------------
    ideas = state.get("ideas") or {}
    if state["flow"] == "ideas" and ideas.get("niches"):
        st.markdown('<div class="rf-section">12 faceless niches</div>',
                    unsafe_allow_html=True)
        rows = ideas["niches"]
        for start in range(0, len(rows), 2):
            cols = st.columns(2)
            for col, niche in zip(cols, rows[start:start + 2]):
                with col, st.container(border=True):
                    cpm = niche["cpm"]
                    st.markdown(f"**{niche['name']}**")
                    st.markdown(
                        badge(f"${cpm['low']:.1f}–${cpm['high']:.0f}", "violet")
                        + badge(cpm["label"], "cyan"), unsafe_allow_html=True)
                    st.caption(niche["format"])
                    for pro in niche["pros"]:
                        st.markdown(f"- ✅ {pro}")
                    for con in niche["cons"]:
                        st.markdown(f"- ⚠️ {con}")
                    if niche["monetization"]:
                        st.caption(f"💰 {niche['monetization']}")
                    if st.button("🔍 Validate this one", key=f"ns_pick_{niche['name'][:40]}",
                                 width="stretch"):
                        state["topic"] = niche["name"]
                        state["flow"] = "validate"
                        state["assessment"] = None
                        state["buckets"] = None
                        state["recon"] = None
                        st.session_state.pop("ns_topic", None)
                        st.session_state.pop("ns_flow", None)
                        st.rerun()

    # ---- Step 2: the scorecard --------------------------------------------
    assessment = state.get("assessment")
    if not assessment:
        return

    render_scout_scorecard(assessment)
    render_scout_recon(state)

    # ---- Step 3: the content plan -----------------------------------------
    with st.container(border=True):
        st.markdown("#### ③ Content buckets")
        st.caption("Seven recurring pillars, three researched episodes under each — "
                   "with the hook, the psychological angle, and a complete SEO package.")

        if st.button("🗂️ Generate 7 content buckets", type="primary", width="stretch",
                     key="ns_buckets"):
            note = st.empty()
            try:
                with st.spinner("Planning 21 episodes..."):
                    state["buckets"] = niche_engine.content_buckets(
                        state["topic"], assessment, progress=lambda m: note.caption(m))
            except niche_engine.NicheError as exc:
                note.empty()
                st.error(str(exc), icon="🚫")
                st.stop()
            note.empty()
            st.rerun()

    buckets = state.get("buckets") or {}
    pillars = buckets.get("pillars") or []
    if not pillars:
        return

    band = assessment["cpm"]["band"]
    total = sum(len(p["topics"]) for p in pillars)
    st.markdown(
        f'<div class="rf-section">{len(pillars)} pillars · {total} episodes</div>',
        unsafe_allow_html=True)

    for index, pillar in enumerate(pillars):
        with st.expander(f"▸ {pillar['pillar']} — {pillar['premise'][:90]}",
                         expanded=index == 0):
            for slot, topic in enumerate(pillar["topics"]):
                with st.container(border=True):
                    st.markdown(f"**{topic['hook_title']}**")
                    st.caption(f"🧠 {topic['why_it_works']}")
                    tags = topic.get("tags") or []
                    st.markdown(
                        badge(f"{len(tags)} tags", "cyan")
                        + badge(f"{sum(len(t) for t in tags)}/{niche_engine.TAGS_TOTAL_LIMIT} chars",
                                "green" if sum(len(t) for t in tags) <= niche_engine.TAGS_TOTAL_LIMIT
                                else "amber"),
                        unsafe_allow_html=True)
                    _scout_topic_actions(topic, band, f"{index}_{slot}", allowed)

# ---------------------------------------------------------------------------
# Dashboard
#
# The landing view. Everything on it is either a live reading or a way into a
# mode -- there is no content here that is only decoration, because a home
# screen you scroll past is a home screen that costs a click on every visit.
#
# The KPI bar is deliberately NOT redrawn here: render_command_center() is
# already painted above every page by main(), and a second copy would collide
# on the purge button's widget key.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Magic Studio: one prompt, one style, one finished video
#
# The planner lives in magic_studio.py; this is the part that has to touch
# Streamlit. Each style calls the same pipeline its manual mode calls rather
# than a copy of it -- a second render path would be the one without the tests
# on it, and it would drift within a week.
# ---------------------------------------------------------------------------

# Narration voices for the one-click flow, per provider. They are NOT
# interchangeable: edge-tts takes Microsoft neural voice ids
# ("en-US-ChristopherNeural") and Gemini TTS takes its own short names
# ("Charon"). Handing one to the other fails with a 400 at request time --
# after the script has been written and paid for, which is the worst moment for
# it. Each runner below passes the constant that matches the provider its
# engine actually uses.
MAGIC_GEMINI_VOICE = "Charon"      # build_minimalist_video, run_batch_job
MAGIC_EDGE_VOICE = DEFAULT_VOICE   # generate_slide_voiceovers, via _synthesize


def magic_context() -> dict[str, Any]:
    """
    Everything a render needs, copied off session_state on this thread.

    The worker cannot read session_state -- there is no ScriptRunContext on it
    -- so the snapshot has to be taken here, while we are still inside the
    script run, and handed over as plain data.
    """
    audio = dict(st.session_state.get("audio_settings") or {})
    voice = dict(st.session_state.get("voice_settings") or {})
    exports = user_exports()
    os.makedirs(exports, exist_ok=True)

    return {
        "exports": exports,
        "aspect": str(st.session_state.get("render_aspect") or next(iter(ASPECT_RATIOS))),
        "fit": str(st.session_state.get("render_fit") or DEFAULT_FIT),
        "transition": str(st.session_state.get("render_transition") or "crossfade"),
        "transition_dur": float(st.session_state.get("render_transition_dur") or 0.5),
        "fps": int(st.session_state.get("render_fps") or 24),
        "watermark": str(st.session_state.get("render_watermark") or ""),
        "target": str(st.session_state.get("magic_target") or DEFAULT_TARGET),
        "music_style": str(audio.get("style") or "lofi"),
        "music_volume": float(audio.get("volume", 0.8)),
        "voice_volume": float(voice.get("volume", 1.0)),
        "music_duck": float(voice.get("music_duck", 0.28)),
        "sfx_enabled": bool(voice.get("sfx_enabled", True)),
        "sfx_volume": float(voice.get("sfx_volume", 0.55)),
        "rate_pct": int(voice.get("rate_pct", 12)),
        "pitch_hz": int(voice.get("pitch_hz", 0)),
    }


# Narration voices for the one-click flow, per provider. They are NOT
# interchangeable: edge-tts takes Microsoft neural voice ids
# ("en-US-ChristopherNeural") and Gemini TTS takes its own short names
# ("Charon"). Handing one to the other fails with a 400 at request time --
# after the script has been written and paid for, which is the worst moment for
# it. Each runner below passes the constant that matches the provider its
# engine actually uses.
MAGIC_GEMINI_VOICE = "Charon"      # build_minimalist_video, run_batch_job
MAGIC_EDGE_VOICE = DEFAULT_VOICE   # generate_slide_voiceovers

# What a runner is handed: (prompt, context, report). `report` is
# MagicJob.report -- a plain callable, safe to hold across threads.
Report = Callable[..., None]


def _magic_out(ctx: dict[str, Any], prefix: str) -> str:
    return os.path.join(ctx["exports"], f"{prefix}_{int(time.time())}.mp4")


def _magic_minimalist(prompt: str, ctx: dict[str, Any], report: Report) -> dict[str, Any]:
    """
    Vector animation: scene spec, narration, drawn frames, encode.

    The engine this project leans on hardest. Every frame is computed and every
    audio layer synthesized, so there is no third-party rights holder anywhere
    in the output -- which is what makes it the one engine that cannot be
    caught by a reused-content review.
    """
    concept = magic_studio.scene_concept(prompt)

    report("script", 0.2, "Writing Hook \u2014 designing the metaphor...")
    spec = generate_scene_spec(concept)
    report("script", 1.0)

    out_path = _magic_out(ctx, "minimalist")

    def on_step(step: int, total: int, message: str) -> None:
        # The engine synthesizes narration first and then draws, so its early
        # steps are the voice stage and the rest is picture.
        share = step / max(1, total)
        if share < 0.25:
            report("voice", share / 0.25, f"Synthesizing Voice \u2014 {message}")
        elif share < 0.75:
            report("visual", (share - 0.25) / 0.5, f"Animating \u2014 {message}")
        else:
            report("render", (share - 0.75) / 0.25, f"Finalizing Render \u2014 {message}")

    result = build_minimalist_video(
        spec, out_path,
        fps=int(ctx["fps"]),
        bgm=True, bgm_volume=0.30, sfx=True, narrate=True,
        voice=MAGIC_GEMINI_VOICE,
        progress_callback=on_step,
    )

    entry = append_ledger(ctx["exports"], {
        "video_name": os.path.basename(out_path), "video_path": out_path,
        "duration": float(result["duration"]),
        "licence": "own", "licence_reference": "",
        "source_title": "Procedural vector animation",
        "source_author": "", "source_url": "", "source_provider": "reelforge",
        "tts_provider": str(result.get("tts_provider") or ""),
        "voice": str(result.get("voice") or ""),
        "script_model": str(spec.get("source") or "gemini"),
        "ai_disclosed": True,
        "script": str(spec.get("thesis") or concept),
        "magic_prompt": prompt,
    })
    return {"path": out_path, "duration": float(result["duration"]),
            "script": str(spec.get("thesis") or concept), "entry": entry}


def _magic_duel(prompt: str, ctx: dict[str, Any], report: Report) -> dict[str, Any]:
    """Comparison reel: matchup, photographs, stat rounds, encode."""
    report("script", 0.2, "Writing Hook \u2014 building the matchup...")
    brief = magic_studio.duel_brief(prompt)
    report("script", 1.0)

    duel: dict[str, Any] = {"preset": "", "layout": "stacked",
                            "headline": brief["headline"], "cta": brief["cta"],
                            "rounds": brief["rounds"]}

    report("visual", 0.1, "Animating \u2014 finding photographs...")
    for index, side in enumerate(("a", "b")):
        item = dict(brief[side])
        try:
            # Not the cached search: st.cache_data needs a script context.
            item["image"], item["credit"] = fetch_photo(item["name"])
        except Exception as exc:
            item["image"] = fallback_backdrop()
            item["credit"] = f"Photo unavailable ({type(exc).__name__})"
        item["query"] = item["name"]
        duel[side] = item
        report("visual", 0.1 + 0.3 * (index + 1))

    slides = build_duel_slides(duel)

    report("voice", 0.3, "Synthesizing Voice \u2014 narrating the rounds...")
    generate_slide_voiceovers(
        slides, voice=MAGIC_EDGE_VOICE,
        rate=f"{int(ctx['rate_pct']):+d}%", pitch=f"{int(ctx['pitch_hz']):+d}Hz",
    )

    # The same two engine calls run_render_pipeline makes, without the widgets
    # it draws around them.
    report("render", 0.02, "Finalizing Render \u2014 building the music bed...")
    runtime = sum(float(s.get("duration") or 3.5) for s in slides)
    stereo, rate = generate_synth_music(style=ctx["music_style"], duration=runtime + 2.0)
    music_path = save_wav_to_file(
        stereo, rate, os.path.join(ctx["exports"], f"music_{int(time.time())}.wav"))

    out_path = _magic_out(ctx, "duel")

    def on_step(step: int, total: int, message: str) -> None:
        report("render", 0.05 + 0.95 * (step / max(1, total)),
               f"Finalizing Render \u2014 {message}")

    result = build_reel_video(
        slides=slides,
        aspect_ratio_name=ctx["aspect"],
        fit_mode=ctx["fit"],
        transition_type=ctx["transition"],
        transition_dur=ctx["transition_dur"],
        audio_path=music_path,
        audio_volume=ctx["music_volume"],
        watermark_text=ctx["watermark"],
        fps=int(ctx["fps"]),
        output_path=out_path,
        progress_callback=on_step,
        voiceover_volume=ctx["voice_volume"],
        music_duck=ctx["music_duck"],
        sfx_enabled=ctx["sfx_enabled"],
        sfx_volume=ctx["sfx_volume"],
    )

    script = " ".join(str(s.get("voiceover") or "") for s in slides).strip()
    entry = append_ledger(ctx["exports"], {
        "video_name": os.path.basename(out_path), "video_path": out_path,
        "duration": float(result["duration"]),
        "licence": DEFAULT_LICENCE, "licence_reference": "",
        "source_title": duel["headline"],
        "source_author": "", "source_url": "", "source_provider": "pexels",
        "tts_provider": "edge", "voice": MAGIC_EDGE_VOICE,
        "script_model": str(brief.get("model") or ""),
        "ai_disclosed": True, "script": script, "magic_prompt": prompt,
    })
    return {"path": out_path, "duration": float(result["duration"]),
            "script": script, "entry": entry}


def _magic_commentary(prompt: str, ctx: dict[str, Any], report: Report) -> dict[str, Any]:
    """
    Licensed footage under an editorial script, captions burned in.

    There is no commentary_engine.py: this pipeline is split between
    gemini_engine (which watches the clip and writes), video_engine (which
    reframes, loops, captions and encodes) and demo_data (which finds the
    licensed footage). run_batch_job is the orchestration, and it is already
    free of Streamlit by design because Batch Studio runs it unattended.
    """
    stages = {"footage": ("script", 0.15), "script": ("script", 0.9),
              "voice": ("voice", 0.5), "render": ("render", 0.05)}

    def on_stage(name: str) -> None:
        key, within = stages.get(name, ("render", 0.5))
        report(key, within)

    job = _batch_job(prompt)
    settings = {
        "aspect": ctx["aspect"], "fit": ctx["fit"], "fps": int(ctx["fps"]),
        "watermark": ctx["watermark"], "target": ctx["target"],
        "angle": "suspense", "tts_provider": "gemini",
        "voice": MAGIC_GEMINI_VOICE, "kinetic": True, "bgm": True,
        "ai_disclosed": True, "exports": ctx["exports"],
    }
    job = run_batch_job(job, settings, on_stage=on_stage)
    return {"path": str(job["video_path"]), "duration": float(job["duration"]),
            "script": str(job["script"]),
            "entry": find_entry(ctx["exports"], str(job["video_name"])) or {}}


def _magic_atmosphere(prompt: str, ctx: dict[str, Any], report: Report) -> dict[str, Any]:
    """Long-form ambient: bed, texture, drifting canvas, stream-copy mux."""
    plan = magic_studio.atmosphere_plan(prompt)
    workspace = ctx["exports"]

    report("script", 1.0, "Writing Hook \u2014 choosing the soundscape...")

    canvas = {"rain_window": "rain_glass", "thunderstorm": "deep_dark",
              "fireplace": "ember", "stream": "mist",
              "brown_noise": "deep_dark"}.get(plan["bed"], ambient_engine.DEFAULT_CANVAS)

    report("visual", 0.2, "Animating \u2014 painting the canvas...")
    still = ambient_engine.build_canvas_preset(canvas, workspace)

    out_path = _magic_out(ctx, "atmosphere")

    def on_progress(message: str) -> None:
        low = message.lower()
        if "sound" in low or "audio" in low or "seed" in low:
            report("voice", 0.5, f"Synthesizing Voice \u2014 {message}")
        elif "visual" in low or "frame" in low:
            report("visual", 0.7, f"Animating \u2014 {message}")
        else:
            report("render", 0.4, f"Finalizing Render \u2014 {message}")

    result = ambient_engine.render_atmosphere(
        bed=plan["bed"], texture=plan["texture"], visual_source=still,
        duration_key=plan["duration"], output_path=out_path, workspace=workspace,
        progress=on_progress,
    )

    entry = append_ledger(workspace, {
        "video_name": os.path.basename(out_path), "video_path": out_path,
        "duration": float(result["duration"]),
        "licence": "own", "licence_reference": "",
        "source_title": "Synthesized ambient soundscape",
        "source_author": "", "source_url": "", "source_provider": "reelforge",
        "tts_provider": "", "voice": "", "script_model": "procedural",
        "ai_disclosed": True, "script": "", "magic_prompt": prompt,
    })
    return {"path": out_path, "duration": float(result["duration"]),
            "script": "", "entry": entry}


_MAGIC_RUNNERS: dict[str, Callable[..., dict[str, Any]]] = {
    "minimalist": _magic_minimalist,
    "duel": _magic_duel,
    "commentary": _magic_commentary,
    "atmosphere": _magic_atmosphere,
}


def start_magic_job(prompt: str, style_key: str,
                    ctx: dict[str, Any]) -> magic_studio.MagicJob:
    """
    Starts the render on a worker thread and returns immediately.

    Daemon, so a render in flight never stops the server from shutting down.
    Nothing inside `work` may call st.* -- see the note on MagicJob.
    """
    job = magic_studio.MagicJob(prompt, style_key)
    runner = _MAGIC_RUNNERS[style_key]

    def work() -> None:
        try:
            result = runner(prompt, ctx, job.report)
            path = str(result.get("path") or "")
            if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
                raise FileNotFoundError(
                    "The render finished but produced no file.")
            with open(path, "rb") as handle:
                data = handle.read()
            job.finish({
                "prompt": prompt, "style": style_key, "path": path,
                "name": os.path.basename(path), "bytes": data,
                "duration": float(result.get("duration") or 0.0),
                "script": str(result.get("script") or ""),
                "entry": result.get("entry") or {},
            })
        except Exception as exc:                              # noqa: BLE001
            job.fail(exc)

    thread = threading.Thread(target=work, name=f"magic-{style_key}", daemon=True)
    job.thread = thread
    thread.start()
    return job


# How long a script run waits before drawing the progress again. Short enough
# that the bar moves, long enough that the poll is not the load.
MAGIC_POLL_SECONDS = 0.7


def render_magic_studio(allowed: Sequence[str]) -> None:
    """
    One engine, four starting points, one button.

    This used to offer four styles behind a style picker. It offers one now,
    because only one of them produces something a reused-content review cannot
    touch: every frame computed, every audio layer synthesized, no third-party
    rights holder anywhere in the file. The other runners are still here and
    still tested -- their modes moved to the archive drawer -- but a landing
    page that asks which engine you want is a landing page that has not decided
    what the product is.
    """
    if magic_studio.PRIMARY_STYLE not in allowed:
        return

    style = magic_studio.style(magic_studio.PRIMARY_STYLE)
    low = int(gemini_engine.SCENE_MIN_SECONDS)
    high = int(gemini_engine.SCENE_MAX_SECONDS)

    st.markdown('<div class="rf-magic">'
                f'<div class="rf-magic-title">\u2728 Magic Studio \u00b7 '
                f'{style["label"]}</div>'
                '<div class="rf-magic-sub">Type a topic and press one button. '
                'Every frame is drawn from code and every audio layer is '
                'synthesized, so there is nothing in the file to license and '
                'nothing to claim.</div>'
                '</div>', unsafe_allow_html=True)

    job = st.session_state.get("magic_job")
    if job is not None:
        render_magic_progress(job, allowed)
        return

    # Quick topics. Buttons rather than a pills widget on purpose: these set
    # the contents of another widget, and a selection widget would then hold a
    # second, competing piece of state for the same choice.
    st.caption("Start from one of these, or write your own:")
    for column, topic in zip(st.columns(len(magic_studio.QUICK_TOPICS)),
                             magic_studio.QUICK_TOPICS):
        with column:
            if st.button(topic, key=f"magic_topic_{topic[:18]}", width="stretch"):
                # Written before the text input is drawn on the next run, which
                # is the only order in which a widget accepts a new value for a
                # key it already owns.
                st.session_state["magic_prompt"] = topic
                st.rerun()

    prompt = st.text_input(
        "Your topic",
        key="magic_prompt",
        placeholder=magic_studio.PLACEHOLDER,
        label_visibility="collapsed",
    )

    st.markdown(
        badge(f"\U0001f512 Locked to {low}\u2013{high}s", "green")
        + badge("TikTok Rewards eligible", "green")
        + badge("100% procedural \u00b7 zero copyright risk", "violet"),
        unsafe_allow_html=True,
    )
    st.caption(
        f"Runtime is fixed at {low}\u2013{high} seconds. Creator Rewards counts "
        f"nothing at or under {TIKTOK_REWARDS_MIN_SECONDS:.0f}s, so the script is "
        f"written to a word budget that clears it rather than trimmed to fit "
        f"afterwards \u2014 trimming is what cuts a narration off mid-sentence.")

    if st.button("\u2728 Render Monetized Short (1-Click)", key="magic_go",
                 type="primary", width="stretch", disabled=not prompt.strip()):
        st.session_state.pop("magic_result", None)
        st.session_state.pop("magic_error", None)
        st.session_state["magic_job"] = start_magic_job(
            prompt.strip(), magic_studio.PRIMARY_STYLE, magic_context())
        st.rerun()

    error = st.session_state.get("magic_error")
    if error:
        st.error(f"That did not render: {error}", icon="\u26a0\ufe0f")

    result = st.session_state.get("magic_result")
    if result:
        render_magic_result(result, allowed)


def render_magic_progress(job: Any, allowed: Sequence[str]) -> None:
    """
    Draws where the render has got to, then schedules the next look.

    This function returns in milliseconds. That is the whole point: the render
    is on another thread, so the script run that draws this is short and the
    browser stays responsive for the whole job.
    """
    state = job.snapshot()

    if state["done"]:
        st.session_state.pop("magic_job", None)
        if state["error"]:
            st.session_state["magic_error"] = state["error"]
        else:
            st.session_state["magic_result"] = state["result"]
            notify_complete(
                "Render finished",
                f"{state['result']['name']} \u2014 "
                f"{state['result']['duration']:.0f}s in {state['elapsed']:.0f}s")
        st.rerun()
        return

    with st.container(border=True):
        st.markdown(f"**{magic_studio.style(job.style)['label']}** \u00b7 {job.prompt}")
        suffix = f"  \u00b7  ~{state['eta']:.0f}s left" if state["eta"] else ""
        st.progress(state["fraction"], text=f"{state['message']}{suffix}")
        st.caption(state["caption"])
        st.caption(f"{state['elapsed']:.0f}s elapsed \u00b7 rendering in the "
                   f"background \u2014 this page stays responsive, and you can "
                   f"leave it open.")

        if st.button("Cancel", key="magic_cancel", width="stretch"):
            # The thread is a daemon and ffmpeg has its own timeout, so
            # abandoning it is safe; there is no way to interrupt a render
            # mid-encode without leaving a part-written file behind.
            st.session_state.pop("magic_job", None)
            st.session_state["magic_error"] = "Cancelled. The render was abandoned."
            st.rerun()

    time.sleep(MAGIC_POLL_SECONDS)
    st.rerun()


def render_magic_result(result: dict[str, Any], allowed: Sequence[str]) -> None:
    """The finished video, and the things anyone wants to do with it."""
    with st.container(border=True):
        st.markdown(f"#### \u2705 {result['name']}")
        st.video(result["bytes"])

        chip = dashboard_view.duration_chip(result["duration"])
        earns = result["duration"] > TIKTOK_REWARDS_MIN_SECONDS
        st.markdown(
            badge(magic_studio.style(result["style"])["label"], "violet")
            + (badge(chip, "cyan") if chip else "")
            + badge(_human_bytes(len(result["bytes"])), "")
            + badge("TikTok Rewards eligible" if earns
                    else f"under {TIKTOK_REWARDS_MIN_SECONDS:.0f}s \u2014 no TikTok payout",
                    "green" if earns else "amber"),
            unsafe_allow_html=True,
        )

        download, reveal, upload = st.columns(3)
        with download:
            st.download_button("\U0001f4e5 Download MP4", data=result["bytes"],
                               file_name=result["name"], mime="video/mp4",
                               width="stretch", key="magic_download")
        with reveal:
            if st.button("\U0001f4c2 Open in Explorer", key="magic_reveal",
                         width="stretch",
                         help="Opens the folder on the machine running Streamlit, "
                              "which is the server \u2014 not necessarily the "
                              "device you are reading this on."):
                problem = reveal_in_file_manager(result["path"])
                if problem:
                    st.error(f"Could not open the folder: {problem}")
                else:
                    st.toast("Opened on the server's desktop.", icon="\U0001f4c2")
        with upload:
            can_publish = "atmosphere" in allowed
            if st.button("\U0001f680 Upload to YouTube/TikTok",
                         key="magic_upload", width="stretch", disabled=not can_publish,
                         help="Opens the publisher with this file selected."
                              if can_publish else
                              "Publishing lives in Atmosphere Studio, which your "
                              "role does not have."):
                st.session_state["publish_target"] = result["path"]
                go_to_mode("atmosphere")

        st.caption(f"`{result['path']}`")

        if result.get("script"):
            render_viral_scorecard(result["script"], "magic", result.get("entry") or {})


def render_dashboard(modes: Sequence[str]) -> None:
    """
    The landing page, and the only page most people need.

    Everything that answers "is this machine healthy" rather than "what am I
    making" now lives in Admin: encoder presets, disk usage, the scratch purge
    and the API key panel. They were useful while this was being built and they
    are noise to someone who wants a video.
    """
    allowed = list(modes)
    credits = dashboard_view.credit_status(user_exports(), current_user())

    st.markdown(
        '<div class="rf-hero">'
        '<div class="rf-hero-title">ReelForge Studio</div>'
        '<div class="rf-hero-lede">Autonomous Viral Faceless Production Suite</div>'
        f'<div class="rf-hero-badges">'
        + badge(f"\u26a1 Credits: {credits['remaining']} / {credits['quota']}"
                f" | {credits['tier_label']} Tier",
                "green" if credits["remaining"] > 0 else "amber")
        + '</div></div>',
        unsafe_allow_html=True,
    )

    render_magic_studio(allowed)
    divider()

    # ---- the three-step path ----------------------------------------------
    section("The production path")
    for column, step in zip(st.columns(3), dashboard_view.STEPS):
        with column:
            lanes = ""
            for lane_label, lane_modes in step["lanes"]:
                chips = "".join(
                    badge(MODE_LABELS[m].split(" ", 1)[-1], "cyan")
                    for m in lane_modes if m in allowed
                )
                if chips:
                    lanes += f'<div class="rf-step-lane">{lane_label}</div>{chips}'

            st.markdown(
                f'<div class="rf-step"><div class="rf-step-n">{step["number"]}</div>'
                f'<div class="rf-step-t">{step["title"]}</div>'
                f'<div class="rf-step-b">{step["blurb"]}</div>{lanes}</div>',
                unsafe_allow_html=True,
            )
            for target in step["targets"]:
                if target not in allowed:
                    continue
                if st.button(f"{MODE_LABELS[target]} \u2192",
                             key=f"step_{step['number']}_{target}", width="stretch"):
                    go_to_mode(target)

    divider()

    # ---- the engine grid ---------------------------------------------------
    section("Studio engines")
    cards = dashboard_view.engines_for(allowed)
    for row_start in range(0, len(cards), 4):
        row = cards[row_start:row_start + 4]
        # Always four columns, even on a short final row: three cards stretched
        # across the full width read as a different, larger tier of thing.
        columns = st.columns(4)
        for column, card in zip(columns, row):
            with column:
                st.markdown(
                    f'<div class="rf-engine"><div class="rf-engine-head">'
                    f'<div class="rf-engine-icon">{card["icon"]}</div>'
                    f'<div class="rf-engine-name">{card["name"]}</div></div>'
                    + badge(card["format"], "violet")
                    + f'<div class="rf-engine-use">{card["use"]}</div></div>',
                    unsafe_allow_html=True,
                )
                if st.button("Open Engine \u2192", key=f"open_{card['mode']}",
                             width="stretch"):
                    go_to_mode(card["mode"])

    divider()
    render_dashboard_activity(allowed)
    divider()
    render_monetization_tips()


def render_dashboard_activity(allowed: Sequence[str]) -> None:
    """The three most recent renders, with a runtime chip and a download."""
    section("Recent activity")
    recent = list_exports()[:3]

    if not recent:
        st.caption("Nothing rendered yet. Start at step one and the last three "
                   "finished videos will appear here.")
        return

    for column, item in zip(st.columns(3), recent):
        with column:
            details = export_details(item["path"], item["mtime"], item["bytes"])
            if details["thumb"] is not None:
                st.image(details["thumb"], width="stretch")
            else:
                st.markdown('<div class="rf-thumb-blank">\u25b6</div>',
                            unsafe_allow_html=True)

            chip = dashboard_view.duration_chip(details["duration"])
            ratio = aspect_tag(details["width"], details["height"])
            st.markdown(
                badge(item["kind"], "violet")
                + (badge(chip, "cyan") if chip else "")
                + (badge(ratio, "") if ratio else ""),
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="rf-export-name" title="{item["name"]}">{item["name"]}</div>'
                f'<div class="rf-export-meta">{_human_bytes(item["bytes"])} \u00b7 '
                f'{time.strftime("%d %b %Y, %H:%M", time.localtime(item["mtime"]))}</div>',
                unsafe_allow_html=True,
            )

            if item["bytes"] <= DOWNLOAD_LIMIT_BYTES:
                with open(item["path"], "rb") as handle:
                    st.download_button("\u2b07 Download", data=handle.read(),
                                       file_name=item["name"], mime="video/mp4",
                                       width="stretch", key=f"dash_dl_{item['name']}")
            else:
                st.button("\u2b07 Download", key=f"dash_dl_{item['name']}",
                          width="stretch", disabled=True,
                          help=f"{_human_bytes(item['bytes'])} is too large to push "
                               f"through the browser \u2014 copy it from disk instead.")

    if "library" in allowed:
        if st.button("\U0001f4c1 Open the Exports Library \u2192", key="dash_to_library"):
            go_to_mode("library")


def render_monetization_tips() -> None:
    """Platform compliance, folded away until someone wants it."""
    with st.expander("\U0001f4b0 Platform monetization rules "
                     "\u2014 read before you publish"):
        for group in dashboard_view.platform_rules():
            st.markdown(f"**{group['platform']}**")
            if group["threshold"]:
                st.caption(group["threshold"])
            for rule in group["rules"]:
                st.markdown(f"- {rule}")
            st.markdown("")


MODE_SUBTITLES: dict[str, str] = {
    "dashboard": "Autonomous Viral Faceless Production Suite \u2014 describe a video, pick a style, and one button renders every engine end to end.",
    "commentary": "Drop a raw clip, let Gemini analyze the visual beats, and publish "
                  "high-retention editorial commentary.",
    "minimalist": "Generate original, code-driven 2D vector psychology and finance "
                  "metaphors with zero copyright risk.",
    "narrative": "Produce multi-scene narrative shorts with consistent characters, "
                 "dynamic drift, and cinematic pacing.",
    "batch": "Queue video concepts in bulk and let ReelForge script, voice, and render "
             "in the background.",
    "reel": "Create hook-driven vertical listicles and fact reels with kinetic subtitles "
            "and curated loops.",
    "duel": "Build split-screen comparison duels with animated stat reveal meters and "
            "audience voting polls.",
    "atmosphere": "Synthesize multi-hour ambient soundscapes and publish directly to "
                  "YouTube with automated SEO.",
    "scout": "Find a faceless niche worth entering, measure how crowded it is, and turn "
             "it into a season of episodes.",
    "library": "Preview, download, reveal in explorer, and manage rendered media across "
               "all engines.",
    "admin": "Manage accounts, API keys and every workspace on this deployment.",
}

MODE_LABELS: dict[str, str] = {
    "dashboard": "\U0001f3e0 Studio Dashboard",
    "commentary": "🎙️ Commentary Machine",
    "minimalist": "◼️ Minimalist Motion",
    "narrative": "📖 Narrative Studio",
    "batch": "📦 Batch Studio",
    "reel": "🎬 Reel Studio",
    "duel": "⚔️ Versus Duel",
    "atmosphere": "🌙 Atmosphere Studio",
    "scout": "🎯 Niche Scout",
    "library": "📁 Exports Library",
    "admin": "⚙️ Admin Settings",
}


# The landing view when nothing else has been chosen -- on first sign-in, and
# on a browser refresh that drops the session.
DEFAULT_MODE = "dashboard"

# The sidebar. Three routes are exposed and everything else is folded away,
# because the app has one job now -- a monetizable vector short -- and the
# Dashboard does that job itself.
#
# Archiving is a menu decision and nothing more. Every mode below still routes,
# still renders and is still tested; the accordion changes where you click to
# reach it, not whether it works.
PRIMARY_MODES: tuple[str, ...] = ("dashboard", "library", "admin")

# Minimalist Motion's manual page leads the drawer rather than sitting among
# the experiments: it is the engine the Dashboard drives, and filing the
# primary engine under "Experimental" would be a lie told by a heading. Niche
# Scout is there for the opposite reason -- it is not an experiment either, but
# it is not one of the three exposed routes, so this is where it lands.
ARCHIVED_LEAD: tuple[str, ...] = ("minimalist", "scout")

ARCHIVED_MODES: tuple[str, ...] = ("commentary", "narrative", "batch", "reel",
                                   "duel", "atmosphere")

ARCHIVE_LABEL = "📦 Archived Labs (Experimental)"

# Every registered mode, in menu order. A test asserts this covers the registry
# exactly once, so a new mode cannot be added without being given a home.
NAV_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("", PRIMARY_MODES),
    (ARCHIVE_LABEL, ARCHIVED_LEAD + ARCHIVED_MODES),
)


def go_to_mode(mode: str) -> None:
    """
    Switches the active view and reruns.

    `app_mode` is plain session state rather than a widget key, which is what
    makes this safe to call from a dashboard card: a widget whose key already
    exists in session state ignores any value written to it, so a pills-backed
    menu would have gone on showing the old mode selected while the page
    rendered the new one. Buttons carry no value to fight with.
    """
    st.session_state["app_mode"] = mode
    st.rerun()


def _nav_button(mode: str, active: str) -> None:
    if st.button(MODE_LABELS.get(mode, mode), key=f"nav_{mode}", width="stretch",
                 type="primary" if mode == active else "secondary",
                 help=MODE_SUBTITLES.get(mode, "")):
        go_to_mode(mode)


def render_mode_nav(modes: Sequence[str], active: str) -> None:
    """Three routes, then a drawer holding everything else."""
    for mode in PRIMARY_MODES:
        if mode in modes:
            _nav_button(mode, active)

    drawer = [m for m in ARCHIVED_LEAD + ARCHIVED_MODES if m in modes]
    if not drawer:
        return

    divider()
    # Open when you are standing in one of them, so the highlighted mode is
    # never hidden behind a collapsed heading.
    with st.expander(ARCHIVE_LABEL, expanded=active in drawer):
        lead = [m for m in ARCHIVED_LEAD if m in modes]
        if lead:
            st.caption("The vector engine the Dashboard drives, and the niche "
                       "research that feeds it. Not experiments \u2014 they are "
                       "here because the main list is down to three routes.")
            for mode in lead:
                _nav_button(mode, active)
            divider()
        for mode in ARCHIVED_MODES:
            if mode in modes:
                _nav_button(mode, active)


def main() -> None:
    # Nothing renders before this. An unauthenticated visitor gets the login
    # card and no other markup at all -- no brand, no sidebar, no filenames.
    if not require_login():
        return

    # Housekeeping: clear stale render intermediates once per session.
    sweep_scratch_files()

    role = auth.normalise_role(current_user().get("role"))
    modes = list(auth.allowed_modes(role))

    # The header is drawn before the sidebar, but the subtitle has to name the
    # mode the sidebar is about to show. Streamlit writes a widget's value into
    # session state *before* the rerun that follows a click, so by the time this
    # line runs "app_mode" already holds the newly picked mode -- there is no
    # need for a placeholder, and no one-rerun lag.
    requested = str(st.session_state.get("app_mode") or DEFAULT_MODE)
    active = requested if requested in modes else (
        DEFAULT_MODE if DEFAULT_MODE in modes else modes[0])
    # Written back so the nav highlight, the header subtitle and the routed
    # page below can never disagree about which mode is on screen.
    st.session_state["app_mode"] = active
    denied = requested if requested != active else ""

    st.markdown(
        '<div class="rf-brand"><div class="rf-logo">🎬</div>'
        '<div><div class="rf-title">ReelForge Studio</div></div></div>'
        f'<div class="rf-sub">{MODE_SUBTITLES.get(active, "")}</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        render_identity_bar()
        divider()
        st.markdown('<div class="rf-section">Production Mode</div>', unsafe_allow_html=True)
        # The menu is built from the role, so a creator is never offered an
        # engine they cannot open.
        render_mode_nav(modes, active)
        mode = active

        divider()
        st.markdown('<div class="rf-section">Format</div>', unsafe_allow_html=True)
        aspect_name = st.selectbox(
            "Aspect ratio", list(ASPECT_RATIOS.keys()), index=0, key="aspect_pick",
        )
        fit_mode = pick(
            "Image fit", list(FIT_MODES.keys()), DEFAULT_FIT, "fit_mode",
            format_func=lambda x: FIT_MODES[x],
            help="How a clip that is not already 9:16 fills the frame. "
                 "Crop fills it edge to edge and loses the sides; Blur fill keeps "
                 "the whole picture over a blurred plate; Pad adds black bars.",
        )
        fps = pick("FPS", [24, 30, 60], 24, "fps_pick", format_func=str)

        divider()
        st.markdown('<div class="rf-section">Motion</div>', unsafe_allow_html=True)
        transition_type = st.selectbox(
            "Transition", ["crossfade", "slide_left", "slide_bottom", "fade_black", "cut"],
            format_func=lambda x: {
                "crossfade": "Crossfade", "slide_left": "Slide left",
                "slide_bottom": "Slide up", "fade_black": "Fade black", "cut": "Hard cut",
            }[x],
            index=0,
        )
        transition_dur = st.slider("Transition (s)", 0.2, 1.5, 0.5, 0.1)

        divider()
        st.markdown('<div class="rf-section">Branding</div>', unsafe_allow_html=True)
        watermark_text = st.text_input(
            "Watermark handle", value="@viral_reels", key="watermark_input",
            help="Burned into the top-right of every render. Clear it for no watermark.",
        )

    # Mirror the render settings into session state so panels that render on
    # their own (the duel build button) use exactly what the sidebar shows.
    st.session_state["render_aspect"] = aspect_name
    st.session_state["render_fit"] = fit_mode
    st.session_state["render_transition"] = transition_type
    st.session_state["render_transition_dur"] = transition_dur
    st.session_state["render_fps"] = fps
    st.session_state["render_watermark"] = watermark_text

    # The sidebar already filtered the menu; this catches a stale `app_mode`
    # left in session state from a previous account on the same browser.
    if denied:
        st.error(f"Your role does not have access to that view. "
                 f"Showing {MODE_LABELS[mode]}.", icon="🚫")

    # The landing page. No render card and no guidance banner: it is a way in
    # to the engines, not one of them.
    if mode == "dashboard":
        render_dashboard(modes)
        return

    if mode == "admin":
        render_admin_studio()
        return

    if mode == "library":
        render_exports_library()
        return

    # Research rather than production: no render card, no ETA, no guidance
    # banner about how long a render takes.
    if mode == "scout":
        render_niche_scout()
        return

    # Every production page opens with the same three things: what this engine
    # is for, the work, then what came out of it last time.
    render_mode_guide(mode)

    # The commentary machine is a single linear flow -- no tabs to get lost in.
    if mode == "commentary":
        render_commentary_studio()
    elif mode == "minimalist":
        render_minimalist_studio()
    elif mode == "narrative":
        render_narrative_studio()
    elif mode == "batch":
        render_batch_studio()
    elif mode == "atmosphere":
        render_atmosphere_studio()
    else:
        if mode == "duel":
            tabs = st.tabs(["⚔️ Versus Duel", "🖼️ Slide Studio", "🎵 Audio", "👁️ Preview", "🚀 Export"])
            with tabs[0]:
                render_duel_studio()
        else:
            tabs = st.tabs(["🤖 AI Auto-Creator", "🖼️ Slide Studio", "🎵 Audio", "👁️ Preview", "🚀 Export"])
            with tabs[0]:
                render_auto_creator(aspect_name)

        with tabs[1]:
            render_slide_studio()
        with tabs[2]:
            render_audio_studio()
        with tabs[3]:
            render_preview(aspect_name, fps)
        with tabs[4]:
            render_export(aspect_name, fit_mode, transition_type, transition_dur, fps, watermark_text)


if __name__ == "__main__":
    main()

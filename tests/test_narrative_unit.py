"""Unit tests: entity extraction, storyboard JSON parsing, retiming."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import narrative_engine as ne

# ---- 1. tolerant JSON parsing ---------------------------------------------
print("1. storyboard JSON parsing")
GOOD = {"title": "T", "beats": [{"line": "One."}, {"line": "Two."}]}
cases = [
    ("bare object", json.dumps(GOOD)),
    ("fenced", "```json\n" + json.dumps(GOOD) + "\n```"),
    ("fenced, no lang", "```\n" + json.dumps(GOOD) + "\n```"),
    ("prose wrapped", "Here is the episode:\n" + json.dumps(GOOD) + "\nHope that helps."),
    ("trailing comma", '{"title": "T", "beats": [{"line": "One."},]}'),
    ("nested trailing comma", '{"title": "T", "protagonist": {"name": "A",}, "beats": [{"line": "X"}]}'),
]
for name, text in cases:
    parsed = ne.parse_narrative(text)
    ok = bool(parsed.get("beats"))
    print(f"   {name:22} -> parsed={ok} beats={len(parsed.get('beats') or [])}")
    assert ok, name

for name, text in (("empty", ""), ("not json", "I could not do that."),
                   ("a list", "[1,2,3]"), ("null", "null")):
    parsed = ne.parse_narrative(text)
    print(f"   {name:22} -> {parsed!r}")
    assert parsed == {}, name
print("   fenced, wrapped and trailing-comma JSON all recovered; junk returns {}\n")

# ---- 2. entity extraction --------------------------------------------------
print("2. entity extraction")
RAW = {
    "protagonist": {"name": "Daniel", "age": 28,
                    "appearance": "  lean   build,\n short dark hair, tired eyes ",
                    "clothing": "grey hoodie", "demeanour": "quiet"},
    "environments": [
        {"tag": "Suburban Home!", "name": "Suburban Home Interior", "description": "narrow bedroom"},
        {"tag": "suburban_home", "name": "duplicate tag", "description": "should be dropped"},
        {"name": "Downtown Office at Night", "description": "glass, sodium light"},
    ],
    "objects": [{"tag": "old sedan", "name": "Old Sedan", "description": "rusted saloon"}],
}
ents = ne.extract_entities(RAW)
print(f"   protagonist : {ents['protagonist']['name']}, {ents['protagonist']['age']}")
print(f"   appearance  : {ents['protagonist']['appearance']!r}")
assert ents["protagonist"]["age"] == "28", "non-string age not coerced"
assert "  " not in ents["protagonist"]["appearance"], "whitespace not collapsed"

tags = [e["tag"] for e in ents["environments"]]
print(f"   environments: {tags}")
assert tags[0] == "suburban_home", "tag not slugified"
assert len(tags) == len(set(tags)), "duplicate tags survived"
assert len(tags) == 2, f"expected the duplicate dropped, got {tags}"
assert ents["objects"][0]["tag"] == "old_sedan"

empty = ne.extract_entities({})
print(f"   from {{}}      : protagonist={empty['protagonist']['name']!r} "
      f"environments={[e['tag'] for e in empty['environments']]}")
assert empty["environments"], "no default environment"
assert empty["protagonist"]["appearance"], "no default appearance"

ref = ne.character_reference(ents)
print(f"   reference   : {ref[:78]}...")
assert "Daniel" in ref and "grey hoodie" in ref
print("   tags slugified and deduped, missing fields defaulted\n")

# ---- 3. storyboard segmentation --------------------------------------------
print("3. storyboard segmentation")
RAW["beats"] = [
    {"line": "NARRATOR: I moved back in at twenty three.", "environment": "suburban_home",
     "objects": ["old_sedan"], "framing": "Wide establishing shot", "beat_type": "setup"},
    {"line": "[beat] Nobody tells you how quiet that is.", "environment": "nonexistent_tag"},
    {"line": "", "environment": "suburban_home"},
    "A bare string beat, which some models return.",
    {"line": "The years did the work.", "objects": ["ghost_prop", "old_sedan"]},
]
board = ne.build_storyboard(RAW, ents, "short")
for s in board:
    print(f"   #{s['index']} {s['duration']:4.1f}s  env={s['environment']:14} "
          f"objs={s['objects']}  {s['line'][:40]!r}")

assert len(board) == 4, f"empty beat not dropped: {len(board)}"
assert not board[0]["line"].startswith("NARRATOR"), "speaker label not stripped"
assert not board[1]["line"].startswith("["), "stage direction not stripped"
assert board[1]["environment"] == "suburban_home", "unknown env tag not defaulted"
assert board[3]["objects"] == ["old_sedan"], "unknown object tag not filtered"
assert all(ne.MIN_SEGMENT_SECONDS <= s["duration"] <= ne.MAX_SEGMENT_SECONDS for s in board)

starts = [s["start"] for s in board]
assert starts == sorted(starts), "segments out of order"
for a, b in zip(board, board[1:]):
    assert abs(a["end"] - b["start"]) < 0.01, "gap or overlap between segments"
print(f"   {len(board)} segments, contiguous, runtime {ne.storyboard_runtime(board)}s\n")

# ---- 4. retiming against real word timings ---------------------------------
print("4. retiming against word timings")
words = []
clock = 0.0
for segment in board:
    for _ in range(segment["words"]):
        words.append({"word": "x", "start": clock, "end": clock + 0.5})
        clock += 0.5

retimed = ne.retime_storyboard(board, words)
print(f"   estimated runtime {ne.storyboard_runtime(board):5.2f}s -> "
      f"measured {ne.storyboard_runtime(retimed):5.2f}s (voice is {clock:.2f}s)")
assert abs(ne.storyboard_runtime(retimed) - clock) < 0.6, "retimed board does not match the voice"
for a, b in zip(retimed, retimed[1:]):
    assert abs(a["end"] - b["start"]) < 0.01, "retimed segments not contiguous"

# no timings (Gemini TTS) must leave the board alone rather than zero it
untouched = ne.retime_storyboard(board, [])
assert ne.storyboard_runtime(untouched) == ne.storyboard_runtime(board)
print("   with no word timings the estimated board is kept, not zeroed\n")

# ---- 5. image prompts carry the world --------------------------------------
print("5. image prompt consistency")
prompts = [ne.image_prompt(s, ents, "nordic_noir") for s in board]
for p in prompts[:2]:
    print(f"   {p[:96]}...")
assert all("Daniel" in p for p in prompts), "character reference missing from a prompt"
assert all("nordic noir" in p.lower() for p in prompts), "aesthetic missing from a prompt"
assert all("no watermark" in p.lower() for p in prompts)
card = ne.character_reference(ents)
assert all(card in p for p in prompts), "the character card is not verbatim in every prompt"
print(f"   the same {len(card)}-char character card appears verbatim in all "
      f"{len(prompts)} prompts\n")

# ---- 6. duration estimation -------------------------------------------------
print("6. duration estimation")
for text, note in (("Four words go here.", "short"),
                   ("A much longer sentence, with clauses, that should take noticeably more time to read aloud.", "long"),
                   ("", "empty")):
    print(f"   {ne.estimate_seconds(text):5.2f}s  {note}")
assert ne.estimate_seconds("") == 0.0
assert ne.estimate_seconds("a b c d e f g h i j") > ne.estimate_seconds("a b c")
print()

# ---- 7. the offline fallback is complete ------------------------------------
print("7. offline fallback")
ep = ne.normalise_narrative(None, "POV you lived with your parents in your 20s")
print(f"   title={ep['title']!r} segments={len(ep['segments'])} runtime={ep['runtime']}s")
print(f"   entities: {len(ep['entities']['environments'])} environments, "
      f"{len(ep['entities']['objects'])} objects")
assert ep["segments"] and ep["runtime"] > 10
assert ep["script"] and ep["source"] == "fallback"
assert ep["entities"]["protagonist"]["appearance"]

print("\nNARRATIVE UNIT TESTS PASSED")

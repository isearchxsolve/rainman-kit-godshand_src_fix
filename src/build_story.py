#!/usr/bin/env python3
"""THE ONE PLACE TO EDIT THE STORY.

Edit story.yaml, then this regenerates shots_<film>.json (the file the render
engine reads). Everything a person changes -- shots, action, who, camera,
timing, look -- lives in story.yaml. This script fills in all the boilerplate
fields the engine needs so you never touch the big JSON by hand.

Usage:  python3 build_story.py story.yaml shots_confrontation.json

If a base JSON already exists, its globals/character_design are used as a
fallback so nothing critical is lost when a field is omitted from story.yaml.
"""
import json, os, sys, yaml

DEFAULT_RENDER = {
    "fps": 24, "width": 1280, "height": 720, "base_model": "Wan2.2-S2V",
    "quant": "Q8", "steps": 30, "cfg": 5.5, "sampler": "uni_pc",
    "scheduler": "beta", "interpolate_to_fps": 48, "upscale": 1.75,
    "rejection_samples": 3,
}

def build(story_path, out_json):
    story = yaml.safe_load(open(story_path))
    base = {}
    if os.path.exists(out_json):
        try: base = json.load(open(out_json))
        except Exception: base = {}
    base_g = base.get("globals", {})
    base_cd = base.get("character_design", {})

    look = story.get("look", base_g.get("tone", ""))
    avoid = story.get("avoid", base_g.get("negative_prompt", ""))

    # globals: start from defaults, layer base json, layer story.render, set tone/negative
    g = dict(DEFAULT_RENDER)
    g.update({k: v for k, v in base_g.items() if k not in ("tone", "negative_prompt")})
    g.update(story.get("render", {}) or {})
    g["tone"] = look
    g["negative_prompt"] = avoid

    # characters + character_design
    characters, character_design = {}, {}
    for name, c in (story.get("characters", {}) or {}).items():
        clook = c.get("look", "")
        characters[name] = {"refs": "refs/" + name, "tokens": clook, "anchor": c.get("anchor", name + ".png")}
        # preserve any rich design summary from the base, else use the look line
        character_design[name] = base_cd.get(name, {"summary": clook}) if base_cd.get(name) else {"summary": clook}

    shots = []
    for s in story.get("shots", []):
        shot = {
            "id": s["id"],
            "dur": s.get("secs", 4),
            "chars": list(s.get("who", [])),
            "talking": bool(s.get("talking", False)),
            "vo": bool(s.get("voiceover", False)),
            "camera": s.get("camera", ""),
            "mode": s.get("mode", "A"),
            "song": s.get("song", ""),
            "prompt": s.get("action", ""),
            "palette": s.get("palette", ""),
            "negative": s.get("avoid", avoid),
            "tone": s.get("look", look),
            "movement": s.get("movement", s.get("song", "")),
        }
        if s.get("fx") is not None: shot["fx"] = s["fx"]
        if s.get("intensity") is not None: shot["intensity"] = s["intensity"]
        if s.get("takes") is not None: shot["rejection_samples"] = s["takes"]
        shots.append(shot)

    out = {
        "film": story.get("film", base.get("film", "UNTITLED")),
        "note": story.get("note", base.get("note", "")),
        "globals": g,
        "character_design": character_design,
        "characters": characters,
        "shots": shots,
    }
    with open(out_json, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"[build_story] wrote {out_json}: {len(shots)} shots, {len(characters)} characters, look/avoid applied")
    return out

if __name__ == "__main__":
    story_path = sys.argv[1] if len(sys.argv) > 1 else "story.yaml"
    out_json = sys.argv[2] if len(sys.argv) > 2 else "shots_confrontation.json"
    build(story_path, out_json)

#!/usr/bin/env python3
"""One-time: turn the sprawling shots_<film>.json into a single, human-friendly
story.yaml that a person can actually edit. Run once to seed story.yaml."""
import json, sys, collections, yaml

def most_common(vals):
    return collections.Counter(vals).most_common(1)[0][0]

def main(src_json, out_yaml):
    d = json.load(open(src_json))
    g = d["globals"]; shots = d["shots"]
    master_look = most_common([s["tone"] for s in shots])
    master_avoid = most_common([s["negative"] for s in shots])

    chars = {}
    cd = d.get("character_design", {}); cc = d.get("characters", {})
    # map each character to the real anchor by role order (man=first male, etc.)
    anchor_by_char = {"jack": "man_anchor.png", "julie": "woman_anchor.png"}
    for name in cc:
        chars[name] = {
            "anchor": anchor_by_char.get(name, name + ".png"),
            "look": cc[name].get("tokens", ""),
        }

    story = {
        "film": d.get("film", "UNTITLED"),
        "note": d.get("note", ""),
        "look": master_look,
        "avoid": master_avoid,
        "render": {k: g[k] for k in ("fps","width","height","base_model","quant","steps","cfg","sampler","scheduler","interpolate_to_fps","upscale","rejection_samples") if k in g},
        "characters": chars,
        "shots": [],
    }
    for s in shots:
        sh = {
            "id": s["id"],
            "secs": s["dur"],
            "who": list(s.get("chars", [])),
            "camera": s.get("camera", ""),
            "song": s.get("song", ""),
            "action": s.get("prompt", ""),
            "palette": s.get("palette", ""),
        }
        if s.get("talking"): sh["talking"] = True
        if s.get("vo"): sh["voiceover"] = True
        # only carry overrides when they differ from the master
        if s.get("tone") and s["tone"] != master_look: sh["look"] = s["tone"]
        if s.get("negative") and s["negative"] != master_avoid: sh["avoid"] = s["negative"]
        if s.get("fx"): sh["fx"] = s["fx"]
        if s.get("intensity") is not None: sh["intensity"] = s["intensity"]
        if s.get("rejection_samples") is not None: sh["takes"] = s["rejection_samples"]
        story["shots"].append(sh)

    with open(out_yaml, "w") as f:
        yaml.safe_dump(story, f, sort_keys=False, allow_unicode=True, width=100000, default_flow_style=False)
    print(f"wrote {out_yaml}: {len(story['shots'])} shots, {len(chars)} characters")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "shots_confrontation.json",
         sys.argv[2] if len(sys.argv)>2 else "story.yaml")

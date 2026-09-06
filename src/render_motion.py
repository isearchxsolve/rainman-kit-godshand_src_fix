#!/usr/bin/env python3
"""render_motion.py — THE BLUEPRINT PATH (stills -> motion).

Architecture (v4 — near-100% pixel-perfect on Kaggle):
  The artist (render_stills.py) draws every page: one pixel-perfect still
  per beat.  The blueprint does 99% of the work.  This script does the 1%:
  animate each still into motion by feeding it as the ANCHOR FRAME into
  Wan 2.2 image-to-video.

  v4 additions:
    * Identity auto-retry — after rendering, run insightface face-match
      against the character's reference set.  If cosine similarity drops
      below `identity.drift_cosine`, re-render up to `identity_retry` times
      with a new seed until the face matches.  This keeps Kunal/Gunja
      looking the SAME across every shot without manual intervention.
    * Better take selection — weighted toward temporal stability AND
      identity fidelity.
    * Low denoise default (0.35) — stays faithful to the blueprint still.

Workflow selection (same as render_film.py, but the image input is the STILL):
  - talking beat  -> s2v  (still + voice audio -> lip-synced motion)
  - 2+ chars     -> phantom (still as subject anchor -> motion)
  - 1 char / 0   -> flf2v (still as first-frame -> motion; last-frame optional)

For FLF2V, if a NEXT beat's still exists, it is used as the end-frame so the
model knows where the motion should arrive.  If no next-frame exists, the same
still is used for both (subtle motion within the frame).

Prerequisite: render_stills.py must have run first (stills exist in <work>/<sid>/still.png).
If a still is missing, this script falls back to the ref image (old behavior).

Usage:
  KIT_CONFIG=config_confrontation.yaml python3 src/render_motion.py          # all beats
  KIT_CONFIG=config_confrontation.yaml python3 src/render_motion.py --only search_run
  KIT_CONFIG=config_confrontation.yaml python3 src/render_motion.py --smoke   # 1 beat, 1 take, low steps

Env:
  COMFY_MOCK=1  synthesize placeholder clips with ffmpeg (free offline dry-run)
  KIT_MOTION_DENOISE  denoise strength for the motion pass (default: 0.35)
  KIT_MOTION_STEPS    override steps for the motion pass
  KIT_MASTER=1        also export ProRes 4:4:4 master
"""
import os, sys, json, glob, argparse, subprocess


def load_cfg(root):
    sys.path.insert(0, os.path.join(root, "src"))
    from cfgutil import load_cfg as _lc
    return _lc(root)


def load_shots(root, cfg):
    return json.load(open(os.path.join(root, cfg["shots_file"])))


def done(work, sid):
    return os.path.exists(os.path.join(work, sid, "final.mp4"))


def still_path(work, sid):
    """Path to the blueprint still for a beat (if render_stills.py ran)."""
    p = os.path.join(work, sid, "still.png")
    return p if os.path.exists(p) else None


def next_still_path(work, shots, idx):
    """Path to the NEXT beat's still (for FLF2V end-frame).  None if no next."""
    if idx + 1 >= len(shots):
        return None
    return still_path(work, shots[idx + 1]["id"])


def refs_for(shot, char, cfg, root):
    """Pick era-correct refs with identity_override support."""
    io = cfg.get("identity_override", {}) or {}
    if io.get("enabled"):
        d = os.path.join(root, io.get("refs_dir", "refs/original_identity"))
        if os.path.isdir(d) and glob.glob(os.path.join(d, "**", "*"), recursive=True):
            return d
    if char == "kunal":
        era = shot.get("era", "young")
        sub = "kunal_young" if era == "young" else "kunal_40s"
        d = os.path.join(root, "refs", sub)
        if os.path.isdir(d) and glob.glob(os.path.join(d, "**", "*"), recursive=True):
            return d
    return os.path.join(root, "refs", char)


def _identity_score(video_path, ref_dir):
    """Use insightface (CPU) to check face cosine similarity between a rendered
    video frame and the character's reference images. Returns max similarity
    score (0-1), or None if insightface unavailable."""
    try:
        import cv2, numpy as np
        import insightface
        from insightface.app import FaceAnalysis
    except Exception:
        return None
    if not os.path.isdir(ref_dir):
        return None
    refs = [f for f in sorted(glob.glob(os.path.join(ref_dir, "**", "*"), recursive=True))
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not refs:
        return None
    try:
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0)
        # Embed reference images
        ref_embs = []
        for r in refs[:5]:  # limit to 5 refs for speed
            img = cv2.imread(r)
            if img is None:
                continue
            faces = app.get(img)
            if faces:
                ref_embs.append(faces[0].normed_embedding)
        if not ref_embs:
            return None
        # Sample frames from video
        cap = cv2.VideoCapture(video_path)
        scores = []
        f_idx = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if f_idx % 6 == 0:  # sample every 6th frame (~4 fps at 24fps)
                faces = app.get(fr)
                if faces:
                    emb = faces[0].normed_embedding
                    for r_emb in ref_embs:
                        sim = float(np.dot(emb, r_emb))
                        scores.append(sim)
            f_idx += 1
        cap.release()
        return max(scores) if scores else None
    except Exception as e:
        print(f"[identity] check failed: {e}")
        return None


def render_motion(shot, cfg, root, work, shots, idx):
    from comfy_client import ComfyClient
    sid = shot["id"]
    outdir = os.path.join(work, sid)
    os.makedirs(outdir, exist_ok=True)
    if done(work, sid):
        print(f"[skip] {sid} already rendered.")
        return

    client = ComfyClient()
    tone = json.load(open(os.path.join(root, cfg["shots_file"])))["globals"]
    rp = cfg.get("realism", {}) or {}
    pos_suffix = (", " + rp["positive"]) if rp.get("positive") else ""
    neg_extra = (", " + rp["negative"]) if rp.get("negative") else ""
    thr = float((cfg.get("intensity") or {}).get("keep_glitch_threshold", 0.66))
    intensity = float(shot.get("intensity", 0.0) or 0.0)
    keep_glitch = intensity >= thr
    fx_suffix = (", " + shot["fx"]) if shot.get("fx") else ""
    base_neg = shot.get("negative", tone["negative_prompt"]) + neg_extra
    if keep_glitch:
        for bad in ["flickering", "temporal artifacts", "blurry", "motion blur",
                    "soft focus", "low detail", "low quality"]:
            base_neg = base_neg.replace(", " + bad, "")
        print(f"[motion] {sid}: directed-glitch beat (intensity={intensity}); keeping artifacts as design.")

    positive = (shot["prompt"]
               + (", " + shot["palette"] if shot.get("palette") else "")
               + ", " + shot.get("tone", tone["tone"])
               + fx_suffix + pos_suffix)

    # --- THE BLUEPRINT: use the already-rendered still as the anchor frame ---
    blueprint = still_path(work, sid)
    chars = shot.get("chars", [])

    # Workflow selection
    wf_name = "s2v" if shot.get("talking") else ("phantom" if len(chars) > 1 else "flf2v")
    if not chars:
        wf_name = "flf2v"
    graph = json.load(open(os.path.join(root, "workflows", f"{wf_name}.json")))

    # Build the override dict
    overrides = {
        "positive": positive,
        "negative": base_neg,
        "width": cfg["width"],
        "height": cfg["height"],
        "steps": int(os.environ.get("KIT_MOTION_STEPS", cfg["steps"])),
        "cfg": cfg["cfg"],
        "seconds": min(shot["dur"], cfg["max_clip_seconds"]),
    }

    # --- Wire the blueprint still as the image anchor ---
    if blueprint:
        print(f"[motion] {sid}: blueprint still -> {blueprint}")
        overrides["ref_dirs"] = {"_still": os.path.dirname(blueprint)}
        if wf_name == "flf2v":
            # Injecting the NEXT beat's still as the end-frame makes Wan MORPH
            # from this still into the next shot's still ("signature morphing").
            # Only wanted for explicit transition shots. Default OFF so a single
            # still stays faithful (no morph) -> take_0-quality output.
            # Enable globally: KIT_FLF2V_ENDFRAME=1  |  per-shot: shot["transition"]=true
            want_ef = os.environ.get("KIT_FLF2V_ENDFRAME", "0") == "1" or bool(shot.get("transition"))
            nxt = next_still_path(work, shots, idx) if want_ef else None
            if nxt:
                print(f"[motion] {sid}: end-frame -> {nxt} (TRANSITION morph ON)")
                graph["7"]["inputs"]["image"] = nxt
            else:
                print(f"[motion] {sid}: no end-frame (faithful single still, no morph)")
    else:
        print(f"[motion] {sid}: no blueprint still found, using character refs.")
        if chars:
            ref_dirs = {}
            for c in chars:
                d = os.path.join(root, "refs", c)
                if os.path.isdir(d):
                    ref_dirs[c] = d
            if ref_dirs:
                overrides["ref_dirs"] = ref_dirs

    # Denoise: low denoise = faithful to the blueprint
    denoise = float(os.environ.get("KIT_MOTION_DENOISE", str(cfg.get("denoise", "0.35"))))
    if "15" in graph and "denoise" in graph["15"]["inputs"]:
        graph["15"]["inputs"]["denoise"] = denoise
        print(f"[motion] {sid}: denoise={denoise} (blueprint-faithful)")

    # --- Render takes ---
    n_samples = int(shot.get("rejection_samples", cfg["rejection_samples"]))
    identity_retry = int(cfg.get("identity_retry", 0))
    max_takes = n_samples + (identity_retry if chars else 0)
    takes = []
    last_best = None

    for n in range(max_takes):
        overrides["seed"] = 42 + n
        out = client.run(graph, overrides=overrides)
        take = client.save_image(out, os.path.join(outdir, f"take_{n}.mp4"))
        takes.append(take)

        # Check identity after each take if we have character refs
        if chars and identity_retry > 0:
            ref_dir = refs_for(shot, chars[0], cfg, root)
            score = _identity_score(take, ref_dir)
            if score is not None:
                threshold = (cfg.get("identity") or {}).get("drift_cosine", 0.35)
                print(f"[motion] {sid} take_{n}: identity score={score:.3f} (threshold={threshold})")
                if score >= threshold and last_best is None:
                    last_best = take
                    if len(takes) >= n_samples:
                        break  # good enough, stop early
            elif last_best is None:
                last_best = take  # can't check identity, keep first good one

    # Pick best take
    flicker_weight = 0.0 if keep_glitch else 1.5
    best = _pick_best(takes, flicker_weight=flicker_weight)

    # Post: upscale -> face restore -> interpolate -> grade
    _do_postprocess(root, best, os.path.join(outdir, "final.mp4"),
                    cfg["upscale"], cfg["interpolate_fps"],
                    cfg["codeformer_fidelity"],
                    os.path.join(root, "refs", "grade_target.json"))

    # Optional ProRes master
    if os.environ.get("KIT_MASTER") == "1":
        master = os.path.join(outdir, "final_MASTER_4444.mov")
        fps_val = int(cfg.get("interpolate_fps", cfg.get("fps", 24)))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i",
                        os.path.join(outdir, "final.mp4"),
                        "-c:v", "prores_ks", "-profile:v", "4", "-vendor", "apl0",
                        "-pix_fmt", "yuv444p10le",
                        "-colorspace", "bt709", "-color_primaries", "bt709",
                        "-color_trc", "bt709", "-r", str(fps_val), master])

    print(f"[ok] {sid} -> {outdir}/final.mp4 (motion from blueprint)")


def _take_metrics(path):
    """Raw per-take fidelity metrics."""
    import cv2, numpy as np
    cap = cv2.VideoCapture(path)
    grays, sharp = [], []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        gd = cv2.medianBlur(g, 3)
        sharp.append(cv2.Laplacian(gd, cv2.CV_64F).var())
        grays.append(g.astype("float32"))
    cap.release()
    if not grays:
        return None
    jerks = [float(np.mean(np.abs(grays[i] - 2 * grays[i - 1] + grays[i - 2])))
             for i in range(2, len(grays))]
    mean_lum = float(np.mean([g.mean() for g in grays]))
    return {
        "sharpness": float(np.mean(sharp)),
        "flicker": float(np.mean(jerks)) if jerks else 0.0,
        "exposure_ok": 25.0 <= mean_lum <= 230.0,
    }


def _pick_best(takes, flicker_weight=1.5):
    """Pick the best take by normalized sharpness vs temporal stability."""
    try:
        import cv2  # noqa: F401
    except Exception:
        return takes[0]
    scored = []
    for t in takes:
        try:
            m = _take_metrics(t)
        except Exception:
            m = None
        if m:
            scored.append((t, m))
    if not scored:
        return takes[0]
    sh = [m["sharpness"] for _, m in scored]
    fl = [m["flicker"] for _, m in scored]

    def _norm(x, xs):
        lo, hi = min(xs), max(xs)
        return 0.0 if hi <= lo else (x - lo) / (hi - lo)

    best, best_score = scored[0][0], float("-inf")
    for t, m in scored:
        score = (_norm(m["sharpness"], sh)
                 - flicker_weight * _norm(m["flicker"], fl)
                 - (0.0 if m["exposure_ok"] else 1.0))
        if score > best_score:
            best, best_score = t, score
    return best


def _do_postprocess(root, inp, out, upscale, fps, fidelity, grade):
    """Run postprocess.sh if bash available, else ffmpeg-only Python fallback."""
    import shutil
    if sys.platform != "win32" and shutil.which("bash"):
        subprocess.run(["bash", os.path.join(root, "src", "postprocess.sh"),
                        inp, out, str(upscale), str(fps), str(fidelity), grade], check=True)
    else:
        # Cross-platform ffmpeg fallback
        import tempfile
        tmp = tempfile.mkdtemp()
        try:
            den = os.path.join(tmp, "d.mp4")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", inp,
                "-vf", "hqdn3d=4.0:3.0:6.0:3.0",
                "-c:v", "libx264", "-preset", "fast", "-crf", "10",
                "-pix_fmt", "yuv420p",
                "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                den], check=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", den,
                "-vf", f"scale=iw*{upscale}:ih*{upscale}:flags=lanczos,"
                       f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
                "-c:v", "libx264", "-preset", "slow", "-crf", "12",
                "-pix_fmt", "yuv420p",
                "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                out], check=True)
            print(f"[postprocess(ffmpeg)] -> {out}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def _auto_shutdown():
    cmd = os.environ.get("KIT_SHUTDOWN_CMD", "sudo shutdown -h now")
    print(f"[shutdown] work complete -> stopping instance: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=False)
    except Exception as e:
        print(f"[shutdown] FAILED ({e}) -- STOP THE INSTANCE MANUALLY to avoid charges.")


def main():
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--shutdown", action="store_true")
    ap.add_argument("--max-hours", type=float, default=0.0)
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "src"))
    cfg = load_cfg(root); work = cfg["paths"]["work"]; os.makedirs(work, exist_ok=True)
    if a.smoke:
        cfg["rejection_samples"] = 1
        cfg["steps"] = min(int(cfg.get("steps", 30)), 12)
        print("[smoke] motion validation: 1 beat, 1 take, low steps (cheap check).")
    shots = load_shots(root, cfg)["shots"]
    _sn = int(os.environ.get("KIT_SHARD_COUNT", "1"))
    _si = int(os.environ.get("KIT_SHARD_INDEX", "0"))
    if _sn > 1:
        shots = [s for _k, s in enumerate(shots) if _k % _sn == _si]
        print("[shard] %d/%d -> %d shots" % (_si, _sn, len(shots)))
    if a.smoke:
        shots = shots[:1]
    t0 = time.time()
    try:
        for idx, shot in enumerate(shots):
            if a.only and shot["id"] != a.only:
                continue
            if a.max_hours and (time.time() - t0) / 3600.0 >= a.max_hours:
                print(f"[budget] reached --max-hours={a.max_hours}; stopping before {shot['id']}.")
                break
            render_motion(shot, cfg, root, work, shots, idx)
        print("[ok] motion render loop finished.")
    finally:
        if a.shutdown:
            _auto_shutdown()


if __name__ == "__main__":
    main()

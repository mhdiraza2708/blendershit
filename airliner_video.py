#!/usr/bin/env python3
"""
airliner_video.py - render a video of the airliner switching between its five modes.

Builds the aircraft with airliner.py (same folder), then animates
    finished -> X-ray -> cutaway -> exploded -> laid out -> reassembled
with a slow orbit on every mode and a live part count in the caption. Each frame is
computed from the timeline directly (nothing is keyframed), rendered to PNG, and the
frames are encoded to H.264 with Blender's own FFmpeg.

Usage:
    blender -b -P airliner_video.py -- --out renders/airliner_modes.mp4
    python airliner_video.py --out renders/airliner_modes.mp4          # PyPI bpy

Options:
    --out PATH          video file (default renders/airliner_modes.mp4)
    --res WxH           default 1280x720
    --fps N             default 24
    --engine {eevee,cycles}   default eevee (much faster; needs OpenGL, software is fine)
    --samples N         render samples (default 12 for EEVEE, 24 for Cycles)
    --frames A-B        render only this frame range (for previews), no encoding
Frames go to <out without extension>_frames/; frames that already exist are skipped, so an
interrupted render resumes where it stopped.
"""

import argparse
import math
import os
import sys
import time
import zlib

import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import airliner as A  # noqa: E402

# (kind, mode or (from, to), seconds)
TIMELINE = [
    ("hold", "FINISHED", 3.0),
    ("go", ("FINISHED", "XRAY"), 1.4),
    ("hold", "XRAY", 3.0),
    ("go", ("XRAY", "CUTAWAY"), 2.0),
    ("hold", "CUTAWAY", 3.0),
    ("go", ("CUTAWAY", "EXPLODED"), 2.6),
    ("hold", "EXPLODED", 3.0),
    ("go", ("EXPLODED", "LAYOUT"), 3.6),
    ("hold", "LAYOUT", 3.0),
    ("go", ("LAYOUT", "FINISHED"), 4.4),
    ("hold", "FINISHED", 2.0),
]
ORBIT = {"FINISHED": 2.0, "XRAY": 2.0, "CUTAWAY": 1.6, "EXPLODED": 1.6, "LAYOUT": 0.0}  # degrees / s
CAM_SHARE = {("EXPLODED", "LAYOUT"): 0.55, ("LAYOUT", "FINISHED"): 0.5}   # camera arrives early
BUILD_ORDER = ["Structure", "Cargo & systems", "Flight deck", "Cabin", "Landing gear", "Engines",
               "Wings", "Tail", "Fuselage"]
LABEL = {"FINISHED": "Finished", "XRAY": "X-ray", "CUTAWAY": "Cutaway", "EXPLODED": "Exploded",
         "LAYOUT": "Laid out"}
STUDIO_BG, XRAY_BG = Vector((0.62, 0.68, 0.76)), Vector((0.004, 0.012, 0.03))


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def clamp01(x):
    return min(max(x, 0.0), 1.0)


def mat(v):
    return Matrix([v[0:4], v[4:8], v[8:12], v[12:16]])


def jitter(name, amount):
    return (zlib.crc32(name.encode()) % 1000) / 1000 * amount


class Part:
    def __init__(self, o, lift):
        self.o = o
        self.cat = o["ap_cat"]
        self.cut = bool(o["ap_cut"])
        self.shell = bool(o["ap_shell"])
        self.home = mat(o["ap_home"]).decompose()
        xpl = Vector(o["ap_xpl"])
        self.expl = (Matrix.Translation(xpl + lift) @ mat(o["ap_home"])).decompose()
        self.lay = mat(o["ap_lay"]).decompose()
        d = xpl.normalized() if xpl.length > 1e-6 else Vector((0, 1, 0))
        self.away = (Matrix.Translation(d * 9.0 + Vector((0, 0, 2.5))) @ mat(o["ap_home"])).decompose()
        self.j = jitter(o.name, 1.0)


def blend(a, b, t, arc=0.0):
    loc = a[0].lerp(b[0], t)
    loc.z += arc * math.sin(math.pi * t)
    return Matrix.LocRotScale(loc, a[1].slerp(b[1], t), a[2].lerp(b[2], t))


def segment(t):
    t0 = 0.0
    for kind, what, dur in TIMELINE:
        if t < t0 + dur:
            return kind, what, (t - t0) / dur, t0, dur
        t0 += dur
    kind, what, dur = TIMELINE[-1]
    return kind, what, 1.0, t0 - dur, dur


def total_seconds():
    return sum(d for _, _, d in TIMELINE)


class Director:
    def __init__(self, scene):
        self.sc = scene
        lift = Vector((0, 0, scene.get("ap_xlift", 5.5)))
        self.parts = [Part(o, lift) for o in scene.objects if "ap_part" in o]
        cuts = sorted((p for p in self.parts if p.cut), key=lambda p: (p.home[0].x, p.j))
        for i, p in enumerate(reversed(cuts)):         # nose first
            p.rank = i / max(1, len(cuts) - 1)
        self.ghost = A.MAT["ghost"]
        self.xmats = {c: A.MAT.get(f"AP X-ray {c}") for c in A.CATEGORIES}
        self.labels = {o["ap_cat_label"]: o for o in scene.objects if "ap_cat_label" in o}
        self.hud = {o["ap_hud"]: o for o in scene.objects if "ap_hud" in o}
        self.ground = next(o for o in scene.objects if "ap_ground" in o)
        self.bg = scene.world.node_tree.nodes["ap_bg"]
        self.cams = {m: self._loc_target(m, mat(scene["ap_cam_" + m])) for m in LABEL}
        self.xray_on = None

    # ---- materials -------------------------------------------------------------
    def set_xray(self, on):
        if on == self.xray_on:
            return
        self.xray_on = on
        for p in self.parts:
            m = (self.ghost if p.shell else self.xmats.get(p.cat)) if on else None
            for s in p.o.material_slots:
                if m:
                    s.link = "OBJECT"
                    s.material = m
                else:
                    s.link = "DATA"
        self.ground.hide_render = on
        for o in self.hud.values():
            em = o.active_material.node_tree.nodes["Emission"]
            em.inputs["Color"].default_value = (0.6, 0.9, 1.0, 1) if on else (0.02, 0.025, 0.04, 1)

    # ---- camera: moved as (position, look-at point) so it always faces the action --
    @staticmethod
    def _loc_target(mode, M):
        loc = M.translation.copy()
        fwd = -Vector(M.col[2][:3]).normalized()
        if mode == "LAYOUT":
            return loc, loc + fwd * ((A.GROUND - loc.z) / fwd.z)
        return loc, loc + fwd * fwd.dot(Vector((0, 0, -1)) - loc)

    def cam_hold(self, mode, secs):
        R = Matrix.Rotation(math.radians(ORBIT[mode] * secs), 3, "Z")
        loc, tgt = self.cams[mode]
        return R @ loc, R @ tgt

    # ---- one frame -------------------------------------------------------------
    def apply(self, t):
        kind, what, p, t0, dur = segment(t)
        show_labels = set()
        if kind == "hold":
            mode = what
            cam = self.cam_hold(mode, p * dur)
            pose = {"EXPLODED": "expl", "LAYOUT": "lay"}.get(mode, "home")
            for q in self.parts:
                q.o.matrix_world = Matrix.LocRotScale(*getattr(q, pose))
                q.o.hide_render = mode == "CUTAWAY" and q.cut
            self.set_xray(mode == "XRAY")
            bg = XRAY_BG if mode == "XRAY" else STUDIO_BG
            title = mode
            if mode == "LAYOUT":
                show_labels = set(self.labels)
        else:
            a, b = what
            prev_hold = next(d for k, w, d in TIMELINE if k == "hold" and w == a)
            la, ta = self.cam_hold(a, prev_hold)
            lb, tb = self.cams[b]
            pc = ease(p / CAM_SHARE.get((a, b), 1.0))
            cam = (la.lerp(lb, pc), ta.lerp(tb, pc))
            title = b if p >= 0.5 else a
            bg = STUDIO_BG
            if (a, b) == ("FINISHED", "XRAY"):
                for q in self.parts:
                    q.o.matrix_world = Matrix.LocRotScale(*q.home)
                    q.o.hide_render = False
                self.set_xray(p >= 0.5)
                bg = STUDIO_BG.lerp(XRAY_BG, ease((p - 0.3) / 0.4))
            elif (a, b) == ("XRAY", "CUTAWAY"):
                self.set_xray(p < 0.04)
                bg = XRAY_BG.lerp(STUDIO_BG, ease(p / 0.04))
                for q in self.parts:
                    if q.cut:
                        k = clamp01((p - 0.15 - q.rank * 0.5) / 0.3)
                        q.o.matrix_world = blend(q.home, q.away, ease(k))
                        q.o.hide_render = k >= 1.0
                    else:
                        q.o.matrix_world = Matrix.LocRotScale(*q.home)
                        q.o.hide_render = False
            elif (a, b) == ("CUTAWAY", "EXPLODED"):
                self.set_xray(False)
                for q in self.parts:
                    d = BUILD_ORDER.index(q.cat) / len(BUILD_ORDER) * 0.35 + q.j * 0.08
                    k = ease((p - d) / 0.55)
                    q.o.matrix_world = blend(q.away if q.cut else q.home, q.expl, k)
                    q.o.hide_render = q.cut and p <= 0.0
            elif (a, b) == ("EXPLODED", "LAYOUT"):
                self.set_xray(False)
                for q in self.parts:
                    d = A.CATEGORIES.index(q.cat) / len(A.CATEGORIES) * 0.5 + q.j * 0.08
                    k = (p - d) / 0.4
                    q.o.matrix_world = blend(q.expl, q.lay, ease(k), arc=4.0)
                    q.o.hide_render = False
                for c in self.labels:
                    if p >= A.CATEGORIES.index(c) / len(A.CATEGORIES) * 0.5 + 0.45:
                        show_labels.add(c)
            elif (a, b) == ("LAYOUT", "FINISHED"):
                self.set_xray(False)
                for q in self.parts:
                    d = BUILD_ORDER.index(q.cat) / len(BUILD_ORDER) * 0.6 + q.j * 0.06
                    k = (p - d) / 0.3
                    q.o.matrix_world = blend(q.lay, q.home, ease(k), arc=7.0)
                    q.o.hide_render = False
                for c in self.labels:
                    if p < BUILD_ORDER.index(c) / len(BUILD_ORDER) * 0.6:
                        show_labels.add(c)
        self.sc.camera.matrix_world = A.look_at(*cam)
        self.bg.inputs["Color"].default_value = (*bg, 1)
        for c, o in self.labels.items():
            o.hide_render = c not in show_labels
        tot = len(self.parts)
        vis = sum(1 for q in self.parts if not q.o.hide_render)
        self.hud["title"].data.body = "AIRLINER  /  " + LABEL[title].upper()
        self.hud["count"].data.body = f"{vis} of {tot} parts shown" if vis != tot else f"{tot} parts"


def encode_video(frames_dir, out, fps, res):
    """Encode the PNG frames to H.264 with Blender's own FFmpeg (via the sequencer)."""
    files = sorted(f for f in os.listdir(frames_dir) if f.startswith("frame_") and f.endswith(".png"))
    enc = bpy.data.scenes.new("Encode")
    se = enc.sequence_editor_create()
    strips = se.strips if hasattr(se, "strips") else se.sequences
    strip = strips.new_image("frames", os.path.join(frames_dir, files[0]), 1, 1)
    for f in files[1:]:
        strip.elements.append(f)
    enc.frame_start, enc.frame_end = 1, len(files)
    enc.render.fps = fps
    enc.render.resolution_x, enc.render.resolution_y = res
    enc.render.resolution_percentage = 100
    im = enc.render.image_settings
    if hasattr(im, "media_type"):         # Blender 5
        im.media_type = "VIDEO"
    im.file_format = "FFMPEG"
    ff = enc.render.ffmpeg
    ff.format, ff.codec = "MPEG4", "H264"
    ff.constant_rate_factor = "HIGH"
    ff.ffmpeg_preset = "GOOD"
    enc.view_settings.view_transform = "Standard"   # the frames are already tone-mapped
    enc.view_settings.look = "None"
    enc.render.filepath = out
    enc.render.use_file_extension = False
    bpy.ops.render.render(animation=True, scene=enc.name)
    bpy.data.scenes.remove(enc)
    print(f"[video] wrote {out}")


def parse_args():
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    elif sys.argv and sys.argv[0].endswith(".py") and bpy.app.background:
        argv = sys.argv[1:]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="renders/airliner_modes.mp4")
    p.add_argument("--res", default="1280x720")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--engine", default="eevee", choices=["eevee", "cycles"])
    p.add_argument("--samples", type=int)
    p.add_argument("--frames")
    return p.parse_args(argv)


def main():
    args = parse_args()
    sc = A.build()
    A.install_ui()
    for o in sc.objects:                  # let the director find layout labels by category
        if "ap_label" in o:
            o["ap_cat_label"] = o.name.replace("AP Label ", "").split(".")[0]
    w, h = (int(x) for x in args.res.lower().split("x"))
    r = sc.render
    r.resolution_x, r.resolution_y = w, h
    if args.engine == "eevee":
        engines = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
        r.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
        sc.eevee.taa_render_samples = args.samples or 12
    else:
        r.engine = "CYCLES"
        sc.cycles.samples = args.samples or 24
        r.use_persistent_data = True
    r.image_settings.file_format = "PNG"

    d = Director(sc)
    n = int(round(total_seconds() * args.fps))
    first, last = 0, n - 1
    if args.frames:
        first, last = (int(x) for x in args.frames.split("-"))
    out = os.path.abspath(args.out)
    frames_dir = os.path.splitext(out)[0] + "_frames"
    os.makedirs(frames_dir, exist_ok=True)
    t_start = time.time()
    done = 0
    for f in range(first, last + 1):
        path = os.path.join(frames_dir, f"frame_{f:04d}.png")
        if os.path.exists(path):
            continue
        d.apply(f / args.fps)
        r.filepath = path
        bpy.ops.render.render(write_still=True, scene=sc.name)
        done += 1
        el = time.time() - t_start
        print(f"[video] frame {f + 1}/{n}  {el / done:.1f}s/frame  ~{el / done * (last - f) / 60:.0f} min left",
              flush=True)
    if not args.frames:
        encode_video(frames_dir, out, args.fps, (w, h))


if __name__ == "__main__":
    main()

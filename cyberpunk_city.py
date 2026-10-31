#!/usr/bin/env python3
"""
cyberpunk_city.py - a rain-soaked cyberpunk street at night, for Blender + Cycles.

One file builds the whole scene: a street canyon of procedurally generated buildings
(shopfronts, ledges, balconies, AC units, pipes, rooftop clutter), towers and a hazy
megastructure skyline behind them, neon signage in Japanese and English (tube lettering,
lightbox blade signs, LED billboards, a hologram), street lamps, lantern strings,
overhead cables, a skybridge, cars, pedestrians with umbrellas, flying cars,
searchlights, rain streaks, steam, and volumetric fog.

Lit windows use an interior-mapping shader, so every room has depth and parallax.
Road, pavement, walls and metal use CC0 scanned textures from Poly Haven
(polyhaven.com), downloaded once into a cache folder. Without network access (or
with --offline) the script falls back to procedural materials.

Usage
-----
Blender GUI (4.2 or newer):
    Scripting workspace -> Text > Open... -> cyberpunk_city.py -> Run Script
    The city is built into a new scene called "Cyberpunk City". Press F12 to render.

Command line:
    blender -b -P cyberpunk_city.py -- --preset final --render city.png
    blender -b -P cyberpunk_city.py -- --save city.blend

Python with the `bpy` module from PyPI (pip install bpy):
    python cyberpunk_city.py --preset draft --render city.png

Options (put them after "--" when using the blender binary):
    --preset {draft,preview,final,ultra}   quality preset (default: preview)
    --res WxH, --samples N                 override the preset
    --seed N                               city layout seed (default: 4)
    --gpu                                  render on the GPU (OptiX/CUDA/HIP/Metal/oneAPI)
    --no-fog, --no-rain                    skip the volumetric haze / rain streaks
    --offline                              never download textures (procedural fallback)
    --tex-dir PATH                         texture cache (default: ./textures next to the script)
    --render PATH                          render a still (.exr = linear HDR)
    --save PATH                            save the generated .blend
    --pack                                 with --save: make the .blend self-contained
                                           (sign text as meshes, textures packed)
    --threads N                            CPU render threads (default: all)
"""

import argparse
import glob
import json
import math
import os
import random
import sys
import time
import urllib.request

import bpy
import numpy as np
from mathutils import Vector

BLENDER_5 = bpy.app.version >= (5, 0, 0)

# -----------------------------------------------------------------------------
#  Art direction
# -----------------------------------------------------------------------------

PRESETS = {
    "draft":   dict(res=(960, 540), samples=24),
    "preview": dict(res=(1600, 900), samples=48),
    "final":   dict(res=(2560, 1440), samples=96),     # 1440p
    "ultra":   dict(res=(3840, 2160), samples=160),    # 4K, GPU territory
}

LOOK = dict(
    cam_loc=(-1.6, 0.0, 1.35),
    cam_yaw=-2.5,          # degrees, + turns left
    cam_tilt=9.0,          # degrees, + looks up
    cam_roll=0.0,
    lens=22.0,
    shift_y=0.04,
    fstop=2.8,
    focus=14.0,
    exposure=-0.7,
    look="AgX - High Contrast",
    fog_low=0.0045,        # density of the haze below 30 m
    fog_high=0.0018,       # and above
    fog_aniso=0.55,
    rain_count=42000,
    rain_tilt=(0.10, 0.0),  # wind: x, y lean of the rain
)

STREET = dict(
    road_half=5.6,         # carriageway half width
    walk=3.6,              # pavement width
    length=760.0,          # street length (camera at y=0, looking +y)
    cross_y=(168.0, 186.0),  # a cross street
)

NEON = {
    "magenta": "#ff2a8a", "pink": "#ff5fc4", "cyan": "#18e6ff", "teal": "#00ffc8",
    "red": "#ff1f3a", "orange": "#ff7a1a", "amber": "#ffb000", "yellow": "#ffe14a",
    "violet": "#a84bff", "blue": "#2f6bff", "lime": "#9dff3a", "white": "#ffe9d6",
}

JP_WORDS = ["ラーメン", "居酒屋", "ホテル", "薬局", "カラオケ", "寿司", "焼肉", "麻雀", "質屋",
            "診療所", "電脳", "義体", "整備工場", "営業中", "二十四時間", "電気街", "未来",
            "酒場", "占い", "銭湯", "喫茶", "漫画", "遊技場", "宿", "餃子", "うどん", "龍",
            "夜市", "金融", "修理", "神経接続", "記憶屋", "義眼", "闇医者", "電子", "銀河"]
EN_WORDS = ["HOTEL", "BAR", "NOODLES", "OPEN 24H", "CYBERWARE", "KARAOKE", "ARCADE",
            "CLINIC", "RAMEN", "LIQUOR", "NEURAL", "DINER", "PAWN", "MOTEL", "SUSHI",
            "REPAIRS", "IMPLANTS", "LOUNGE", "NOVA", "OKAMI", "KAGE", "ZENITH", "CHROME"]
BRANDS = ["KAGEMORI", "NOVA-TEK", "OKAMI", "SYNAPSE", "HELIX", "YUREI", "ZENITH",
          "RAIJIN", "MIRAI", "KITSUNE"]

TEXTURES = {   # key -> Poly Haven asset id (CC0)
    "asphalt": "asphalt_02",
    "pavers": "concrete_pavers_03",
    "concrete": "concrete_wall_006",
    "tiles": "concrete_tile_facade",
    "rust": "rusty_metal_02",
    "shutter": "rusty_metal_shutter",
}


def log(msg):
    print(f"[city {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def hexcol(h, alpha=1.0):
    h = h.lstrip("#")
    srgb = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb]
    return (lin[0], lin[1], lin[2], alpha)


def kelvin(t):
    """Approximate linear RGB of a black body (normalised to max 1)."""
    t = t / 100.0
    r = 255 if t <= 66 else 329.7 * (t - 60) ** -0.1332
    g = 99.47 * math.log(t) - 161.1 if t <= 66 else 288.1 * (t - 60) ** -0.0755
    b = 255 if t >= 66 else (0 if t <= 19 else 138.5 * math.log(t - 10) - 305.0)
    c = [min(max(v, 0), 255) / 255.0 for v in (r, g, b)]
    c = [v ** 2.2 for v in c]
    m = max(c)
    return (c[0] / m, c[1] / m, c[2] / m, 1.0)


def nrm(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


# -----------------------------------------------------------------------------
#  Mesh construction
# -----------------------------------------------------------------------------

_ATTR_WIDTH = {"FLOAT": 1, "INT": 1, "FLOAT_VECTOR": 3, "FLOAT_COLOR": 4}
_ATTR_KEY = {"FLOAT": "value", "INT": "value", "FLOAT_VECTOR": "vector", "FLOAT_COLOR": "color"}


def set_attribute(me, name, values, domain, kind):
    attr = me.attributes.new(name=name, type=kind, domain=domain)
    dtype = np.int32 if kind == "INT" else np.float32
    attr.data.foreach_set(_ATTR_KEY[kind], np.ascontiguousarray(values, dtype=dtype).ravel())


def mesh_from_arrays(name, verts, faces_flat, sizes):
    me = bpy.data.meshes.new(name)
    verts = np.ascontiguousarray(verts, dtype=np.float32).reshape(-1)
    faces_flat = np.ascontiguousarray(faces_flat, dtype=np.int32)
    sizes = np.ascontiguousarray(sizes, dtype=np.int32)
    me.vertices.add(len(verts) // 3)
    me.vertices.foreach_set("co", verts)
    if len(sizes):
        starts = np.zeros(len(sizes), dtype=np.int32)
        starts[1:] = np.cumsum(sizes)[:-1]
        me.loops.add(len(faces_flat))
        me.loops.foreach_set("vertex_index", faces_flat)
        me.polygons.add(len(sizes))
        me.polygons.foreach_set("loop_start", starts)
        if bpy.app.version < (4, 0, 0):
            me.polygons.foreach_set("loop_total", sizes)
    me.update(calc_edges=True)
    return me


class MeshBuilder:
    """Collects quads/tris (with UVs, material indices and face attributes) from many
    small pieces and builds a single Blender mesh. Faces of one add() call share a size."""

    def __init__(self):
        self.verts, self.faces, self.uvs, self.mats, self.fattr = [], [], [], [], []
        self.nv = 0

    def add(self, verts, faces, uvs=None, mat=0, fattrs=None):
        verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
        faces = np.asarray(faces, dtype=np.int64)
        if faces.ndim == 1:
            faces = faces[None]
        if uvs is None:
            uvs = np.zeros(faces.shape + (2,))
        self.verts.append(verts)
        self.faces.append(faces + self.nv)
        self.uvs.append(np.asarray(uvs, dtype=np.float64).reshape(faces.shape + (2,)))
        self.mats.append(np.broadcast_to(np.asarray(mat, dtype=np.int32), (len(faces),)))
        self.fattr.append(fattrs or {})
        self.nv += len(verts)

    def build(self, name, materials=(), smooth=False):
        if not self.faces:
            return None
        V = np.concatenate(self.verts)
        # group by polygon size, keeping order inside each group
        sizes = np.concatenate([np.full(len(f), f.shape[1], dtype=np.int32) for f in self.faces])
        flat = np.concatenate([f.ravel() for f in self.faces])
        me = mesh_from_arrays(name, V, flat, sizes)
        uvl = me.uv_layers.new(name="UVMap")
        uvl.data.foreach_set("uv", np.concatenate([u.reshape(-1, 2) for u in self.uvs]).astype(np.float32).ravel())
        me.polygons.foreach_set("material_index", np.concatenate(self.mats).astype(np.int32))
        kinds = {}
        for c in self.fattr:
            for k, (kind, _) in c.items():
                kinds[k] = kind
        for nm, kind in kinds.items():
            w = _ATTR_WIDTH[kind]
            parts = []
            for c, f in zip(self.fattr, self.faces):
                shape = (len(f),) if w == 1 else (len(f), w)
                if nm in c:
                    parts.append(np.broadcast_to(np.asarray(c[nm][1], dtype=np.float64), shape))
                else:
                    parts.append(np.zeros(shape))
            set_attribute(me, nm, np.concatenate(parts), "FACE", kind)
        for m in materials:
            me.materials.append(m)
        if smooth:
            me.shade_smooth()
        me.update()
        return me


_BOX_FACES = np.array([(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
_BOX_CORNERS = np.array([(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                         (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], dtype=np.float64) * 0.5


def world_uv(P, n):
    """Metre-scale UVs: (x, y) on horizontal faces, (along-wall, z) on walls. The wall
    direction is (-ny, nx, 0), the same tangent the facade shader uses."""
    if abs(n[2]) > 0.5:
        return P[:, :2]
    t = np.array([-n[1], n[0], 0.0])
    t /= max(np.linalg.norm(t), 1e-9)
    return np.stack([P @ t, P[:, 2]], axis=1)


def add_box(mb, c, s, mat, rz=0.0, fattrs=None, faces=None):
    """Axis-aligned (optionally Z-rotated) box from centre and size. `faces` picks
    a subset of the six faces (indices into _BOX_FACES: 0 bottom, 1 top, 2 -y, 3 +x, 4 +y, 5 -x)."""
    V = _BOX_CORNERS * np.asarray(s, dtype=np.float64)
    if rz:
        cz, sz = math.cos(rz), math.sin(rz)
        V = V @ np.array([[cz, sz, 0], [-sz, cz, 0], [0, 0, 1]])
    V = V + np.asarray(c, dtype=np.float64)
    F = _BOX_FACES if faces is None else _BOX_FACES[list(faces)]
    uvs = []
    for f in F:
        P = V[f]
        n = np.cross(P[1] - P[0], P[2] - P[0])
        n /= max(np.linalg.norm(n), 1e-12)
        uvs.append(world_uv(P, n))
    mb.add(V, F, np.array(uvs), mat, fattrs)


def tube(pts, rad, sides=8, cap=True):
    """Tube along a polyline. Returns verts, quads, uvs."""
    P = np.asarray(pts, dtype=np.float64)
    R = np.broadcast_to(np.asarray(rad, dtype=np.float64), (len(P),))
    n = len(P)
    T = np.empty_like(P)
    T[1:-1] = P[2:] - P[:-2]
    T[0] = P[1] - P[0]
    T[-1] = P[-1] - P[-2]
    T = nrm(T)
    ref = np.array([1.0, 0.0, 0.0]) if abs(T[0][2]) > 0.9 else np.array([0.0, 0.0, 1.0])
    N = nrm(ref[None, :] - (T @ ref)[:, None] * T)
    B = np.cross(T, N)
    th = np.arange(sides) * (2 * np.pi / sides)
    ring = np.cos(th)[None, :, None] * N[:, None, :] + np.sin(th)[None, :, None] * B[:, None, :]
    V = (P[:, None, :] + R[:, None, None] * ring).reshape(-1, 3)
    i = np.arange(n - 1)[:, None]
    j = np.arange(sides)[None, :]
    j1 = (j + 1) % sides
    Q = np.stack([i * sides + j, i * sides + j1, (i + 1) * sides + j1, (i + 1) * sides + j], -1).reshape(-1, 4)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    u0 = np.broadcast_to(j / sides, (n - 1, sides))
    u1 = np.broadcast_to((j + 1) / sides, (n - 1, sides))
    v0 = np.broadcast_to(s[:-1, None], (n - 1, sides))
    v1 = np.broadcast_to(s[1:, None], (n - 1, sides))
    UV = np.stack([np.stack([u0, v0], -1), np.stack([u1, v0], -1),
                   np.stack([u1, v1], -1), np.stack([u0, v1], -1)], axis=2).reshape(-1, 4, 2)
    if cap:
        # close the ends with quads fanned to a centre ring (keeps everything quads)
        extra_v, extra_q, extra_uv = [], [], []
        base = len(V)
        for end, idx in ((0, 0), (1, n - 1)):
            c = P[idx]
            extra_v.append(c)
            ci = base + len(extra_v) - 1
            r0 = idx * sides
            for k in range(0, sides, 2):
                a, b, cc = r0 + k, r0 + (k + 1) % sides, r0 + (k + 2) % sides
                extra_q.append((ci, cc, b, a) if end == 0 else (ci, a, b, cc))
                extra_uv.append([(0, 0), (0, 0), (0, 0), (0, 0)])
        V = np.vstack([V, extra_v])
        Q = np.vstack([Q, extra_q])
        UV = np.vstack([UV, np.array(extra_uv, dtype=np.float64)])
    return V, Q, UV


def add_tube(mb, pts, rad, sides, mat, cap=True, fattrs=None):
    V, Q, UV = tube(pts, rad, sides, cap)
    mb.add(V, Q, UV, mat, fattrs)


def add_sphere(mb, c, r, mat, nu=12, nv=8, scale=(1, 1, 1)):
    th = np.linspace(0, np.pi, nv + 1)
    ph = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    T, Ph = np.meshgrid(th, ph, indexing="ij")
    V = np.stack([np.sin(T) * np.cos(Ph), np.sin(T) * np.sin(Ph), np.cos(T)], -1).reshape(-1, 3)
    V = V * r * np.asarray(scale) + np.asarray(c)
    i = np.arange(nv)[:, None]
    j = np.arange(nu)[None, :]
    j1 = (j + 1) % nu
    Q = np.stack([i * nu + j, (i + 1) * nu + j, (i + 1) * nu + j1, i * nu + j1], -1).reshape(-1, 4)
    UV = np.zeros((len(Q), 4, 2))
    mb.add(V, Q, UV, mat)


def new_object(name, data, coll):
    ob = bpy.data.objects.new(name, data)
    coll.objects.link(ob)
    return ob


# -----------------------------------------------------------------------------
#  Node-tree helper
# -----------------------------------------------------------------------------

_SOCKET_ALIASES = {"Fac": "Factor", "Factor": "Fac"}


class NB:
    """Tiny helper to build node trees tersely."""

    def __init__(self, tree):
        self.tree, self.nodes, self.links = tree, tree.nodes, tree.links

    @staticmethod
    def sock(sockets, key):
        if isinstance(key, int):
            return sockets[key]
        for k in (key, _SOCKET_ALIASES.get(key)):
            if k is None:
                continue
            for s in sockets:
                if s.identifier == k:
                    return s
            match = [s for s in sockets if s.name == k]
            for s in match:
                if s.enabled:
                    return s
            if match:
                return match[0]
        raise KeyError(f"socket {key!r} not in {[s.name for s in sockets]}")

    def out(self, x, key=None):
        if isinstance(x, bpy.types.NodeSocket):
            return x
        if key is not None:
            return self.sock(x.outputs, key)
        for s in x.outputs:
            if s.enabled:
                return s
        return x.outputs[0]

    def set(self, node, key, value):
        s = self.sock(node.inputs, key)
        if isinstance(value, (bpy.types.NodeSocket, bpy.types.Node)):
            self.links.new(self.out(value), s)
            return
        if isinstance(value, str):
            value = hexcol(value)
        dv = s.default_value
        if hasattr(dv, "__len__"):
            if isinstance(value, (int, float)):
                value = (value,) * len(dv)
                if len(dv) == 4:
                    value = value[:3] + (1.0,)
            elif len(value) == 3 and len(dv) == 4:
                value = tuple(value) + (1.0,)
            elif len(value) == 4 and len(dv) == 3:
                value = tuple(value[:3])
        s.default_value = value

    def node(self, idname, inputs=None, **props):
        n = self.nodes.new(idname)
        for k, v in props.items():
            setattr(n, k, v)
        for k, v in (inputs or {}).items():
            self.set(n, k, v)
        return n

    def math(self, op, a, b=0.0, c=None, clamp=False):
        n = self.node("ShaderNodeMath", operation=op, use_clamp=clamp)
        self.set(n, 0, a)
        self.set(n, 1, b)
        if c is not None:
            self.set(n, 2, c)
        return n.outputs[0]

    # arithmetic sugar
    def add(self, a, b): return self.math("ADD", a, b)
    def sub(self, a, b): return self.math("SUBTRACT", a, b)
    def mul(self, a, b): return self.math("MULTIPLY", a, b)
    def div(self, a, b): return self.math("DIVIDE", a, b)
    def gt(self, a, b): return self.math("GREATER_THAN", a, b)
    def lt(self, a, b): return self.math("LESS_THAN", a, b)
    def mn(self, a, b): return self.math("MINIMUM", a, b)
    def mx(self, a, b): return self.math("MAXIMUM", a, b)
    def frac(self, a): return self.math("FRACT", a)
    def floor(self, a): return self.math("FLOOR", a)
    def absf(self, a): return self.math("ABSOLUTE", a)

    def band(self, x, lo, hi):
        """1 where lo < x < hi."""
        return self.mul(self.gt(x, lo), self.lt(x, hi))

    def vmath(self, op, a, b=None, scale=None):
        n = self.node("ShaderNodeVectorMath", operation=op)
        self.set(n, 0, a)
        if b is not None:
            self.set(n, 1, b)
        if scale is not None:
            self.set(n, "Scale", scale)
        return n.outputs[1] if op in ("DOT_PRODUCT", "LENGTH", "DISTANCE") else n.outputs[0]

    def mapr(self, v, a, b, c=0.0, d=1.0, clamp=True, interp="LINEAR"):
        n = self.node("ShaderNodeMapRange", clamp=clamp, interpolation_type=interp)
        for k, val in (("Value", v), ("From Min", a), ("From Max", b), ("To Min", c), ("To Max", d)):
            self.set(n, k, val)
        return self.sock(n.outputs, "Result")

    def mix(self, fac, a, b, blend="MIX"):
        n = self.node("ShaderNodeMix", data_type="RGBA", blend_type=blend)
        self.set(n, "Factor_Float", fac)
        self.set(n, "A_Color", a)
        self.set(n, "B_Color", b)
        return self.sock(n.outputs, "Result_Color")

    def mixf(self, fac, a, b):
        n = self.node("ShaderNodeMix", data_type="FLOAT")
        self.set(n, "Factor_Float", fac)
        self.set(n, "A_Float", a)
        self.set(n, "B_Float", b)
        return self.sock(n.outputs, "Result_Float")

    def ramp(self, fac, stops, interp="LINEAR"):
        n = self.node("ShaderNodeValToRGB")
        cr = n.color_ramp
        cr.interpolation = interp
        conv = [(p, hexcol(c) if isinstance(c, str) else tuple(c)) for p, c in stops]
        cr.elements[0].position, cr.elements[0].color = conv[0]
        cr.elements[1].position, cr.elements[1].color = conv[-1]
        for p, c in conv[1:-1]:
            cr.elements.new(p).color = c
        self.set(n, 0, fac)
        return n.outputs[0]

    def noise(self, vec, scale, detail=4.0, rough=0.5, distortion=0.0, dims="3D"):
        n = self.node("ShaderNodeTexNoise", noise_dimensions=dims)
        if vec is not None:
            self.set(n, "Vector", vec)
        for k, val in (("Scale", scale), ("Detail", detail), ("Roughness", rough), ("Distortion", distortion)):
            self.set(n, k, val)
        return n

    def voronoi(self, vec, scale, feature="F1", randomness=1.0):
        n = self.node("ShaderNodeTexVoronoi", voronoi_dimensions="3D", feature=feature)
        if vec is not None:
            self.set(n, "Vector", vec)
        self.set(n, "Scale", scale)
        self.set(n, "Randomness", randomness)
        return n

    def white(self, vec):
        n = self.node("ShaderNodeTexWhiteNoise", noise_dimensions="3D")
        self.set(n, "Vector", vec)
        return n

    def attr(self, name, out="Fac"):
        n = self.node("ShaderNodeAttribute", attribute_name=name)
        return self.sock(n.outputs, out)

    def sep(self, vec):
        n = self.node("ShaderNodeSeparateXYZ")
        self.set(n, 0, vec)
        return n

    def comb(self, x, y, z):
        return self.node("ShaderNodeCombineXYZ", {0: x, 1: y, 2: z}).outputs[0]

    def bump(self, height, strength, distance, normal=None):
        n = self.node("ShaderNodeBump", {"Height": height, "Strength": strength, "Distance": distance})
        if normal is not None:
            self.set(n, "Normal", normal)
        return n.outputs[0]

    def principled(self, **kw):
        names = {"base": "Base Color", "rough": "Roughness", "normal": "Normal", "metal": "Metallic",
                 "spec": "Specular IOR Level", "coat": "Coat Weight", "coat_rough": "Coat Roughness",
                 "emit": "Emission Color", "emit_str": "Emission Strength", "alpha": "Alpha",
                 "trans": "Transmission Weight", "ior": "IOR", "sss": "Subsurface Weight"}
        return self.node("ShaderNodeBsdfPrincipled", {names[k]: v for k, v in kw.items()})

    def output(self, surface=None, volume=None):
        o = self.node("ShaderNodeOutputMaterial")
        if surface is not None:
            self.set(o, "Surface", surface)
        if volume is not None:
            self.set(o, "Volume", volume)
        return o


def new_material(name):
    m = bpy.data.materials.new(name)
    if not BLENDER_5:
        m.use_nodes = True
    m.node_tree.nodes.clear()
    return m, NB(m.node_tree)


# -----------------------------------------------------------------------------
#  Scanned textures (Poly Haven, CC0) with a procedural fallback
# -----------------------------------------------------------------------------

class TextureLibrary:
    MAPS = {"diff": "Diffuse", "rough": "Rough", "disp": "Displacement"}

    def __init__(self, folder, offline, res="2k"):
        self.folder, self.offline, self.res = folder, offline, res
        self.sets = {}
        self.images = {}

    def _get(self, url, path):
        req = urllib.request.Request(url, headers={"User-Agent": "cyberpunk-city-blender/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)

    def fetch(self, key, asset):
        paths = {m: os.path.join(self.folder, f"{asset}_{m}_{self.res}.jpg") for m in self.MAPS}
        if all(os.path.exists(p) for p in paths.values()):
            self.sets[key] = paths
            return
        if self.offline:
            log(f"texture {asset}: offline, using a procedural stand-in")
            return
        try:
            os.makedirs(self.folder, exist_ok=True)
            req = urllib.request.Request(f"https://api.polyhaven.com/files/{asset}",
                                         headers={"User-Agent": "cyberpunk-city-blender/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                files = json.load(r)
            for m, api_key in self.MAPS.items():
                if not os.path.exists(paths[m]):
                    self._get(files[api_key][self.res]["jpg"]["url"], paths[m])
            self.sets[key] = paths
            log(f"texture {asset} ready")
        except Exception as e:                        # offline, renamed asset, ...
            log(f"texture {asset} unavailable ({e.__class__.__name__}); using a procedural stand-in")

    def fetch_all(self):
        for key, asset in TEXTURES.items():
            self.fetch(key, asset)

    def image(self, path, color):
        if path not in self.images:
            img = bpy.data.images.load(path, check_existing=True)
            img.colorspace_settings.name = "sRGB" if color else "Non-Color"
            self.images[path] = img
        return self.images[path]

    def nodes(self, nb, key, vec, fallback="#5a5a5a", fb_scale=3.0):
        """(color, roughness, height) sockets for texture `key` sampled at `vec`."""
        s = self.sets.get(key)
        if s is None:
            n1 = nb.noise(vec, fb_scale, 8, 0.6)
            n2 = nb.noise(vec, fb_scale * 9, 4, 0.5)
            col = nb.mix(nb.mapr(nb.out(n1, "Fac"), 0.3, 0.7), hexcol(fallback),
                         tuple(c * 0.6 for c in hexcol(fallback)[:3]))
            rough = nb.mapr(nb.out(n2, "Fac"), 0.3, 0.7, 0.55, 0.85)
            h = nb.add(nb.out(n1, "Fac"), nb.mul(nb.out(n2, "Fac"), 0.3))
            return col, rough, h
        out = []
        for m in ("diff", "rough", "disp"):
            t = nb.node("ShaderNodeTexImage", interpolation="Linear")
            t.image = self.image(s[m], m == "diff")
            nb.set(t, "Vector", vec)
            out.append(nb.out(t, "Color"))
        return out[0], nb.math("ADD", out[1], 0.0), nb.math("ADD", out[2], 0.0)


# -----------------------------------------------------------------------------
#  Materials
# -----------------------------------------------------------------------------

def _uv_meters(nb):
    return nb.out(nb.node("ShaderNodeUVMap", uv_map="UVMap"), "UV")


def _puddles(nb, uv, scale, bias):
    """0..1 puddle mask (sharpish edges) from metre UVs."""
    n1 = nb.out(nb.noise(uv, scale, 6, 0.55), "Fac")
    n2 = nb.out(nb.noise(uv, scale * 5.0, 3, 0.5), "Fac")
    h = nb.add(n1, nb.mul(nb.sub(n2, 0.5), 0.12))
    h = nb.add(h, bias)
    return nb.mapr(h, 0.56, 0.585, 0.0, 1.0, interp="SMOOTHSTEP")


def _ripples(nb, uv):
    """Height field of rain-drop ripple rings."""
    vo = nb.voronoi(nb.vmath("MULTIPLY", uv, (2.2, 2.2, 1.0)), 1.0)
    d = nb.out(vo, "Distance")
    rnd = nb.sep(nb.out(vo, "Color"))
    phase = nb.mul(nb.out(rnd, 0), 12.0)
    ring = nb.math("SINE", nb.sub(nb.mul(d, 70.0), phase))
    fade = nb.mapr(d, 0.0, 0.35, 1.0, 0.0)
    life = nb.mapr(nb.out(rnd, 1), 0.0, 1.0, 0.2, 1.0)
    return nb.mul(nb.mul(ring, fade), life)


def mat_road(tex):
    m, nb = new_material("Road_WetAsphalt")
    uv = _uv_meters(nb)
    vec = nb.vmath("MULTIPLY", uv, (1 / 3.5, 1 / 3.5, 1))
    col, rough, h = tex.nodes(nb, "asphalt", vec, "#3b3a38", 4.0)
    sx = nb.sep(uv)
    x, y = nb.out(sx, 0), nb.out(sx, 1)
    ax = nb.absf(x)
    rh = STREET["road_half"]
    on_road = nb.lt(ax, rh)
    # paint: dashed centre line, solid edge lines, two zebra crossings
    centre = nb.mul(nb.lt(ax, 0.075), nb.lt(nb.frac(nb.div(y, 9.0)), 0.55))
    edge = nb.band(ax, rh - 0.42, rh - 0.28)
    zebra_z = nb.add(nb.band(y, 9.0, 13.5), nb.band(y, STREET["cross_y"][0] - 5.5, STREET["cross_y"][0] - 1.0))
    zebra = nb.mul(nb.mul(zebra_z, nb.lt(nb.frac(nb.div(x, 1.1)), 0.5)), nb.lt(ax, rh - 0.6))
    stop = nb.band(y, 14.2, 14.6)
    paint = nb.math("MINIMUM", nb.add(nb.add(centre, edge), nb.add(zebra, nb.mul(stop, nb.lt(x, 0)))), 1.0)
    wear = nb.mapr(nb.out(nb.noise(uv, 1.7, 6, 0.65), "Fac"), 0.35, 0.6)
    paint = nb.mul(nb.mul(paint, wear), on_road)
    # wetness: tyre tracks a little smoother, gutters and dips hold water
    lane = nb.frac(nb.div(nb.add(x, rh), rh))
    tracks = nb.add(nb.band(lane, 0.2, 0.36), nb.band(lane, 0.64, 0.8))
    gutter = nb.mapr(ax, rh - 1.2, rh, 0.0, 0.08)
    pud = _puddles(nb, uv, 0.22, nb.add(nb.mul(tracks, 0.03), gutter))
    wet_col = nb.mix(0.55, col, (0.0, 0.0, 0.0))
    paint_col = nb.mix(nb.lt(ax, 0.1), (0.62, 0.6, 0.56), (0.75, 0.48, 0.08))    # centre line is yellow
    base = nb.mix(paint, wet_col, paint_col)
    base = nb.mix(pud, base, nb.mix(0.6, base, (0, 0, 0)))
    r = nb.mapr(rough, 0.0, 1.0, 0.18, 0.5)
    r = nb.mixf(tracks, r, nb.mul(r, 0.75))
    r = nb.mixf(paint, r, 0.32)
    r = nb.mixf(pud, r, 0.015)
    bump = nb.bump(h, nb.mul(nb.sub(1.0, pud), 0.55), 0.02)
    rip = _ripples(nb, uv)
    bump = nb.bump(rip, nb.mul(pud, 0.12), 0.004, bump)
    bsdf = nb.principled(base=base, rough=r, normal=bump, spec=0.5)
    nb.output(bsdf)
    return m


def mat_pavement(tex):
    m, nb = new_material("Pavement_Wet")
    uv = _uv_meters(nb)
    vec = nb.vmath("MULTIPLY", uv, (1 / 2.0, 1 / 2.0, 1))
    col, rough, h = tex.nodes(nb, "pavers", vec, "#6a6863", 2.0)
    pud = _puddles(nb, nb.vmath("MULTIPLY", uv, (1.7, 1.7, 1)), 0.25, -0.02)
    base = nb.mix(0.5, col, (0, 0, 0))
    base = nb.mix(pud, base, nb.mix(0.55, base, (0, 0, 0)))
    r = nb.mapr(rough, 0, 1, 0.22, 0.55)
    r = nb.mixf(pud, r, 0.02)
    bump = nb.bump(h, nb.mul(nb.sub(1.0, pud), 0.6), 0.02)
    bump = nb.bump(_ripples(nb, uv), nb.mul(pud, 0.1), 0.004, bump)
    nb.output(nb.principled(base=base, rough=r, normal=bump))
    return m


def mat_concrete(tex, name="Concrete_Wet", tint="#8c8a86", scale=3.0, key="concrete"):
    m, nb = new_material(name)
    uv = _uv_meters(nb)
    vec = nb.vmath("MULTIPLY", uv, (1 / scale, 1 / scale, 1))
    col, rough, h = tex.nodes(nb, key, vec, tint, 1.5)
    streak = nb.out(nb.noise(nb.vmath("MULTIPLY", uv, (3.0, 0.12, 1)), 2.0, 5, 0.6), "Fac")
    wet = nb.mapr(streak, 0.4, 0.65)
    base = nb.mix(0.5, col, hexcol(tint), "MULTIPLY")
    base = nb.mix(nb.mul(wet, 0.5), base, (0, 0, 0))
    r = nb.mixf(wet, nb.mapr(rough, 0, 1, 0.45, 0.85), 0.2)
    nb.output(nb.principled(base=base, rough=r, normal=nb.bump(h, 0.5, 0.02)))
    return m


def mat_metal(tex, name, color, rough=0.35, metal=1.0, key="rust", rust=0.0):
    m, nb = new_material(name)
    uv = _uv_meters(nb)
    col, rr, h = tex.nodes(nb, key, nb.vmath("MULTIPLY", uv, (0.7, 0.7, 1)), "#5b4636", 3.0)
    mask = nb.mapr(nb.out(nb.noise(uv, 1.2, 6, 0.6), "Fac"), 0.62 - rust, 0.7 - rust)
    base = nb.mix(mask, color, col)
    r = nb.mixf(mask, rough, nb.mapr(rr, 0, 1, 0.5, 0.9))
    met = nb.mixf(mask, metal, 0.2)
    nb.output(nb.principled(base=base, rough=r, metal=met, normal=nb.bump(h, nb.mul(mask, 0.4), 0.01)))
    return m


def mat_simple(name, color, rough=0.5, metal=0.0, coat=0.0, spec=0.5):
    m, nb = new_material(name)
    nb.output(nb.principled(base=color, rough=rough, metal=metal, coat=coat, spec=spec))
    return m


def mat_glass_dark(name="Glass_Dark", tint=(0.01, 0.012, 0.015), rough=0.02):
    m, nb = new_material(name)
    nb.output(nb.principled(base=tint, rough=rough, spec=0.7, coat=0.0))
    return m


def mat_emit(name, color, strength):
    m, nb = new_material(name)
    em = nb.node("ShaderNodeEmission", {"Color": color, "Strength": strength})
    nb.output(em)
    return m


def mat_neon(name, color, strength):
    """Neon tube: emission with a hot white-ish core and small per-object flicker/aging."""
    m, nb = new_material(name)
    oi = nb.node("ShaderNodeObjectInfo")
    rnd = nb.out(oi, "Random")
    s = nb.mul(nb.mapr(rnd, 0, 1, 0.75, 1.15), strength)
    lw = nb.node("ShaderNodeLayerWeight", {"Blend": 0.35})
    core = nb.mix(nb.mapr(nb.out(lw, "Facing"), 0.0, 0.6, 0.35, 0.0), color, (1, 0.95, 0.92))
    em = nb.node("ShaderNodeEmission", {"Color": core, "Strength": s})
    nb.output(em)
    return m


def mat_lightbox(name, color, strength, dirty=True):
    """Backlit acrylic sign face: bright, slightly uneven, a glossy dark rim."""
    m, nb = new_material(name)
    uv = _uv_meters(nb)
    # uneven backlight (the tubes behind the acrylic) and grime
    hot = nb.mapr(nb.out(nb.noise(uv, 0.6, 2, 0.5), "Fac"), 0.35, 0.65, 0.7, 1.2)
    dirt = nb.mapr(nb.out(nb.noise(uv, 3.0, 6, 0.6), "Fac"), 0.3, 0.7, 0.7, 1.0) if dirty else 1.0
    s = nb.mul(nb.mul(hot, dirt), strength)
    nb.output(nb.principled(base=color, rough=0.25, emit=color, emit_str=s))
    return m


def mat_led_screen(name, design, colors, strength):
    """Procedural LED billboard: animated-looking ad graphics plus a visible pixel grid."""
    m, nb = new_material(name)
    uv = _uv_meters(nb)
    gen = nb.out(nb.node("ShaderNodeTexCoord"), "Generated")
    g = nb.sep(gen)
    gx, gz = nb.out(g, 0), nb.out(g, 2)
    c1, c2, c3 = (hexcol(c) for c in colors)
    if design == 0:          # diagonal gradient with liquid waves
        w = nb.out(nb.noise(nb.vmath("MULTIPLY", gen, (2, 2, 4)), 1.5, 4, 0.6, 1.5), "Fac")
        f = nb.mapr(nb.add(nb.mul(gx, 0.6), nb.add(nb.mul(gz, 0.6), nb.mul(w, 0.6))), 0.3, 1.3)
        img = nb.mix(f, nb.mix(0.85, c1, (0, 0, 0)), c2)
        band = nb.mapr(nb.absf(nb.sub(w, 0.55)), 0.0, 0.04, 1.0, 0.0)
        img = nb.mix(band, img, c3)
    elif design == 1:        # big glowing ring on dark
        cx = nb.sub(gx, 0.5)
        cz = nb.mul(nb.sub(gz, 0.55), 1.6)
        r = nb.math("SQRT", nb.add(nb.mul(cx, cx), nb.mul(cz, cz)))
        ring = nb.math("POWER", nb.mapr(nb.absf(nb.sub(r, 0.32)), 0.0, 0.12, 1.0, 0.0), 2.0)
        bg = nb.mix(gz, c3, (0.0, 0.0, 0.0))
        img = nb.mix(ring, bg, c1)
        img = nb.mix(nb.mul(nb.lt(r, 0.2), 0.6), img, c2)
    else:                    # glitch bars
        bars = nb.out(nb.white(nb.comb(0.0, nb.floor(nb.mul(gz, 38.0)), 3.3)), "Value")
        shift = nb.frac(nb.add(gx, nb.mul(bars, 0.3)))
        img = nb.mix(nb.lt(shift, 0.5), c1, c2)
        img = nb.mix(nb.gt(bars, 0.85), img, c3)
        img = nb.mix(nb.lt(bars, 0.12), img, (0, 0, 0))
    # scanline + pixel grid
    px = nb.vmath("MULTIPLY", uv, (28.0, 28.0, 1.0))
    sp = nb.sep(px)
    grid = nb.mul(nb.lt(nb.frac(nb.out(sp, 0)), 0.72), nb.lt(nb.frac(nb.out(sp, 1)), 0.72))
    grid = nb.mapr(grid, 0, 1, 0.35, 1.0)
    scan = nb.mapr(nb.math("SINE", nb.mul(gz, 220.0)), -1, 1, 0.85, 1.0)
    s = nb.mul(nb.mul(grid, scan), strength)
    nb.output(nb.principled(base=(0.01, 0.01, 0.01), rough=0.15, emit=img, emit_str=s))
    return m


def mat_hologram(name, color, strength):
    m, nb = new_material(name)
    tc = nb.node("ShaderNodeTexCoord")
    ob = nb.sep(nb.out(tc, "Object"))
    scan = nb.mapr(nb.math("SINE", nb.mul(nb.out(ob, 2), 26.0)), -1, 1, 0.35, 1.0)
    noi = nb.mapr(nb.out(nb.noise(nb.out(tc, "Object"), 0.8, 3, 0.5), "Fac"), 0.3, 0.7, 0.5, 1.2)
    lw = nb.node("ShaderNodeLayerWeight", {"Blend": 0.5})
    rim = nb.mapr(nb.out(lw, "Facing"), 0.0, 1.0, 0.25, 1.0)
    em = nb.node("ShaderNodeEmission", {"Color": color, "Strength": nb.mul(nb.mul(nb.mul(scan, noi), rim), strength)})
    tr = nb.node("ShaderNodeBsdfTransparent")
    add = nb.node("ShaderNodeAddShader")
    nb.set(add, 0, tr)
    nb.set(add, 1, em)
    nb.output(add)
    return m


def mat_facade(tex):
    """Building walls with windows. Per-face attributes (written by the building
    generator) choose the window grid, which rooms are lit and the wall tint.
    Lit windows use interior mapping: the view ray is intersected with a virtual room
    behind the glass, so walls, floor and ceiling show proper parallax."""
    m, nb = new_material("Facade")
    tc = nb.node("ShaderNodeTexCoord")
    P = nb.out(tc, "Object")
    geo = nb.node("ShaderNodeNewGeometry")
    N = nb.out(geo, "Normal")
    I = nb.out(geo, "Incoming")
    W = nb.attr("win_w")
    H = nb.attr("floor_h")
    z0 = nb.attr("z0")
    htop = nb.attr("htop")
    litp = nb.attr("lit")
    brand = nb.attr("brand")
    mxm = nb.attr("mx")
    mbm = nb.attr("mb")
    mtm = nb.attr("mt")
    warmth = nb.attr("warmth")
    glassy = nb.attr("glassy")
    tint = nb.attr("tint", "Color")
    sN = nb.sep(N)
    Nx, Ny, Nz = nb.out(sN, 0), nb.out(sN, 1), nb.out(sN, 2)
    t = nb.comb(nb.mul(Ny, -1.0), Nx, 0.0)
    u = nb.sub(nb.vmath("DOT_PRODUCT", P, t), nb.attr("u0"))
    sP = nb.sep(P)
    v = nb.out(sP, 2)
    vert = nb.lt(nb.absf(Nz), 0.5)
    cu = nb.div(u, W)
    cv = nb.div(nb.sub(v, z0), H)
    fu, fv = nb.frac(cu), nb.frac(cv)
    iu, iv = nb.floor(cu), nb.floor(cv)
    inband = nb.mul(nb.gt(v, z0), nb.lt(v, nb.sub(htop, 0.9)))
    win = nb.mul(nb.mul(nb.band(fu, mxm, nb.sub(1.0, mxm)), nb.band(fv, mbm, nb.sub(1.0, mtm))),
                 nb.mul(vert, inband))
    # frame: a slightly larger rectangle around the glass
    fr = 0.035
    winx = nb.mul(nb.mul(nb.band(fu, nb.sub(mxm, fr), nb.sub(nb.add(1.0, fr), mxm)),
                         nb.band(fv, nb.sub(mbm, fr), nb.sub(nb.add(1.0, fr), mtm))),
                  nb.mul(vert, inband))
    # a mullion in the middle of wide windows
    mull = nb.mul(nb.lt(nb.absf(nb.sub(fu, 0.5)), nb.div(0.025, W)), nb.gt(W, 2.2))
    frame = nb.mx(nb.sub(winx, win), nb.mul(mull, win))
    glass = nb.mul(win, nb.sub(1.0, mull))

    # per-window random numbers
    seedv = nb.comb(iu, iv, nb.mul(brand, 517.0))
    w1 = nb.white(seedv)
    r1 = nb.out(w1, "Value")
    rc = nb.sep(nb.out(w1, "Color"))
    r2, r3, r4 = nb.out(rc, 0), nb.out(rc, 1), nb.out(rc, 2)
    w2 = nb.white(nb.vmath("ADD", seedv, (0.37, 0.71, 3.3)))
    q = nb.sep(nb.out(w2, "Color"))
    r5, r6, r7 = nb.out(q, 0), nb.out(q, 1), nb.out(q, 2)
    lit = nb.lt(r1, litp)
    tint_key = nb.mapr(nb.add(nb.mul(r2, 0.75), nb.mul(warmth, 0.35)), 0.0, 1.1)
    room_col = nb.ramp(tint_key, [
        (0.00, (0.75, 0.86, 1.0, 1)),       # cold office LED
        (0.16, (0.86, 1.0, 0.92, 1)),       # greenish fluorescent
        (0.28, (1.0, 0.92, 0.80, 1)),       # neutral
        (0.48, (1.0, 0.70, 0.42, 1)),       # warm
        (0.70, (1.0, 0.56, 0.26, 1)),       # tungsten
        (0.86, (1.0, 0.30, 0.62, 1)),       # pink party light
        (0.93, (0.45, 0.30, 1.0, 1)),       # violet gamer room
    ], "CONSTANT")
    bright = nb.mul(nb.mapr(r3, 0, 1, 0.25, 1.25), nb.attr("glow"))

    # --- interior mapping -------------------------------------------------
    depth = 4.5
    d = nb.vmath("SCALE", I, scale=-1.0)
    du = nb.div(nb.vmath("DOT_PRODUCT", d, t), W)
    dv = nb.div(nb.out(nb.sep(d), 2), H)
    dn = nb.div(nb.mul(nb.vmath("DOT_PRODUCT", d, N), -1.0), depth)
    dn = nb.mx(dn, 0.02)
    tx = nb.div(nb.sub(nb.gt(du, 0.0), fu), nb.add(du, nb.mul(nb.math("SIGN", du), 1e-5)))
    ty = nb.div(nb.sub(nb.gt(dv, 0.0), fv), nb.add(dv, nb.mul(nb.math("SIGN", dv), 1e-5)))
    tx = nb.mx(tx, 0.0)
    ty = nb.mx(ty, 0.0)
    tz = nb.div(1.0, dn)
    txy = nb.mn(tx, ty)
    tm = nb.mn(txy, tz)
    hx = nb.add(fu, nb.mul(du, tm))
    hy = nb.add(fv, nb.mul(dv, tm))
    hz = nb.mul(dn, tm)
    is_back = nb.lt(tz, nb.add(txy, 1e-4))
    is_y = nb.mul(nb.sub(1.0, is_back), nb.lt(ty, tx))
    is_floor = nb.mul(is_y, nb.lt(dv, 0.0))
    is_ceil = nb.mul(is_y, nb.gt(dv, 0.0))
    is_side = nb.mx(nb.sub(nb.sub(1.0, is_back), is_y), 0.0)
    # back wall: some furniture silhouettes and a picture / screen
    furn_lo = nb.mul(nb.lt(hy, nb.mapr(r5, 0, 1, 0.18, 0.42)), nb.band(hx, nb.mul(r6, 0.5), nb.add(nb.mul(r6, 0.5), 0.45)))
    pic = nb.mul(nb.band(hy, 0.48, 0.72), nb.band(hx, nb.mul(r7, 0.6), nb.add(nb.mul(r7, 0.6), 0.25)))
    backw = nb.mapr(nb.out(nb.noise(nb.comb(hx, hy, nb.add(iu, nb.mul(iv, 7.0))), 3.0, 3, 0.5), "Fac"), 0.3, 0.7, 0.55, 0.85)
    backw = nb.mul(backw, nb.sub(1.0, nb.mul(furn_lo, 0.85)))
    backw = nb.add(backw, nb.mul(pic, nb.mapr(r5, 0, 1, -0.3, 1.2)))
    # ceiling: a light fixture in the middle of the room
    fx = nb.sub(hx, 0.5)
    fz = nb.sub(hz, 0.45)
    lamp = nb.math("EXPONENT", nb.mul(nb.add(nb.mul(fx, fx), nb.mul(fz, fz)), -14.0))
    panel = nb.mul(nb.band(hx, 0.36, 0.64), nb.band(hz, 0.3, 0.52))     # a ceiling light panel
    ceil = nb.add(nb.add(0.22, nb.mul(lamp, 0.7)), nb.mul(panel, 2.2))
    shade = nb.add(nb.add(nb.mul(is_back, backw), nb.mul(is_floor, 0.28)),
                   nb.add(nb.mul(is_ceil, ceil), nb.mul(is_side, nb.mapr(hz, 0, 1, 0.75, 0.45))))
    shade = nb.mul(shade, nb.mapr(hz, 0.0, 1.0, 1.0, 0.55))
    # curtains (from the sides) or blinds (from the top), at the glass plane
    gu = nb.div(nb.sub(fu, mxm), nb.mx(nb.sub(1.0, nb.mul(mxm, 2.0)), 0.05))
    gv = nb.div(nb.sub(fv, mbm), nb.mx(nb.sub(nb.sub(1.0, mbm), mtm), 0.05))
    cover = nb.mul(r5, 0.5)
    curt = nb.mul(nb.lt(r4, 0.4), nb.mx(nb.lt(gu, cover), nb.gt(gu, nb.sub(1.0, cover))))
    folds = nb.mapr(nb.math("SINE", nb.mul(gu, 90.0)), -1, 1, 0.65, 1.0)
    blind = nb.mul(nb.band(r4, 0.4, 0.68), nb.mul(nb.gt(gv, nb.sub(1.0, nb.mul(r6, 1.1))),
                                              nb.lt(nb.frac(nb.mul(gv, nb.mul(H, 11.0))), 0.78)))
    shade = nb.mixf(curt, shade, nb.mul(folds, 0.55))
    shade = nb.mixf(blind, shade, 0.4)
    interior = nb.mix(1.0, room_col, nb.comb(shade, shade, shade), "MULTIPLY")
    # curtains are dyed fabric: tint them
    curt_tint = nb.ramp(r7, [(0, (1.0, 0.75, 0.5, 1)), (0.4, (1.0, 0.9, 0.75, 1)), (0.7, (0.6, 0.8, 1.0, 1)),
                             (0.9, (1.0, 0.5, 0.45, 1))], "CONSTANT")
    interior = nb.mix(curt, interior, nb.mix(1.0, interior, curt_tint, "MULTIPLY"))
    lit_e = nb.mul(nb.mul(lit, bright), glass)
    # a few dark rooms flicker blue with a TV
    tv = nb.mul(nb.mul(nb.sub(1.0, lit), nb.lt(r6, 0.06)), glass)
    emis = nb.mix(lit_e, (0, 0, 0), interior)
    emis = nb.mix(nb.mul(tv, 0.3), emis, (0.2, 0.35, 1.0))

    # --- wall --------------------------------------------------------------
    hz_ = nb.lt(vert, 0.5)
    wallv = nb.node("ShaderNodeMix", data_type="VECTOR")
    nb.set(wallv, "Factor_Float", hz_)
    nb.set(wallv, "A_Vector", nb.comb(nb.div(u, 3.0), nb.div(v, 3.0), 0.0))
    nb.set(wallv, "B_Vector", nb.comb(nb.div(nb.out(sP, 0), 3.0), nb.div(nb.out(sP, 1), 3.0), 0.0))
    wvec = nb.sock(wallv.outputs, "Result_Vector")
    tile_sel = nb.gt(nb.frac(nb.mul(brand, 7.31)), 0.55)
    c1, r1t, h1 = tex.nodes(nb, "concrete", wvec, "#77736d", 1.2)
    c2, r2t, h2 = tex.nodes(nb, "tiles", nb.vmath("MULTIPLY", wvec, (1.5, 1.5, 1)), "#9a958c", 3.0)
    wcol = nb.mix(tile_sel, c1, c2)
    wr = nb.mixf(tile_sel, r1t, r2t)
    wh = nb.mixf(tile_sel, h1, h2)
    wcol = nb.mix(1.0, wcol, tint, "MULTIPLY")
    streak = nb.out(nb.noise(nb.comb(nb.mul(u, 2.5), nb.mul(v, 0.09), brand), 2.0, 5, 0.6), "Fac")
    wet = nb.mapr(streak, 0.42, 0.66)
    wcol = nb.mix(nb.mul(wet, 0.55), wcol, (0, 0, 0))
    wr = nb.mixf(wet, nb.mapr(wr, 0, 1, 0.45, 0.85), 0.18)
    # glassy towers: dark spandrel glass between the windows
    wcol = nb.mix(glassy, wcol, (0.012, 0.014, 0.018))
    wr = nb.mixf(glassy, wr, 0.06)

    base = nb.mix(frame, wcol, (0.03, 0.03, 0.032))
    base = nb.mix(glass, base, (0.004, 0.005, 0.006))
    rough = nb.mixf(frame, wr, 0.3)
    rough = nb.mixf(glass, rough, nb.mapr(r7, 0, 1, 0.01, 0.06))
    metal = nb.mx(frame, nb.mul(glassy, nb.sub(1.0, glass)))
    bump = nb.bump(wh, nb.mul(nb.sub(1.0, winx), 0.45), 0.02)
    bsdf = nb.principled(base=base, rough=rough, metal=metal, normal=bump, spec=0.6,
                         emit=emis, emit_str=0.9)
    nb.output(bsdf)
    return m


def mat_rain():
    m, nb = new_material("Rain")
    tr = nb.node("ShaderNodeBsdfTransparent")
    gl = nb.node("ShaderNodeBsdfGlossy", {"Roughness": 0.15})
    em = nb.node("ShaderNodeEmission", {"Color": (0.7, 0.8, 1.0), "Strength": 0.12})
    gen = nb.out(nb.node("ShaderNodeTexCoord"), "UV")
    gv = nb.out(nb.sep(gen), 1)
    taper = nb.math("SINE", nb.mul(gv, math.pi))       # fade the streak ends
    a1 = nb.node("ShaderNodeAddShader")
    nb.set(a1, 0, gl)
    nb.set(a1, 1, em)
    mx = nb.node("ShaderNodeMixShader")
    nb.set(mx, 0, nb.mul(taper, 0.75))
    nb.set(mx, 1, tr)
    nb.set(mx, 2, a1)
    nb.output(mx)
    return m


def mat_fog(density, aniso, color=(0.75, 0.78, 0.9)):
    m, nb = new_material(f"Fog_{density:g}")
    v = nb.node("ShaderNodeVolumePrincipled", {"Density": density, "Anisotropy": aniso, "Color": color})
    nb.output(volume=v)
    return m


def mat_steam():
    m, nb = new_material("Steam")
    gen = nb.out(nb.node("ShaderNodeTexCoord"), "Generated")
    g = nb.sep(gen)
    cx = nb.sub(nb.out(g, 0), 0.5)
    cy = nb.sub(nb.out(g, 1), 0.5)
    z = nb.out(g, 2)
    r = nb.math("SQRT", nb.add(nb.mul(cx, cx), nb.mul(cy, cy)))
    width = nb.mapr(z, 0, 1, 0.12, 0.5)
    radial = nb.mapr(nb.div(r, width), 0.3, 1.0, 1.0, 0.0, interp="SMOOTHSTEP")
    n = nb.out(nb.noise(nb.vmath("MULTIPLY", gen, (3, 3, 1.5)), 2.5, 6, 0.6, 0.6), "Fac")
    dens = nb.mul(nb.mul(radial, nb.mapr(n, 0.4, 0.75)), nb.mapr(z, 0.0, 1.0, 1.0, 0.0))
    v = nb.node("ShaderNodeVolumePrincipled", {"Density": nb.mul(dens, 6.0), "Anisotropy": 0.4,
                                               "Color": (0.9, 0.9, 0.92)})
    nb.output(volume=v)
    return m


class Materials:
    def __init__(self, tex):
        self.tex = tex
        self.facade = mat_facade(tex)
        self.road = mat_road(tex)
        self.pavement = mat_pavement(tex)
        self.concrete = mat_concrete(tex)
        self.concrete_dark = mat_concrete(tex, "Concrete_Dark", "#4a4844", 2.0)
        self.metal_dark = mat_metal(tex, "Metal_Dark", (0.025, 0.026, 0.028), 0.35, 1.0, rust=0.05)
        self.metal_rust = mat_metal(tex, "Metal_Rusty", (0.08, 0.07, 0.065), 0.45, 0.9, rust=0.25)
        self.metal_paint = mat_metal(tex, "Metal_Painted", (0.18, 0.19, 0.2), 0.4, 0.0, rust=0.1)
        self.shutter = mat_metal(tex, "Shutter", (0.3, 0.31, 0.32), 0.45, 0.6, key="shutter", rust=0.4)
        self.ac = mat_metal(tex, "AC_Unit", (0.42, 0.42, 0.40), 0.5, 0.0, rust=0.15)
        self.cable = mat_simple("Cable", (0.006, 0.006, 0.006), 0.35)
        self.glass_dark = mat_glass_dark()
        self.plastic_black = mat_simple("Plastic_Black", (0.01, 0.01, 0.01), 0.18, coat=0.5)
        self.fabric = mat_simple("Fabric_Dark", (0.02, 0.02, 0.022), 0.8, spec=0.3)
        self.skin = mat_simple("Skin", (0.35, 0.22, 0.17), 0.45)
        self.rubber = mat_simple("Rubber", (0.015, 0.015, 0.015), 0.6)
        self.chrome = mat_simple("Chrome", (0.85, 0.85, 0.87), 0.06, 1.0)
        self.umbrella_clear = self._umbrella()
        self.rain = mat_rain()
        self.lamp = mat_emit("Lamp_LED", kelvin(4800), 30.0)
        self.lamp_sodium = mat_emit("Lamp_Sodium", kelvin(2200), 22.0)
        self.red_beacon = mat_emit("Beacon_Red", hexcol("#ff1a1a"), 60.0)
        self.head = mat_emit("Headlight", kelvin(5500), 60.0)
        self.tail = mat_emit("Taillight", hexcol("#ff0a14"), 25.0)
        self.lantern = mat_lightbox("Lantern", hexcol("#ff3b1f"), 9.0)
        self.neon = {k: mat_neon(f"Neon_{k}", hexcol(c), 22.0) for k, c in NEON.items()}
        self.box = {k: mat_lightbox(f"Lightbox_{k}", hexcol(c), 5.0) for k, c in NEON.items()}
        self.text_dark = mat_simple("Sign_Letters_Dark", (0.02, 0.018, 0.02), 0.3)
        self.paint = {}

    def _umbrella(self):
        m, nb = new_material("Umbrella_Clear")
        nb.output(nb.principled(base=(0.9, 0.95, 1.0), rough=0.12, trans=0.9, ior=1.45, alpha=0.65))
        return m

    def car_paint(self, color):
        if color not in self.paint:
            m, nb = new_material(f"CarPaint_{color}")
            nb.output(nb.principled(base=color, rough=0.32, metal=0.6, coat=1.0, coat_rough=0.03))
            self.paint[color] = m
        return self.paint[color]


# -----------------------------------------------------------------------------
#  Signs and text
# -----------------------------------------------------------------------------

def find_font(cands):
    for c in cands:
        for p in glob.glob(os.path.expanduser(c)):
            if os.path.exists(p):
                return p
    return None


CJK_FONTS = [
    "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf", "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK*.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK*.ttc", "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "C:/Windows/Fonts/YuGothB.ttc", "C:/Windows/Fonts/meiryob.ttc", "C:/Windows/Fonts/msgothic.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf", "~/.fonts/*CJK*", "~/Library/Fonts/*CJK*",
]
LATIN_FONTS = [
    "/usr/share/fonts/opentype/inter/Inter-Bold.otf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", "C:/Windows/Fonts/arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf",
]
SCRIPT_FONTS = [
    "/usr/share/fonts/truetype/freefont/FreeSansBoldOblique.ttf",
    "/usr/share/fonts/opentype/inter/InterDisplay-BoldItalic.otf",
    "C:/Windows/Fonts/arialbi.ttf", "/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf",
]


class SignMaker:
    def __init__(self, coll, mats, rng):
        self.coll, self.mats, self.rng = coll, mats, rng
        cjk = find_font(CJK_FONTS)
        lat = find_font(LATIN_FONTS)
        scr = find_font(SCRIPT_FONTS) or lat
        self.cjk = bpy.data.fonts.load(cjk, check_existing=True) if cjk else None
        self.latin = bpy.data.fonts.load(lat, check_existing=True) if lat else None
        self.script = bpy.data.fonts.load(scr, check_existing=True) if scr else None
        log(f"fonts: CJK={os.path.basename(cjk) if cjk else 'none (Latin-only signs)'}, "
            f"Latin={os.path.basename(lat) if lat else 'Blender default'}")
        self.count = 0

    def word(self, prefer_jp=True):
        if self.cjk and (prefer_jp and self.rng.random() < 0.72):
            return self.rng.choice(JP_WORDS), True
        return self.rng.choice(EN_WORDS + BRANDS), False

    def text(self, body, loc, rot, size, mat, font=None, extrude=0.02, tube=0.0, align="CENTER",
             vertical=False, spacing=1.0):
        cu = bpy.data.curves.new(f"txt{self.count}", "FONT")
        self.count += 1
        if vertical:
            body = "\n".join(body.replace(" ", ""))
            cu.space_line = 0.88 * spacing
        cu.body = body
        f = font or self.latin
        if f:
            cu.font = f
        cu.size = size
        cu.align_x = align
        cu.align_y = "CENTER"
        cu.space_character = spacing
        if tube > 0:
            cu.fill_mode = "NONE"
            cu.bevel_depth = tube
            cu.bevel_resolution = 2
            cu.resolution_u = 6
        else:
            cu.extrude = extrude
            cu.bevel_depth = min(extrude * 0.3, 0.01)
            cu.bevel_resolution = 1
            cu.resolution_u = 4
        cu.materials.append(mat)
        ob = new_object(cu.name, cu, self.coll)
        ob.location = loc
        ob.rotation_euler = rot
        return ob

    def fit_text(self, ob, max_w, max_h):
        """Uniformly scale a text object so it fits max_w x max_h (local x, y)."""
        # update through the object's own scene: in the GUI the script builds a new scene,
        # and the context view layer may still belong to the previous one
        (ob.users_scene[0].view_layers[0] if ob.users_scene else bpy.context.view_layer).update()
        dims = ob.dimensions
        if dims.x <= 0 or dims.y <= 0:
            return
        s = min(max_w / dims.x, max_h / dims.y, 1.0)
        ob.scale = (s, s, 1.0)


def face_rot(normal):
    """Euler rotation for a text object whose front faces `normal` (one of the four
    horizontal axis directions, as a (nx, ny) tuple)."""
    nx, ny = normal
    gamma = math.atan2(nx, -ny)
    return (math.pi / 2, 0.0, gamma)


# -----------------------------------------------------------------------------
#  City generation
# -----------------------------------------------------------------------------

class City:
    def __init__(self, opts, coll, mats):
        self.opts, self.coll, self.M = opts, coll, mats
        self.rng = random.Random(opts.seed)
        self.nprng = np.random.default_rng(opts.seed)
        self.signs = SignMaker(self.sub("Signs"), mats, self.rng)
        # mesh builders: one per material group (index = slot in materials list)
        self.mb_bldg = MeshBuilder()
        self.bldg_mats = [mats.facade, mats.concrete, mats.concrete_dark, mats.metal_dark, mats.metal_rust,
                          mats.ac, mats.shutter, mats.glass_dark, mats.metal_paint]
        self.mb_detail = MeshBuilder()
        self.mb_emit = MeshBuilder()
        self.emit_mats = []
        self.lights = []
        self.lamp_posts = []
        self.screens = self.sub("Screens")

    def sub(self, name):
        c = bpy.data.collections.new(name)
        self.coll.children.link(c)
        return c

    def emit_index(self, mat):
        if mat not in self.emit_mats:
            self.emit_mats.append(mat)
        return self.emit_mats.index(mat)

    # ---- building blocks ---------------------------------------------------
    def facade_attrs(self, b, z0=None, htop=None, lit=None):
        a = {
            "win_w": ("FLOAT", b["win_w"]), "floor_h": ("FLOAT", b["floor_h"]),
            "z0": ("FLOAT", b["z0"] if z0 is None else z0), "htop": ("FLOAT", b["h"] if htop is None else htop),
            "lit": ("FLOAT", b["lit"] if lit is None else lit), "brand": ("FLOAT", b["brand"]),
            "mx": ("FLOAT", b["mx"]), "mb": ("FLOAT", b["mb"]), "mt": ("FLOAT", b["mt"]),
            "warmth": ("FLOAT", b["warmth"]), "glassy": ("FLOAT", b["glassy"]),
            "tint": ("FLOAT_COLOR", b["tint"]), "glow": ("FLOAT", b.get("glow", 1.0)),
            "u0": ("FLOAT", b.get("u0", 0.0)),
        }
        return a

    def new_building_params(self, h, style=None):
        r = self.rng
        style = style if style is not None else r.choices(["punched", "ribbon", "curtain", "resi"],
                                                         [0.35, 0.2, 0.15, 0.3])[0]
        p = dict(style=style, h=h, brand=r.random(), z0=r.uniform(4.2, 5.2),
                 floor_h=r.uniform(3.1, 3.6), warmth=r.random(), glassy=0.0)
        if style == "punched":
            p.update(win_w=r.uniform(1.5, 2.6), mx=r.uniform(0.14, 0.24), mb=r.uniform(0.22, 0.32),
                     mt=r.uniform(0.1, 0.16), lit=r.uniform(0.3, 0.55))
        elif style == "ribbon":
            p.update(win_w=r.uniform(1.4, 2.0), mx=0.0, mb=r.uniform(0.3, 0.4), mt=r.uniform(0.08, 0.12),
                     lit=r.uniform(0.35, 0.6))
        elif style == "curtain":
            p.update(win_w=r.uniform(1.4, 1.8), mx=0.03, mb=0.05, mt=0.05, lit=r.uniform(0.2, 0.45),
                     glassy=1.0, warmth=r.uniform(0.0, 0.25), floor_h=r.uniform(3.6, 4.1))
        else:   # residential: small windows, balconies, AC units
            p.update(win_w=r.uniform(2.6, 3.4), mx=r.uniform(0.2, 0.3), mb=r.uniform(0.3, 0.42),
                     mt=r.uniform(0.12, 0.18), lit=r.uniform(0.35, 0.6), floor_h=r.uniform(2.8, 3.1),
                     warmth=r.uniform(0.35, 1.0))
        g = r.uniform(0.55, 1.05)
        hue = r.choice([(1.0, 0.97, 0.92), (0.9, 0.95, 1.0), (1.0, 0.92, 0.88), (0.95, 0.95, 0.95),
                        (0.85, 0.9, 0.85), (1.0, 0.85, 0.75)])
        p["tint"] = (hue[0] * g, hue[1] * g, hue[2] * g, 1.0)
        return p

    def add_mass(self, b, x0, x1, y0, y1, z0, z1, top=True, recess=0.0):
        """A building volume carrying the facade material. With `recess`, the ground floor
        (up to b["z0"]) is set back by that much from the street face at x0, leaving
        room for shopfronts."""
        if recess > 0 and z0 < b["z0"] < z1:
            sgn = 1.0 if x1 > x0 else -1.0
            add_box(self.mb_bldg, ((x0 + sgn * recess + x1) / 2, (y0 + y1) / 2, b["z0"] / 2),
                    (abs(x1 - x0) - recess, abs(y1 - y0), b["z0"]), 0, fattrs=self.facade_attrs(b, htop=b["z0"]),
                    faces=[2, 3, 4, 5])
            z0 = b["z0"]
        c = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
        s = (abs(x1 - x0), abs(y1 - y0), z1 - z0)
        faces = None if top else [0, 2, 3, 4, 5]
        add_box(self.mb_bldg, c, s, 0, fattrs=self.facade_attrs(b, htop=z1), faces=faces)
        # parapet & roof slab
        if top:
            t = 0.25
            add_box(self.mb_bldg, (c[0], c[1], z1 + 0.5), (s[0], s[1], 1.0), 1,
                    faces=[2, 3, 4, 5])
            add_box(self.mb_bldg, (c[0], c[1], z1 + 0.98), (s[0], s[1], 0.04), 2, faces=[1])
            add_box(self.mb_bldg, (c[0], c[1], z1 + 0.6), (s[0] - 2 * t, s[1] - 2 * t, 0.04), 2, faces=[1])

    def roof_clutter(self, x0, x1, y0, y1, z, near):
        r = self.rng
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w, d = abs(x1 - x0), abs(y1 - y0)
        for _ in range(r.randint(1, 4)):
            bx = cx + r.uniform(-0.3, 0.3) * w
            by = cy + r.uniform(-0.3, 0.3) * d
            sz = (r.uniform(1.5, 4), r.uniform(1.5, 4), r.uniform(1.2, 3))
            add_box(self.mb_bldg, (bx, by, z + sz[2] / 2), sz, r.choice([4, 8, 1]))
        if r.random() < 0.5:     # water tank
            bx, by = cx + r.uniform(-0.25, 0.25) * w, cy + r.uniform(-0.25, 0.25) * d
            add_tube(self.mb_detail, [(bx, by, z), (bx, by, z + 3.0)], 1.3, 14, 0)
            for k in range(4):
                a = k * math.pi / 2 + 0.6
                add_tube(self.mb_detail, [(bx + 1.0 * math.cos(a), by + 1.0 * math.sin(a), z - 1.5),
                                          (bx + 1.0 * math.cos(a), by + 1.0 * math.sin(a), z)], 0.08, 6, 1)
        # antenna mast with a red beacon
        if r.random() < 0.65:
            bx, by = cx + r.uniform(-0.3, 0.3) * w, cy + r.uniform(-0.3, 0.3) * d
            hh = r.uniform(5, 18)
            add_tube(self.mb_detail, [(bx, by, z), (bx, by, z + hh)], 0.12, 6, 1)
            add_sphere(self.mb_emit, (bx, by, z + hh + 0.2), 0.22, self.emit_index(self.M.red_beacon), 8, 6)

    def facade_greebles(self, b, side, fx, y0, y1, z_from, z_to):
        """Ledges, fins, balconies, AC units and pipes on a street-facing facade at x=fx
        (wall normal points to -side)."""
        r = self.rng
        n = -side
        W, H = b["win_w"], b["floor_h"]
        floors = int((z_to - b["z0"]) / H)
        style = b["style"]
        # floor ledges
        if style in ("punched", "ribbon") and r.random() < 0.7:
            dep = r.uniform(0.18, 0.45)
            for k in range(floors + 1):
                z = b["z0"] + k * H
                if z < z_from or z > z_to - 0.5:
                    continue
                add_box(self.mb_bldg, (fx + n * dep / 2, (y0 + y1) / 2, z + 0.06), (dep, y1 - y0, 0.16), 1)
        # vertical fins
        if style == "curtain" or (style == "ribbon" and r.random() < 0.4):
            dep = r.uniform(0.25, 0.6)
            k0 = math.ceil(y0 / W)
            for k in range(k0, int(y1 / W) + 1):
                y = k * W
                if y0 + 0.2 < y < y1 - 0.2:
                    add_box(self.mb_bldg, (fx + n * dep / 2, y, (z_from + z_to) / 2), (dep, 0.12, z_to - z_from), 3)
        # balconies / AC units per window cell (the facade shader puts window k at u in [kW, (k+1)W])
        # u = dot(P, t) with t = (-ny, nx, 0) = (0, n, 0)  ->  u = n*y
        for k in range(floors):
            z = b["z0"] + k * H
            if z < max(z_from, 5.0) or z > z_to - H:
                continue
            ku0, ku1 = sorted((n * y0, n * y1))
            for ku in range(math.ceil(ku0 / W), int(ku1 / W)):
                uc = (ku + 0.5) * W
                yc = uc * n
                if not (y0 + 0.6 < yc < y1 - 0.6):
                    continue
                roll = r.random()
                if style == "resi" and roll < 0.55:
                    bw, bd = W * 0.92, r.uniform(0.9, 1.3)
                    add_box(self.mb_bldg, (fx + n * bd / 2, yc, z + 0.08), (bd, bw, 0.16), 1)
                    add_box(self.mb_bldg, (fx + n * (bd - 0.03), yc, z + 0.62), (0.05, bw, 0.95), 8)
                    for sgn in (-1, 1):
                        add_box(self.mb_bldg, (fx + n * bd / 2, yc + sgn * bw / 2, z + 0.62), (bd, 0.05, 0.95), 8)
                    if r.random() < 0.35:   # laundry / clutter box on the balcony
                        add_box(self.mb_bldg, (fx + n * 0.4, yc + r.uniform(-0.3, 0.3) * bw, z + 0.55),
                                (0.4, 0.5, 0.75), r.choice([5, 4, 8]))
                    if r.random() < 0.5:   # AC on the balcony
                        add_box(self.mb_bldg, (fx + n * 0.3, yc - 0.3 * bw, z + 0.45), (0.5, 0.8, 0.6), 5)
                elif roll < (0.22 if style != "curtain" else 0.0):
                    ay = yc + r.uniform(-0.3, 0.3) * W
                    add_box(self.mb_bldg, (fx + n * 0.3, ay, z + b["mb"] * H * 0.5 + 0.1), (0.6, 0.85, 0.6), 5)
                    # condensate pipe
                    if r.random() < 0.3:
                        add_tube(self.mb_detail, [(fx + n * 0.05, ay + 0.35, z + 0.1), (fx + n * 0.05, ay + 0.35, z - H + 0.4)], 0.02, 5, 1)
        # vertical pipes near the building edges
        for _ in range(r.randint(0, 3)):
            py = r.uniform(y0 + 0.3, y1 - 0.3)
            rad = r.uniform(0.05, 0.13)
            add_tube(self.mb_detail, [(fx + n * (rad + 0.05), py, 0.2), (fx + n * (rad + 0.05), py, z_to)], rad, 8, r.choice([0, 1]))

    def shopfront(self, b, side, fx, y0, y1):
        """Ground floor shops along [y0, y1] of the facade at x=fx."""
        r = self.rng
        n = -side
        S = self.signs
        width = y1 - y0
        k = max(1, int(width / r.uniform(4.5, 8.0)))
        edges = np.linspace(y0, y1, k + 1)
        gh = b["z0"]
        for i in range(k):
            a, c = edges[i] + 0.25, edges[i + 1] - 0.25
            ym = (a + c) / 2
            w = c - a
            # pillars between shops
            add_box(self.mb_bldg, (fx + n * 0.2, edges[i], gh / 2), (0.4, 0.5, gh), 2)
            kind = r.choices(["open", "open", "shutter", "open_awning"], [0.4, 0.2, 0.15, 0.25])[0]
            recess = 0.5
            gx = fx - n * recess
            if kind == "shutter":
                add_box(self.mb_bldg, (fx - n * 0.05, ym, 1.5), (0.08, w, 3.0), 6)
            else:
                # shop window: a facade-material face with one big lit "room"
                shop = dict(b)
                shop.update(win_w=w, floor_h=3.2, z0=0.15, mx=0.015, mb=0.02, mt=0.03, lit=1.0,
                            brand=r.random(), warmth=r.random(), glassy=0.0, glow=r.uniform(1.3, 2.2))
                shop["u0"] = min(n * a, n * c)      # start the window grid at this shop's edge
                add_box(self.mb_bldg, (gx + n * 0.03 - n * 0.05, ym, 1.75), (0.1, w, 3.2), 0,
                        fattrs=self.facade_attrs(shop, z0=0.15, htop=3.35 + 0.9, lit=1.0), faces=[3 if n > 0 else 5])
                # recess reveal (side walls, ceiling), and a step
                for yy in (a, c):
                    add_box(self.mb_bldg, (fx - n * recess / 2, yy, 1.75), (recess, 0.06, 3.5), 2)
                add_box(self.mb_bldg, (fx - n * recess / 2, ym, 3.45), (recess, w, 0.1), 2)
                # frame of the shop window
                add_box(self.mb_bldg, (gx + n * 0.02, ym, 0.12), (0.12, w, 0.24), 3)
                add_box(self.mb_bldg, (gx + n * 0.02, ym, 3.3), (0.12, w, 0.1), 3)
                add_box(self.mb_bldg, (gx + n * 0.02, ym + r.uniform(-0.2, 0.2) * w, 1.7), (0.1, 0.08, 3.2), 3)
                # interior light spilling onto the pavement
                L = bpy.data.lights.new(f"shop{len(self.lights)}", "AREA")
                L.shape = "RECTANGLE"
                L.size, L.size_y = w * 0.9, 0.6
                L.energy = r.uniform(25, 70) * w
                L.color = kelvin(r.uniform(2800, 5500))[:3]
                lo = new_object(L.name, L, self.coll)
                lo.location = (gx + n * 0.15, ym, 3.0)
                lo.rotation_euler = Vector((n, 0.0, -0.8)).to_track_quat("-Z", "Y").to_euler()
                self.lights.append(lo)
                L.use_shadow = True
            # sign band over the shop
            col = r.choice(list(NEON))
            style = r.random()
            band_c = (fx + n * 0.12, ym, gh - 0.55)
            if style < 0.5:
                add_box(self.mb_emit, band_c, (0.22, w * 0.96, 0.95), self.emit_index(self.M.box[col]))
                word, jp = S.word()
                ob = S.text(word, (fx + n * 0.24, ym, gh - 0.55), face_rot((n, 0)), 0.6, self.M.text_dark,
                            font=S.cjk if jp else S.latin, extrude=0.02)
                S.fit_text(ob, w * 0.85, 0.7)
            else:
                add_box(self.mb_bldg, band_c, (0.22, w * 0.96, 0.95), 3)
                word, jp = S.word(prefer_jp=r.random() < 0.6)
                font = S.cjk if jp else (S.script if r.random() < 0.5 else S.latin)
                tube = 0.0 if jp and r.random() < 0.5 else 0.018
                ob = S.text(word, (fx + n * 0.26, ym, gh - 0.55), face_rot((n, 0)), 0.65, self.M.neon[col],
                            font=font, extrude=0.015, tube=tube)
                S.fit_text(ob, w * 0.85, 0.7)
            if kind == "open_awning":
                ad = r.uniform(1.2, 1.9)
                verts = [(fx, a, gh - 1.1), (fx, c, gh - 1.1), (fx + n * ad, c, gh - 1.6), (fx + n * ad, a, gh - 1.6)]
                self.mb_bldg.add(verts, [(0, 1, 2, 3) if n > 0 else (3, 2, 1, 0)], None, 8)
                add_box(self.mb_bldg, (fx + n * ad, ym, gh - 1.75), (0.04, w, 0.3), 8)

    def blade_sign(self, side, fx, y, z0, length):
        """A vertical sign sticking out of the facade at (fx, y), readable from up and
        down the street."""
        r = self.rng
        S = self.signs
        n = -side
        depth = r.uniform(1.0, 1.7)
        th = 0.24
        cx = fx + n * (depth / 2 + 0.35)
        col = r.choice(list(NEON))
        lightbox = r.random() < 0.5
        word, jp = S.word()
        if not jp:
            word = word.replace(" ", "")
        zc = z0 + length / 2
        # brackets
        for zz in (z0 + 0.4, z0 + length - 0.4):
            add_box(self.mb_bldg, (fx + n * 0.2, y, zz), (0.4, 0.1, 0.1), 3)
        if lightbox:
            add_box(self.mb_emit, (cx, y, zc), (depth, th, length), self.emit_index(self.M.box[col]))
            tmat = self.M.text_dark if r.random() < 0.6 else self.M.box[r.choice(list(NEON))]
            tex_kw = dict(extrude=0.02)
        else:
            add_box(self.mb_bldg, (cx, y, zc), (depth, th, length), 3)
            tmat = self.M.neon[col]
            tex_kw = dict(extrude=0.012, tube=0.0 if jp else 0.02)
            # neon outline tube
            inset = 0.08
            pts = [(cx - depth / 2 + inset, 0, z0 + inset), (cx + depth / 2 - inset, 0, z0 + inset),
                   (cx + depth / 2 - inset, 0, z0 + length - inset), (cx - depth / 2 + inset, 0, z0 + length - inset),
                   (cx - depth / 2 + inset, 0, z0 + inset)]
            omat = self.M.neon[r.choice(list(NEON))]
            for sgn in (-1, 1):
                p = [(px, y + sgn * (th / 2 + 0.03), pz) for px, _, pz in pts]
                add_tube(self.mb_emit, p, 0.02, 6, self.emit_index(omat), cap=False)
        font = S.cjk if jp else S.latin
        for sgn in (-1, 1):
            ob = S.text(word, (cx, y + sgn * (th / 2 + 0.005), zc), face_rot((0, sgn)), depth * 0.7, tmat,
                        font=font, vertical=True, **tex_kw)
            S.fit_text(ob, depth * 0.8, length * 0.92)

    def billboard(self, c, w, h, normal, design=None):
        """An LED screen on a steel frame, facing the horizontal `normal`."""
        r = self.rng
        nx, ny = normal
        design = r.randint(0, 2) if design is None else design
        cols = r.sample([k for k in NEON if k != "white"], 3)
        self.n_screens = getattr(self, "n_screens", 0) + 1
        mat = mat_led_screen(f"LED_{self.n_screens}", design, [NEON[k] for k in cols], r.uniform(3.0, 5.0))
        smb = MeshBuilder()
        if abs(nx) > 0.5:
            add_box(smb, c, (0.3, w, h), 0)
            add_box(self.mb_bldg, (c[0] - nx * 0.35, c[1], c[2]), (0.4, w + 0.4, h + 0.4), 3)
        else:
            add_box(smb, c, (w, 0.3, h), 0)
            add_box(self.mb_bldg, (c[0], c[1] - ny * 0.35, c[2]), (w + 0.4, 0.4, h + 0.4), 3)
        new_object(f"Screen{self.n_screens}", smb.build(f"Screen{self.n_screens}", [mat]), self.screens)
        # a slogan over the graphics
        S = self.signs
        word, jp = S.word(prefer_jp=r.random() < 0.5)
        if r.random() < 0.6:
            word = r.choice(BRANDS)
            jp = False
        tm = mat_emit(f"LEDtext{len(self.emit_mats)}", (1, 1, 1), r.uniform(5, 9))
        ob = S.text(word, (c[0] + nx * 0.17, c[1] + ny * 0.17, c[2] - h * 0.25), face_rot(normal), h * 0.2, tm,
                    font=S.cjk if jp else S.latin, extrude=0.01)
        S.fit_text(ob, w * 0.85, h * 0.25)

    # ---- the street -------------------------------------------------------
    def build_street_side(self, side):
        r = self.rng
        rh, wk = STREET["road_half"], STREET["walk"]
        base_fx = rh + wk
        y = -30.0
        idx = 0
        while y < STREET["length"]:
            cy0, cy1 = STREET["cross_y"]
            if cy0 - 1 < y < cy1:
                y = cy1 + 0.5
                continue
            w = r.uniform(9.0, 24.0)
            y1 = min(y + w, STREET["length"])
            if y < cy0 < y1:
                y1 = cy0 - 0.5
            if y1 - y < 4:
                y = y1 + 0.5
                continue
            dist = max(y, 0.0)
            fx = side * (base_fx + r.uniform(0.0, 0.6))
            depth = r.uniform(14, 26)
            tall = r.random()
            if tall < 0.18:
                h = r.uniform(70, 150)
            elif tall < 0.45:
                h = r.uniform(35, 70)
            else:
                h = r.uniform(14, 34)
            b = self.new_building_params(h)
            x_out = fx + side * depth
            near = dist < 180
            if h > 50 and r.random() < 0.7:
                # podium + set-back tower
                hp = r.uniform(12, 26)
                b["h"] = hp
                self.add_mass(b, fx, x_out, y, y1, 0, hp, recess=0.5 if near else 0.0)
                tb = self.new_building_params(h, r.choice(["curtain", "ribbon", "punched"]))
                sb = r.uniform(2, 5)
                self.add_mass(tb, fx + side * sb, x_out, y + 0.5, y1 - 0.5, hp, h)
                self.roof_clutter(fx + side * sb, x_out, y + 0.5, y1 - 0.5, h + 1, near)
                if near:
                    self.facade_greebles(b, side, fx, y, y1, 0, hp)
                    self.facade_greebles(tb, side, fx + side * sb, y + 0.5, y1 - 0.5, hp, h)
            else:
                self.add_mass(b, fx, x_out, y, y1, 0, h, recess=0.5 if near else 0.0)
                self.roof_clutter(fx, x_out, y, y1, h + 1, near)
                if near:
                    self.facade_greebles(b, side, fx, y, y1, 0, h)
            # side wall facing the camera gets windows too (it's the same material)
            if near:
                self.shopfront(b, side, fx, y, y1)
                # blade signs
                for _ in range(r.choices([0, 1, 2, 3], [0.15, 0.4, 0.3, 0.15])[0]):
                    sy = r.uniform(y + 0.8, y1 - 0.8)
                    z0 = r.uniform(b["z0"] + 0.4, b["z0"] + 6.0)
                    L = min(r.uniform(4.0, 11.0), b["h"] - z0 - 1.0)
                    if L > 2.5:
                        self.blade_sign(side, fx, sy, z0, L)
                # billboards on some taller buildings, facing up the street
                if b["h"] > 22 and r.random() < 0.35 and y > 15:
                    bw, bh = r.uniform(3.5, 5.0), r.uniform(6, 11)
                    bz = r.uniform(12, max(13, b["h"] - bh / 2 - 2))
                    self.billboard((fx - side * (bw / 2 + 0.4), y - 0.3, bz), bw, bh, (0, -1))
                # facade billboards (seen in perspective)
                elif b["h"] > 18 and r.random() < 0.3:
                    bw, bh = r.uniform(6, min(14, y1 - y - 1)), r.uniform(4, 8)
                    bz = r.uniform(9, max(10, b["h"] - bh / 2 - 2))
                    self.billboard((fx - side * 0.5, (y + y1) / 2, bz), bw, bh, (-side, 0))
            # alleys between some buildings
            gap = r.choice([0.0, 0.0, 0.3, 2.5]) if near else 0.2
            if gap > 1.0 and near:
                # a warm light deep in the alley
                L = bpy.data.lights.new(f"alley{idx}", "POINT")
                L.energy = r.uniform(150, 500)
                L.color = r.choice([kelvin(2600)[:3], hexcol(NEON[r.choice(list(NEON))])[:3]])
                L.shadow_soft_size = 0.3
                lo = new_object(L.name, L, self.coll)
                lo.location = (fx + side * r.uniform(5, 12), y1 + gap / 2, r.uniform(2.5, 5))
            y = y1 + gap
            idx += 1

    def build_cross_street_corners(self):
        """Buildings lining the cross street, so the gap reads as a street."""
        r = self.rng
        cy0, cy1 = STREET["cross_y"]
        for side in (-1, 1):
            x = side * (STREET["road_half"] + STREET["walk"] + 30)
            for k in range(6):
                for yy in ((cy0 - 30, cy0 - 3.5), (cy1 + 3.5, cy1 + 30)):
                    h = r.uniform(25, 110)
                    b = self.new_building_params(h)
                    x0 = x + side * k * 22
                    self.add_mass(b, x0, x0 + side * 20, yy[0], yy[1], 0, h)

    def build_backdrop(self):
        """Second-row towers behind the street front and the far skyline."""
        r = self.rng
        rh, wk = STREET["road_half"], STREET["walk"]
        for side in (-1, 1):
            y = -20.0
            while y < STREET["length"] + 100:
                w = r.uniform(18, 40)
                x0 = side * (rh + wk + r.uniform(28, 45))
                x1 = x0 + side * r.uniform(18, 35)
                h = r.uniform(60, 260) * (1.0 + 0.6 * min(y / 400.0, 1.0))
                b = self.new_building_params(h, r.choice(["curtain", "ribbon", "punched", "punched"]))
                b["z0"] = 0.0
                self.add_mass(b, x0, x1, y, y + w, 0, h)
                if r.random() < 0.5:
                    hb = h + r.uniform(10, 40)
                    tb = dict(b)
                    self.add_mass(tb, x0 + side * 4, x1 - side * 4, y + 4, y + w - 4, h, hb)
                    h = hb
                add_sphere(self.mb_emit, ((x0 + x1) / 2, y + w / 2, h + 2.0), 0.6, self.emit_index(self.M.red_beacon), 8, 6)
                if r.random() < 0.3:
                    bw, bh = r.uniform(12, 25), r.uniform(18, 40)
                    self.billboard(((x0 + x1) / 2, y - 0.4, r.uniform(40, max(41, h - bh))), bw, bh, (0, -1))
                y += w + r.uniform(4, 14)
        # far skyline at the end of the street and beyond
        for _ in range(150):
            x = r.uniform(-700, 700)
            y = r.uniform(STREET["length"] - 40, STREET["length"] + 1300)
            if abs(x) < 30 and y < STREET["length"] + 100:
                continue
            w, d = r.uniform(25, 70), r.uniform(25, 70)
            h = r.uniform(80, 420) * (1.0 - min(abs(x) / 1400, 0.5))
            b = self.new_building_params(h, r.choice(["curtain", "ribbon", "punched"]))
            b["z0"] = 0.0
            b["win_w"] *= 1.3
            b["floor_h"] *= 1.15
            self.add_mass(b, x - w / 2, x + w / 2, y, y + d, 0, h)
            add_sphere(self.mb_emit, (x, y + d / 2, h + 3), 1.4, self.emit_index(self.M.red_beacon), 8, 6)
        # the megastructure that closes the vista
        y = STREET["length"] + 160
        b = self.new_building_params(900, "curtain")
        b["z0"] = 0.0
        b["lit"] = 0.22
        self.add_mass(b, -90, 90, y, y + 140, 0, 520)
        self.add_mass(dict(b, brand=0.77), -55, 55, y + 20, y + 120, 520, 760)
        self.add_mass(dict(b, brand=0.31), -25, 25, y + 45, y + 95, 760, 900)
        self.billboard((0, y - 1.0, 230), 120, 70, (0, -1), design=1)
        self.billboard((-60, y - 1.0, 420), 50, 120, (0, -1), design=2)
        S = self.signs
        if S.cjk:
            tm = mat_neon("Neon_mega", hexcol(NEON["red"]), 30.0)
            S.text("電脳都市", (60, y - 2.0, 420), face_rot((0, -1)), 28.0, tm, font=S.cjk,
                        vertical=True, extrude=0.5)

    def build_ground(self):
        M = self.M
        rh, wk = STREET["road_half"], STREET["walk"]
        L = STREET["length"]
        mb = MeshBuilder()
        # road surface (one big quad, the cross street included)
        y0, y1 = -40.0, L + 400.0
        add_box(mb, (0, (y0 + y1) / 2, -0.05), (1400, y1 - y0, 0.1), 0, faces=[1])
        cy0, cy1 = STREET["cross_y"]
        for side in (-1, 1):
            for (a, c) in ((y0, cy0 - 0.2), (cy1 + 0.2, L)):
                # pavement and curb
                xm = side * (rh + wk / 2 + 0.15)
                add_box(mb, (xm, (a + c) / 2, 0.075), (wk + 0.3 + 1.0, c - a, 0.15), 1, faces=[1])
                add_box(mb, (side * (rh + 0.09), (a + c) / 2, 0.075), (0.18, c - a, 0.15), 2,
                        faces=[3 if side < 0 else 5, 1])
        me = mb.build("Ground", [M.road, M.pavement, M.concrete])
        new_object("Ground", me, self.coll)
        # manholes and drains
        for (x, y) in ((1.8, 7.0), (-2.6, 31.0), (2.1, 63.0), (-1.5, 120.0)):
            add_tube(self.mb_detail, [(x, y, -0.01), (x, y, 0.012)], 0.38, 20, 0)
        for y in np.arange(-10, 160, 12.0):
            for side in (-1, 1):
                add_box(self.mb_detail, (side * (rh - 0.2), y, 0.005), (0.35, 0.9, 0.02), 0)

    def build_street_furniture(self):
        r = self.rng
        M = self.M
        rh, wk = STREET["road_half"], STREET["walk"]
        # street lamps
        for side in (-1, 1):
            for y in np.arange(4.0 if side > 0 else 16.0, 330.0, 26.0):
                x = side * (rh + 0.45)
                add_tube(self.mb_detail, [(x, y, 0.15), (x, y, 7.5)], 0.09, 10, 0)
                arm = [(x, y, 7.4), (x - side * 0.8, y, 7.8), (x - side * 2.2, y, 7.9)]
                add_tube(self.mb_detail, arm, 0.05, 8, 0)
                hx = x - side * 2.3
                add_box(self.mb_detail, (hx, y, 7.95), (0.9, 0.36, 0.14), 0)
                sodium = r.random() < 0.3
                add_box(self.mb_emit, (hx, y, 7.87), (0.75, 0.26, 0.02),
                        self.emit_index(M.lamp_sodium if sodium else M.lamp))
                L = bpy.data.lights.new(f"lamp{len(self.lights)}", "SPOT")
                L.energy = 2600 if not sodium else 2000
                L.color = (kelvin(2200) if sodium else kelvin(4800))[:3]
                L.spot_size = math.radians(115)
                L.spot_blend = 0.6
                L.shadow_soft_size = 0.25
                lo = new_object(L.name, L, self.coll)
                lo.location = (hx, y, 7.8)
                self.lights.append(lo)
        # bollards & vending machines & trash near the camera
        for y in np.arange(2.0, 120.0, 3.2):
            for side in (-1, 1):
                if r.random() < 0.3:
                    add_tube(self.mb_detail, [(side * (rh + 0.35), y, 0.15), (side * (rh + 0.35), y, 1.0)], 0.07, 8, 0)
        for (side, y) in ((-1, 6.5), (1, 19.0), (-1, 42.0), (1, 58.0)):
            x = side * (rh + wk - 0.55)
            add_box(self.mb_detail, (x, y, 1.0), (0.85, 1.05, 1.85), 2)
            add_box(self.mb_emit, (x - side * 0.43, y, 1.15), (0.02, 0.9, 1.25),
                    self.emit_index(self.vending_mat(r.choice(["cyan", "white", "red"]))))
            add_box(self.mb_detail, (x - side * 0.44, y, 0.35), (0.04, 0.6, 0.18), 3)
        for _ in range(4):       # piles of rubbish bags against the walls
            side = r.choice((-1, 1))
            y0 = r.uniform(8, 90)
            for k in range(r.randint(2, 5)):
                x = side * (rh + wk - r.uniform(0.25, 0.7))
                y = y0 + r.uniform(-0.7, 0.7)
                sz = r.uniform(0.25, 0.38)
                add_sphere(self.mb_detail, (x, y, sz * 0.6 + 0.15), sz, 4, 10, 7,
                           scale=(r.uniform(0.9, 1.2), r.uniform(0.9, 1.3), r.uniform(0.6, 0.85)))

    def vending_mat(self, col):
        key = f"Vending_{col}"
        if key in bpy.data.materials:
            return bpy.data.materials[key]
        m, nb = new_material(key)
        uv = _uv_meters(nb)
        s = nb.sep(nb.vmath("MULTIPLY", uv, (5.5, 4.0, 1)))
        cell = nb.mul(nb.lt(nb.frac(nb.out(s, 0)), 0.78), nb.lt(nb.frac(nb.out(s, 1)), 0.7))
        w = nb.white(nb.comb(nb.floor(nb.out(s, 0)), nb.floor(nb.out(s, 1)), 1.0))
        prod = nb.ramp(nb.out(w, "Value"), [(0, "#ff3050"), (0.25, "#3080ff"), (0.5, "#ffd040"),
                                            (0.75, "#30ff90"), (1, "#ffffff")], "CONSTANT")
        bg = hexcol(NEON[col]) if col != "white" else (1, 1, 1, 1)
        img = nb.mix(cell, bg, prod)
        nb.output(nb.principled(base=(0.02, 0.02, 0.02), rough=0.05, emit=img, emit_str=nb.mapr(cell, 0, 1, 2.5, 4.0)))
        return m

    def build_overhead(self):
        """Cables crossing the street, lantern strings, a skybridge, hanging signs."""
        r = self.rng
        M = self.M
        fx = STREET["road_half"] + STREET["walk"]
        for _ in range(55):
            y = r.uniform(-5, 260)
            z1 = r.uniform(6, 16)
            z2 = z1 + r.uniform(-2.5, 2.5)
            y2 = y + r.uniform(-6, 6)
            sag = r.uniform(0.4, 2.0)
            t = np.linspace(0, 1, 24)
            P = np.stack([-fx + 2 * fx * t, y + (y2 - y) * t, z1 + (z2 - z1) * t - sag * 4 * t * (1 - t)], 1)
            add_tube(self.mb_detail, P, r.uniform(0.012, 0.03), 5, 3, cap=False)
        # lantern strings
        for y in (24.0, 31.0, 38.0):
            z = 7.5
            t = np.linspace(0, 1, 30)
            sag = 1.2
            P = np.stack([-fx + 2 * fx * t, np.full_like(t, y), z - sag * 4 * t * (1 - t)], 1)
            add_tube(self.mb_detail, P, 0.012, 5, 3, cap=False)
            for k in range(2, 28, 2):
                p = P[k]
                add_sphere(self.mb_emit, (p[0], p[1], p[2] - 0.45), 0.26, self.emit_index(M.lantern), 12, 8,
                           scale=(1, 1, 1.35))
                add_tube(self.mb_detail, [p, (p[0], p[1], p[2] - 0.12)], 0.006, 4, 3, cap=False)
        # skybridge
        y0 = 96.0
        z0 = 21.0
        add_box(self.mb_bldg, (0, y0, z0), (2 * fx + 1, 6.0, 0.6), 2)
        add_box(self.mb_bldg, (0, y0, z0 + 4.2), (2 * fx + 1, 6.4, 0.5), 2)
        b = dict(win_w=2.0, floor_h=4.0, z0=z0 + 0.3, h=z0 + 5.2, lit=0.85, brand=0.123, mx=0.03, mb=0.06,
                 mt=0.08, warmth=0.1, glassy=1.0, tint=(1, 1, 1, 1), style="curtain")
        add_box(self.mb_bldg, (0, y0, z0 + 2.2), (2 * fx, 5.6, 3.7), 0, fattrs=self.facade_attrs(b, htop=z0 + 5.4),
                faces=[2, 4])
        add_box(self.mb_emit, (0, y0 - 3.25, z0 - 0.15), (2 * fx, 0.08, 0.08), self.emit_index(M.neon["cyan"]))
        self.billboard((0, y0 - 3.4, z0 + 7.5), 9.0, 4.0, (0, -1), design=0)
        # a hanging street sign across the road
        S = self.signs
        add_box(self.mb_emit, (0, 52.0, 9.8), (7.0, 0.25, 1.4), self.emit_index(M.box["amber"]))
        for sx in (-3.3, 3.3):
            add_tube(self.mb_detail, [(sx, 52.0, 10.5), (sx * 1.6, 52.0, 14.5)], 0.01, 4, 3, cap=False)
        if S.cjk:
            ob = S.text("中央通り  CENTRAL AVE", (0, 51.86, 9.8), face_rot((0, -1)), 0.8, M.text_dark,
                        font=S.cjk, extrude=0.01)
            S.fit_text(ob, 6.5, 1.0)

    def build_hologram(self):
        """A giant translucent hologram over the street: three rings round a glowing core and a word."""
        coll = self.coll
        col = hexcol(NEON["cyan"])
        mat = mat_hologram("Hologram", col, 3.5)
        c = Vector((-2.0, 78.0, 33.0))
        mb = MeshBuilder()
        for k, (rad, tilt) in enumerate(((5.0, 0.3), (4.2, -0.5), (3.4, 1.1))):
            t = np.linspace(0, 2 * np.pi, 96)
            ring = np.stack([rad * np.cos(t), rad * np.sin(t) * math.cos(tilt), rad * np.sin(t) * math.sin(tilt)], 1)
            add_tube(mb, ring + np.array(c), 0.07, 6, 0, cap=False)
        add_sphere(mb, tuple(c), 2.0, 0, 24, 16)
        me = mb.build("Hologram", [mat])
        new_object("Hologram", me, coll)
        S = self.signs
        word = "未来" if S.cjk else "MIRAI"
        tm = mat_hologram("Hologram_text", hexcol(NEON["pink"]), 6.0)
        S.text(word, (c.x, c.y - 0.2, c.z - 7.0), face_rot((0, -1)), 3.0, tm,
               font=S.cjk or S.latin, extrude=0.05)
        # projector light cone
        L = bpy.data.lights.new("holo_projector", "SPOT")
        L.energy = 4000
        L.color = col[:3]
        L.spot_size = math.radians(25)
        L.spot_blend = 0.4
        lo = new_object("holo_projector", L, coll)
        lo.location = (c.x, c.y, 0.5)
        lo.rotation_euler = (math.pi, 0.0, 0.0)      # spots shine along local -Z: flip to point up

    def build_cars(self):
        r = self.rng
        M = self.M
        specs = [(-3.0, 18.0, 1), (3.4, 27.0, -1), (-2.9, 45.0, 1), (2.8, 70.0, -1),
                 (-2.7, 95.0, 1), (3.0, 130.0, -1), (-2.8, 160.0, 1)]
        for i, (x, y, d) in enumerate(specs):
            col = r.choice([(0.01, 0.01, 0.012), (0.6, 0.6, 0.62), (0.25, 0.01, 0.02), (0.03, 0.06, 0.1),
                            (0.12, 0.12, 0.13), (0.8, 0.35, 0.02)])
            self.car(f"Car{i}", (x, y), d, M.car_paint(col), headlights=i % 3 != 2)

    def car(self, name, xy, heading, paint, headlights=True):
        """A boxy future sedan, built along +y and turned by `heading` (+1 / -1)."""
        M = self.M
        mb = MeshBuilder()
        Lh, Wh = 2.35, 0.95
        add_box(mb, (0, 0, 0.62), (2 * Wh, 2 * Lh, 0.62), 0)
        # cabin: a tapered prism
        zc0, zc1 = 0.93, 1.42
        cab = [(-Wh * 0.92, -1.2, zc0), (Wh * 0.92, -1.2, zc0), (Wh * 0.92, 1.0, zc0), (-Wh * 0.92, 1.0, zc0),
               (-Wh * 0.72, -0.7, zc1), (Wh * 0.72, -0.7, zc1), (Wh * 0.72, 0.35, zc1), (-Wh * 0.72, 0.35, zc1)]
        mb.add(cab, _BOX_FACES[[1, 2, 3, 4, 5]], None, 1)
        for sx in (-1, 1):
            for sy in (-1, 1):
                add_tube(mb, [(sx * (Wh - 0.12), sy * 1.45, 0.36), (sx * (Wh + 0.04), sy * 1.45, 0.36)], 0.36, 16, 2)
        if headlights:
            add_box(mb, (0, Lh + 0.005, 0.78), (2 * Wh * 0.85, 0.02, 0.07), 3)
        add_box(mb, (0, -Lh - 0.005, 0.8), (2 * Wh * 0.9, 0.02, 0.06), 4)
        me = mb.build(name, [paint, M.glass_dark, M.rubber, M.head, M.tail])
        ob = new_object(name, me, self.coll)
        ob.location = (xy[0], xy[1], 0.0)
        ob.rotation_euler = (0, 0, 0 if heading > 0 else math.pi)
        bev = ob.modifiers.new("Bevel", "BEVEL")
        bev.width = 0.07
        bev.segments = 3
        bev.limit_method = "ANGLE"
        if headlights:
            for sx in (-0.6, 0.6):
                L = bpy.data.lights.new(name + "_hl", "SPOT")
                L.energy = 900
                L.color = kelvin(5500)[:3]
                L.spot_size = math.radians(40)
                L.spot_blend = 0.5
                L.shadow_soft_size = 0.05
                lo = new_object(L.name, L, self.coll)
                lo.parent = ob
                lo.location = (sx, Lh + 0.05, 0.78)
                lo.rotation_euler = (math.radians(-84), 0, 0)
        L = bpy.data.lights.new(name + "_tl", "POINT")
        L.energy = 12
        L.color = (1, 0.02, 0.03)
        L.shadow_soft_size = 0.2
        lo = new_object(L.name, L, self.coll)
        lo.parent = ob
        lo.location = (0, -Lh - 0.2, 0.8)

    def build_people(self):
        r = random.Random(self.opts.seed * 7919 + 1)     # own stream: layout tweaks don't reshuffle people
        M = self.M
        rh, wk = STREET["road_half"], STREET["walk"]
        mb = MeshBuilder()
        mats = [M.fabric, M.skin, M.umbrella_clear, M.plastic_black, M.chrome]
        led_mats = {}
        mbl = MeshBuilder()
        led_list = []
        spots = []
        for _ in range(56):
            side = r.choice((-1, 1))
            y = 5.0 + 110.0 * r.random() ** 1.5
            x = side * (rh + r.uniform(0.6, wk - 0.8))
            spots.append((x, y))
        spots += [(-4.0, 14.0), (2.5, 22.0)]      # two crossing the zebra / road
        for (x, y) in spots:
            h = r.uniform(1.62, 1.88)
            s = h / 1.75
            yaw = r.uniform(0, 2 * math.pi)
            dx, dy = math.cos(yaw) * 0.1 * s, math.sin(yaw) * 0.1 * s
            # legs, coat, torso, head
            for sg in (-1, 1):
                lx, ly = x - dy * sg, y + dx * sg
                stride = r.uniform(-0.12, 0.12) * sg
                add_tube(mb, [(lx + stride * math.cos(yaw), ly + stride * math.sin(yaw), 0.16), (lx, ly, 0.85 * s)],
                         [0.065 * s, 0.085 * s], 8, 0)
            add_tube(mb, [(x, y, 0.62 * s), (x, y, 1.0 * s), (x, y, 1.42 * s), (x, y, 1.5 * s)],
                     [0.25 * s, 0.21 * s, 0.22 * s, 0.13 * s], 10, 0)
            add_sphere(mb, (x, y, 1.62 * s), 0.11 * s, 1, 10, 8, scale=(0.9, 0.95, 1.12))
            # hood / hair
            add_sphere(mb, (x, y, 1.65 * s), 0.12 * s, 0 if r.random() < 0.5 else 3, 10, 8, scale=(1, 1, 1.0))
            if r.random() < 0.75:
                # umbrella: canopy cone + shaft (some with an LED shaft, Blade Runner style)
                uh = h + r.uniform(0.25, 0.4)
                ur = r.uniform(0.5, 0.62)
                cx, cy = x + r.uniform(-0.1, 0.1), y + r.uniform(-0.1, 0.1)
                nseg = 16
                th = np.linspace(0, 2 * np.pi, nseg, endpoint=False)
                rim = np.stack([cx + ur * np.cos(th), cy + ur * np.sin(th), np.full(nseg, uh - 0.22)], 1)
                apex = np.array([[cx, cy, uh]])
                V = np.vstack([apex, rim])
                F = [(0, 1 + (k + 1) % nseg, 1 + k) for k in range(nseg)]
                clear = r.random() < 0.6
                mb.add(V, F, None, 2 if clear else 3)
                led = clear and r.random() < 0.6
                if led:
                    col = r.choice(["cyan", "pink", "violet", "teal", "amber"])
                    if col not in led_mats:
                        led_mats[col] = len(led_list)
                        led_list.append(M.neon[col])
                    add_tube(mbl, [(cx, cy, 1.0 * s), (cx, cy, uh)], 0.012, 6, led_mats[col])
                    # LED ring on the canopy rim
                    add_tube(mbl, np.vstack([rim, rim[:1]]), 0.006, 4, led_mats[col], cap=False)
                else:
                    add_tube(mb, [(cx, cy, 1.0 * s), (cx, cy, uh)], 0.008, 5, 4)
        me = mb.build("People", mats)
        new_object("People", me, self.coll)
        if led_list:
            me = mbl.build("Umbrella_LEDs", led_list)
            new_object("Umbrella_LEDs", me, self.coll)

    def build_flyers(self):
        r = self.rng
        M = self.M
        for i in range(7):
            x = r.uniform(-25, 25)
            y = r.uniform(70, 420)
            z = r.uniform(30, 110)
            mb = MeshBuilder()
            add_box(mb, (0, 0, 0), (2.4, 5.0, 1.1), 0)
            add_box(mb, (0, -0.2, 0.75), (1.8, 2.4, 0.5), 1)
            add_box(mb, (0, 2.51, 0.1), (1.9, 0.02, 0.12), 2)
            add_box(mb, (0, -2.51, 0.1), (2.0, 0.02, 0.15), 3)
            add_box(mb, (0, 0, -0.56), (2.0, 4.2, 0.02), 4)
            under = M.neon[r.choice(["cyan", "violet", "pink"])]
            me = mb.build(f"Flyer{i}", [M.car_paint((0.05, 0.05, 0.06)), M.glass_dark, M.head, M.tail, under])
            ob = new_object(f"Flyer{i}", me, self.coll)
            ob.location = (x, y, z)
            ob.rotation_euler = (r.uniform(-0.05, 0.05), r.uniform(-0.1, 0.1), r.uniform(-0.3, 0.3) + (math.pi if r.random() < 0.5 else 0))
            if i < 2:      # a police spinner sweeping the street with a searchlight
                L = bpy.data.lights.new(f"Flyer{i}_spot", "SPOT")
                L.energy = 60000
                L.color = kelvin(6500)[:3]
                L.spot_size = math.radians(9)
                L.spot_blend = 0.3
                L.shadow_soft_size = 0.1
                lo = new_object(L.name, L, self.coll)
                lo.location = (x, y, z - 0.7)
                tgt = Vector((r.uniform(-4, 4), y - r.uniform(5, 25), 0))
                lo.rotation_euler = (tgt - Vector(lo.location)).to_track_quat("-Z", "Y").to_euler()
        # rooftop searchlights sweeping the sky
        for (x, y, z, tx, ty) in ((-40, 220, 120, -10, 320), (55, 300, 150, 90, 420)):
            L = bpy.data.lights.new("searchlight", "SPOT")
            L.energy = 250000
            L.color = kelvin(6000)[:3]
            L.spot_size = math.radians(3.5)
            L.spot_blend = 0.2
            lo = new_object(L.name, L, self.coll)
            lo.location = (x, y, z)
            tgt = Vector((tx, ty, z + 300))
            lo.rotation_euler = (tgt - Vector(lo.location)).to_track_quat("-Z", "Y").to_euler()

    def build_rain(self, cam_loc, cam_dir):
        """Pre-stretched rain streaks (the 1/50 s exposure of falling drops) in front
        of the camera, each a thin quad turned to face the lens."""
        rng = self.nprng
        n = LOOK["rain_count"]
        # uniform in the view frustum: distance ~ d^2
        d = 0.5 + 34.5 * rng.random(n) ** (1 / 3)
        hfov, vfov = math.radians(46), math.radians(30)
        az = rng.uniform(-hfov, hfov, n)
        el = rng.uniform(-vfov, vfov, n) + math.radians(LOOK["cam_tilt"])
        fwd = np.array(cam_dir)
        yaw0 = math.atan2(fwd[0], fwd[1])
        dirs = np.stack([np.sin(yaw0 - az) * np.cos(el), np.cos(yaw0 - az) * np.cos(el), np.sin(el)], 1)
        C = np.array(cam_loc) + dirs * d[:, None]
        keep = (C[:, 2] > 0.05) & (C[:, 2] < 40)
        C, d = C[keep], d[keep]
        m = len(C)
        fall = nrm(np.array([LOOK["rain_tilt"][0], LOOK["rain_tilt"][1], -1.0]))
        fall = nrm(fall[None, :] + rng.normal(0, 0.02, (m, 3)))
        L = rng.uniform(0.25, 0.65, m)
        w = rng.uniform(0.0012, 0.0025, m) * (1 + d / 18)
        view = nrm(C - np.array(cam_loc))
        S = nrm(np.cross(fall, view))
        a = C - fall * (L / 2)[:, None]
        b = C + fall * (L / 2)[:, None]
        V = np.stack([a - S * (w / 2)[:, None], a + S * (w / 2)[:, None], b + S * (w / 2)[:, None], b - S * (w / 2)[:, None]], 1).reshape(-1, 3)
        F = np.arange(4 * m).reshape(-1, 4)
        UV = np.broadcast_to(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64), (m, 4, 2))
        mb = MeshBuilder()
        mb.add(V, F, UV, 0)
        me = mb.build("Rain", [self.M.rain])
        ob = new_object("Rain", me, self.coll)
        ob.visible_shadow = False
        try:
            ob.visible_volume_scatter = False
        except AttributeError:
            pass
        log(f"rain: {m:,} streaks")

    def build_steam(self):
        mat = mat_steam()
        for i, (x, y, hgt) in enumerate(((1.8, 7.0, 3.5), (-2.6, 31.0, 4.5), (7.5, 40.0, 5.0))):
            mb = MeshBuilder()
            add_box(mb, (0, 0, 0.5), (1, 1, 1), 0)
            me = mb.build(f"Steam{i}", [mat])
            ob = new_object(f"Steam{i}", me, self.coll)
            ob.location = (x, y, 0.0)
            ob.scale = (2.4, 2.4, hgt)
            ob.rotation_euler = (0, 0.12, 0.4 * i)
            ob.display_type = "BOUNDS"

    def finish(self):
        me = self.mb_bldg.build("Buildings", self.bldg_mats)
        new_object("Buildings", me, self.sub("Buildings"))
        me = self.mb_detail.build("Details", [self.M.metal_dark, self.M.metal_rust, self.M.metal_paint,
                                              self.M.cable, self.M.plastic_black])
        if me:
            new_object("Details", me, self.coll)
        me = self.mb_emit.build("Lights_Emissive", self.emit_mats)
        if me:
            new_object("Lights_Emissive", me, self.coll)


# -----------------------------------------------------------------------------
#  World, fog, camera, render
# -----------------------------------------------------------------------------

def setup_world(scene):
    world = bpy.data.worlds.new("Night_Sky")
    scene.world = world
    if not BLENDER_5:
        world.use_nodes = True
    nb = NB(world.node_tree)
    world.node_tree.nodes.clear()
    tc = nb.node("ShaderNodeTexCoord")
    D = nb.out(tc, "Generated")
    z = nb.out(nb.sep(D), 2)
    # light-polluted night sky: magenta-orange glow at the horizon, deep blue above
    sky = nb.ramp(nb.mapr(z, -0.05, 0.6), [(0.0, (0.06, 0.025, 0.04, 1)), (0.12, (0.05, 0.018, 0.045, 1)),
                                          (0.35, (0.012, 0.008, 0.025, 1)), (1.0, (0.002, 0.003, 0.008, 1))])
    # low cloud deck lit from below
    cl = nb.out(nb.noise(nb.vmath("MULTIPLY", D, (3, 3, 9)), 2.0, 8, 0.62, 0.3), "Fac")
    clouds = nb.mul(nb.mapr(cl, 0.45, 0.75), nb.mapr(z, 0.05, 0.5, 1.0, 0.2))
    sky = nb.mix(nb.mul(clouds, 0.6), sky, (0.09, 0.03, 0.06))
    bg = nb.node("ShaderNodeBackground", {"Color": sky, "Strength": 1.0})
    out = nb.node("ShaderNodeOutputWorld")
    nb.set(out, "Surface", bg)


def add_fog(coll, opts):
    if opts.no_fog:
        return
    for name, (z0, z1), dens in (("Fog_Low", (0.0, 30.0), LOOK["fog_low"]),
                                 ("Fog_High", (30.0, 220.0), LOOK["fog_high"])):
        mb = MeshBuilder()
        add_box(mb, (0, 380, (z0 + z1) / 2), (260, 840, z1 - z0), 0)
        me = mb.build(name, [mat_fog(dens, LOOK["fog_aniso"])])
        ob = new_object(name, me, coll)
        ob.display_type = "BOUNDS"


def setup_camera(coll):
    cam = bpy.data.cameras.new("Camera")
    cam.lens = LOOK["lens"]
    cam.sensor_width = 36.0
    cam.sensor_fit = "HORIZONTAL"
    cam.clip_start = 0.05
    cam.clip_end = 5000.0
    cam.shift_y = LOOK["shift_y"]
    cam.dof.use_dof = True
    cam.dof.focus_distance = LOOK["focus"]
    cam.dof.aperture_fstop = LOOK["fstop"]
    cam.dof.aperture_blades = 6
    cam.dof.aperture_ratio = 1.15         # a hint of anamorphic oval bokeh
    ob = new_object("Camera", cam, coll)
    for idb in (ob, cam):           # take the plain name even if a default "Camera" was around
        other = (bpy.data.objects if idb is ob else bpy.data.cameras).get("Camera")
        if other is not None and other is not idb:
            other.name = "Camera_old"
        idb.name = "Camera"
    ob.location = LOOK["cam_loc"]
    ob.rotation_euler = (math.radians(90 + LOOK["cam_tilt"]), math.radians(LOOK["cam_roll"]),
                         math.radians(LOOK["cam_yaw"]))
    return ob


def pick_device(want_gpu):
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except (KeyError, AttributeError):
        return "CPU"
    if want_gpu:
        for backend in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
            try:
                prefs.compute_device_type = backend
            except TypeError:
                continue
            prefs.refresh_devices()
            gpus = [d for d in prefs.devices if d.type == backend]
            if gpus:
                for d in prefs.devices:
                    d.use = d.type == backend
                log(f"GPU rendering via {backend}: {', '.join(d.name for d in gpus)}")
                return "GPU"
        log("no usable GPU found - rendering on the CPU")
    return "CPU"


def setup_render(scene, opts):
    scene.render.engine = "CYCLES"
    c = scene.cycles
    c.device = pick_device(opts.gpu)
    c.samples = opts.samples
    c.use_adaptive_sampling = True
    c.adaptive_threshold = 0.02
    c.use_denoising = True
    try:
        c.denoiser = "OPENIMAGEDENOISE"
        c.denoising_input_passes = "RGB_ALBEDO_NORMAL"
        c.denoising_prefilter = "ACCURATE"
        c.denoising_quality = "HIGH"
    except (TypeError, AttributeError):
        pass
    c.max_bounces = 10
    c.diffuse_bounces = 3
    c.glossy_bounces = 4          # wet streets need reflections of reflections
    c.transmission_bounces = 6
    c.volume_bounces = 0
    c.transparent_max_bounces = 24     # rain streaks stack up
    c.sample_clamp_direct = 0.0
    c.sample_clamp_indirect = 8.0
    c.blur_glossy = 0.5
    c.caustics_reflective = False
    c.caustics_refractive = False
    c.use_light_tree = True
    try:
        c.volume_step_rate = 4.0
    except AttributeError:
        pass
    r = scene.render
    r.resolution_x, r.resolution_y = opts.res
    r.resolution_percentage = 100
    r.filter_size = 1.2
    if opts.threads:
        r.threads_mode = "FIXED"
        r.threads = opts.threads
    r.image_settings.file_format = "PNG"
    r.image_settings.color_depth = "8"
    vs = scene.view_settings
    vs.view_transform = "AgX"
    try:
        vs.look = LOOK["look"]
    except TypeError:
        pass
    vs.exposure = LOOK["exposure"]


def _try_input(node, key, value):
    try:
        NB.sock(node.inputs, key).default_value = value
        return True
    except (KeyError, TypeError, ValueError):
        return False


def setup_compositor(scene):
    """Bloom on the neon, a touch of lateral chromatic aberration, a vignette."""
    w, h = scene.render.resolution_x, scene.render.resolution_y
    if hasattr(scene, "compositing_node_group"):                       # Blender 5.x
        ng = bpy.data.node_groups.new("City Finish", "CompositorNodeTree")
        ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        scene.compositing_node_group = ng
        N, L = ng.nodes, ng.links
        final_input = N.new("NodeGroupOutput").inputs[0]
    else:
        scene.use_nodes = True
        N, L = scene.node_tree.nodes, scene.node_tree.links
        N.clear()
        final_input = N.new("CompositorNodeComposite").inputs["Image"]
    rl = N.new("CompositorNodeRLayers")
    rl.scene = scene

    glare = N.new("CompositorNodeGlare")
    if not _try_input(glare, "Type", "Bloom"):
        try:
            glare.glare_type = "BLOOM"
        except TypeError:
            glare.glare_type = "FOG_GLOW"
    if not _try_input(glare, "Quality", "High"):
        glare.quality = "HIGH"
    if not _try_input(glare, "Highlights Threshold", 1.0):
        glare.threshold = 1.0
    if not _try_input(glare, "Strength", 0.4):
        glare.mix = -0.6
    if not _try_input(glare, "Size", 0.9):
        glare.size = 8

    lens = N.new("CompositorNodeLensdist")
    _try_input(lens, "Dispersion", 0.018)
    _try_input(lens, "Distortion", -0.012)
    if not _try_input(lens, "Fit", True):
        lens.use_fit = True

    ell = N.new("CompositorNodeEllipseMask")
    if not _try_input(ell, "Size", (0.95, 0.95)):
        ell.mask_width, ell.mask_height = 0.95, 0.95
    blur = N.new("CompositorNodeBlur")
    if not _try_input(blur, "Size", (w * 0.25, h * 0.25)):
        blur.filter_type = "GAUSS"
        blur.size_x, blur.size_y = int(w * 0.25), int(h * 0.25)
    try:
        vig = N.new("CompositorNodeMixRGB")
        vig.blend_type = "MULTIPLY"
        vig.inputs[0].default_value = 0.45
        vig_a, vig_b, vig_out = vig.inputs[1], vig.inputs[2], vig.outputs[0]
    except RuntimeError:
        vig = N.new("ShaderNodeMix")
        vig.data_type, vig.blend_type = "RGBA", "MULTIPLY"
        NB.sock(vig.inputs, "Factor_Float").default_value = 0.45
        vig_a, vig_b = NB.sock(vig.inputs, "A_Color"), NB.sock(vig.inputs, "B_Color")
        vig_out = NB.sock(vig.outputs, "Result_Color")
    L.new(rl.outputs["Image"], glare.inputs["Image"])
    L.new(NB.sock(glare.outputs, "Image"), lens.inputs["Image"])
    L.new(lens.outputs["Image"], vig_a)
    L.new(ell.outputs["Mask"], blur.inputs["Image"])
    L.new(blur.outputs["Image"], vig_b)
    L.new(vig_out, final_input)


def prepare_scene():
    if bpy.context.window is not None:
        scene = bpy.data.scenes.new("Cyberpunk City")
        bpy.context.window.scene = scene
    else:
        scene = bpy.context.scene
        for ob in list(scene.objects):
            bpy.data.objects.remove(ob, do_unlink=True)
        for c in list(scene.collection.children):
            scene.collection.children.unlink(c)
        try:
            bpy.data.orphans_purge(do_recursive=True)
        except Exception:
            pass
        scene.name = "Cyberpunk City"
    return scene


def open_in_camera_view():
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                for space in area.spaces:
                    if space.type == "VIEW_3D" and space.region_3d is not None:
                        space.region_3d.view_perspective = "CAMERA"


def make_portable():
    """Turn the sign text into meshes (the .blend then needs no font files) and pack
    the textures into the file."""
    dg = bpy.context.evaluated_depsgraph_get()
    fonts = [o for o in bpy.data.objects if o.type == "FONT"]
    for ob in fonts:
        me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg), preserve_all_data_layers=True)
        new = bpy.data.objects.new(ob.name + "_mesh", me)
        new.matrix_world = ob.matrix_world.copy()
        for c in ob.users_collection:
            c.objects.link(new)
        bpy.data.objects.remove(ob, do_unlink=True)
    for cu in [c for c in bpy.data.curves if c.users == 0]:
        bpy.data.curves.remove(cu)
    for f in list(bpy.data.fonts):
        bpy.data.fonts.remove(f)
    for img in bpy.data.images:
        if img.source == "FILE" and img.packed_file is None:
            img.pack()
    log(f"portable: {len(fonts)} text objects -> meshes, textures packed")


def build_city(opts):
    t0 = time.time()
    scene = prepare_scene()
    root = bpy.data.collections.new("City")
    scene.collection.children.link(root)
    tex = TextureLibrary(opts.tex_dir, opts.offline)
    tex.fetch_all()
    mats = Materials(tex)
    city = City(opts, root, mats)
    city.build_ground()
    for side in (-1, 1):
        city.build_street_side(side)
    city.build_cross_street_corners()
    city.build_backdrop()
    city.build_street_furniture()
    city.build_overhead()
    city.build_hologram()
    city.build_cars()
    city.build_people()
    city.build_flyers()
    city.build_steam()
    cam = setup_camera(root)
    scene.camera = cam
    bpy.context.view_layer.update()
    if not opts.no_rain:
        fwd = cam.matrix_world.to_3x3() @ Vector((0, 0, -1))
        city.build_rain(tuple(cam.location), tuple(fwd))
    city.finish()
    add_fog(root, opts)
    setup_world(scene)
    setup_render(scene, opts)
    setup_compositor(scene)
    open_in_camera_view()
    log(f"city built in {time.time() - t0:.0f} s: {len(scene.objects)} objects, "
        f"{city.signs.count} text signs, {len(city.lights)} lamps")
    return scene


# -----------------------------------------------------------------------------
#  Entry point
# -----------------------------------------------------------------------------

def parse_args(argv=None):
    if argv is None:
        if "--" in sys.argv:
            argv = sys.argv[sys.argv.index("--") + 1:]
        elif not bpy.app.binary_path:
            argv = sys.argv[1:]
        else:
            argv = []
    p = argparse.ArgumentParser(description="Cyberpunk city street for Blender + Cycles")
    p.add_argument("--preset", choices=sorted(PRESETS), default="preview")
    p.add_argument("--seed", type=int, default=4)
    p.add_argument("--res", type=str, default=None)
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--no-fog", action="store_true")
    p.add_argument("--no-rain", action="store_true")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--tex-dir", type=str, default=None)
    p.add_argument("--render", type=str, default=None)
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--pack", action="store_true", help="with --save: text to meshes, textures packed")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--gpu", action="store_true")
    a = p.parse_args(argv)
    pr = PRESETS[a.preset]
    a.res = tuple(int(v) for v in a.res.lower().split("x")) if a.res else pr["res"]
    a.samples = a.samples or pr["samples"]
    if a.tex_dir is None:
        here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
        if not os.access(here, os.W_OK):
            here = os.path.expanduser("~")
        a.tex_dir = os.path.join(here, "textures")
    return a


def main():
    opts = parse_args()
    log(f"Blender {bpy.app.version_string}, preset={opts.preset}, seed={opts.seed}")
    scene = build_city(opts)
    if opts.save:
        if opts.pack:
            make_portable()
        bpy.ops.wm.save_as_mainfile(filepath=bpy.path.abspath(opts.save), compress=True)
        log(f"saved {opts.save}")
    if opts.render:
        scene.render.filepath = bpy.path.abspath(opts.render)
        if opts.render.lower().endswith(".exr"):
            scene.render.image_settings.file_format = "OPEN_EXR"
            scene.render.image_settings.color_depth = "16"
        log(f"rendering {opts.res[0]}x{opts.res[1]} @ {opts.samples} samples ...")
        t0 = time.time()
        bpy.ops.render.render(write_still=True, scene=scene.name)
        log(f"wrote {opts.render} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

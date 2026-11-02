#!/usr/bin/env python3
"""
airliner.py - a twin-engine passenger airliner for Blender, built from 807 separate parts.

Every part is its own object: fuselage skin panels, frames, stringers, floor beams, windows,
doors, seats, overhead bins, galleys, wing skins, ribs and spars, slats, flaps, spoilers,
ailerons, winglets, engine cowls, fan blades, compressor and turbine stages, landing gear
struts and wheels, and so on. No textures or models are downloaded: everything is code.

Five display modes (sidebar: View3D > N panel > "Airliner" tab):
    Finished    the complete aircraft
    X-ray       outer skin turns to glass; structure and systems glow in colour by system
    Cutaway     a stepped section of skin is removed on the left side
    Exploded    parts fly apart along their assembly directions (slider for the amount)
    Laid out    every part is laid flat on the floor, grouped and labelled by system
The panel and the viewport overlay show a live part count for the current mode.

Usage
-----
Blender GUI (4.2 or newer):
    Scripting workspace -> Text > Open... -> airliner.py -> Run Script
    Then press N in the 3D viewport and open the "Airliner" tab.

Command line:
    blender -b -P airliner.py -- --save airliner.blend
    blender -b -P airliner.py -- --mode xray --render xray.png
    blender -b -P airliner.py -- --render-all renders/        # one PNG per mode

Python with the `bpy` module from PyPI (pip install bpy):
    python airliner.py --render-all renders/

Options (put them after "--" when using the blender binary):
    --mode {finished,xray,cutaway,exploded,layout}   mode to leave the scene in (default finished)
    --explode F          explode distance multiplier (default 1.0)
    --render PATH        render the chosen mode to PATH
    --render-all DIR     render all five modes to DIR/airliner_<mode>.png
    --res WxH            render resolution (default 1920x1080)
    --samples N          Cycles samples (default 64)
    --no-hud             leave the mode/part-count caption out of renders
    --save PATH          save the .blend (the mode panel is stored inside it)

Opening a saved .blend: the mode panel lives in the text block "airliner_ui.py". Blender asks
before running scripts in a file; press "Allow Execution" (or open the text block and press
Run Script) and the "Airliner" tab appears.
"""

import argparse
import math
import os
import sys
import time
import warnings

import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector

warnings.filterwarnings("ignore", category=DeprecationWarning)

TAU = 2.0 * math.pi
RAD = math.radians

# -----------------------------------------------------------------------------
#  Main dimensions (metres). Nose points +X, left wing +Y, up +Z.
# -----------------------------------------------------------------------------
X_NOSE = 18.8          # x of the nose tip; "station" s is metres aft of it
FUS_LEN = 37.6
R_A, R_B = 1.98, 2.07  # fuselage half-width / half-height
GROUND = -4.15         # z of the apron

WING_LE_X0 = X_NOSE - 13.0   # leading edge x at the centreline
WING_Z0 = -1.30
TAN_LE = math.tan(RAD(27.0))
DIHEDRAL = RAD(5.0)
KINK_Y, TIP_Y, TIP_CHORD = 6.3, 16.9, 1.55

ENG_Y = 5.75

CABIN_FLOOR = -0.45
CARGO_FLOOR = -1.80

# Livery band boundaries, as cos(theta) on the fuselage cross-section (1 = top, -1 = keel)
LIVERY_CUTS = [0.07, -0.02, -0.05, -0.09, -0.45]

XRAY_COLORS = {"Structure": (0.75, 0.85, 1.0), "Cabin": (1.0, 0.72, 0.22), "Flight deck": (1.0, 0.5, 0.2),
               "Engines": (1.0, 0.33, 0.12), "Landing gear": (0.95, 0.3, 0.85), "Cargo & systems": (0.35, 1.0, 0.45)}

CATEGORIES = ["Fuselage", "Wings", "Tail", "Engines", "Landing gear",
              "Structure", "Cabin", "Flight deck", "Cargo & systems"]


# =============================================================================
#  Small geometry toolkit
# =============================================================================
class Buf:
    """Accumulates vertices/faces/material indices for one mesh."""

    def __init__(self):
        self.v, self.f, self.m = [], [], []

    def add(self, vf, mi=0):
        verts, faces = vf
        o = len(self.v)
        self.v.extend(tuple(p) for p in verts)
        for face in faces:
            self.f.append([i + o for i in face])
            self.m.append(mi)
        return self


def V(*a):
    return Vector(a if len(a) == 3 else a[0])


def lin(a, b, step, minimum=1):
    n = max(minimum, int(math.ceil(abs(b - a) / step - 1e-9)))
    return [a + (b - a) * k / n for k in range(n + 1)]


def grid(rows, wrap=False, flip=False):
    """Quad grid from rows of points. Rows whose points coincide collapse to a pole."""
    verts, idx = [], []
    for row in rows:
        p0 = Vector(row[0])
        if all((Vector(p) - p0).length < 1e-7 for p in row):
            verts.append(tuple(p0))
            idx.append([len(verts) - 1] * len(row))
        else:
            start = len(verts)
            verts.extend(tuple(p) for p in row)
            idx.append(list(range(start, start + len(row))))
    faces = []
    ncol = len(rows[0])
    for i in range(len(rows) - 1):
        for j in range(ncol if wrap else ncol - 1):
            j1 = (j + 1) % ncol
            q = [idx[i][j], idx[i][j1], idx[i + 1][j1], idx[i + 1][j]]
            out = []
            for k in q:
                if not out or out[-1] != k:
                    out.append(k)
            if len(out) > 1 and out[0] == out[-1]:
                out.pop()
            if len(out) >= 3:
                faces.append(out[::-1] if flip else out)
    return verts, faces


def box(c, s, R=None):
    hx, hy, hz = s[0] / 2, s[1] / 2, s[2] / 2
    pts = [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
           (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]
    if R is not None:
        pts = [tuple(R @ Vector(p)) for p in pts]
    v = [(c[0] + p[0], c[1] + p[1], c[2] + p[2]) for p in pts]
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return v, f


def _basis(axis):
    a = Vector(axis).normalized()
    t = Vector((0, 0, 1)) if abs(a.z) < 0.9 else Vector((1, 0, 0))
    u = (t - a * t.dot(a)).normalized()
    return a, u, a.cross(u)


def cyl(p0, p1, r, n=16, r1=None, caps=True):
    p0, p1 = Vector(p0), Vector(p1)
    a, u, w = _basis(p1 - p0)
    r1 = r if r1 is None else r1
    ring0 = [p0 + (u * math.cos(TAU * k / n) + w * math.sin(TAU * k / n)) * r for k in range(n)]
    ring1 = [p1 + (u * math.cos(TAU * k / n) + w * math.sin(TAU * k / n)) * r1 for k in range(n)]
    v = [tuple(p) for p in ring0 + ring1]
    f = [(k, (k + 1) % n, n + (k + 1) % n, n + k) for k in range(n)]
    if caps:
        f.append(tuple(range(n - 1, -1, -1)))
        f.append(tuple(range(n, 2 * n)))
    return v, f


def lathe(profile, origin, axis, up, a0=0.0, a1=TAU, n=48, loop=False):
    """Revolve a (d, r) profile about `axis` through `origin`. phi=0 points along `up`.
    loop=True treats the profile as a closed cross-section (a ring-shaped solid)."""
    origin = Vector(origin)
    axis = Vector(axis).normalized()
    up = Vector(up)
    up = (up - axis * up.dot(axis)).normalized()
    side = up.cross(axis)
    full = abs((a1 - a0) - TAU) < 1e-6
    nseg = n if full else max(2, int(math.ceil(n * (a1 - a0) / TAU)))
    angles = [a0 + (a1 - a0) * k / nseg for k in range(nseg if full else nseg + 1)]
    prof = list(profile) + ([profile[0]] if loop else [])
    rows = []
    for d, r in prof:
        rows.append([tuple(origin + axis * d + (up * math.cos(p) + side * math.sin(p)) * r)
                     for p in angles])
    verts, faces = grid(rows, wrap=full)
    if loop and not full:     # caps on the cut faces of a partial revolution
        for p in (angles[0], angles[-1]):
            start = len(verts)
            for d, r in profile:
                verts.append(tuple(origin + axis * d + (up * math.cos(p) + side * math.sin(p)) * r))
            faces.append(list(range(start, len(verts))))
    return verts, faces


def prism(profile, a0, a1, axis="X"):
    """Extrude a closed 2D profile. axis X: profile is (y, z); axis Y: profile is (x, z)."""
    def P(t, p):
        return (t, p[0], p[1]) if axis == "X" else (p[0], t, p[1])
    n = len(profile)
    v = [P(a0, p) for p in profile] + [P(a1, p) for p in profile]
    f = [(k, (k + 1) % n, n + (k + 1) % n, n + k) for k in range(n)]
    f.append(tuple(range(n - 1, -1, -1)))
    f.append(tuple(range(n, 2 * n)))
    return v, f


def ngon_cap(points):
    return [tuple(p) for p in points], [list(range(len(points)))]


def superellipse(w, h, n=24, e=4.0, plane="XZ", off=0.0):
    pts = []
    for k in range(n):
        t = TAU * k / n
        c, s = math.cos(t), math.sin(t)
        x = w / 2 * math.copysign(abs(c) ** (2 / e), c)
        z = h / 2 * math.copysign(abs(s) ** (2 / e), s)
        pts.append((x, off, z) if plane == "XZ" else (x, z, off))
    return pts


def face_normal(verts, face):
    a, b, c = (Vector(verts[i]) for i in face[:3])
    return (b - a).cross(c - b)


# =============================================================================
#  Fuselage shape
# =============================================================================
def pchip(xs, ys):
    """Monotone cubic through the points; flat at the last point (joins the constant section)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    h = np.diff(xs)
    d = np.diff(ys) / h
    m = np.zeros_like(ys)
    m[0] = d[0]
    for k in range(1, len(xs) - 1):
        if d[k - 1] * d[k] > 0:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])

    def f(x):
        if x <= xs[0]:
            return float(ys[0])
        if x >= xs[-1]:
            return float(ys[-1])
        k = int(np.searchsorted(xs, x)) - 1
        t = (x - xs[k]) / h[k]
        return float((2 * t ** 3 - 3 * t ** 2 + 1) * ys[k] + (t ** 3 - 2 * t ** 2 + t) * h[k] * m[k]
                     + (-2 * t ** 3 + 3 * t ** 2) * ys[k + 1] + (t ** 3 - t ** 2) * h[k] * m[k + 1])
    return f


# Nose profile: crown line, keel line and half-width against station (radome tip at s = 0)
NOSE_TOP = pchip([0, 0.06, 0.2, 0.5, 1.0, 1.7, 2.3, 3.0, 3.8, 5.0, 6.5, 8.0],
                 [-0.35, -0.15, 0.04, 0.25, 0.50, 0.75, 0.95, 1.40, 1.70, 1.93, 2.05, R_B])
NOSE_BOT = pchip([0, 0.06, 0.2, 0.5, 1.0, 1.7, 2.6, 3.6, 5.0, 8.0],
                 [-0.35, -0.56, -0.76, -1.02, -1.32, -1.60, -1.84, -1.98, -R_B, -R_B])
NOSE_W = pchip([0, 0.06, 0.2, 0.5, 1.0, 1.7, 2.6, 3.6, 4.8, 6.2, 8.0],
               [0.0, 0.24, 0.45, 0.71, 1.0, 1.30, 1.58, 1.80, 1.94, R_A, R_A])


def fus_section(s):
    """(z_centre, half_width, half_height) of the fuselage at station s."""
    top, bot, a = R_B, -R_B, R_A
    if s < 8.0:
        top, bot, a = NOSE_TOP(s), NOSE_BOT(s), NOSE_W(s)
    if s > 25.0:
        t = min((s - 25.0) / (FUS_LEN - 25.0), 1.0)
        bot = -R_B + (R_B + 0.95) * t ** 1.8
        t2 = min(max((s - 28.0) / (FUS_LEN - 28.0), 0.0), 1.0)
        top = R_B - 0.52 * t2 ** 2
        a = (top - bot) * 0.5 * (R_A / R_B)
    return (top + bot) / 2, a, (top - bot) / 2


def fus_point(s, th, off=0.0):
    zc, a, b = fus_section(s)
    return (X_NOSE - s, (a + off) * math.sin(th), zc + (b + off) * math.cos(th))


def theta_at_height(s, z):
    zc, a, b = fus_section(s)
    return math.acos(max(-1.0, min(1.0, (z - zc) / b)))


def interior_half_width(s, z, margin=0.13):
    zc, a, b = fus_section(s)
    q = 1.0 - ((z - zc) / b) ** 2
    return max(0.0, a * math.sqrt(max(q, 0.0)) - margin)


def livery_rgb(c, s):
    WHITE, NAVY, TEAL, BELLY = (0.80, 0.81, 0.82), (0.006, 0.02, 0.085), (0.0, 0.22, 0.40), (0.30, 0.32, 0.35)
    if c < -0.45:
        return BELLY
    if s >= 5.0:
        if -0.09 <= c < -0.05:
            return TEAL
        if -0.02 <= c < 0.07:
            return NAVY
    return WHITE


def theta_samples(th0, th1, step=RAD(4.5), wrap=False):
    ths = lin(th0, th1, step)
    for c in LIVERY_CUTS:
        a = math.acos(c)
        for cand in (a, -a, TAU - a, TAU + a, -TAU + a):
            if th0 + 1e-4 < cand < th1 - 1e-4:
                ths.append(cand)
    ths = sorted(set(round(t, 7) for t in ths))
    if wrap:
        ths = ths[:-1]
    return ths


def fus_patch(s0, s1, th0, th1, off=0.0, ds=0.5, wrap=False):
    ss = lin(s0, s1, ds if s0 > 5 else ds / 2)
    ths = theta_samples(th0, th1, wrap=wrap)
    rows = [[fus_point(s, t, off) for t in ths] for s in ss]
    v, f = grid(rows, wrap=wrap)
    # per-face livery colour (face-corner domain gives crisp stripe edges)
    cols = []
    for face in f:
        cen = sum((Vector(v[i]) for i in face), Vector()) / len(face)
        s = X_NOSE - cen.x
        zc, a, b = fus_section(s)
        c = max(-1.0, min(1.0, (cen.z - zc) / max(b, 1e-6)))
        cols.append(livery_rgb(c, s))
    return v, f, cols


# =============================================================================
#  Lifting surfaces
# =============================================================================
def naca_half(u, t):
    u = min(max(u, 0.0), 1.0)
    return 5 * t * (0.2969 * math.sqrt(u) - 0.1260 * u - 0.3516 * u * u + 0.2843 * u ** 3 - 0.1015 * u ** 4)


class Surf:
    """A lofted wing-like surface. Each section: q (span parameter), le, chord, chord dir, normal, t/c."""

    def __init__(self, secs, camber=0.0, cpos=0.4):
        self.secs, self.m, self.p = secs, camber, cpos

    def frame(self, q):
        S = self.secs
        if q <= S[0]["q"]:
            k, f = 0, 0.0
        elif q >= S[-1]["q"]:
            k, f = len(S) - 2, 1.0
        else:
            k = max(i for i in range(len(S) - 1) if S[i]["q"] <= q)
            f = (q - S[k]["q"]) / (S[k + 1]["q"] - S[k]["q"])
        A, B = S[k], S[k + 1]
        L = lambda key: A[key] * (1 - f) + B[key] * f
        return (A["le"].lerp(B["le"], f), L("c"), A["cd"].lerp(B["cd"], f).normalized(),
                A["n"].lerp(B["n"], f).normalized(), L("t"))

    def camber(self, u):
        m, p = self.m, self.p
        if m == 0:
            return 0.0
        return m / p ** 2 * (2 * p * u - u * u) if u < p else m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * u - u * u)

    def pt(self, q, u, side, off=0.0):
        le, c, cd, n, t = self.frame(q)
        h = self.camber(u) + side * naca_half(u, t)
        return le + cd * (u * c) + n * (h * c + side * off)

    def qs(self, q0, q1, step=0.6):
        out = lin(q0, q1, step)
        out += [s["q"] for s in self.secs if q0 + 1e-6 < s["q"] < q1 - 1e-6]
        return sorted(set(out))

    @staticmethod
    def us(u0, u1, n):
        if u0 <= 1e-9:   # cluster towards the leading edge
            return [u1 * (1 - math.cos(0.5 * math.pi * k / n)) for k in range(n + 1)]
        return [u0 + (u1 - u0) * (1 - math.cos(math.pi * k / n)) / 2 for k in range(n + 1)]


def surf_patch(S, q0, q1, u0, u1, side, nu=10, off=0.0, step=0.6):
    us = Surf.us(u0, u1, nu)
    qs = S.qs(q0, q1, step)
    rows = [[tuple(S.pt(q, u, side, off)) for u in us] for q in qs]
    v, f = grid(rows)
    want = S.frame((q0 + q1) / 2)[3] * side
    if face_normal(v, f[len(f) // 2]).dot(want) < 0:
        f = [fc[::-1] for fc in f]
    return v, f


def surf_solid(S, q0, q1, u0, u1, nu=10, step=0.6):
    us = Surf.us(u0, u1, nu)
    qs = S.qs(q0, q1, step)
    rows = []
    for q in qs:
        upper = [tuple(S.pt(q, u, +1)) for u in reversed(us)]
        lower = [tuple(S.pt(q, u, -1)) for u in us]
        if u0 <= 1e-9:
            lower = lower[1:]
        rows.append(upper + lower)
    v, f = grid(rows, wrap=True)
    for row in (rows[0], rows[-1]):
        start = len(v)
        v.extend(row)
        f.append(list(range(start, len(v))))
    return v, f


def surf_rib(S, q, u0, u1, nu=10, inset=0.035):
    us = Surf.us(u0, u1, nu)
    up = [tuple(S.pt(q, u, +1, -inset)) for u in us]
    lo = [tuple(S.pt(q, u, -1, -inset)) for u in reversed(us)]
    return up + lo, [list(range(len(up) + len(lo)))]


def surf_spar(S, q0, q1, u, inset=0.035, step=0.6):
    rows = [[tuple(S.pt(q, u, +1, -inset)), tuple(S.pt(q, u, -1, -inset))] for q in S.qs(q0, q1, step)]
    return grid(rows)


# =============================================================================
#  Scene / material helpers
# =============================================================================
MAT = {}


def _principled(mat):
    mat.use_nodes = True
    nt = mat.node_tree
    return nt, next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")


def pbr(name, color, metal=0.0, rough=0.5, coat=0.0, emit=None, estr=0.0, attr=None, sheen=0.0):
    mat = bpy.data.materials.new(name)
    nt, b = _principled(mat)
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Metallic"].default_value = metal
    b.inputs["Roughness"].default_value = rough
    if coat:
        b.inputs["Coat Weight"].default_value = coat
        b.inputs["Coat Roughness"].default_value = 0.08
    if sheen:
        b.inputs["Sheen Weight"].default_value = sheen
    if emit:
        b.inputs["Emission Color"].default_value = (*emit, 1)
        b.inputs["Emission Strength"].default_value = estr
    if attr:
        a = nt.nodes.new("ShaderNodeAttribute")
        a.attribute_name = attr
        a.location = (-300, 200)
        nt.links.new(a.outputs["Color"], b.inputs["Base Color"])
    mat.diffuse_color = (*[c ** (1 / 2.2) for c in color], 1)
    mat.metallic, mat.roughness = metal, rough
    MAT[name] = mat
    return mat


def emission_mat(name, color, strength=1.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.name = "Emission"
    em.inputs["Color"].default_value = (*color, 1)
    em.inputs["Strength"].default_value = strength
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    mat.diffuse_color = (*color, 1)
    MAT[name] = mat
    return mat


def ghost_mat():
    """X-ray glass: transparent in the middle, glowing cyan towards silhouettes."""
    mat = bpy.data.materials.new("AP X-ray skin")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.25
    pw = nt.nodes.new("ShaderNodeMath")
    pw.operation = "POWER"
    pw.inputs[1].default_value = 1.6
    mad = nt.nodes.new("ShaderNodeMath")
    mad.operation = "MULTIPLY_ADD"
    mad.inputs[1].default_value = 0.75
    mad.inputs[2].default_value = 0.05
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (0.25, 0.75, 1.0, 1)
    em.inputs["Strength"].default_value = 1.4
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lw.outputs["Facing"], pw.inputs[0])
    nt.links.new(pw.outputs[0], mad.inputs[0])
    nt.links.new(mad.outputs[0], mix.inputs["Fac"])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(em.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "BLENDED"
    if hasattr(mat, "use_transparency_overlap"):
        mat.use_transparency_overlap = True
    mat.use_backface_culling = False
    mat.diffuse_color = (0.3, 0.75, 1.0, 0.12)
    mat.use_fake_user = True        # unused outside X-ray mode; keep it in saved files
    MAT["ghost"] = mat
    return mat


def build_materials():
    pbr("livery", (0.8, 0.8, 0.8), rough=0.28, coat=0.35, attr="livery")
    pbr("wing", (0.50, 0.52, 0.55), metal=0.25, rough=0.38)
    pbr("belly", (0.30, 0.32, 0.35), rough=0.4, coat=0.2)
    pbr("tail", (0.006, 0.02, 0.085), rough=0.3, coat=0.4)
    pbr("nacelle", (0.80, 0.81, 0.82), rough=0.3, coat=0.35)
    pbr("lip", (0.85, 0.86, 0.88), metal=1.0, rough=0.16)
    pbr("titanium", (0.42, 0.43, 0.46), metal=1.0, rough=0.28)
    pbr("metal", (0.66, 0.67, 0.70), metal=1.0, rough=0.33)
    pbr("hot", (0.42, 0.30, 0.20), metal=1.0, rough=0.45)
    pbr("chrome", (0.95, 0.95, 0.96), metal=1.0, rough=0.05)
    pbr("primer", (0.20, 0.42, 0.33), rough=0.55)
    pbr("alu", (0.70, 0.71, 0.74), metal=0.85, rough=0.38)
    pbr("glass", (0.01, 0.014, 0.025), rough=0.04, coat=1.0)
    pbr("seal", (0.06, 0.06, 0.07), rough=0.6)
    pbr("rubber", (0.018, 0.018, 0.02), rough=0.8)
    pbr("fabric_y", (0.03, 0.07, 0.22), rough=0.95, sheen=0.4)
    pbr("fabric_j", (0.22, 0.025, 0.04), rough=0.9, sheen=0.4)
    pbr("leather", (0.03, 0.03, 0.035), rough=0.45)
    pbr("plastic", (0.78, 0.77, 0.73), rough=0.45)
    pbr("plastic_dk", (0.12, 0.125, 0.14), rough=0.5)
    pbr("carpet", (0.07, 0.09, 0.16), rough=1.0)
    pbr("steel", (0.78, 0.78, 0.80), metal=1.0, rough=0.25)
    pbr("container", (0.62, 0.63, 0.66), metal=0.9, rough=0.42)
    pbr("composite", (0.10, 0.10, 0.11), rough=0.45)
    pbr("gear", (0.55, 0.56, 0.58), metal=0.6, rough=0.4)
    pbr("screen", (0.0, 0.0, 0.0), rough=0.2, emit=(0.15, 0.55, 1.0), estr=3.0)
    pbr("nav_red", (0.5, 0.0, 0.0), rough=0.1, emit=(1.0, 0.02, 0.02), estr=8.0)
    pbr("nav_green", (0.0, 0.5, 0.1), rough=0.1, emit=(0.02, 1.0, 0.15), estr=8.0)
    pbr("ground", (0.42, 0.43, 0.45), rough=0.7)
    for cat, col in XRAY_COLORS.items():     # X-ray mode: internals glow in their system colour
        m = pbr(f"AP X-ray {cat}", col, rough=0.5, emit=col, estr=0.45)
        m.use_fake_user = True
    emission_mat("label", (0.02, 0.025, 0.04), 1.0)
    emission_mat("hud", (0.02, 0.025, 0.04), 1.0)
    ghost_mat()


COLL = {}
PARTS = []


def collection(name, parent=None):
    c = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(c)
    return c


def _finish_mesh(me, mats, mat_idx, smooth, recalc):
    for m in mats:
        me.materials.append(MAT[m])
    if mat_idx is not None and len(mats) > 1:
        me.polygons.foreach_set("material_index", mat_idx)
    flags = smooth if isinstance(smooth, list) else [smooth] * len(me.polygons)
    me.polygons.foreach_set("use_smooth", flags)
    if recalc:
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(me)
        bm.free()
    me.update()


def new_mesh(name, data, mats, smooth=True, recalc=True, colors=None, center=True):
    if isinstance(data, Buf):
        verts, faces, mi = data.v, data.f, data.m
    else:
        verts, faces = data
        mi = None
    arr = np.asarray(verts, dtype=float)
    c = (arr.min(0) + arr.max(0)) / 2 if center else np.zeros(3)
    arr = arr - c
    me = bpy.data.meshes.new(name)
    me.from_pydata(arr.tolist(), [], [list(f) for f in faces])
    if colors is not None:
        ca = me.color_attributes.new("livery", "FLOAT_COLOR", "CORNER")
        flat = []
        for poly, col in zip(me.polygons, colors):
            flat.extend((*col, 1.0) * poly.loop_total)
        ca.data.foreach_set("color", flat)
    _finish_mesh(me, mats, mi, smooth, recalc)
    return me, Vector(c)


def thicken(data, th, colors=None):
    """Give an open, outward-facing surface a real thickness (inwards). The rim gets its own
    flat-shaded vertices so the smooth outer skin keeps clean normals right up to its edge."""
    if isinstance(data, Buf):
        verts, faces, mats = data.v, data.f, data.m
    else:
        verts, faces = data
        mats = [0] * len(faces)
    P = np.asarray(verts, float)
    n = len(P)
    vn = np.zeros_like(P)
    for fc in faces:
        q = P[list(fc)]
        nrm = np.zeros(3)
        for i in range(len(fc)):          # Newell normal
            a, b = q[i], q[(i + 1) % len(fc)]
            nrm += np.array([(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]),
                             (a[0] - b[0]) * (a[1] + b[1])])
        vn[list(fc)] += nrm
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    inner = P - vn * th
    out = Buf()
    out.v = [tuple(p) for p in P] + [tuple(p) for p in inner]
    src, flags = [], []
    for i, fc in enumerate(faces):
        out.f.append(list(fc)); out.m.append(mats[i]); src.append(i); flags.append(True)
    for i, fc in enumerate(faces):
        out.f.append([k + n for k in reversed(fc)]); out.m.append(mats[i]); src.append(i); flags.append(True)
    edges = {}
    for i, fc in enumerate(faces):
        for k in range(len(fc)):
            a, b = fc[k], fc[(k + 1) % len(fc)]
            key = (min(a, b), max(a, b))
            edges.setdefault(key, []).append((a, b, i))
    for key, uses in edges.items():
        if len(uses) != 1:
            continue
        a, b, i = uses[0]
        base = len(out.v)
        out.v += [tuple(P[b]), tuple(P[a]), tuple(inner[a]), tuple(inner[b])]
        out.f.append([base, base + 1, base + 2, base + 3]); out.m.append(mats[i]); src.append(i)
        flags.append(False)
    cols = [colors[i] for i in src] if colors is not None else None
    return out, cols, flags


def add_part(name, data, mats, cat, *, shell=False, cut=False, xpl=(0, 0, 0), smooth=True,
             solid=0.0, recalc=True, colors=None, mesh=None, loc=None, rot=None, coll=None):
    """Create one part object and tag it for the mode system."""
    if mesh is None and solid and smooth is True and not recalc:
        data, colors, smooth = thicken(data, solid, colors)
        solid = 0.0
    if mesh is None:
        mesh, c = new_mesh(name, data, mats if isinstance(mats, (list, tuple)) else [mats],
                           smooth, recalc, colors)
        loc = c if loc is None else loc
    ob = bpy.data.objects.new(name, mesh)
    ob.location = loc if loc is not None else (0, 0, 0)
    if rot is not None:
        ob.rotation_euler = rot
    (coll or COLL[cat]).objects.link(ob)
    if solid:
        mod = ob.modifiers.new("Skin thickness", "SOLIDIFY")
        mod.thickness = solid
        mod.offset = -1.0
        mod.use_even_offset = True
    ob["ap_part"] = 1
    ob["ap_cat"] = cat
    ob["ap_shell"] = int(shell)
    ob["ap_cut"] = int(cut)
    ob["ap_xpl"] = list(map(float, xpl))
    PARTS.append(ob)
    return ob


# =============================================================================
#  Fuselage
# =============================================================================
SECTIONS = [1.6, 3.2, 5.0, 7.0, 9.5, 12.0, 14.5, 17.0, 19.5, 22.0, 24.5, 27.0, 29.5, 32.0, 34.8]
TAIL_CONE_S = 34.8
MID_S = 18.8

DOORS = [  # name, s0, s1, z0, z1, sides
    ("Passenger door 1", 5.30, 6.15, CABIN_FLOOR + 0.02, 1.40, "LR"),
    ("Passenger door 2", 27.45, 28.30, CABIN_FLOOR + 0.02, 1.40, "LR"),
    ("Overwing exit 1", 14.95, 15.45, -0.08, 0.95, "LR"),
    ("Overwing exit 2", 15.75, 16.25, -0.08, 0.95, "LR"),
    ("Forward cargo door", 8.0, 9.8, -1.85, -0.78, "R"),
    ("Aft cargo door", 22.6, 24.4, -1.85, -0.78, "R"),
]


def fus_quad(corners, sg, off, n=6):
    """Patch of the skin bounded by four (station, angle-in-degrees) corners, bilinear between."""
    (s0, t0), (s1, t1), (s2, t2), (s3, t3) = corners
    rows = []
    for j in range(n + 1):
        v = j / n
        row = []
        for i in range(n + 1):
            u = i / n
            s_ = (1 - u) * (1 - v) * s0 + u * (1 - v) * s1 + u * v * s2 + (1 - u) * v * s3
            t_ = (1 - u) * (1 - v) * t0 + u * (1 - v) * t1 + u * v * t2 + (1 - u) * v * t3
            row.append(fus_point(s_, sg * RAD(t_), off))
        rows.append(row)
    v, f = grid(rows)
    p = Vector(v[f[0][0]])
    zc = fus_section(X_NOSE - p.x)[0]
    if face_normal(v, f[0]).dot(Vector((0, p.y, p.z - zc))) < 0:
        f = [fc[::-1] for fc in f]
    return v, f


def radial(th):
    return Vector((0.0, math.sin(th), math.cos(th)))


def long_spread(s, k):
    return Vector((-(s - MID_S) * k, 0, 0))


def cut_region(s0, s1, k):
    """Which fuselage skin panels the cutaway removes (stepped section on the left side)."""
    if s0 < 5.0 - 1e-6 or s1 > 32.0 + 1e-6:
        return False
    if k in (1, 2):
        return True
    if k == 0:
        return 7.0 - 1e-6 <= s0 and s1 <= 27.0 + 1e-6
    if k == 3:
        return (7.0 - 1e-6 <= s0 and s1 <= 12.0 + 1e-6) or (19.5 - 1e-6 <= s0 and s1 <= 27.0 + 1e-6)
    return False


def cut_point(s, th):
    """Is a point on the skin inside the cutaway opening?"""
    for i in range(len(SECTIONS) - 1):
        if SECTIONS[i] <= s < SECTIONS[i + 1]:
            k = int(((th + RAD(22.5)) % TAU) // RAD(45))
            return cut_region(SECTIONS[i], SECTIONS[i + 1], k)
    return False


def build_fuselage():
    cat = "Fuselage"
    # radome + panels + tail cone
    v, f, cols = fus_patch(0.0, SECTIONS[0], 0, TAU, wrap=True)
    add_part("Radome", (v, f), ["livery"], cat, shell=True, colors=cols, recalc=False,
             solid=0.03, xpl=(5.0, 0, 0))
    for i in range(len(SECTIONS) - 1):
        s0, s1 = SECTIONS[i], SECTIONS[i + 1]
        for k in range(8):
            th0 = RAD(-22.5 + 45 * k)
            th1 = th0 + RAD(45)
            v, f, cols = fus_patch(s0, s1, th0, th1)
            thc = (th0 + th1) / 2
            side = "top" if k == 0 else ("keel" if k == 4 else ("L" if k < 4 else "R"))
            name = f"Skin panel {i + 1:02d}-{k + 1} ({side})"
            sc = (s0 + s1) / 2
            xpl = radial(thc) * 3.2 + long_spread(sc, 0.3)
            add_part(name, (v, f), ["livery"], cat, shell=True, cut=cut_region(s0, s1, k),
                     colors=cols, recalc=False, solid=0.03, xpl=xpl)
    v, f, cols = fus_patch(TAIL_CONE_S, FUS_LEN, 0, TAU, wrap=True)
    end = [fus_point(FUS_LEN, t) for t in theta_samples(0, TAU, wrap=True)]
    v2, f2 = ngon_cap(end[::-1])
    o = len(v)
    v = v + v2
    f = f + [[i + o for i in fc] for fc in f2]
    cols = cols + [(0.3, 0.32, 0.35)]
    add_part("Tail cone", (v, f), ["livery"], cat, shell=True, colors=cols, recalc=False,
             solid=0.03, xpl=(-5.0, 0, 1.0))

    # doors: a dark seal plus a painted plug, both following the skin
    for name, s0, s1, z0, z1, sides in DOORS:
        for sd in sides:
            sg = 1 if sd == "L" else -1
            sm = (s0 + s1) / 2
            t0, t1 = theta_at_height(sm, z1), theta_at_height(sm, z0)
            buf = Buf()
            seal_rows = [[fus_point(s, sg * t, 0.006) for t in lin(t0 - 0.012, t1 + 0.012, RAD(4))]
                         for s in lin(s0 - 0.02, s1 + 0.02, 0.25)]
            buf.add(grid(seal_rows, flip=sg < 0), 1)
            dv, df, dc = [], [], []
            rows = [[fus_point(s, sg * t, 0.012) for t in lin(t0, t1, RAD(4))] for s in lin(s0, s1, 0.25)]
            gv, gf = grid(rows, flip=sg < 0)
            buf.add((gv, gf), 0)
            cols = []
            for fc in buf.f:
                cen = sum((Vector(buf.v[i]) for i in fc), Vector()) / len(fc)
                s = X_NOSE - cen.x
                zc, a, b = fus_section(s)
                cols.append(livery_rgb((cen.z - zc) / b, s))
            # face order: seal faces first then door faces, matching buf.f
            thc = sg * (t0 + t1) / 2
            add_part(f"{name} {sd}", buf, ["livery", "seal"], cat, shell=True, colors=cols,
                     recalc=False, cut=(sd == "L" and cut_point(sm, thc)),
                     xpl=radial(thc) * 4.6 + long_spread(sm, 0.3))

    # passenger windows (one shared mesh, ~70 objects)
    wbuf = Buf()
    wbuf.add(ngon_cap(superellipse(0.27, 0.37, 28, 3.2, "XZ", 0.004)), 1)
    wbuf.add(ngon_cap(superellipse(0.22, 0.32, 28, 3.2, "XZ", 0.009)), 0)
    wme, _ = new_mesh("Window", wbuf, ["glass", "seal"], smooth=False, recalc=False, center=False)
    door_ranges = [(d[1] - 0.22, d[2] + 0.22) for d in DOORS if d[5] == "LR"]
    s = 6.95
    n = 0
    while s < 27.2:
        if not any(a <= s <= b for a, b in door_ranges):
            n += 1
            for sd, sg in (("L", 1), ("R", -1)):
                th = sg * theta_at_height(s, 0.38)
                p = Vector(fus_point(s, th))
                add_part(f"Window {n:02d}{sd}", None, None, cat, mesh=wme, loc=p,
                         rot=(math.pi / 2 - th, 0, 0), shell=True,
                         cut=(sd == "L" and cut_point(s, th)),
                         xpl=radial(th) * 5.4 + long_spread(s, 0.3))
        s += 0.533

    # flight-deck windshields: dark frame + glass, as quads in (station, angle) space
    panes = [("Windshield", [(2.30, 2.5), (2.55, 28.0), (3.05, 26.0), (2.98, 2.5)]),
             ("Side window", [(2.62, 31.0), (3.28, 50.0), (3.75, 40.0), (3.12, 29.0)]),
             ("Aft side window", [(3.36, 52.0), (4.15, 57.5), (4.15, 44.0), (3.82, 42.0)])]
    for name, corners in panes:
        cs = sum(c[0] for c in corners) / 4
        ct = sum(c[1] for c in corners) / 4
        frame = [(cs + (a - cs) * 1.07 + (0.02 if a > cs else -0.02), ct + (t - ct) * 1.12) for a, t in corners]
        for sd, sg in (("L", 1), ("R", -1)):
            buf = Buf()
            buf.add(fus_quad(frame, sg, 0.006), 1)
            buf.add(fus_quad(corners, sg, 0.012), 0)
            thc = sg * RAD(ct)
            add_part(f"{name} {sd}", buf, ["glass", "seal"], cat, shell=True, recalc=False,
                     xpl=radial(thc) * 4.0 + Vector((4.0, 0, 0)))

    # belly fairing (wing-to-body)
    rows = []
    for s in lin(10.4, 22.6, 0.4):
        w = max(0.0, math.sin(math.pi * (s - 10.4) / 12.2)) ** 0.6
        row = []
        for th in lin(RAD(96), RAD(264), RAD(4)):
            g = max(0.0, math.sin(math.pi * (th - RAD(96)) / RAD(168))) ** 0.7
            zc, a, b = fus_section(s)
            row.append((X_NOSE - s, (a + 0.012 + 0.38 * w * g) * math.sin(th),
                        zc + (b + 0.012 + 0.30 * w * g) * math.cos(th)))
        rows.append(row)
    v, f = grid(rows)
    if face_normal(v, f[len(f) // 2]).z > 0:
        f = [fc[::-1] for fc in f]
    add_part("Belly fairing", (v, f), ["belly"], cat, shell=True, recalc=False, solid=0.03,
             xpl=(0, 0, -5.5))


# =============================================================================
#  Airframe structure (fuselage)
# =============================================================================
def build_fuselage_structure():
    cat = "Structure"
    n = 56
    s = 2.5
    k = 0
    while s <= 34.6:
        zc, a, b = fus_section(s)
        rings = []
        for dx, inset in ((0.025, 0.03), (0.025, 0.14), (-0.025, 0.14), (-0.025, 0.03)):
            rings.append([(X_NOSE - s + dx, (a - inset) * math.sin(TAU * j / n),
                           zc + (b - inset) * math.cos(TAU * j / n)) for j in range(n)])
        v, f = grid(rings + [rings[0]], wrap=True)
        k += 1
        add_part(f"Frame {k:02d}", (v, f), ["primer"], cat, smooth=True,
                 xpl=long_spread(s, 0.3) + Vector((0, 0, 0)))
        s += 1.0

    # stringers: radial webs following the skin
    # stringers are spliced at the production breaks: forward, centre and aft fuselage
    for sec, (s0, s1) in zip("FCA", ((2.6, 12.0), (12.04, 24.5), (24.54, 34.4))):
        for j in range(22):
            th = RAD(360 / 22 * j + 8.0)
            rows = [[fus_point(s, th, -0.03), fus_point(s, th, -0.085)] for s in lin(s0, s1, 0.8)]
            v, f = grid(rows)
            add_part(f"Stringer {sec}{j + 1:02d}", (v, f), ["alu"], cat, solid=0.025, smooth=False,
                     xpl=radial(th) * 2.0 + long_spread((s0 + s1) / 2, 0.3))

    # cabin floor panels and beams
    stations = [2.8, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.2]
    for i in range(len(stations) - 1):
        s0, s1 = stations[i] + 0.01, stations[i + 1] - 0.01
        hw = min(interior_half_width(s, CABIN_FLOOR, 0.15) for s in (s0, s1, (s0 + s1) / 2))
        v, f = box(((X_NOSE - s0 + X_NOSE - s1) / 2, 0, CABIN_FLOOR - 0.02), (s1 - s0, 2 * hw, 0.04))
        add_part(f"Cabin floor panel {i + 1:02d}", (v, f), ["carpet"], cat, smooth=False,
                 xpl=Vector((0, 0, 7.0)) + long_spread((s0 + s1) / 2, 0.3))
    s, k = 3.0, 0
    while s <= 30.0:
        hw = interior_half_width(s, CABIN_FLOOR - 0.12, 0.10)
        v, f = box((X_NOSE - s, 0, CABIN_FLOOR - 0.12), (0.06, 2 * hw, 0.17))
        k += 1
        add_part(f"Floor beam {k:02d}", (v, f), ["alu"], cat, smooth=False,
                 xpl=Vector((0, 0, 0.6)) + long_spread(s, 0.3))
        s += 1.0

    # pressure bulkheads
    zc, a, b = fus_section(SECTIONS[0] + 0.05)
    pts = [(X_NOSE - SECTIONS[0] - 0.05, (a - 0.05) * math.sin(t), zc + (b - 0.05) * math.cos(t))
           for t in lin(0, TAU, RAD(6))[:-1]]
    v, f = ngon_cap(pts)
    add_part("Forward pressure bulkhead", (v, f), ["primer"], cat, solid=0.03, smooth=False,
             xpl=(7.0, 0, 0))
    s_ab = 30.2
    zc, a, b = fus_section(s_ab)
    rows = []
    for k in range(9):
        r = math.cos(0.5 * math.pi * k / 8)
        d = 0.75 * math.sin(0.5 * math.pi * k / 8)
        rows.append([(X_NOSE - s_ab - d, (a - 0.05) * r * math.sin(t), zc + (b - 0.05) * r * math.cos(t))
                     for t in lin(0, TAU, RAD(6))[:-1]])
    v, f = grid(rows, wrap=True)
    add_part("Aft pressure bulkhead", (v, f), ["primer"], cat, solid=0.03, recalc=True,
             xpl=(-6.5, 0, 0))

    v, f = box((X_NOSE - 16.5, 0, -1.99), (7.5, 0.3, 0.12))
    add_part("Keel beam", (v, f), ["alu"], cat, smooth=False, xpl=(0, 0, -3.2))


# =============================================================================
#  Wings
# =============================================================================
def wing_te_x(y):
    if y <= KINK_Y:
        return WING_LE_X0 - 7.0 - 0.07 * y
    t0 = WING_LE_X0 - 7.0 - 0.07 * KINK_Y
    t1 = WING_LE_X0 - TIP_Y * TAN_LE - TIP_CHORD
    return t0 + (t1 - t0) * (y - KINK_Y) / (TIP_Y - KINK_Y)


def wing_surf(sign):
    secs = []
    for y, t, inc in ((0.0, 0.155, 2.0), (1.4, 0.15, 2.0), (KINK_Y, 0.12, 0.8), (TIP_Y, 0.10, -1.0)):
        xle = WING_LE_X0 - y * TAN_LE
        c = xle - wing_te_x(y)
        cd = Vector((-math.cos(RAD(inc)), 0, -math.sin(RAD(inc))))
        n0 = Vector((0, -sign * math.sin(DIHEDRAL), math.cos(DIHEDRAL)))
        n = (n0 - cd * n0.dot(cd)).normalized()
        secs.append(dict(q=y, le=Vector((xle, sign * y, WING_Z0 + y * math.tan(DIHEDRAL))),
                         c=c, cd=cd, n=n, t=t))
    return Surf(secs, camber=0.02)


def wing_lower_z(S, y, x):
    le, c, cd, n, t = S.frame(y)
    u = min(max((le.x - x) / c, 0.0), 1.0)
    return S.pt(y, u, -1).z


def build_wing(sign, S):
    cat = "Wings"
    sd = "L" if sign > 0 else "R"
    base = Vector((0, sign * 7.5, 0))
    left = sign > 0
    spans = [(1.4, KINK_Y), (KINK_Y, 11.6), (11.6, TIP_Y)]
    names = ["inboard", "mid", "outboard"]
    for (q0, q1), nm in zip(spans, names):
        add_part(f"Wing upper skin {nm} {sd}", surf_patch(S, q0, q1, 0.13, 0.60, +1, 12), ["wing"], cat,
                 shell=True, cut=(left and nm != "outboard"), recalc=False, solid=0.02,
                 xpl=base + Vector((0, 0, 1.8)))
        add_part(f"Wing lower skin {nm} {sd}", surf_patch(S, q0, q1, 0.13, 0.72, -1, 12), ["wing"], cat,
                 shell=True, recalc=False, solid=0.02, xpl=base + Vector((0, 0, -1.8)))
    for i, (q0, q1) in enumerate([(1.4, 3.0), (6.2, 6.7), (12.7, TIP_Y)]):
        add_part(f"Wing trailing-edge panel {i + 1} {sd}", surf_patch(S, q0, q1, 0.60, 0.72, +1, 4),
                 ["wing"], cat, shell=True, cut=(left and i < 2), recalc=False, solid=0.02,
                 xpl=base + Vector((0, 0, 1.8)))
    for i, (q0, q1) in enumerate([(1.4, 3.6), (5.4, 6.1), (16.6, TIP_Y)]):
        add_part(f"Fixed leading edge {i + 1} {sd}", surf_solid(S, q0, q1, 0.0, 0.13, 10), ["wing"], cat,
                 shell=True, xpl=base + Vector((1.6, 0, 0)))
    for i, (q0, q1) in enumerate([(3.6, 5.4), (6.1, 8.8), (8.8, 11.5), (11.5, 14.1), (14.1, 16.6)]):
        add_part(f"Slat {i + 1} {sd}", surf_solid(S, q0, q1, 0.0, 0.13, 10), ["wing"], cat,
                 shell=True, xpl=base + Vector((2.6, 0, -0.2)))
    te = [("Trailing-edge fairing 1", 1.4, 2.0), ("Inboard flap", 2.0, 6.4), ("Outboard flap", 6.5, 12.8),
          ("Trailing-edge fairing 2", 12.8, 13.0), ("Aileron", 13.0, 16.4),
          ("Trailing-edge fairing 3", 16.4, TIP_Y)]
    for name, q0, q1 in te:
        mov = name.endswith("flap") or name == "Aileron"
        add_part(f"{name} {sd}", surf_solid(S, q0, q1, 0.72, 1.0, 8), ["wing"], cat, shell=True,
                 xpl=base + Vector((-2.6 if mov else -1.4, 0, -0.5 if mov else 0)))
    sp = [(3.0, 6.2), (6.7, 8.2), (8.2, 9.7), (9.7, 11.2), (11.2, 12.7)]
    for i, (q0, q1) in enumerate(sp):
        add_part(f"Spoiler {i + 1} {sd}", surf_patch(S, q0, q1, 0.60, 0.72, +1, 4, off=0.006), ["wing"],
                 cat, shell=True, cut=left, recalc=False, solid=0.02, xpl=base + Vector((-0.6, 0, 3.0)))

    # flap-track fairings (canoes)
    for i, y in enumerate((4.3, 8.6, 11.4)):
        le, c, cd, n, t = S.frame(y)
        x0 = le.x - 0.55 * c
        L = 0.55 * c + 1.1
        z = wing_lower_z(S, y, x0 - 0.4 * L) - 0.10
        prof = [(0, 0), (0.12 * L, 0.11), (0.35 * L, 0.18), (0.65 * L, 0.17), (0.88 * L, 0.10), (L, 0)]
        v, f = lathe(prof, (x0, sign * y, z), (-1, 0, 0), (0, 0, 1), n=16)
        add_part(f"Flap-track fairing {i + 1} {sd}", (v, f), ["wing"], cat, shell=True,
                 xpl=base + Vector((-1.0, 0, -2.6)))

    # winglet
    le, c, cd, n, t = S.frame(TIP_Y)
    secs = [dict(q=0.0, le=le.copy(), c=c, cd=cd, n=n, t=t)]
    R = 0.55
    q = 0.0
    prev = le.copy()
    for phi in (25, 50, 70, 80):
        ph = RAD(phi)
        p = Vector((le.x - 0.35 * phi / 80, sign * (TIP_Y + R * math.sin(ph)), le.z + R * (1 - math.cos(ph))))
        q += (p - prev).length
        prev = p
        nn = Vector((0, -sign * math.sin(ph + DIHEDRAL), math.cos(ph + DIHEDRAL)))
        secs.append(dict(q=q, le=p, c=1.55 - 0.55 * phi / 80, cd=Vector((-1, 0, 0)), n=nn, t=0.095))
    ph = RAD(80) + DIHEDRAL
    top = prev + Vector((-1.25, sign * 2.1 * math.sin(RAD(10)), 2.1 * math.cos(RAD(10))))
    q += (top - prev).length
    secs.append(dict(q=q, le=top, c=0.42, cd=Vector((-1, 0, 0)),
                     n=Vector((0, -sign * math.sin(ph), math.cos(ph))), t=0.08))
    W = Surf(secs, camber=0.02)
    add_part(f"Winglet {sd}", surf_solid(W, 0.0, q, 0.0, 1.0, 14, step=0.25), ["tail"], cat, shell=True,
             xpl=base + Vector((0, sign * 2.6, 1.2)))
    tip = S.pt(TIP_Y - 0.15, 0.02, 0)
    v, f = lathe([(0, 0), (0.05, 0.07), (0.16, 0.075), (0.2, 0)], tip + Vector((0.1, 0, 0)),
                 (-1, 0, 0), (0, 0, 1), n=12)
    add_part(f"Navigation light {sd}", (v, f), ["nav_red" if left else "nav_green"], cat, shell=True,
             xpl=base + Vector((1.5, sign * 3.5, 0)))

    # internal structure
    for nm, u in (("Front spar", 0.15), ("Rear spar", 0.65)):
        add_part(f"{nm} {sd}", surf_spar(S, 1.45, TIP_Y - 0.1, u), ["alu"], "Structure", smooth=False,
                 recalc=False, solid=0.035, xpl=base + Vector((1.0 if u < 0.5 else -1.0, 0, 0)))
    y, k = 1.8, 0
    while y < TIP_Y - 0.3:
        k += 1
        add_part(f"Wing rib {k:02d} {sd}", surf_rib(S, y, 0.155, 0.645), ["primer"], "Structure",
                 smooth=False, recalc=False, solid=0.025, xpl=base + Vector((0, sign * (y - 1.8) * 0.12, 0)))
        y += 0.75


def build_center_box(SL, SR):
    b = Buf()
    b.add(surf_solid(SL, 0.0, 1.45, 0.13, 0.72, 10))
    b.add(surf_solid(SR, 0.0, 1.45, 0.13, 0.72, 10))
    add_part("Centre wing box", b, ["primer"], "Structure", xpl=(0, 0, -1.6))


# =============================================================================
#  Tail
# =============================================================================
def build_tail():
    cat = "Tail"
    h1 = 6.4
    secs = [dict(q=0.0, le=Vector((-10.6, 0, 1.4)), c=6.0, cd=Vector((-1, 0, 0)), n=Vector((0, 1, 0)), t=0.11),
            dict(q=h1, le=Vector((-10.6 - h1 * math.tan(RAD(35)), 0, 1.4 + h1)), c=2.0,
                 cd=Vector((-1, 0, 0)), n=Vector((0, 1, 0)), t=0.10)]
    S = Surf(secs)
    up = Vector((0, 0, 4.8))
    add_part("Fin leading edge", surf_solid(S, 0.0, 6.3, 0.0, 0.12, 10), ["tail"], cat, shell=True,
             xpl=up + Vector((1.4, 0, 0)))
    add_part("Fin skin L", surf_patch(S, 0.0, 6.3, 0.12, 0.68, +1, 12), ["tail"], cat, shell=True, cut=True,
             recalc=False, solid=0.02, xpl=up + Vector((0, 1.5, 0)))
    add_part("Fin skin R", surf_patch(S, 0.0, 6.3, 0.12, 0.68, -1, 12), ["tail"], cat, shell=True,
             recalc=False, solid=0.02, xpl=up + Vector((0, -1.5, 0)))
    add_part("Fin trailing-edge fairing", surf_solid(S, 0.0, 0.6, 0.68, 1.0, 8), ["tail"], cat, shell=True,
             xpl=up + Vector((-1.2, 0, 0)))
    add_part("Rudder", surf_solid(S, 0.6, 6.3, 0.68, 1.0, 8), ["tail"], cat, shell=True,
             xpl=up + Vector((-2.2, 0, 0)))
    add_part("Fin tip", surf_solid(S, 6.3, h1, 0.0, 1.0, 14), ["tail"], cat, shell=True,
             xpl=up + Vector((0, 0, 1.2)))
    for nm, u in (("Fin front spar", 0.15), ("Fin rear spar", 0.65)):
        add_part(nm, surf_spar(S, 0.1, 6.25, u), ["alu"], "Structure", smooth=False, recalc=False,
                 solid=0.03, xpl=up + Vector((0.6 if u < 0.5 else -0.6, 0, 0)))
    for k in range(6):
        add_part(f"Fin rib {k + 1}", surf_rib(S, 0.9 + k * 0.9, 0.155, 0.645), ["primer"], "Structure",
                 smooth=False, recalc=False, solid=0.025, xpl=up + Vector((0, 0, k * 0.15)))
    # dorsal fillet
    D = Surf([dict(q=0.0, le=Vector((-7.4, 0, 1.92)), c=3.4, cd=Vector((-1, 0, 0)), n=Vector((0, 1, 0)), t=0.10),
              dict(q=1.0, le=Vector((-11.25, 0, 2.55)), c=1.1, cd=Vector((-1, 0, 0)), n=Vector((0, 1, 0)),
                   t=0.14)])
    add_part("Dorsal fin", surf_solid(D, 0.0, 1.0, 0.0, 1.0, 10, step=0.2), ["tail"], cat, shell=True,
             xpl=up + Vector((2.2, 0, -1.6)))

    # horizontal stabiliser
    for sign in (1, -1):
        sd = "L" if sign > 0 else "R"
        g = RAD(6)
        n0 = Vector((0, -sign * math.sin(g), math.cos(g)))
        H = Surf([dict(q=0.0, le=Vector((-13.0, 0, 0.8)), c=3.7, cd=Vector((-1, 0, 0)), n=n0, t=0.11),
                  dict(q=6.2, le=Vector((-13.0 - 6.2 * math.tan(RAD(32)), sign * 6.2, 0.8 + 6.2 * math.tan(g))),
                       c=1.3, cd=Vector((-1, 0, 0)), n=n0, t=0.09)])
        b = Vector((-2.5, sign * 3.6, 1.0))
        add_part(f"Stabiliser leading edge {sd}", surf_solid(H, 0.0, 6.0, 0.0, 0.12, 10), ["wing"], cat,
                 shell=True, xpl=b + Vector((1.3, 0, 0)))
        add_part(f"Stabiliser upper skin {sd}", surf_patch(H, 0.0, 6.0, 0.12, 0.70, +1, 12), ["wing"], cat,
                 shell=True, cut=(sign > 0), recalc=False, solid=0.02, xpl=b + Vector((0, 0, 1.3)))
        add_part(f"Stabiliser lower skin {sd}", surf_patch(H, 0.0, 6.0, 0.12, 0.70, -1, 12), ["wing"], cat,
                 shell=True, recalc=False, solid=0.02, xpl=b + Vector((0, 0, -1.3)))
        add_part(f"Stabiliser trailing-edge fairing {sd}", surf_solid(H, 0.0, 1.2, 0.70, 1.0, 8), ["wing"],
                 cat, shell=True, xpl=b + Vector((-1.0, 0, 0)))
        add_part(f"Elevator {sd}", surf_solid(H, 1.2, 6.0, 0.70, 1.0, 8), ["wing"], cat, shell=True,
                 xpl=b + Vector((-2.0, 0, 0)))
        add_part(f"Stabiliser tip {sd}", surf_solid(H, 6.0, 6.2, 0.0, 1.0, 12), ["wing"], cat, shell=True,
                 xpl=b + Vector((0, sign * 1.2, 0)))
        for nm, u in (("front", 0.15), ("rear", 0.65)):
            add_part(f"Stabiliser {nm} spar {sd}", surf_spar(H, 0.05, 5.95, u), ["alu"], "Structure",
                     smooth=False, recalc=False, solid=0.03,
                     xpl=b + Vector((0.5 if u < 0.5 else -0.5, 0, 0)))
        for k in range(5):
            add_part(f"Stabiliser rib {k + 1} {sd}", surf_rib(H, 1.0 + k * 1.0, 0.155, 0.645), ["primer"],
                     "Structure", smooth=False, recalc=False, solid=0.025, xpl=b)

    v, f = cyl((X_NOSE - FUS_LEN + 0.25, 0, fus_section(FUS_LEN)[0]),
               (X_NOSE - FUS_LEN - 0.25, 0, fus_section(FUS_LEN)[0]), 0.17, 20)
    add_part("APU exhaust", (v, f), ["hot"], "Fuselage", shell=True, xpl=(-6.5, 0, 1.0))


# =============================================================================
#  Engines
# =============================================================================
def stage(d, r_in, r_hub, r_tip, nblades, chord, stagger, x_f, c_y, c_z, thick=0.012, disk_w=0.05):
    b = Buf()
    org = Vector((x_f, c_y, c_z))
    ax = Vector((-1, 0, 0))
    if r_in is not None:
        b.add(lathe([(d - disk_w / 2, r_in), (d - disk_w / 2, r_hub), (d + disk_w / 2, r_hub),
                     (d + disk_w / 2, r_in)], org, ax, (0, 0, 1), n=32, loop=True), 0)
    st = RAD(stagger)
    for k in range(nblades):
        ph = TAU * k / nblades
        rad = Vector((0, math.sin(ph), math.cos(ph)))
        tan = Vector((0, math.cos(ph), -math.sin(ph)))
        cd = ax * math.cos(st) + tan * math.sin(st)
        tn = cd.cross(rad).normalized()
        pts = []
        for r in (r_hub - 0.005, r_tip):
            cen = org + ax * d + rad * r
            for a_, b_ in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                pts.append(tuple(cen + cd * (a_ * chord / 2) + tn * (b_ * thick / 2)))
        f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        b.add((pts, f), 1 if r_in is not None else 0)
    return b


def fan_blade_mesh():
    """One fan blade in engine-local coordinates (axis +X forward, blade along +Z)."""
    rows = []
    rh, rt = 0.33, 0.862
    for k in range(9):
        t = k / 8
        r = rh + (rt - rh) * t
        beta = RAD(28 + 34 * t)
        c = 0.21 + 0.15 * t
        cd = Vector((-math.cos(beta), math.sin(beta), 0))
        tn = Vector((math.sin(beta), math.cos(beta), 0))
        th = 0.022 * (1 - 0.7 * t)
        sweep = Vector((0.03 * t * t, 0, 0))
        cen = Vector((0, 0, r)) + sweep
        ring = []
        for f, w in ((-0.5, 0), (-0.2, 1), (0.2, 0.75), (0.5, 0), (0.2, -0.35), (-0.2, -0.5)):
            ring.append(tuple(cen + cd * (f * c) + tn * (w * th)))
        rows.append(ring)
    v, f = grid(rows, wrap=True)
    for row in (rows[0], rows[-1]):
        s = len(v)
        v.extend(row)
        f.append(list(range(s, len(v))))
    me, _ = new_mesh("Fan blade", (v, f), ["titanium"], smooth=True, center=False)
    return me


def build_engine(sign, S, blade_me):
    cat = "Engines"
    sd = "L" if sign > 0 else "R"
    y_e = sign * ENG_Y
    wz = WING_Z0 + ENG_Y * math.tan(DIHEDRAL)
    z_e = wz - 1.55
    x_f = (WING_LE_X0 - ENG_Y * TAN_LE) + 2.5
    org = Vector((x_f, y_e, z_e))
    AX, UP = (-1, 0, 0), (0, 0, 1)
    G = Vector((0, sign * 7.5, -4.4))
    mid = 2.3

    def axial(d):
        return Vector((-(d - mid) * 1.25, 0, 0))

    def P(name, data, mats, d, shell=False, cut=False, extra=(0, 0, 0), smooth=True, coll=None):
        return add_part(f"{name} {sd}", data, mats, cat, shell=shell, cut=cut, smooth=smooth,
                        xpl=G + axial(d) + Vector(extra))

    # lathe angle 0 = top; with the axis pointing aft, angles 0..pi sweep the -Y side
    halves = (("outboard" if sign > 0 else "inboard", math.pi, TAU, 1),
              ("inboard" if sign > 0 else "outboard", 0.0, math.pi, -1))

    P("Inlet lip", lathe([(0.0, 0.93), (0.03, 0.975), (0.1, 1.01), (0.3, 1.04), (0.3, 0.865), (0.1, 0.86),
                          (0.03, 0.875)], org, AX, UP, n=64, loop=True), ["lip"], 0.0, shell=True,
      extra=(0.8, 0, 0))
    P("Inlet cowl", lathe([(0.3, 1.04), (0.65, 1.055), (1.0, 1.06), (1.0, 0.875), (0.3, 0.865)], org, AX, UP,
                          n=64, loop=True), ["nacelle"], 0.6, shell=True, extra=(0.4, 0, 0))
    for nm, a0, a1, sg in halves:
        hs = Vector((0, sg * 1.8, 0))
        is_cut = (sign > 0 and sg > 0)
        P(f"Fan cowl ({nm})", lathe([(1.0, 1.06), (1.6, 1.065), (2.2, 1.055), (2.2, 0.915), (1.0, 0.915)],
                                    org, AX, UP, a0, a1, 64, True), ["nacelle"], 1.6, shell=True, cut=is_cut,
          extra=hs * 1.3)
        P(f"Fan case ({nm})", lathe([(1.0, 0.875), (2.15, 0.88), (2.15, 0.905), (1.0, 0.905)], org, AX, UP,
                                    a0, a1, 64, True), ["composite"], 1.55, shell=True, cut=is_cut, extra=hs * 0.6)
        P(f"Thrust reverser ({nm})", lathe([(2.2, 1.055), (2.8, 1.0), (3.4, 0.86), (3.4, 0.82), (2.8, 0.80),
                                            (2.2, 0.88)], org, AX, UP, a0, a1, 64, True), ["nacelle"], 2.8,
          shell=True, cut=is_cut, extra=hs * 1.3)
        P(f"Core cowl ({nm})", lathe([(2.0, 0.50), (2.6, 0.565), (3.3, 0.53), (3.95, 0.455), (3.95, 0.43),
                                      (3.3, 0.50), (2.6, 0.535), (2.0, 0.475)], org, AX, UP, a0, a1, 48, True),
          ["metal"], 3.0, shell=True, cut=is_cut, extra=hs * 0.5)
    P("Primary nozzle", lathe([(3.9, 0.45), (4.25, 0.37), (4.25, 0.35), (3.9, 0.43)], org, AX, UP, n=48,
                              loop=True), ["hot"], 4.1, shell=True)
    P("Exhaust plug", lathe([(3.95, 0.30), (4.3, 0.27), (4.65, 0.16), (4.95, 0.0)], org, AX, UP, n=40),
      ["hot"], 4.45, shell=True, extra=(-0.6, 0, 0))
    P("Spinner", lathe([(0.80, 0.0), (0.92, 0.15), (1.08, 0.26), (1.27, 0.335)], org, AX, UP, n=40),
      ["composite"], 1.0, extra=(1.2, 0, 0))
    P("Fan disk", lathe([(1.26, 0.18), (1.26, 0.338), (1.52, 0.345), (1.52, 0.18)], org, AX, UP, n=40,
                        loop=True), ["titanium"], 1.4)
    for k in range(24):
        ph = TAU * k / 24
        loc = org + Vector((-1.39, 0, 0))
        rad = Vector((0, -math.sin(ph), math.cos(ph)))
        add_part(f"Fan blade {k + 1:02d} {sd}", None, None, cat, mesh=blade_me, loc=loc, rot=(ph, 0, 0),
                 xpl=G + axial(1.39) + rad * 1.3)
    P("Outlet guide vanes", stage(1.83, None, 0.49, 0.876, 44, 0.09, 8, x_f, y_e, z_e, 0.01)
      .add(lathe([(1.76, 0.47), (1.76, 0.495), (1.9, 0.495), (1.9, 0.47)], org, AX, UP, n=48, loop=True)),
      ["metal"], 1.83, smooth=False)
    for i, d in enumerate((1.62, 1.72, 1.82)):
        P(f"Booster stage {i + 1}", stage(d - 0.0, 0.12, 0.30, 0.46 - i * 0.01, 40, 0.06, 35, x_f, y_e, z_e),
          ["titanium", "titanium"], d - 0.4, smooth=False, extra=(0, 0, 0.25 * i))
    for i in range(5):
        d = 2.05 + i * 0.15
        rt = 0.41 - i * 0.017
        P(f"HP compressor stage {i + 1}", stage(d, 0.16, 0.26 + i * 0.008, rt, 46, 0.05, 40, x_f, y_e, z_e),
          ["titanium", "titanium"], d, smooth=False)
    P("Combustor", lathe([(2.80, 0.31), (2.80, 0.42), (3.15, 0.43), (3.18, 0.37), (3.15, 0.31)], org, AX, UP,
                         n=48, loop=True), ["hot"], 2.98, extra=(0, 0, 0.6))
    P("HP turbine stage", stage(3.25, 0.15, 0.31, 0.40, 60, 0.045, -45, x_f, y_e, z_e),
      ["hot", "hot"], 3.25, smooth=False)
    for i in range(4):
        d = 3.40 + i * 0.12
        P(f"LP turbine stage {i + 1}", stage(d, 0.12, 0.31, 0.43 + i * 0.01, 70, 0.045, -40, x_f, y_e, z_e),
          ["hot", "hot"], d + 0.25 * i, smooth=False)
    P("LP shaft", lathe([(1.3, 0.045), (4.0, 0.045), (4.0, 0.07), (1.3, 0.07)], org, AX, UP, n=20, loop=True),
      ["steel"], 2.6, extra=(0, 0, -0.9))
    P("HP shaft", lathe([(2.0, 0.12), (3.3, 0.12), (3.3, 0.15), (2.0, 0.15)], org, AX, UP, n=24, loop=True),
      ["steel"], 2.6, extra=(0, 0, -0.55))

    # pylon: side profile in XZ, extruded across the span
    xs = [x_f - 0.75 - k * 0.3 for k in range(13)]
    top = [(x, wing_lower_z(S, ENG_Y, x) + 0.06) for x in xs if x < WING_LE_X0 - ENG_Y * TAN_LE - 0.05]
    prof = [(x_f - 0.75, z_e + 1.0), (x_f - 1.4, z_e + 1.32)] + top[:1] + top[1:] \
           + [(top[-1][0] - 0.4, top[-1][1] - 0.35), (x_f - 3.6, z_e + 0.98)]
    prof = prof[::-1]
    v, f = prism(prof, y_e - 0.21, y_e + 0.21, axis="Y")
    P("Pylon", (v, f), ["nacelle"], 2.4, shell=True, smooth=False, extra=(0, 0, 3.6))
    x0, x1 = x_f - 4.0, x_f - 6.1
    prof2 = [(x0, z_e + 0.62), (x0 + 0.2, wing_lower_z(S, ENG_Y, x0 + 0.2) + 0.04),
             (x1, wing_lower_z(S, ENG_Y, x1) + 0.04), (x1 + 0.3, wing_lower_z(S, ENG_Y, x1) - 0.18)]
    v, f = prism(prof2, y_e - 0.15, y_e + 0.15, axis="Y")
    P("Pylon aft fairing", (v, f), ["nacelle"], 5.0, shell=True, smooth=False, extra=(0, 0, 2.2))


# =============================================================================
#  Landing gear
# =============================================================================
def tire(c, r, w, axis=(0, 1, 0)):
    prof = [(-0.36 * w, r), (0.36 * w, r), (0.49 * w, r * 0.94), (0.5 * w, r * 0.76), (0.44 * w, r * 0.57),
            (-0.44 * w, r * 0.57), (-0.5 * w, r * 0.76), (-0.49 * w, r * 0.94)]
    return lathe(prof, c, axis, (0, 0, 1), n=40, loop=True)


def hub(c, r, w, axis=(0, 1, 0)):
    prof = [(-0.42 * w, r * 0.6), (0.42 * w, r * 0.6), (0.42 * w, r * 0.45), (0.2 * w, r * 0.2),
            (0.2 * w, r * 0.1), (-0.3 * w, r * 0.1), (-0.3 * w, r * 0.45), (-0.42 * w, r * 0.45)]
    return lathe(prof, c, axis, (0, 0, 1), n=32, loop=True)


def build_gear(SL, SR):
    cat = "Landing gear"
    # nose gear
    xn = X_NOSE - 5.0
    rt, wt = 0.38, 0.21
    axz = GROUND + rt
    top = -1.65
    D = Vector((0, 0, -3.4))
    add_part("Nose gear strut", cyl((xn, 0, top), (xn, 0, axz + 0.38), 0.09, 16), ["gear"], cat, xpl=D)
    add_part("Nose gear piston", cyl((xn, 0, axz + 0.5), (xn, 0, axz), 0.065, 16), ["chrome"], cat,
             xpl=D + Vector((0, 0, -0.6)))
    add_part("Nose gear drag brace", cyl((xn, 0, top - 0.55), (xn + 0.95, 0, -1.95), 0.045, 10), ["gear"], cat,
             xpl=D + Vector((1.2, 0, 0.4)))
    b = Buf().add(box((xn + 0.12, 0, axz + 0.28), (0.18, 0.05, 0.05)), 0)
    b.add(box((xn + 0.19, 0, axz + 0.18), (0.05, 0.05, 0.22)), 0)
    add_part("Nose gear torque link", b, ["gear"], cat, smooth=False, xpl=D + Vector((0.7, 0, 0)))
    add_part("Nose gear axle", cyl((xn, -0.36, axz), (xn, 0.36, axz), 0.035, 12), ["steel"], cat,
             xpl=D + Vector((0, 0, -0.5)))
    for sd, sg in (("L", 1), ("R", -1)):
        c = (xn, sg * 0.24, axz)
        add_part(f"Nose wheel tyre {sd}", tire(c, rt, wt), ["rubber"], cat, xpl=D + Vector((0, sg * 1.5, -0.4)))
        add_part(f"Nose wheel hub {sd}", hub(c, rt, wt), ["metal"], cat, xpl=D + Vector((0, sg * 0.9, -0.4)))
        add_part(f"Nose gear door {sd}", box((xn + 0.35, sg * 0.46, -2.28), (1.9, 0.03, 0.42)), ["belly"], cat,
                 smooth=False, xpl=D + Vector((0, sg * 1.4, 0.8)))

    # main gear
    for sign, S in ((1, SL), (-1, SR)):
        sd = "L" if sign > 0 else "R"
        y0 = sign * 3.8
        x0 = 0.25
        rt, wt = 0.57, 0.36
        axz = GROUND + rt
        top = wing_lower_z(S, 3.8, x0) + 0.15
        D = Vector((0, sign * 3.0, -3.6))
        add_part(f"Main gear strut {sd}", cyl((x0, y0, top), (x0, y0, axz + 0.55), 0.14, 18), ["gear"], cat,
                 xpl=D)
        add_part(f"Main gear piston {sd}", cyl((x0, y0, axz + 0.75), (x0, y0, axz), 0.10, 18), ["chrome"], cat,
                 xpl=D + Vector((0, 0, -0.7)))
        add_part(f"Main gear side brace {sd}", cyl((x0, y0, top - 0.7), (x0, sign * 1.55, -1.45), 0.06, 10),
                 ["gear"], cat, xpl=D + Vector((0, -sign * 1.2, 0.5)))
        b = Buf().add(box((x0 + 0.17, y0, axz + 0.45), (0.26, 0.07, 0.07)), 0)
        b.add(box((x0 + 0.27, y0, axz + 0.28), (0.07, 0.07, 0.3)), 0)
        add_part(f"Main gear torque link {sd}", b, ["gear"], cat, smooth=False, xpl=D + Vector((0.9, 0, 0)))
        add_part(f"Main gear axle {sd}", cyl((x0, y0 - 0.62, axz), (x0, y0 + 0.62, axz), 0.06, 14), ["steel"],
                 cat, xpl=D + Vector((0, 0, -0.6)))
        for wn, dy in (("outer", 0.45), ("inner", -0.45)):
            c = (x0, y0 + sign * dy, axz)
            add_part(f"Main wheel tyre {wn} {sd}", tire(c, rt, wt), ["rubber"], cat,
                     xpl=D + Vector((0, sign * dy * 4.0, -0.6)))
            add_part(f"Main wheel hub {wn} {sd}", hub(c, rt, wt), ["metal"], cat,
                     xpl=D + Vector((0, sign * dy * 2.4, -0.6)))
        h = top - (axz + 0.7)
        add_part(f"Main gear leg door {sd}", box((x0 - 0.05, y0 + sign * 0.24, axz + 0.7 + h / 2), (0.95, 0.03, h)),
                 ["wing"], cat, smooth=False, xpl=D + Vector((0, sign * 1.6, 0)))


# =============================================================================
#  Cabin, flight deck, cargo
# =============================================================================
def seat_mesh(name, w, fabric, biz=False):
    b = Buf()
    d = 0.62 if biz else 0.48
    hb = 1.12 if biz else 1.05
    lean = Matrix.Rotation(RAD(-12), 3, "Y")
    b.add(box((0.02, 0, 0.44), (d, w - 0.06, 0.13)), 0)                          # cushion
    b.add(box((-d / 2 + 0.02, 0, 0.82 + (hb - 1.05) / 2), (0.10, w - 0.06, hb - 0.42), lean), 0)   # back
    b.add(box((-d / 2 - 0.03, 0, hb - 0.02), (0.11, w - 0.14, 0.17), lean), 2)    # headrest cover
    for sy in (-1, 1):
        b.add(box((0.05, sy * (w / 2 - 0.02), 0.62), (d * 0.85, 0.05, 0.05)), 1)  # armrest
        b.add(box((0.10, sy * (w / 2 - 0.05), 0.20), (0.05, 0.04, 0.40)), 1)       # legs
        b.add(box((-0.15, sy * (w / 2 - 0.05), 0.20), (0.05, 0.04, 0.40)), 1)
    b.add(box((-0.02, 0, 0.36), (0.3, w - 0.1, 0.05)), 1)                         # frame beam
    me, _ = new_mesh(name, b, [fabric, "plastic_dk", "plastic"], smooth=False, recalc=True, center=False)
    return me


def build_cabin():
    cat = "Cabin"
    UPX = Vector((0, 0, 7.6))
    y_me = seat_mesh("Economy seat", 0.46, "fabric_y")
    j_me = seat_mesh("Business seat", 0.60, "fabric_j", biz=True)
    letters = "ABCDEF"
    row = 0
    for s in (8.6, 9.55, 10.5):
        row += 1
        for li, y in zip("ACDF", (1.17, 0.55, -0.55, -1.17)):
            add_part(f"Seat {row}{li}", None, None, cat, mesh=j_me, loc=(X_NOSE - s, y, CABIN_FLOOR),
                     xpl=UPX + long_spread(s, 0.3))
    s = 11.7
    while s <= 25.95:
        if 14.6 < s < 16.3:
            s = 16.45
            continue
        row += 1
        for li, y in zip(letters, (1.40, 0.94, 0.48, -0.48, -0.94, -1.40)):
            add_part(f"Seat {row}{li}", None, None, cat, mesh=y_me, loc=(X_NOSE - s, y, CABIN_FLOOR),
                     xpl=UPX + long_spread(s, 0.3))
        s += 0.79

    # overhead bins
    prof = [(0.86, 1.62), (0.86, 1.24), (0.96, 1.12), (1.38, 1.12), (1.30, 1.40), (1.05, 1.62)]
    s, k = 8.15, 0
    while s + 1.58 <= 27.3:
        k += 1
        for sd, sg in (("L", 1), ("R", -1)):
            p = [(sg * y, z) for y, z in prof]
            if sg < 0:
                p = p[::-1]
            v, f = prism(p, X_NOSE - s - 1.56, X_NOSE - s, axis="X")
            add_part(f"Overhead bin {k:02d}{sd}", (v, f), ["plastic"], cat, smooth=False, cut=(sg > 0),
                     xpl=Vector((0, sg * 1.0, 9.2)) + long_spread(s, 0.3))
        s += 1.58

    # galleys, lavatories, partitions
    def unit(name, s0, s1, y0, y1, z1, mats=("plastic",), x_off=(0, 0, 0)):
        prof = [(y0, CABIN_FLOOR), (y1, CABIN_FLOOR), (y1, z1 - 0.35 if abs(y1) > abs(y0) else z1), (y0, z1)]
        if abs(y1) > 1.0:
            prof = [(y0, CABIN_FLOOR), (y1, CABIN_FLOOR), (y1, 1.0), (y1 - math.copysign(0.32, y1), z1), (y0, z1)]
        if y1 < y0:
            prof = prof[::-1]
        v, f = prism(prof, X_NOSE - s1, X_NOSE - s0, axis="X")
        add_part(name, (v, f), list(mats), cat, smooth=False, xpl=Vector(x_off) + UPX + long_spread(s0, 0.3))

    unit("Forward galley", 6.35, 7.75, -0.32, -1.38, 1.5, x_off=(0, -1.0, 0))
    unit("Forward lavatory", 6.35, 7.75, 0.32, 1.38, 1.5, x_off=(0, 1.0, 0))
    unit("Aft lavatory L", 26.25, 27.25, 0.32, 1.30, 1.5, x_off=(0, 1.0, 0))
    unit("Aft lavatory R", 26.25, 27.25, -0.32, -1.30, 1.5, x_off=(0, -1.0, 0))
    unit("Aft galley L", 28.5, 29.6, 0.12, 1.22, 1.45, x_off=(-1.0, 0.8, 0))
    unit("Aft galley R", 28.5, 29.6, -0.12, -1.22, 1.45, x_off=(-1.0, -0.8, 0))
    for nm, s0 in (("Cabin divider", 11.1),):
        v, f = box((X_NOSE - s0, 0.9, 0.55), (0.05, 0.8, 1.9))
        add_part(f"{nm} L", (v, f), ["plastic"], cat, smooth=False, xpl=UPX + long_spread(s0, 0.3))
        v, f = box((X_NOSE - s0, -0.9, 0.55), (0.05, 0.8, 1.9))
        add_part(f"{nm} R", (v, f), ["plastic"], cat, smooth=False, xpl=UPX + long_spread(s0, 0.3))


def build_flight_deck():
    cat = "Flight deck"
    UPX = Vector((3.5, 0, 7.6))
    s_b = 4.95
    zc, a, b = fus_section(s_b)
    pts = [(X_NOSE - s_b, (a - 0.08) * math.sin(t), zc + (b - 0.08) * math.cos(t))
           for t in lin(-theta_at_height(s_b, CABIN_FLOOR), theta_at_height(s_b, CABIN_FLOOR), RAD(6))]
    buf = Buf().add(ngon_cap(pts[::-1]), 0)
    buf.add(box((X_NOSE - s_b + 0.03, 0, CABIN_FLOOR + 0.98), (0.04, 0.82, 1.95)), 1)
    add_part("Flight deck bulkhead + door", buf, ["plastic", "plastic_dk"], cat, smooth=False,
             xpl=UPX + Vector((-1.8, 0, 0)))
    pm = seat_mesh("Pilot seat", 0.58, "leather", biz=True)
    for sd, y in (("Captain", 0.56), ("First officer", -0.56)):
        add_part(f"{sd} seat", None, None, cat, mesh=pm, loc=(X_NOSE - 3.95, y, CABIN_FLOOR), xpl=UPX)
    add_part("Observer seat", None, None, cat, mesh=pm, loc=(X_NOSE - 4.6, 0.0, CABIN_FLOOR),
             xpl=UPX + Vector((-1.0, 0, 0)))
    tilt = Matrix.Rotation(RAD(-14), 3, "Y")
    b = Buf().add(box((X_NOSE - 3.05, 0, 0.36), (0.12, 1.72, 0.62), tilt), 0)
    for y in (-0.62, -0.21, 0.21, 0.62):
        b.add(box((X_NOSE - 3.05 + 0.065 * math.cos(RAD(14)), y, 0.36 + 0.065 * math.sin(RAD(14))),
                  (0.01, 0.32, 0.30), tilt), 1)
    add_part("Main instrument panel", b, ["plastic_dk", "screen"], cat, smooth=False, xpl=UPX + Vector((1.2, 0, 0)))
    add_part("Glareshield", box((X_NOSE - 3.0, 0, 0.72), (0.38, 1.6, 0.1)), ["plastic_dk"], cat, smooth=False,
             xpl=UPX + Vector((1.2, 0, 0.7)))
    b = Buf().add(box((X_NOSE - 3.55, 0, -0.08), (0.9, 0.34, 0.62)), 0)
    b.add(box((X_NOSE - 3.25, 0, 0.235), (0.28, 0.26, 0.01)), 1)
    add_part("Centre pedestal", b, ["plastic_dk", "screen"], cat, smooth=False, xpl=UPX + Vector((0, 0, -0.8)))
    b = Buf().add(box((X_NOSE - 3.85, 0, 1.3), (0.95, 0.85, 0.08), Matrix.Rotation(RAD(10), 3, "Y")), 0)
    add_part("Overhead panel", b, ["plastic_dk"], cat, smooth=False, xpl=UPX + Vector((0, 0, 1.4)))
    for sd, y in (("L", 0.98), ("R", -0.98)):
        b = Buf().add(box((X_NOSE - 3.75, y, 0.12), (0.3, 0.2, 0.1)), 0)
        b.add(cyl((X_NOSE - 3.75, y, 0.17), (X_NOSE - 3.72, y, 0.33), 0.022, 10), 0)
        add_part(f"Sidestick {sd}", b, ["plastic_dk"], cat, smooth=False,
                 xpl=UPX + Vector((0, 0.6 if y > 0 else -0.6, 0)))


def build_cargo_systems():
    cat = "Cargo & systems"
    for nm, s0, s1 in (("Forward cargo floor", 6.6, 12.3), ("Aft cargo floor", 21.0, 26.3)):
        hw = min(interior_half_width(s, CARGO_FLOOR, 0.12) for s in (s0, s1))
        add_part(nm, box((X_NOSE - (s0 + s1) / 2, 0, CARGO_FLOOR - 0.02), (s1 - s0, 2 * hw, 0.04)),
                 ["alu"], cat, smooth=False, xpl=(0, 0, -3.0))
    prof = [(-0.78, CARGO_FLOOR), (0.78, CARGO_FLOOR), (1.18, -1.38), (1.18, -0.72), (-1.18, -0.72), (-1.18, -1.38)]
    k = 0
    for s in (6.75, 8.35, 9.95, 21.15, 22.75, 24.35):
        k += 1
        v, f = prism(prof, X_NOSE - s - 1.53, X_NOSE - s, axis="X")
        add_part(f"Cargo container {k}", (v, f), ["container"], cat, smooth=False,
                 xpl=Vector((0, 0, -4.2)) + long_spread(s, 0.3))
    zc = fus_section(35.8)[0]
    add_part("Auxiliary power unit", box((X_NOSE - 35.7, 0, zc - 0.05), (1.3, 0.75, 0.6)), ["metal"], cat,
             smooth=False, xpl=(-3.0, 0, 3.0))
    for sd, sg in (("L", 1), ("R", -1)):
        add_part(f"Air-conditioning pack {sd}", box((X_NOSE - 19.5, sg * 0.66, -2.11), (2.0, 0.68, 0.2)),
                 ["metal"], cat, smooth=False, xpl=(0, sg * 1.5, -6.5))


# =============================================================================
#  Studio, cameras, layout
# =============================================================================
def look_at(loc, target, lens=35.0):
    loc, target = Vector(loc), Vector(target)
    q = (target - loc).to_track_quat("-Z", "Y")
    return Matrix.Translation(loc) @ q.to_matrix().to_4x4()


def flat(M):
    return [float(M[i][j]) for i in range(4) for j in range(4)]


def build_studio(scene):
    world = bpy.data.worlds.new("AP World")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes.get("Background")
    bg.name = "ap_bg"
    bg.inputs["Color"].default_value = (0.62, 0.68, 0.76, 1)
    bg.inputs["Strength"].default_value = 0.9

    sun_d = bpy.data.lights.new("AP Sun", "SUN")
    sun_d.energy = 3.2
    sun_d.angle = RAD(4)
    sun_d.color = (1.0, 0.97, 0.92)
    sun = bpy.data.objects.new("AP Sun", sun_d)
    sun.rotation_euler = (RAD(42), RAD(-8), RAD(-128))
    COLL["Studio"].objects.link(sun)

    me = bpy.data.meshes.new("AP Ground")
    me.from_pydata([(-1500, -1500, 0), (1500, -1500, 0), (1500, 1500, 0), (-1500, 1500, 0)], [], [(0, 1, 2, 3)])
    me.materials.append(MAT["ground"])
    g = bpy.data.objects.new("AP Ground", me)
    g.location.z = GROUND
    g["ap_ground"] = 1
    COLL["Studio"].objects.link(g)

    cd = bpy.data.cameras.new("AP Camera")
    cd.lens = 35
    cd.clip_start = 0.1
    cd.clip_end = 2000
    cam = bpy.data.objects.new("AP Camera", cd)
    COLL["Studio"].objects.link(cam)
    scene.camera = cam

    def hud_text(name, body, size, y):
        cu = bpy.data.curves.new(name, "FONT")
        cu.body = body
        cu.size = size
        cu.materials.append(MAT["hud"])
        ob = bpy.data.objects.new(name, cu)
        ob.parent = cam
        ob.location = (-0.49, y, -1.0)
        ob["ap_hud"] = name.split()[-1]
        ob.visible_shadow = False
        COLL["Studio"].objects.link(ob)
        return ob

    hud_text("AP HUD title", "FINISHED", 0.034, 0.235)
    hud_text("AP HUD count", "0 parts", 0.020, 0.205)
    return cam


LAYOUT_ELEV = RAD(72)
HFOV = math.atan(18 / 35)          # half field of view, 35 mm lens, 16:9 frame
VFOV = math.atan(18 / 35 * 9 / 16)


def _pack_block(group, W, gap=0.3, head=2.8):
    """Shelf-pack one category into a block of width W. Returns [(item, x, y)] (y down) and height."""
    out = []
    y = -head
    x, row_d = 0.0, 0.0
    for it in group:
        dx, dy = it["dims"][0], it["dims"][1]
        if x > 0 and x + dx > W:
            y -= row_d + gap
            x, row_d = 0.0, 0.0
        out.append((it, x, y - dy))
        x += dx + gap
        row_d = max(row_d, dy)
    return out, -(y - row_d)


def _pack(items, Wc, ncol, col_gap=4.0, block_gap=2.2):
    """Pack every category into a block, then set the blocks in `ncol` balanced columns."""
    blocks = []
    for cat in CATEGORIES:
        group = [it for it in items if it["cat"] == cat]
        if group:
            pl, h = _pack_block(group, Wc)
            blocks.append((cat, len(group), pl, h))
    # split the ordered blocks into columns, minimising the tallest column
    import itertools
    best = None
    for cuts in itertools.combinations(range(1, len(blocks)), ncol - 1):
        idx = (0,) + cuts + (len(blocks),)
        cols = [blocks[idx[k]:idx[k + 1]] for k in range(ncol)]
        hmax = max(sum(b[3] for b in c) + block_gap * (len(c) - 1) for c in cols)
        if best is None or hmax < best[0]:
            best = (hmax, cols)
    D, cols = best
    Wt = ncol * Wc + (ncol - 1) * col_gap
    place, labels = [], []
    for k, col in enumerate(cols):
        x0 = -Wt / 2 + k * (Wc + col_gap)
        y0 = D / 2
        for cat, n, pl, h in col:
            labels.append((cat, n, x0, y0))
            place += [(it, x0 + x, y0 + y) for it, x, y in pl]
            y0 -= h + block_gap
    return place, labels, Wt, D


def _frame_dist(W, D):
    return max((W * 1.06 / 2) / math.tan(HFOV), (D * math.sin(LAYOUT_ELEV) * 1.16 / 2) / math.tan(VFOV))


def compute_layout(scene):
    """Lay every part flat on the floor, in labelled bands by system (stored per part)."""
    items = []
    for ob in PARTS:
        M = ob.matrix_basis.copy()
        me = ob.data
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        w = co @ np.array(M.to_3x3()).T + np.array(M.translation)
        c = (w.min(0) + w.max(0)) / 2
        ext = w.max(0) - w.min(0)
        R = np.eye(3)
        evals, evecs = np.linalg.eigh(np.cov((w - w.mean(0)).T))
        pext = np.ptp((w - c) @ evecs, axis=0)          # extents along the principal axes
        if pext[0] < 0.35 * pext[1]:        # thin part: lay it flat (smallest principal axis up)
            big, small = evecs[:, 2], evecs[:, 0]
            if np.dot(small, w.mean(0) - c) < 0:
                small = -small
            R = np.vstack([big, np.cross(small, big), small])
        elif ext[2] > 2.0 * max(ext[0], ext[1]):   # tall and slender (struts, rudder): lay it down
            R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)
        elif ext[1] > ext[0] * 1.6:          # long across: turn it to run along X
            R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], float)
        r = (w - c) @ R.T
        lo = r.min(0)
        items.append(dict(ob=ob, cat=ob["ap_cat"], M=M, c=c, R=R, lo=lo, dims=r.max(0) - lo))
    # biggest first; identical parts end up side by side
    items.sort(key=lambda it: (-round(it["dims"][1], 1), -round(it["dims"][0], 1), it["ob"].name))

    wmin = max(it["dims"][0] for it in items) + 0.5
    best = None
    for ncol in (1, 2, 3, 4):
        for k in range(40):
            Wc = wmin * (1 + k * 0.08)
            place, labels, W, D = _pack(items, Wc, ncol)
            d = _frame_dist(W, D)
            if best is None or d < best[0]:
                best = (d, place, labels, W, D)
    d, place, labels, W, D = best

    for it, px, py in place:
        t = Vector((px - it["lo"][0], py - it["lo"][1], GROUND + 0.001 - it["lo"][2]))
        R4 = Matrix([list(it["R"][0]) + [0], list(it["R"][1]) + [0], list(it["R"][2]) + [0], [0, 0, 0, 1]])
        it["ob"]["ap_lay"] = flat(Matrix.Translation(t) @ R4 @ Matrix.Translation(-Vector(it["c"])) @ it["M"])
    for cat, n, x, y in labels:
        cu = bpy.data.curves.new(f"Label {cat}", "FONT")
        cu.body = f"{cat.upper()}   {n}"
        cu.size = 1.5
        cu.materials.append(MAT["label"])
        ob = bpy.data.objects.new(f"AP Label {cat}", cu)
        ob.location = (x, y - 1.6, GROUND + 0.01)
        ob["ap_label"] = 1
        COLL["Layout labels"].objects.link(ob)
    tgt = Vector((0, D * 0.07, GROUND))     # leaves room for the caption in the top-left corner
    loc = tgt + Vector((0, -math.cos(LAYOUT_ELEV), math.sin(LAYOUT_ELEV))) * d * 1.12
    scene["ap_cam_LAYOUT"] = flat(look_at(loc, tgt))


def store_cameras(scene):
    scene["ap_cam_FINISHED"] = flat(look_at((31.0, 31.0, 7.5), (0.5, 0.0, -1.6)))
    scene["ap_cam_XRAY"] = flat(look_at((20.0, 38.0, 17.0), (0.0, 0.0, -0.6)))
    scene["ap_cam_CUTAWAY"] = flat(look_at((19.0, 24.0, 13.5), (-0.5, 1.0, -0.8)))
    scene["ap_cam_EXPLODED"] = flat(look_at((44.0, 50.0, 30.0), (-1.0, 1.0, 4.0)))
    scene["ap_xlift"] = 5.5


# =============================================================================
#  Mode system + UI (stored inside the .blend as a registered text block)
# =============================================================================
UI_SRC = r'''
# Airliner mode panel. Stored in the .blend by airliner.py and registered on load
# (needs "Auto Run Python Scripts", or open this text and press Run Script).
import bpy, blf, time
from mathutils import Matrix, Vector

MODES = [("FINISHED", "Finished", "The complete aircraft"),
         ("XRAY", "X-ray", "Outer skin turns to glass"),
         ("CUTAWAY", "Cutaway", "Remove a section of skin on the left side"),
         ("EXPLODED", "Exploded", "Pull every part away from its neighbours"),
         ("LAYOUT", "Laid out", "Lay every part out flat, grouped by system")]
LABEL = {m[0]: m[1] for m in MODES}
CATS = ["Fuselage", "Wings", "Tail", "Engines", "Landing gear", "Structure", "Cabin", "Flight deck",
        "Cargo & systems"]
_ANIM = {"items": [], "t0": 0.0, "dur": 0.9}


def _mat(v):
    return Matrix([v[0:4], v[4:8], v[8:12], v[12:16]])


def parts(scene):
    return [o for o in scene.objects if "ap_part" in o]


def counts(scene):
    tot, vis, per = 0, 0, {}
    for o in parts(scene):
        tot += 1
        if not o.hide_viewport:
            vis += 1
            c = o.get("ap_cat", "?")
            per[c] = per.get(c, 0) + 1
    return tot, vis, per


def _target(o, mode, f, lift):
    home = _mat(o["ap_home"])
    if mode == "EXPLODED":   # everything also rises so nothing sinks through the floor
        return Matrix.Translation((Vector(o["ap_xpl"]) + lift) * f) @ home
    if mode == "LAYOUT" and "ap_lay" in o:
        return _mat(o["ap_lay"])
    return home




def _tick():
    t = (time.time() - _ANIM["t0"]) / _ANIM["dur"]
    t = min(max(t, 0.0), 1.0)
    e = t * t * (3 - 2 * t)
    for o, l0, r0, s0, l1, r1, s1 in _ANIM["items"]:
        try:
            o.matrix_world = Matrix.LocRotScale(l0.lerp(l1, e), r0.slerp(r1, e), s0.lerp(s1, e))
        except ReferenceError:
            pass
    if t >= 1.0:
        _ANIM["items"] = []
        return None
    return 1 / 60


def apply_mode(scene, mode=None, factor=None, animate=None):
    mode = mode or scene.get("ap_mode_name", "FINISHED")
    if factor is None:
        factor = getattr(scene, "ap_explode", 1.0)
    f = factor
    if animate is None:
        animate = (not bpy.app.background) and getattr(scene, "ap_animate", True)
    ghost = bpy.data.materials.get("AP X-ray skin")
    items = []
    lift = Vector((0, 0, scene.get("ap_xlift", 5.5)))
    for o in parts(scene):
        xm = None
        if mode == "XRAY":
            xm = ghost if o.get("ap_shell", 0) else bpy.data.materials.get("AP X-ray " + o.get("ap_cat", ""))
        for s in o.material_slots:
            if xm:
                s.link = "OBJECT"
                s.material = xm
            else:
                s.link = "DATA"
        hide = bool(mode == "CUTAWAY" and o.get("ap_cut", 0))
        o.hide_viewport = hide
        o.hide_render = hide
        m1 = _target(o, mode, f, lift)
        if animate:
            items.append((o, *o.matrix_world.decompose(), *m1.decompose()))
        else:
            o.matrix_world = m1
    for o in scene.objects:
        if "ap_label" in o:
            o.hide_viewport = o.hide_render = (mode != "LAYOUT")
        if "ap_ground" in o:
            o.hide_viewport = o.hide_render = (mode == "XRAY")
    w = scene.world
    if w and w.node_tree and "ap_bg" in w.node_tree.nodes:
        bg = w.node_tree.nodes["ap_bg"]
        bg.inputs["Color"].default_value = (0.004, 0.012, 0.03, 1) if mode == "XRAY" else (0.62, 0.68, 0.76, 1)
    for o in scene.objects:
        hud = o.active_material if "ap_hud" in o else None
        if hud and hud.node_tree and "Emission" in hud.node_tree.nodes:
            c = (0.6, 0.9, 1.0, 1) if mode == "XRAY" else (0.02, 0.025, 0.04, 1)
            hud.node_tree.nodes["Emission"].inputs["Color"].default_value = c
    cam = scene.camera
    key = "ap_cam_" + mode
    if cam and key in scene and getattr(scene, "ap_move_camera", True):
        cam.matrix_world = _mat(scene[key])
    scene["ap_mode_name"] = mode
    update_hud(scene)
    if items:
        _ANIM["items"] = items
        _ANIM["t0"] = time.time()
        if not bpy.app.timers.is_registered(_tick):
            bpy.app.timers.register(_tick)


def update_hud(scene):
    tot, vis, per = counts(scene)
    mode = scene.get("ap_mode_name", "FINISHED")
    for o in scene.objects:
        if o.get("ap_hud") == "title":
            o.data.body = "AIRLINER  /  " + LABEL[mode].upper()
        elif o.get("ap_hud") == "count":
            o.data.body = (f"{vis} of {tot} parts shown" if vis != tot else f"{tot} parts")


def _on_mode(self, context):
    apply_mode(self, self.ap_mode)


def _on_explode(self, context):
    if self.get("ap_mode_name") == "EXPLODED":
        apply_mode(self, "EXPLODED", self.ap_explode, animate=False)


class AP_PT_panel(bpy.types.Panel):
    bl_label = "Airliner"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Airliner"

    def draw(self, context):
        sc = context.scene
        lay = self.layout
        col = lay.column(align=True)
        col.prop(sc, "ap_mode", expand=True)
        if sc.ap_mode == "EXPLODED":
            lay.prop(sc, "ap_explode", slider=True)
        row = lay.row()
        row.prop(sc, "ap_animate")
        row.prop(sc, "ap_move_camera")
        tot, vis, per = counts(sc)
        box = lay.box()
        box.label(text=f"Parts shown: {vis} / {tot}", icon="MESH_CUBE")
        g = box.grid_flow(columns=2, even_columns=True)
        for c in CATS:
            g.label(text=c)
            g.label(text=str(per.get(c, 0)))


def _draw_overlay():
    sc = bpy.context.scene
    if "ap_mode_name" not in sc:
        return
    tot, vis, per = counts(sc)
    region = bpy.context.region
    blf.size(0, 18)
    blf.color(0, 1, 1, 1, 0.95)
    blf.position(0, 24, region.height - 64, 0)
    blf.draw(0, "AIRLINER  /  " + LABEL[sc["ap_mode_name"]].upper())
    blf.size(0, 14)
    blf.color(0, 0.75, 0.9, 1, 0.95)
    blf.position(0, 24, region.height - 88, 0)
    blf.draw(0, f"{vis} of {tot} parts shown")


def register():
    unregister()
    bpy.utils.register_class(AP_PT_panel)
    bpy.types.Scene.ap_mode = bpy.props.EnumProperty(name="Mode", items=MODES, default="FINISHED",
                                                     update=_on_mode)
    bpy.types.Scene.ap_explode = bpy.props.FloatProperty(name="Explode", default=1.0, min=0.0, max=2.5,
                                                         update=_on_explode)
    bpy.types.Scene.ap_animate = bpy.props.BoolProperty(name="Animate", default=True)
    bpy.types.Scene.ap_move_camera = bpy.props.BoolProperty(name="Move camera", default=True)
    if not bpy.app.background:
        bpy.app.driver_namespace["ap_overlay"] = bpy.types.SpaceView3D.draw_handler_add(
            _draw_overlay, (), "WINDOW", "POST_PIXEL")


def unregister():
    h = bpy.app.driver_namespace.pop("ap_overlay", None)
    if h is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(h, "WINDOW")
        except Exception:
            pass
    try:
        bpy.utils.unregister_class(bpy.types.AP_PT_panel)
    except Exception:
        pass


register()
'''


def install_ui():
    txt = bpy.data.texts.get("airliner_ui.py") or bpy.data.texts.new("airliner_ui.py")
    txt.clear()
    txt.write(UI_SRC.lstrip())
    txt.use_module = True
    ns = {"__name__": "airliner_ui"}
    exec(compile(UI_SRC, "airliner_ui.py", "exec"), ns)
    return ns


# =============================================================================
#  Build
# =============================================================================
def build():
    t0 = time.time()
    if bpy.app.background:            # reuse the startup scene, minus its cube/light/camera
        scene = bpy.context.scene
        for o in list(scene.objects):
            bpy.data.objects.remove(o)
        scene.name = "Airliner"
    else:
        scene = bpy.data.scenes.new("Airliner")
        bpy.context.window.scene = scene
    MAT.clear()
    COLL.clear()
    PARTS.clear()
    build_materials()
    top = bpy.data.collections.new("Airliner")
    scene.collection.children.link(top)
    for c in CATEGORIES:
        COLL[c] = collection(c, top)
    COLL["Studio"] = collection("Studio", scene.collection)
    COLL["Layout labels"] = collection("Layout labels", scene.collection)

    build_fuselage()
    build_fuselage_structure()
    SL, SR = wing_surf(1), wing_surf(-1)
    build_wing(1, SL)
    build_wing(-1, SR)
    build_center_box(SL, SR)
    build_tail()
    blade = fan_blade_mesh()
    build_engine(1, SL, blade)
    build_engine(-1, SR, blade)
    build_gear(SL, SR)
    build_cabin()
    build_flight_deck()
    build_cargo_systems()

    for ob in PARTS:
        ob["ap_home"] = flat(ob.matrix_basis)
    build_studio(scene)
    store_cameras(scene)
    compute_layout(scene)

    r = scene.render
    r.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.cycles.transparent_max_bounces = 64
    scene.cycles.max_bounces = 6
    r.resolution_x, r.resolution_y = 1920, 1080
    r.film_transparent = False
    try:
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass

    print(f"[airliner] built {len(PARTS)} parts in {time.time() - t0:.1f}s")
    return scene


def parse_args():
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    elif sys.argv and sys.argv[0].endswith(".py") and bpy.app.background:
        argv = sys.argv[1:]
    else:
        argv = []
    p = argparse.ArgumentParser(description="Build a multi-part airliner in Blender.")
    p.add_argument("--mode", default="finished", choices=["finished", "xray", "cutaway", "exploded", "layout"])
    p.add_argument("--explode", type=float, default=1.0)
    p.add_argument("--render")
    p.add_argument("--render-all")
    p.add_argument("--res", default="1920x1080")
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--no-hud", action="store_true")
    p.add_argument("--save")
    return p.parse_args(argv)


def render_to(scene, path):
    scene.render.filepath = os.path.abspath(path)
    t = time.time()
    bpy.ops.render.render(write_still=True, scene=scene.name)
    print(f"[airliner] wrote {path} ({time.time() - t:.0f}s)")


def main():
    args = parse_args()
    scene = build()
    ui = install_ui()
    w, h = (int(x) for x in args.res.lower().split("x"))
    scene.render.resolution_x, scene.render.resolution_y = w, h
    scene.cycles.samples = args.samples
    scene.ap_explode = args.explode
    if args.no_hud:
        for o in bpy.data.objects:
            if "ap_hud" in o:
                o.hide_render = True
    if args.render_all:
        os.makedirs(args.render_all, exist_ok=True)
        for m in ("FINISHED", "XRAY", "CUTAWAY", "EXPLODED", "LAYOUT"):
            ui["apply_mode"](scene, m, args.explode, animate=False)
            render_to(scene, os.path.join(args.render_all, f"airliner_{m.lower()}.png"))
    mode = args.mode.upper()
    ui["apply_mode"](scene, mode, args.explode, animate=False)
    scene.ap_mode = mode            # keeps the panel in sync (re-applies the same mode)
    if args.render:
        render_to(scene, args.render)
    if args.save:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save), compress=True)
        print(f"[airliner] saved {args.save}")
    if not bpy.app.background:
        for area in bpy.context.screen.areas if bpy.context.screen else []:
            if area.type == "VIEW_3D":
                for sp in area.spaces:
                    if sp.type == "VIEW_3D":
                        sp.shading.type = "MATERIAL"
                        sp.region_3d.view_perspective = "CAMERA"
                        sp.overlay.show_relationship_lines = False


if __name__ == "__main__":
    main()

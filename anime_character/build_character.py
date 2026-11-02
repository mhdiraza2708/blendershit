#!/usr/bin/env python3
"""
build_character.py - a game-ready anime character for Blender, built entirely by code.

Pipeline
--------
1. Body: box-modelled the way a character artist would do it. A low-resolution quad cage with
   deliberate edge loops (three loops per joint, four-sided finger rings, a diamond-quad groin,
   concentric eye and mouth loops, ear sockets), then one Catmull-Clark subdivision gives the
   game-resolution mesh. The head cage is projected onto a sculpted signed-distance field (a
   lofted head measured off an anime reference) and relaxed around the facial features.
2. Face: socket walls, sphere-cap eyeballs with a generated iris texture, lashes, brows, a
   mouth bag.
3. Hair (scalp cap + draped ribbon locks), pleated skirt, sailor collar with bow, outfit colours.
4. UVs: seams on region borders plus one hidden cut per tube, angle-based unwrap, packed.
5. Rig: game skeleton (97 deform bones incl. spring-bone chains for hair and skirt), IK/FK
   controls, skin weights limited to 4 per vertex, face shape keys (blinks, visemes, smile).
6. Exports: .blend, FBX (deform bones only) and GLB.

Usage
-----
    blender -b -P build_character.py -- --save anime_character.blend
    blender -b -P build_character.py -- --save out.blend --fbx out.fbx --glb out.glb

Tested with Blender 4.5 LTS. Takes about a minute on a 4-core CPU.
"""

import argparse
import math
import os
import sys

import bpy
import bmesh
from mathutils import Matrix, Vector
from mathutils.kdtree import KDTree

V = Vector
X_AXIS, Y_AXIS, Z_AXIS = V((1, 0, 0)), V((0, 1, 0)), V((0, 0, 1))


# =============================================================================
#  Small math helpers
# =============================================================================

def clamp(x, a, b):
    return a if x < a else b if x > b else x


def lerp(a, b, t):
    return a + (b - a) * t


def smoothstep(e0, e1, x):
    t = clamp((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smin(a, b, k):
    """Polynomial smooth minimum (blend radius k)."""
    h = max(k - abs(a - b), 0.0) / k
    return min(a, b) - h * h * k * 0.25


def smax(a, b, k):
    return -smin(-a, -b, k)


def sd_ellipsoid(p, c, r):
    q = V(((p.x - c[0]) / r[0], (p.y - c[1]) / r[1], (p.z - c[2]) / r[2]))
    return (q.length - 1.0) * min(r)


def sd_capsule(p, a, b, r):
    a, b = V(a), V(b)
    pa, ba = p - a, b - a
    h = clamp(pa.dot(ba) / ba.dot(ba), 0.0, 1.0)
    return (pa - ba * h).length - r


def catmull_rom(pts, t, closed=False):
    """Evaluate a uniform Catmull-Rom spline through pts at t in [0, 1]."""
    n = len(pts)
    segs = n if closed else n - 1
    f = clamp(t, 0.0, 1.0) * segs
    i = min(int(f), segs - 1)
    u = f - i

    def P(k):
        if closed:
            return V(pts[k % n])
        return V(pts[clamp(k, 0, n - 1)])

    p0, p1, p2, p3 = P(i - 1), P(i), P(i + 1), P(i + 2)
    return 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * u * u * u)


def angle_wrap(a):
    while a <= -math.pi:
        a += 2 * math.pi
    while a > math.pi:
        a -= 2 * math.pi
    return a


def ortho(v, axis):
    """Component of v perpendicular to axis, normalized."""
    w = v - axis * v.dot(axis)
    return w.normalized()


# =============================================================================
#  Cage: a tiny quad-mesh builder
# =============================================================================

class Cage:
    def __init__(self):
        self.co = []
        self.faces = []
        self.groups = {}      # vertex group name -> set of vertex indices
        self.creases = {}     # (a, b) -> crease value
        self.seams = set()    # (a, b) edges marked as UV seams

    def v(self, p, *groups):
        i = len(self.co)
        self.co.append(V(p))
        for g in groups:
            self.groups.setdefault(g, set()).add(i)
        return i

    def ring(self, pts, *groups):
        return [self.v(p, *groups) for p in pts]

    def tag(self, idx, *groups):
        for g in groups:
            self.groups.setdefault(g, set()).update(idx)

    def quad(self, a, b, c, d):
        self.faces.append((a, b, c, d))

    def bridge(self, A, B, skip=(), closed=True):
        n = len(A)
        assert len(B) == n, (len(A), len(B))
        for k in range(n if closed else n - 1):
            if k in skip:
                continue
            k2 = (k + 1) % n
            self.quad(A[k], A[k2], B[k2], B[k])

    def best_alignment(self, A, B):
        """Return B rotated/reversed so that B[k] lies closest to A[k]."""
        n = len(A)
        best, best_d = None, 1e9
        for cand in (list(B), list(reversed(B))):
            for s in range(n):
                d = sum((self.co[A[k]] - self.co[cand[(k + s) % n]]).length_squared for k in range(n))
                if d < best_d:
                    best_d, best = d, [cand[(k + s) % n] for k in range(n)]
        return best

    def bridge_aligned(self, A, B):
        self.bridge(A, self.best_alignment(A, B))

    def edge_key(self, a, b):
        return (a, b) if a < b else (b, a)

    def crease_loop(self, ring, value, closed=True):
        n = len(ring)
        for k in range(n if closed else n - 1):
            self.creases[self.edge_key(ring[k], ring[(k + 1) % n])] = value

    def seam_loop(self, ring, closed=True):
        n = len(ring)
        for k in range(n if closed else n - 1):
            self.seams.add(self.edge_key(ring[k], ring[(k + 1) % n]))

    def seam_path(self, path):
        for a, b in zip(path, path[1:]):
            self.seams.add(self.edge_key(a, b))


def rect_boundary(W, D):
    """Grid coordinates around the boundary of a W x D rectangle, counter-clockwise
    seen from +Z (or from outside), starting at (0, 0)."""
    return ([(i, 0) for i in range(W)] + [(W, j) for j in range(D)]
            + [(i, D) for i in range(W, 0, -1)] + [(0, j) for j in range(D, 0, -1)])


# =============================================================================
#  Proportions (metres, Z up, the character faces -Y, her left is +X)
# =============================================================================

HEIGHT = 1.60

# torso rings: (z_centre, z_tilt, half_width, front_depth, back_depth, y_offset, squareness)
# z_tilt raises the sides of the ring (the hip ring is higher at the sides than at the crotch)
TORSO = [
    (0.770, 0.030, 0.136, 0.074, 0.090, 0.010, 2.2),   # T0 hip / thigh junction
    (0.855, 0.000, 0.141, 0.080, 0.098, 0.010, 2.2),   # T1 hips + glutes
    (0.928, 0.000, 0.126, 0.077, 0.082, 0.006, 2.2),   # T2 iliac crest
    (0.998, 0.000, 0.101, 0.071, 0.068, 0.004, 2.1),   # T3 waist
    (1.068, 0.000, 0.106, 0.076, 0.072, 0.004, 2.1),   # T4 lower ribs
    (1.137, 0.000, 0.114, 0.082, 0.076, 0.004, 2.1),   # T5 bust
    (1.196, 0.000, 0.118, 0.080, 0.075, 0.004, 2.2),   # T6 armpit (bottom of arm socket)
    (1.254, 0.000, 0.124, 0.071, 0.074, 0.004, 2.2),   # T7 mid socket
]
TW, TD = 4, 2                    # torso cage box: 4 faces across, 2 deep -> 12 around (24 final)
TN = 2 * TW + 2 * TD

SHOULDER = V((0.150, 0.004, 1.266))      # left shoulder joint
ARM_DROP = math.radians(40.0)            # A-pose: arms 40 degrees below horizontal
UPPER_ARM, FOREARM = 0.255, 0.225
HIP = V((0.084, 0.0, 0.835))
KNEE = V((0.078, -0.010, 0.455))
ANKLE = V((0.072, 0.016, 0.080))


# =============================================================================
#  Head shape (signed distance field)
# =============================================================================

HEAD_C = V((0.0, 0.012, 1.462))          # projection centre inside the head


def sd_round_cone(p, a, b, r1, r2):
    """Inigo Quilez's exact round cone (capsule with two radii)."""
    a, b = V(a), V(b)
    ba = b - a
    l2 = ba.dot(ba)
    rr = r1 - r2
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    pa = p - a
    y = pa.dot(ba)
    z = y - l2
    x2 = (pa * l2 - ba * y).length_squared
    y2 = y * y * l2
    z2 = z * z * l2
    k = math.copysign(1.0, rr) * rr * rr * x2
    if math.copysign(1.0, z) * a2 * z2 > k:
        return math.sqrt(x2 + z2) * il2 - r2
    if math.copysign(1.0, y) * a2 * y2 < k:
        return math.sqrt(x2 + y2) * il2 - r1
    return (math.sqrt(x2 * a2 * il2) + y * rr) * il2 - r1


class Profile:
    """A smooth function z -> value through key points (Catmull-Rom, tabulated)."""

    def __init__(self, keys, n=600):
        keys = sorted(keys)
        pts = [V((z, v, 0.0)) for z, v in keys]
        samp = [catmull_rom(pts, k / (n - 1)) for k in range(n)]
        samp.sort(key=lambda q: q.x)
        self.z0, self.z1 = keys[0][0], keys[-1][0]
        self.m = 2000
        self.tab = []
        j = 0
        for k in range(self.m + 1):
            z = self.z0 + (self.z1 - self.z0) * k / self.m
            while j < len(samp) - 2 and samp[j + 1].x < z:
                j += 1
            a, b = samp[j], samp[j + 1]
            u = 0.0 if b.x == a.x else clamp((z - a.x) / (b.x - a.x), 0.0, 1.0)
            self.tab.append(a.y + (b.y - a.y) * u)

    def __call__(self, z):
        f = clamp((z - self.z0) / (self.z1 - self.z0), 0.0, 1.0) * self.m
        i = min(int(f), self.m - 1)
        u = f - i
        return self.tab[i] + (self.tab[i + 1] - self.tab[i]) * u


# Head silhouettes measured off an anime head reference (front and side views).
# crown 1.585, chin 1.358; eye line 56 % down, nose 70 %, mouth 80 %.
# Above the brow the head is an ellipsoid; below it the face profiles take over.
HEAD_TOP = 1.585
CRANIUM_C, CRANIUM_R = V((0.0, 0.020, 1.479)), V((0.0850, 0.0985, 0.106))


def _cranium_keys():
    keys_f, keys_b, keys_w = [], [], []
    for k in range(14):
        z = HEAD_TOP - 0.0003 - (HEAD_TOP - 1.488) * (1 - math.cos(k / 13 * math.pi / 2))
        f = math.sqrt(max(1.0 - ((z - CRANIUM_C.z) / CRANIUM_R.z) ** 2, 0.0))
        keys_f.append((z, CRANIUM_C.y - CRANIUM_R.y * f))
        keys_b.append((z, CRANIUM_C.y + CRANIUM_R.y * f))
        keys_w.append((z, CRANIUM_R.x * f))
    return keys_f, keys_b, keys_w


_KF, _KB, _KW = _cranium_keys()
HEAD_FRONT = Profile(_KF + [(1.468, -0.0780), (1.448, -0.0766), (1.428, -0.0724),
                            (1.412, -0.0674), (1.400, -0.0644), (1.386, -0.0607), (1.373, -0.0566),
                            (1.364, -0.0500), (1.357, -0.036), (1.340, -0.020)])
HEAD_BACK = Profile(_KB + [(1.462, 0.1150), (1.440, 0.1020), (1.420, 0.0830), (1.405, 0.0600),
                           (1.395, 0.0380), (1.380, 0.0300), (1.340, 0.0300)])
HEAD_W = Profile(_KW + [(1.471, 0.0838), (1.451, 0.0805), (1.431, 0.0745), (1.415, 0.0680),
                        (1.400, 0.0605), (1.386, 0.0505), (1.373, 0.0380), (1.363, 0.0250),
                        (1.357, 0.0160), (1.340, 0.010)])
HEAD_TW = Profile([(1.585, 0.50), (1.50, 0.50), (1.45, 0.52), (1.42, 0.56), (1.39, 0.60),
                   (1.36, 0.62), (1.34, 0.62)])
HEAD_BF = Profile([(1.585, 0.50), (1.52, 0.48), (1.47, 0.36), (1.44, 0.38), (1.42, 0.48),
                   (1.40, 0.64), (1.38, 0.74), (1.36, 0.80), (1.34, 0.80)])
HEAD_BB = Profile([(1.585, 0.50), (1.40, 0.50), (1.34, 0.50)])


def sd_head_loft(p):
    # above the brow the loft is exactly the cranium ellipsoid, whose distance estimate is
    # much better (hair and the hair cap are offset from the scalp by distance)
    z = p.z
    if z >= 1.50:
        return sd_ellipsoid(p, CRANIUM_C, CRANIUM_R)
    d = _sd_head_loft(p)
    if z > 1.475:
        return lerp(d, sd_ellipsoid(p, CRANIUM_C, CRANIUM_R), smoothstep(1.475, 1.50, z))
    return d


def _sd_head_loft(p):
    z = p.z
    if z >= HEAD_TOP:
        return z - HEAD_TOP + abs(p.x) + 0.0
    yf, yb, W = HEAD_FRONT(z), HEAD_BACK(z), HEAD_W(z)
    tw, bf, bb = HEAD_TW(z), HEAD_BF(z), HEAD_BB(z)
    L = max(yb - yf, 1e-6)
    t = clamp((p.y - yf) / L, 0.0, 1.0)
    s = 0.5 * t / tw if t < tw else 0.5 + 0.5 * (t - tw) / (1.0 - tw)
    e = bf if s < 0.5 else bb
    hw = W * max(4.0 * s * (1.0 - s), 0.0) ** e
    return max(abs(p.x) - hw, yf - p.y, p.y - yb)


def jaw_underside_z(y):
    """Height of the underside of the jaw: rises from the chin back toward the neck."""
    return 1.3575 + clamp((y + 0.046) * 0.78, 0.0, 0.040)


def head_sdf_base(p):
    d = sd_head_loft(p)
    d = smax(d, jaw_underside_z(p.y) - p.z, 0.018)
    d = smin(d, sd_capsule(p, (0.0, 0.026, 1.29), (0.0, 0.016, 1.43), 0.0365), 0.022)  # neck
    return d


def _front_y(z):
    lo, hi = -0.2, 0.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if head_sdf_base(V((0.0, mid, z))) > 0:
            lo = mid
        else:
            hi = mid
    return hi


NOSE_Z = 1.4275
_NOSE_Y = _front_y(NOSE_Z)


def head_sdf(p):
    d = head_sdf_base(p)
    return smin(d, sd_ellipsoid(p, (0.0, _NOSE_Y + 0.0012, NOSE_Z), (0.0045, 0.0060, 0.0095)), 0.006)


def head_proj_sdf(p):
    """The head with the neck cut off flush under the jaw: what head vertices project onto."""
    return smax(head_sdf(p), jaw_underside_z(p.y) - 0.006 - p.z, 0.004)


def project_ray(origin, direction, sdf, tmax=0.35, step=0.002, fallback=None):
    """Outermost surface point of sdf along origin + t*direction, t in [0, tmax]."""
    direction = direction.normalized()
    t = tmax
    if sdf(origin + direction * t) < 0:
        return origin + direction * t if fallback is None else fallback
    while t > 0 and sdf(origin + direction * t) > 0:
        t -= step
    if t <= 0 and fallback is not None:
        return fallback
    lo, hi = max(t, 0.0), t + step
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if sdf(origin + direction * mid) > 0:
            hi = mid
        else:
            lo = mid
    return origin + direction * (0.5 * (lo + hi))


def project_front(x, z, sdf, y0=-0.3):
    """Project the point (x, ?, z) onto the surface along +Y (from the front)."""
    return project_ray(V((x, y0 + 0.35, z)), V((0, -1, 0)), sdf, tmax=0.35)


# ---- facial features (right side, x < 0; mirrored for the left) -----------------
EYE_C = V((-0.0370, 0.0, 1.4590))      # eye centre (x, -, z)
# eye rim in eye space (u = outward from the nose, v = up), metres
EYE_INNER = (-0.0175, -0.0020)
EYE_OUTER = (0.0195, 0.0015)
EYE_TOP = [(-0.0145, 0.0068), (-0.0085, 0.0129), (-0.0010, 0.0153), (0.0065, 0.0149),
           (0.0128, 0.0118), (0.0172, 0.0066)]
EYE_BOT = [(-0.0128, -0.0085), (-0.0062, -0.0128), (0.0020, -0.0140), (0.0098, -0.0118),
           (0.0160, -0.0064)]
MOUTH_C = V((0.0, 0.0, 1.4040))
MOUTH_HALF_W = 0.0100


def eye_curve(n_half):
    """Rim points: inner corner, n_half-1 top points, outer corner, n_half-1 bottom points."""
    top = [EYE_INNER] + EYE_TOP + [EYE_OUTER]
    bot = [EYE_OUTER] + list(reversed(EYE_BOT)) + [EYE_INNER]
    pts = []
    for k in range(n_half):
        pts.append(catmull_rom(top, k / n_half))
    for k in range(n_half):
        pts.append(catmull_rom(bot, k / n_half))
    return [(p.x, p.y) for p in pts]


def eye_to_world(u, v, side, sdf=head_sdf, inset=0.0):
    x = side * (abs(EYE_C.x) + u)
    p = project_front(x, EYE_C.z + v, sdf)
    return p + V((0, inset, 0))


# =============================================================================
#  Body cage
# =============================================================================

def torso_ring_points(spec):
    z0, ztilt, a, f, b, cy, sq = spec
    pts = []
    for m in range(TN):
        th = math.radians(-90.0 + (m - 2) * 360.0 / TN)
        c, s = math.cos(th), math.sin(th)
        cx = math.copysign(abs(c) ** (2.0 / sq), c)
        sy = math.copysign(abs(s) ** (2.0 / sq), s)
        x = a * cx
        y = cy + (f if s < 0 else b) * sy
        z = z0 + ztilt * abs(c) ** 1.5
        pts.append(V((x, y, z)))
    return pts


def build_torso(cage):
    idx_of = {ij: m for m, ij in enumerate(rect_boundary(TW, TD))}
    F_C, F_L, F_R = idx_of[(2, 0)], idx_of[(3, 0)], idx_of[(1, 0)]       # front centre / beside it
    B_C, B_L, B_R = idx_of[(2, TD)], idx_of[(3, TD)], idx_of[(1, TD)]    # back centre / beside it
    rings = []
    for k, spec in enumerate(TORSO):
        pts = torso_ring_points(spec)
        if k == 1:          # glutes
            for m in (B_L, B_R):
                pts[m].y += 0.020
                pts[m].z -= 0.010
            pts[B_C].y += 0.008
        if k == 0:
            for m in (B_L, B_R):
                pts[m].y += 0.010
        if k == 5:          # bust
            for m in (F_L, F_R):
                pts[m].y -= 0.032
                pts[m].z += 0.004
            pts[F_C].y -= 0.016
        if k == 6:          # upper chest and armpit hollow
            for m in (F_L, F_R):
                pts[m].y -= 0.012
            pts[F_C].y -= 0.006
        if k == 4:
            for m in (F_L, F_R):
                pts[m].y -= 0.006
        rings.append(pts)

    # T8: the shoulder line (explicit, left half; mirrored)
    sh = {
        (2, 0): (0.000, -0.046, 1.300),   # jugular notch
        (3, 0): (0.050, -0.042, 1.312),
        (4, 0): (0.124, -0.034, 1.302),   # front of shoulder
        (4, 1): (0.164, 0.004, 1.306),    # top of shoulder
        (4, 2): (0.124, 0.048, 1.302),    # back of shoulder
        (3, 2): (0.050, 0.060, 1.332),
        (2, 2): (0.000, 0.064, 1.336),    # C7
    }
    t8 = []
    for (i, j) in rect_boundary(TW, TD):
        p = V(sh[(i, j)] if i >= 2 else sh[(TW - i, j)])
        if i < 2:
            p.x = -p.x
        t8.append(p)
    rings.append(t8)

    # create vertices; T7 has no vertices on the sides (the arm sockets live there)
    side_mid = {idx_of[(TW, j)] for j in range(1, TD)} | {idx_of[(0, j)] for j in range(1, TD)}
    left_side = {idx_of[(TW, j)] for j in range(TD + 1)}
    right_side = {idx_of[(0, j)] for j in range(TD + 1)}
    T = []
    for k, pts in enumerate(rings):
        T.append([None if (k == 7 and m in side_mid) else cage.v(p, "torso") for m, p in enumerate(pts)])
    for k in range(len(T) - 1):
        A, B = T[k], T[k + 1]
        for m in range(TN):
            m2 = (m + 1) % TN
            if None in (A[m], A[m2], B[m], B[m2]):
                continue
            if k in (6, 7) and ({m, m2} <= left_side or {m, m2} <= right_side):
                continue                                            # the arm sockets
            cage.quad(A[m], A[m2], B[m2], B[m])

    # top cap: a strip over each shoulder + the neck opening (8 around)
    g = {ij: T[8][m] for ij, m in idx_of.items()}
    p = (0.048, 0.010, 1.348)
    g[(3, 1)] = cage.v(p, "torso")
    g[(1, 1)] = cage.v((-p[0], p[1], p[2]), "torso")
    for j in range(TD):
        cage.quad(g[(0, j + 1)], g[(1, j + 1)], g[(1, j)], g[(0, j)])
        cage.quad(g[(3, j)], g[(4, j)], g[(4, j + 1)], g[(3, j + 1)])
    neck_loop = [g[(1, 0)], g[(2, 0)], g[(3, 0)], g[(3, 1)], g[(3, 2)], g[(2, 2)], g[(1, 2)], g[(1, 1)]]

    def socket(i_side):                                             # 8-vertex arm socket
        ms = [idx_of[(i_side, j)] for j in range(TD + 1)]           # front -> back
        return [T[6][m] for m in ms] + [T[7][ms[-1]]] + [T[8][m] for m in reversed(ms)] + [T[7][ms[0]]]

    sockets = {1: socket(TW), -1: socket(0)}

    # leg openings: a crotch line with a diamond quad at its front and back end, so no vertex
    # in the groin has more than five edges
    gb = {ij: T[0][m] for ij, m in idx_of.items()}
    crotch = [cage.v(q, "torso") for q in ((0.0, -0.036, 0.749), (0.0, 0.002, 0.740), (0.0, 0.040, 0.751))]
    cage.quad(gb[(1, 0)], crotch[0], gb[(3, 0)], gb[(2, 0)])
    cage.quad(gb[(3, 2)], crotch[2], gb[(1, 2)], gb[(2, 2)])
    legs = {
        -1: [gb[(0, 0)], gb[(1, 0)]] + crotch + [gb[(1, 2)], gb[(0, 2)], gb[(0, 1)]],
        1: [gb[(3, 0)], gb[(4, 0)], gb[(4, 1)], gb[(4, 2)], gb[(3, 2)]] + crotch[::-1],
    }
    return {"rings": T, "neck": neck_loop, "sockets": sockets, "legs": legs, "idx_of": idx_of}


def ring_frame_points(center, u1, u2, pts2d, r1, r2):
    return [center + u1 * (a * r1) + u2 * (b * r2) for a, b in pts2d]


def phis_of(points, center, u1, u2):
    return [math.atan2((p - center).dot(u2), (p - center).dot(u1)) for p in points]


def orient_ccw(cage, loop, center, u1, u2):
    """Make the loop counter-clockwise in the (u1, u2) plane."""
    pts = [cage.co[i] for i in loop]
    area = 0.0
    for k in range(len(pts)):
        a = pts[k] - center
        b = pts[(k + 1) % len(pts)] - center
        area += a.dot(u1) * b.dot(u2) - a.dot(u2) * b.dot(u1)
    return loop if area > 0 else list(reversed(loop))


def blend_phis(phi_a, phi_b, t):
    return [pa + angle_wrap(pb - pa) * t for pa, pb in zip(phi_a, phi_b)]


def uniform_phis_like(phis):
    """Evenly spaced angles that best match the given (CCW) angles."""
    n = len(phis)
    step = 2 * math.pi / n
    sx = sum(math.cos(p - k * step) for k, p in enumerate(phis))
    sy = sum(math.sin(p - k * step) for k, p in enumerate(phis))
    p0 = math.atan2(sy, sx)
    return [p0 + k * step for k in range(n)]


def polyline_point(pts, t_len):
    """Point and tangent at arc length t_len along a polyline."""
    acc = 0.0
    for a, b in zip(pts, pts[1:]):
        L = (b - a).length
        if t_len <= acc + L or b is pts[-1]:
            u = clamp((t_len - acc) / L, 0.0, 1.0)
            return a + (b - a) * u, (b - a).normalized()
        acc += L
    return pts[-1], (pts[-1] - pts[-2]).normalized()


# ---------------------------------------------------------------------------- arms
ARM_RINGS = [
    # (arc length from shoulder, r_fwd, r_up, hand_blend)
    (0.030, 0.046, 0.048, 0.0),
    (0.080, 0.039, 0.041, 0.0),
    (0.150, 0.035, 0.034, 0.0),
    (0.212, 0.031, 0.030, 0.0),     # above the elbow
    (0.250, 0.030, 0.029, 0.0),     # elbow
    (0.290, 0.031, 0.030, 0.0),     # below the elbow
    (0.360, 0.029, 0.026, 0.2),
    (0.425, 0.025, 0.020, 0.6),
    (0.480, 0.025, 0.0155, 1.0),    # wrist
]

HAND_LAYOUT = [(1.0, 1.0), (0.5, 1.0), (0.0, 1.0), (-0.5, 1.0), (-1.0, 1.0),
               (-1.0, -1.0), (-0.5, -1.0), (0.0, -1.0), (0.5, -1.0), (1.0, -1.0)]


def arm_frame(side):
    d = V((side * math.cos(ARM_DROP), -0.03, -math.sin(ARM_DROP))).normalized()
    elbow = SHOULDER.copy()
    elbow.x *= side
    elbow = elbow + d * UPPER_ARM + V((0, 0.012, 0))
    wrist = SHOULDER.copy()
    wrist.x *= side
    wrist = wrist + d * (UPPER_ARM + FOREARM) + V((0, -0.004, 0))
    S = SHOULDER.copy()
    S.x *= side
    return S, elbow, wrist


def frame_at(d, side):
    u_fwd = ortho(V((0, -1, 0)), d)
    u_up = (u_fwd.cross(d) * side).normalized()
    return u_fwd, u_up


def build_arm(cage, side, socket):
    S, E, Wr = arm_frame(side)
    path = [S, E, Wr]
    c0, d0 = polyline_point(path, 0.0)
    u_fwd, u_up = frame_at(d0, side)
    sock_center = sum((cage.co[i] for i in socket), V()) / len(socket)
    socket = orient_ccw(cage, socket, sock_center, u_fwd, u_up)
    phi_sock = phis_of([cage.co[i] for i in socket], sock_center, u_fwd, u_up)
    phi_uni = uniform_phis_like(phi_sock)
    prev = socket
    rings = []
    for n, (t, r1, r2, hb) in enumerate(ARM_RINGS):
        c, d = polyline_point(path, t)
        uf, uu = frame_at(d, side)
        phis = blend_phis(phi_sock, phi_uni, smoothstep(0.0, 0.10, t - 0.01))
        pts2d = [(math.cos(p), math.sin(p)) for p in phis]
        pts = ring_frame_points(c, uf, uu, pts2d, r1, r2)
        if n == 4:                              # elbow point sticks out at the back
            for k in range(len(pts)):
                if pts2d[k][0] < -0.7:
                    pts[k] += (-uf) * 0.003
        ring = cage.ring(pts, "arm")
        cage.bridge(prev, ring)
        prev = ring
        rings.append(ring)
    hand = build_hand(cage, side, prev, Wr, (Wr - E).normalized())
    return {"rings": rings, "S": S, "E": E, "W": Wr, "hand": hand}


# --------------------------------------------------------------------------- hands
FINGERS = {
    # name: (phalanx lengths, base half-size (width, thickness), tip half-size, spread deg)
    "index": ((0.030, 0.020, 0.0175), (0.0074, 0.0080), (0.0056, 0.0052), 7.0),
    "middle": ((0.033, 0.022, 0.0185), (0.0076, 0.0082), (0.0058, 0.0054), 1.5),
    "ring": ((0.031, 0.021, 0.0175), (0.0072, 0.0078), (0.0055, 0.0051), -4.0),
    "pinky": ((0.024, 0.017, 0.0155), (0.0064, 0.0070), (0.0049, 0.0046), -10.0),
}
FINGER_ORDER = ["index", "middle", "ring", "pinky"]


def build_hand(cage, side, wrist_ring, W, d_arm):
    # hand frame: continue the forearm, tipped slightly toward the palm
    uf0, uu0 = frame_at(d_arm, side)
    d = (d_arm - uu0 * 0.12).normalized()
    uf, uu = frame_at(d, side)
    # the 8-vertex wrist widens into the 10-vertex palm with one 1->3 cell on the pinky side
    nw = len(wrist_ring)
    wc = sum((cage.co[i] for i in wrist_ring), V()) / nw
    mids = [(cage.co[wrist_ring[c]] + cage.co[wrist_ring[(c + 1) % nw]]) / 2 for c in range(nw)]
    cu = min(range(nw), key=lambda c: (mids[c] - wc).dot(uf))
    splits = [3 if c == cu else 1 for c in range(nw)]
    starts = [sum(splits[:c]) for c in range(nw)]
    m = starts[cu]
    hand_layout = [HAND_LAYOUT[(j - (m + 1) + 4) % 10] for j in range(10)]
    # palm rings: (distance from wrist, half width, half thickness, knuckle curve)
    palm = [(0.020, 0.0275, 0.0135), (0.050, 0.0315, 0.0125), (0.082, 0.0330, 0.0105)]
    prev = wrist_ring
    palm_rings = []
    for n, (t, hw, ht) in enumerate(palm):
        c = W + d * t
        pts = []
        for (a, b) in hand_layout:
            p = c + uf * (a * hw) + uu * (b * ht)
            if n == 2:   # knuckle arc: middle knuckle furthest out, pinky knuckle back
                p += d * (0.006 * (1 - abs(a - 0.1)) - 0.006 * max(0.0, -a))
            if n == 0 and a > 0.9:                       # thenar bulge near the thumb
                p += uf * 0.003 - uu * 0.003
            pts.append(p)
        ring = cage.ring(pts, "hand")
        if n == 0:
            transition_strip(cage, wrist_ring, ring, lambda c: splits[c], lambda c: starts[c], True, "hand")
        else:
            cage.bridge(prev, ring)
        prev = ring
        palm_rings.append(ring)

    # thumb: from the side face between palm ring 0 and 1 (thumb side, a = +1)
    k_top = next(k for k, (a, b) in enumerate(hand_layout) if a == 1.0 and b == 1.0)
    k_bot = next(k for k, (a, b) in enumerate(hand_layout) if a == 1.0 and b == -1.0)
    # remove that face from the palm bridge: rebuild the faces between ring0 and ring1
    r0, r1 = palm_rings[0], palm_rings[1]
    face = None
    for fi, fv in enumerate(cage.faces):
        if set(fv) == {r0[k_top], r0[k_bot], r1[k_top], r1[k_bot]}:
            face = fi
            break
    del cage.faces[face]
    thumb_base = [r0[k_bot], r0[k_top], r1[k_top], r1[k_bot]]
    t_dir = (d * 0.78 + uf * 0.45 - uu * 0.42).normalized()
    t_center = sum((cage.co[i] for i in thumb_base), V()) / 4
    digits = {}
    digits["thumb"] = build_digit(cage, thumb_base, t_center, t_dir, uu, (0.016, 0.026, 0.021, 0.018),
                (0.0092, 0.0090), (0.0066, 0.0058), side, "thumb", curl=(6, 10, 10))

    # four fingers from the knuckle cap (a 4 x 1 grid)
    knuckle = palm_rings[2]
    tops = {hand_layout[k][0]: knuckle[k] for k in range(10) if hand_layout[k][1] > 0}
    bots = {hand_layout[k][0]: knuckle[k] for k in range(10) if hand_layout[k][1] < 0}
    cols = [1.0, 0.5, 0.0, -0.5, -1.0]
    for fi, name in enumerate(FINGER_ORDER):
        a0, a1 = cols[fi], cols[fi + 1]
        base = [bots[a0], tops[a0], tops[a1], bots[a1]]
        lens, b_hs, t_hs, spread = FINGERS[name]
        center = sum((cage.co[i] for i in base), V()) / 4
        rot = Matrix.Rotation(math.radians(spread) * side, 3, uu)
        f_dir = (rot @ d).normalized()
        digits[name] = build_digit(cage, base, center, f_dir, uu, (0.0,) + lens, b_hs, t_hs, side, name,
                    curl=(6, 9, 9))
    knuckle_c = sum((cage.co[i] for i in palm_rings[2]), V()) / 10
    return {"d": d, "uf": uf, "uu": uu, "digits": digits, "knuckles": knuckle_c}


def build_digit(cage, base, center, direction, up_hint, lens, base_hs, tip_hs, side, name, curl):
    """A 4-sided finger tube. lens: (metacarpal-ish offset, phalanx 1, 2, 3).
    Rings: knuckle base, mid p1, joint, mid p2, joint, near tip, tip cap."""
    # orientation of base corners in (width, thickness) space
    w_axis = ortho(up_hint.cross(direction), direction)
    t_axis = direction.cross(w_axis).normalized()
    if t_axis.dot(up_hint) < 0:
        t_axis = -t_axis
    signs = []
    for i in base:
        r = cage.co[i] - center
        signs.append((1 if r.dot(w_axis) > 0 else -1, 1 if r.dot(t_axis) > 0 else -1))

    # joint positions along a gently curling chain
    pts = [center]
    dirs = []
    dcur = direction.copy()
    total = list(lens)
    stations = []        # (point, dir, frac along digit 0..1)
    # chain segments: offset, p1, p2, p3 ; curl applied at the start of p2 and p3 (and p1)
    curl_axes = w_axis
    seg_curl = [0.0, curl[0], curl[1], curl[2]]
    p = center.copy()
    joints = [p.copy()]
    joint_dirs = []
    for L, c_deg in zip(total, seg_curl):
        rot = Matrix.Rotation(-math.radians(c_deg) * 1.0, 3, w_axis if side > 0 else w_axis)
        dcur = (rot @ dcur).normalized()
        joint_dirs.append(dcur.copy())
        p = p + dcur * L
        joints.append(p.copy())
    # ring stations along the chain
    L_total = sum(total)
    def station(s_len):
        acc = 0.0
        for k, L in enumerate(total):
            if L <= 0:
                continue
            if s_len <= acc + L + 1e-9:
                u = (s_len - acc) / L
                return joints[k] + joint_dirs[k] * (u * L), joint_dirs[k]
            acc += L
        return joints[-1], joint_dirs[-1]
    o, l1, l2, l3 = total
    marks = [o + 0.5 * l1, o + l1, o + l1 + 0.5 * l2, o + l1 + l2, o + l1 + l2 + 0.55 * l3,
             o + l1 + l2 + l3]
    prev = base
    rings = []
    for n, s_len in enumerate(marks):
        c, dd = station(s_len)
        frac = s_len / L_total
        hw = lerp(base_hs[0], tip_hs[0], frac)
        ht = lerp(base_hs[1], tip_hs[1], frac)
        wa = ortho(w_axis, dd)
        ta = dd.cross(wa).normalized()
        if ta.dot(t_axis) < 0:
            ta = -ta
        if n == len(marks) - 1:
            hw *= 0.55
            ht *= 0.55
        ring_pts = [c + wa * (sw * hw) + ta * (st * ht) for sw, st in signs]
        if n in (1, 3):  # knuckle bumps on the back of the joints
            ring_pts = [q + ta * (0.0012 if st > 0 else 0.0) for q, (sw, st) in zip(ring_pts, signs)]
        ring = cage.ring(ring_pts, "hand", "finger_" + name)
        cage.bridge(prev, ring)
        prev = ring
        rings.append(ring)
    cage.quad(*prev)          # fingertip cap
    return joints


# ---------------------------------------------------------------------------- legs
LEG_RINGS = [
    # (z, r_side, r_front, extras)
    (0.722, 0.071, 0.077),
    (0.640, 0.066, 0.070),
    (0.560, 0.057, 0.061),
    (0.498, 0.047, 0.050),      # above the knee
    (0.455, 0.045, 0.049),      # knee
    (0.414, 0.043, 0.046),      # below the knee
    (0.338, 0.045, 0.051),      # calf
    (0.250, 0.037, 0.041),
    (0.168, 0.030, 0.033),
    (0.112, 0.029, 0.031),      # above the ankle
]


def leg_center(z, side):
    path = [HIP, KNEE, ANKLE]
    if z >= KNEE.z:
        u = (HIP.z - z) / (HIP.z - KNEE.z)
        p = HIP.lerp(KNEE, u)
    else:
        u = (KNEE.z - z) / (KNEE.z - ANKLE.z)
        p = KNEE.lerp(ANKLE, u)
    return V((side * p.x, p.y, z))


def build_leg(cage, side, top_loop):
    c_top = sum((cage.co[i] for i in top_loop), V()) / len(top_loop)
    u1 = V((0, -1, 0))                 # forward
    u2 = V((side, 0, 0))               # outward
    top_loop = orient_ccw(cage, top_loop, c_top, u1, u2)
    phi_top = phis_of([cage.co[i] for i in top_loop], c_top, u1, u2)
    phi_uni = uniform_phis_like(phi_top)
    prev = top_loop
    rings = []
    for n, (z, rs, rf) in enumerate(LEG_RINGS):
        c = leg_center(z, side)
        if n == 0:
            c.x -= side * 0.006
        blend = 0.6 if n == 0 else 1.0
        phis = blend_phis(phi_top, phi_uni, blend)
        pts = []
        for k, ph in enumerate(phis):
            ca, sa = math.cos(ph), math.sin(ph)      # ca: forward, sa: outward
            r_f = rf
            p = c + u1 * (ca * r_f) + u2 * (sa * rs)
            if n == 4 and ca > 0.8:                   # kneecap
                p += u1 * 0.004
            if n == 6 and ca < -0.3:                  # calf muscle at the back
                p += u1 * (-0.006 * (-ca))
            if n == 6 and sa < -0.5:                  # inner calf
                p += u2 * (-0.003)
            pts.append(p)
        ring = cage.ring(pts, "leg")
        cage.bridge(prev, ring)
        prev = ring
        rings.append(ring)
    foot = build_foot(cage, side, prev)
    return {"rings": rings, "foot": foot}


FOOT_RINGS = [
    # (y, x_offset, z_top, half_width)   sole at z = 0
    (0.058, 0.000, 0.048, 0.023),     # F0 heel back (capped)
    (0.040, 0.000, 0.078, 0.030),     # F1 heel / ankle back
    (0.008, 0.000, 0.084, 0.032),     # F2
    (-0.024, 0.002, 0.074, 0.034),    # F3
    (-0.056, 0.004, 0.058, 0.036),    # F4 instep
    (-0.096, 0.008, 0.044, 0.039),    # F5 ball of the foot
    (-0.134, 0.011, 0.034, 0.035),    # F6 toe box
    (-0.162, 0.012, 0.024, 0.022),    # F7 toe tip (capped)
]
FOOT_RING = rect_boundary(2, 2)      # 8 around: (ix, iz)


def build_foot(cage, side, ankle_ring):
    fr = []
    cx0 = side * ANKLE.x
    for n, (y, xo, ztop, hw) in enumerate(FOOT_RINGS):
        cx = cx0 + side * xo
        ring = []
        for (ix, iz) in FOOT_RING:
            if n == 2 and (ix, iz) == (1, 2):
                ring.append(None)      # inside the ankle opening
                continue
            sx = (ix - 1)
            if iz == 0:
                z = 0.0
                x = cx + sx * hw * 0.92
            elif iz == 1:
                z = ztop * 0.42
                x = cx + sx * hw
            else:
                z = ztop * (1.0 if ix == 1 else 0.86)
                x = cx + sx * hw * 0.80
            if n == 0 and iz == 1:
                z = ztop * 0.5
            ring.append(cage.v((x, y, z), "foot"))
        fr.append(ring)
    nring = len(FOOT_RING)
    for n in range(len(fr) - 1):
        A, B = fr[n], fr[n + 1]
        for k in range(nring):
            k2 = (k + 1) % nring
            if None in (A[k], A[k2], B[k], B[k2]):
                continue
            # skip the top faces that form the ankle opening (rings 1..4, top row)
            if 1 <= n <= 2 and FOOT_RING[k][1] == 2 and FOOT_RING[k2][1] == 2:
                continue
            cage.quad(A[k], A[k2], B[k2], B[k])
    # caps (2x2 grids with a centre vertex)
    for n, sgn in ((0, 1), (len(fr) - 1, -1)):
        ring = fr[n]
        y = FOOT_RINGS[n][0] + sgn * 0.004
        cpt = sum((cage.co[i] for i in ring), V()) / nring
        cpt.y = y
        c = cage.v(cpt, "foot")
        pos = {ij: ring[k] for k, ij in enumerate(FOOT_RING)}
        cage.quad(pos[(0, 0)], pos[(1, 0)], c, pos[(0, 1)])
        cage.quad(pos[(1, 0)], pos[(2, 0)], pos[(2, 1)], c)
        cage.quad(c, pos[(2, 1)], pos[(2, 2)], pos[(1, 2)])
        cage.quad(pos[(0, 1)], c, pos[(1, 2)], pos[(0, 2)])
    # sole edge crease
    for n in range(len(fr) - 1):
        for ij in ((0, 0), (2, 0)):
            k = FOOT_RING.index(ij)
            cage.creases[cage.edge_key(fr[n][k], fr[n + 1][k])] = 0.85
    # ankle opening loop
    pos = [{ij: fr[n][k] for k, ij in enumerate(FOOT_RING)} for n in range(len(fr))]
    opening = [pos[1][(0, 2)], pos[1][(1, 2)], pos[1][(2, 2)], pos[2][(2, 2)],
               pos[3][(2, 2)], pos[3][(1, 2)], pos[3][(0, 2)], pos[2][(0, 2)]]
    cage.bridge_aligned(ankle_ring, opening)
    return {"rings": fr}


# ---------------------------------------------------------------------------- neck + head
# Head topology, laid out the way anime heads are retopologised:
#  * the face is a POLAR grid around the mouth: the lip loops grow outward into the muzzle,
#    pass under the nose, around the nose bridge and cheeks, over the brow and out to the
#    hairline / jaw. Rings 0-3 have 16 spokes (lips), rings 4-9 have 24: four 1->3 transition
#    cells between ring 3 and 4 put the poles in the nasolabial / jaw area, like a real face;
#  * each eye replaces a 3x2-cell block of the outer grid and gets two inset rings, so the lid
#    rim has 20 vertices and its loops flow into the face loops around it;
#  * a HOOD covers the cranium: its rows arc from temple to temple over the head, its columns
#    run from the hairline to the base of the skull; it widens from 6 to 10 columns just behind
#    the hairline and narrows back at the nape (poles under the hair). The ears grow from 1x2
#    sockets in it;
#  * the neck plugs into an 8-vertex ring: the throat (bottom of the face outline) plus the
#    bottom edge of the hood.


def _full(half):
    return half + [-a for a in reversed(half[1:-1])]


IN_SPOKES = [0, 20, 42, 66, 90, 114, 138, 160, 180]                   # left half, degrees from "up"
OUT_SPOKES = [0, 14, 26, 38, 50, 68, 90, 110, 126, 142, 157, 169, 180]
SP_IN, SP_OUT = _full(IN_SPOKES), _full(OUT_SPOKES)                    # 16 and 24 spokes
N_IN, N_OUT = len(SP_IN), len(SP_OUT)
CELL_SPLIT = [1, 3, 1, 1, 1, 3, 1, 1]                                 # inner half-cells -> outer cells
_HALF_MAP = [0]
for _n in CELL_SPLIT:
    _HALF_MAP.append(_HALF_MAP[-1] + _n)                               # inner half index -> outer half index


def in_to_out(k):
    return _HALF_MAP[k] if k <= N_IN // 2 else N_OUT - _HALF_MAP[N_IN - k]


NRING = 10
RING_FRAC = [0.0, 0.05, 0.11, 0.19, 0.29, 0.40, 0.52, 0.64, 0.80, 1.0]
EYE_BLOCK = {1: (1, 4), -1: (20, 23)}       # outer spoke range of each eye block (rings 5..7)
EYE_RINGS = (5, 7)
HOOD_TOP = [3, 2, 1, 0, 23, 22, 21]         # ring-9 spokes along the hairline (P_L ... P_R)
HOOD_ROWS = 8                               # rows from the hairline to the nape
HOOD_COLS = 10                              # columns in the middle rows
HOOD_SPLIT = [1, 3, 1, 1, 3, 1]             # 6 hairline cells -> 10 columns
EAR_C = V((0.0, 0.011, 1.446))
# ring 9 on the left half, spokes 0..9: hairline -> temple -> in front of the ear -> jaw angle
# (projected onto the head). Spokes 10..12 are the throat, on the neck.
R9_GUIDE = [(0.000, -0.070, 1.530), (0.022, -0.068, 1.528), (0.040, -0.062, 1.522),
            (0.055, -0.052, 1.512), (0.067, -0.040, 1.498), (0.076, -0.024, 1.475),
            (0.078, -0.008, 1.440), (0.066, 0.003, 1.408), (0.056, 0.006, 1.396),
            (0.046, 0.007, 1.388), (0.038, 0.000, 1.383)]
R9_NECK_ALPHA = {11: 45, 12: 0}


def neck_ring_point(alpha_deg):
    a = math.radians(alpha_deg)
    dv = V((math.sin(a), -math.cos(a), 0.0))
    z = lerp(1.380, 1.405, smoothstep(-0.9, 0.9, dv.y))
    ay = lerp(0.026, 0.016, (z - 1.29) / 0.14)
    return V((dv.x * 0.0362, ay + dv.y * 0.0362, z))


def head_point(direction):
    return project_ray(HEAD_C, V(direction), head_proj_sdf)


def slerp_dir(a, b, t):
    a, b = a.normalized(), b.normalized()
    om = math.acos(clamp(a.dot(b), -1.0, 1.0))
    if om < 1e-6:
        return a
    return (a * math.sin((1 - t) * om) + b * math.sin(t * om)) / math.sin(om)


def mouth_ring_point(theta_deg, sx, hz_top, hz_bot, inset=0.0):
    th = math.radians(theta_deg)
    x = MOUTH_HALF_W * sx * math.sin(th)
    c = math.cos(th)
    z = MOUTH_C.z + (hz_top * c if c > 0 else hz_bot * c)
    return project_front(x, z, head_sdf) + V((0, inset, 0))


def transition_strip(cage, inner, outer, split, mapping, closed, group="head"):
    """Quads between a row `inner` and a longer row `outer`. Each inner cell becomes 1 outer
    cell, or 3 (the 1->3 pattern: two interior vertices, four quads)."""
    n = len(inner) if closed else len(inner) - 1
    for c in range(n):
        a, b = inner[c], inner[(c + 1) % len(inner)]
        A = mapping(c)
        if split(c) == 1:
            cage.quad(a, b, outer[(A + 1) % len(outer)], outer[A])
            continue
        o = [outer[(A + j) % len(outer)] for j in range(4)]
        pa, pb = cage.co[a], cage.co[b]
        xa = cage.v(pa.lerp(cage.co[o[1]], 0.5), group)
        xb = cage.v(pb.lerp(cage.co[o[2]], 0.5), group)
        cage.quad(o[0], o[1], xa, a)
        cage.quad(a, xa, xb, b)
        cage.quad(b, xb, o[2], o[3])
        cage.quad(o[1], o[2], xb, xa)


def build_head(cage):
    pos_in, pos_out = {}, {}
    for k, th in enumerate(SP_IN):                    # lips and the loops around them (pinned)
        pos_in[0, k] = mouth_ring_point(th, 1.0, 0.0004, 0.0004, inset=0.0015)
        pos_in[1, k] = mouth_ring_point(th, 1.45, 0.0065, 0.0065)
        pos_in[2, k] = mouth_ring_point(th, 2.05, 0.0115, 0.0125)
    for K in range(N_OUT):                            # face outline / throat
        side = 1 if 0 < K < N_OUT // 2 else -1 if K > N_OUT // 2 else 0
        kk = K if K <= N_OUT // 2 else N_OUT - K
        if kk in R9_NECK_ALPHA:
            pos_out[9, K] = neck_ring_point(R9_NECK_ALPHA[kk] * (side or 1))
        else:
            g = V(R9_GUIDE[kk])
            g.x *= side
            pos_out[9, K] = head_point(g - HEAD_C)
    # ring 3 (inner) and rings 4-8 (outer): interpolate along the spokes, ring 2 -> ring 9
    t_of = lambda r: (RING_FRAC[r] - RING_FRAC[2]) / (1.0 - RING_FRAC[2])
    for k in range(N_IN):
        pos_in[3, k] = head_point(slerp_dir(pos_in[2, k] - HEAD_C, pos_out[9, in_to_out(k)] - HEAD_C, t_of(3)))
    for K, th in enumerate(SP_OUT):
        d2 = mouth_ring_point(th, 2.05, 0.0115, 0.0125) - HEAD_C
        for r in range(4, 9):
            pos_out[r, K] = head_point(slerp_dir(d2, pos_out[9, K] - HEAD_C, t_of(r)))
    pos_in[3, 0] = project_front(0.0, NOSE_Z, head_sdf)                # nose tip
    skip = set()
    for (K0, K1) in EYE_BLOCK.values():
        skip.update((6, K) for K in range(K0 + 1, K1))
    Fi = [[cage.v(pos_in[r, k], "head") for k in range(N_IN)] for r in range(4)]
    Fo = {}
    for r in range(4, NRING):
        for K in range(N_OUT):
            if (r, K) in skip:
                continue
            grp = "neckring" if (r == 9 and abs(K - N_OUT // 2) <= 1) else "head"
            Fo[r, K] = cage.v(pos_out[r, K], grp)
    F9 = [Fo[9, K] for K in range(N_OUT)]
    for r in range(3):
        cage.bridge(Fi[r], Fi[r + 1])
    split_in = lambda c: CELL_SPLIT[c] if c < N_IN // 2 else CELL_SPLIT[N_IN - 1 - c]
    transition_strip(cage, Fi[3], [Fo[4, K] for K in range(N_OUT)], split_in, in_to_out, True)
    for r in range(4, NRING - 1):
        for K in range(N_OUT):
            K2 = (K + 1) % N_OUT
            if EYE_RINGS[0] <= r < EYE_RINGS[1] and any(K0 <= K < K1 for (K0, K1) in EYE_BLOCK.values()):
                continue
            cage.quad(Fo[r, K], Fo[r, K2], Fo[r + 1, K2], Fo[r + 1, K])

    # the hood over the cranium
    nside = HOOD_ROWS
    left = [F9[3 + w] for w in range(nside + 1)]                      # P_L ... E_L
    right = [F9[21 - w] for w in range(nside + 1)]                    # P_R ... E_R
    rows = [[F9[K] for K in HOOD_TOP]]
    for w in range(1, nside):
        beta = math.radians(lerp(36.0, 240.0, w / nside))             # the centre column
        dc = V((0.0, -math.cos(beta), math.sin(beta)))
        dl, dr = cage.co[left[w]] - HEAD_C, cage.co[right[w]] - HEAD_C
        half = HOOD_COLS // 2
        row = [left[w]]
        for u in range(1, HOOD_COLS):
            d = slerp_dir(dl, dc, u / half) if u <= half else slerp_dir(dc, dr, (u - half) / half)
            row.append(cage.v(head_point(d), "head"))
        row.append(right[w])
        rows.append(row)
    rows.append([left[nside]] + [cage.v(neck_ring_point(a), "neckring") for a in (90, 135, 180, -135, -90)]
                + [right[nside]])
    hmap = [0]
    for n_ in HOOD_SPLIT:
        hmap.append(hmap[-1] + n_)
    transition_strip(cage, rows[0], rows[1], lambda c: HOOD_SPLIT[c], lambda c: hmap[c], False)
    ear_cells = {(2, 1), (3, 1), (2, HOOD_COLS - 2), (3, HOOD_COLS - 2)}
    for w in range(1, nside - 1):
        for u in range(HOOD_COLS):
            if (w, u) in ear_cells:
                continue
            cage.quad(rows[w][u], rows[w][u + 1], rows[w + 1][u + 1], rows[w + 1][u])
    transition_strip(cage, rows[nside], rows[nside - 1], lambda c: HOOD_SPLIT[c], lambda c: hmap[c], False)

    nape = rows[nside]
    feats = {"nose": Fi[3][0],
             "hole": [F9[11], F9[12], F9[13]] + list(reversed(nape[1:-1])),
             "mouth_rings": Fi[:3],
             "outline": [F9[K] for K in range(N_OUT) if abs(K - N_OUT // 2) > 1]}
    for side, (K0, K1) in EYE_BLOCK.items():
        ks = list(range(K0, K1 + 1)) if side > 0 else list(range(K1, K0 - 1, -1))   # inner -> outer
        r0, r1 = EYE_RINGS
        rm = (r0 + r1) // 2
        feats["eye", side] = ([Fo[rm, ks[0]]] + [Fo[r1, K] for K in ks] + [Fo[rm, ks[-1]]]
                              + [Fo[r0, K] for K in reversed(ks)])
    for side in (1, -1):
        u0 = 1 if side > 0 else HOOD_COLS - 1
        u1 = u0 + 1 if side > 0 else u0 - 1
        # front-top, back-top, back-middle, back-bottom, front-bottom, front-middle
        loop = [rows[2][u0], rows[2][u1], rows[3][u1], rows[4][u1], rows[4][u0], rows[3][u0]]
        feats["ear", side] = (loop, [(-1, 1), (1, 1), (1, 0), (1, -1), (-1, -1), (-1, 0)])
    return feats


EYE_RING_SCALE = [(1.72, 1.95, 0.0015, 0.0008), (1.32, 1.40, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0)]


def place_head_features(cage, feats):
    rim = eye_curve(5)                       # inner, top x4, outer, bottom x4 (outer -> inner)
    for side in (-1, 1):
        rings = []
        for (su, sv, du, dv) in EYE_RING_SCALE:
            rings.append([eye_to_world(u * su + du, v * sv + dv, side) for (u, v) in rim])
        for i, p in zip(feats["eye", side], rings[0]):
            cage.co[i] = p
        feats["eye_rings", side] = rings[1:]
    for side in (-1, 1):
        loop, slots = feats["ear", side]
        for i_v, (a, b) in zip(loop, slots):
            if a and b:
                a, b = a * 0.80, b * 0.80
            y, z = EAR_C.y + a * 0.0090, EAR_C.z + b * 0.0135
            cage.co[i_v] = project_ray(V((0.0, y, z)), V((side, 0, 0)), head_proj_sdf, tmax=0.25)


def cage_neighbors(cage):
    nbrs = {}
    for f in cage.faces:
        n = len(f)
        for k in range(n):
            a, b = f[k], f[(k + 1) % n]
            nbrs.setdefault(a, set()).add(b)
            nbrs.setdefault(b, set()).add(a)
    return nbrs


def relax_head(cage, feats, iters=40):
    """Feature-pinned Laplacian relaxation of the head cage on the head SDF."""
    nbrs = cage_neighbors(cage)
    pinned = {feats["nose"]}
    for side in (-1, 1):
        pinned.update(feats["eye", side])
        pinned.update(feats["ear", side][0])
    for ring in feats["mouth_rings"]:
        pinned.update(ring)
    pinned.update(feats["outline"])
    movable = [i for i in cage.groups["head"] if i not in pinned]
    for _ in range(iters):
        new = {}
        for i in movable:
            nb = nbrs[i]
            new[i] = cage.co[i].lerp(sum((cage.co[j] for j in nb), V()) / len(nb), 0.5)
        for i, p in new.items():
            cage.co[i] = project_ray(HEAD_C, p - HEAD_C, head_proj_sdf, fallback=cage.co[i])


def add_feature_insets(cage, feats):
    for side in (-1, 1):
        prev = feats["eye", side]
        for pts in feats["eye_rings", side]:
            ring = cage.ring(pts, "head", "eye_region")
            cage.bridge(prev, ring)
            prev = ring


def build_ear(cage, side, loop):
    """A simple anime ear from a 6-vertex socket: base ring, helix rim, bowl, two-quad cap."""
    c = sum((cage.co[i] for i in loop), V()) / len(loop)
    n = V((side, 0.30, 0.04)).normalized()
    u_back = ortho(V((0, 1, 0)), n)
    u_up = n.cross(u_back).normalized()
    if u_up.z < 0:
        u_up = -u_up
    thetas = [math.atan2((cage.co[i] - c).dot(u_up), (cage.co[i] - c).dot(u_back)) for i in loop]

    def ring(off_n, cb, rb, cu, ru, tilt):
        pts = []
        for th in thetas:
            cs, sn = math.cos(th), math.sin(th)
            pts.append(c + n * off_n + u_back * (cb + rb * cs + tilt * sn) + u_up * (cu + ru * sn))
        return cage.ring(pts, "ear")

    base = cage.ring([c + (cage.co[i] - c) * 0.95 + n * 0.0040 for i in loop], "ear")
    rim = ring(0.0115, 0.0040, 0.0125, 0.0015, 0.0210, 0.0040)
    bowl = ring(0.0105, 0.0032, 0.0075, 0.0016, 0.0140, 0.0026)
    cage.bridge(loop, base)
    cage.bridge(base, rim)
    cage.bridge(rim, bowl)
    cage.quad(bowl[0], bowl[1], bowl[2], bowl[3])
    cage.quad(bowl[3], bowl[4], bowl[5], bowl[0])


def build_neck(cage, neck_loop, head_hole):
    head_hole = cage.best_alignment(neck_loop, head_hole)
    cen_a = sum((cage.co[i] for i in neck_loop), V()) / len(neck_loop)
    cen_b = sum((cage.co[i] for i in head_hole), V()) / len(head_hole)
    rings = [neck_loop]
    t = 0.5
    c = cen_a.lerp(cen_b, t)
    pts = []
    for ia, ib in zip(neck_loop, head_hole):
        p = cage.co[ia].lerp(cage.co[ib], t)
        dirv = V((p.x - c.x, p.y - c.y, 0))
        if dirv.length > 1e-6:
            p = V((c.x, c.y, p.z)) + dirv.normalized() * lerp(dirv.length, 0.0375, 0.7)
        pts.append(p)
    rings.append(cage.ring(pts, "neck"))
    rings.append(head_hole)
    for a, b in zip(rings, rings[1:]):
        cage.bridge(a, b)
    return rings


# =============================================================================
#  Build the whole cage
# =============================================================================

def build_cage():
    cage = Cage()
    torso = build_torso(cage)
    arms = {s: build_arm(cage, s, torso["sockets"][s]) for s in (-1, 1)}
    legs = {s: build_leg(cage, s, torso["legs"][s]) for s in (-1, 1)}
    feats = build_head(cage)
    place_head_features(cage, feats)
    relax_head(cage, feats)
    add_feature_insets(cage, feats)
    for side in (-1, 1):
        build_ear(cage, side, feats["ear", side][0])
    neck = build_neck(cage, torso["neck"], feats["hole"])
    return cage, {"torso": torso, "arms": arms, "legs": legs, "head": feats, "neck": neck}


# =============================================================================
#  After subdivision: crisp facial features and an even flow on the head
# =============================================================================

def boundary_loops(bm):
    edges = {e for e in bm.edges if e.is_boundary}
    loops = []
    while edges:
        e = edges.pop()
        start, cur = e.verts[0], e.verts[1]
        loop = [start]
        prev_e = e
        while cur is not start:
            loop.append(cur)
            nxt = next(x for x in cur.link_edges if x.is_boundary and x is not prev_e)
            edges.discard(nxt)
            prev_e = nxt
            cur = nxt.other_vert(cur)
        loops.append(loop)
    return loops


def concentric_layers(loop, n):
    layers = [list(loop)]
    seen = {v.index for v in loop}
    for _ in range(n - 1):
        nxt = []
        for v in layers[-1]:
            cand = [e.other_vert(v) for e in v.link_edges if e.other_vert(v).index not in seen]
            nxt.append(cand[0])
        seen.update(v.index for v in nxt)
        layers.append(nxt)
    return layers


def align_layers(layers, k0, forward):
    out = []
    n = len(layers[0])
    for L in layers:
        if forward:
            out.append([L[(k0 + k) % n] for k in range(n)])
        else:
            out.append([L[(k0 - k) % n] for k in range(n)])
    return out


EYE_LAYERS = [(1.00, 1.00, 0.0008), (1.10, 1.15, -0.0003), (1.28, 1.36, 0.0), (1.48, 1.64, 0.0)]
MOUTH_LAYERS = [(1.00, 0.00015, 0.0006), (1.28, 0.0026, 0.0), (1.58, 0.0056, 0.0), (1.86, 0.0088, 0.0)]


def mouth_shape(s, h, n_half):
    """Slots: left corner, upper lip (left->right), right corner, lower lip (right->left)."""
    w = MOUTH_HALF_W * s
    pts = [(-w, 0.0)]
    for k in range(1, n_half):
        x = -w + 2 * w * k / n_half
        e = 1.0 if h < 0.001 else math.sqrt(max(0.0, 1 - (x / w) ** 2))
        pts.append((x, h * e))
    pts.append((w, 0.0))
    for k in range(1, n_half):
        x = w - 2 * w * k / n_half
        e = 1.0 if h < 0.001 else math.sqrt(max(0.0, 1 - (x / w) ** 2))
        pts.append((x, -h * e))
    return [(x, z + 0.0007 * (x / MOUTH_HALF_W) ** 2) for x, z in pts]


def finish_head(ob):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bm.verts.ensure_lookup_table()
    dl = bm.verts.layers.deform.verify()
    gi = {g.name: g.index for g in ob.vertex_groups}

    def wt(v, name):
        return v[dl].get(gi[name], 0.0) if name in gi else 0.0

    head_w = [wt(v, "head") * (1.0 - wt(v, "ear")) for v in bm.verts]
    for v in bm.verts:
        if head_w[v.index] > 1e-3:
            v.co = v.co.lerp(project_ray(HEAD_C, v.co - HEAD_C, head_proj_sdf, fallback=v.co.copy()), head_w[v.index])

    fixed = set()
    features = {"eye": {}, "mouth": None}
    for loop in boundary_loops(bm):
        c = sum((v.co for v in loop), V()) / len(loop)
        if c.z < 1.3 or c.z > 1.6:
            continue
        if abs(c.x) > 0.01:                                 # an eye
            side = 1 if c.x > 0 else -1
            layers = concentric_layers(loop, 5)
            rim = eye_curve(len(loop) // 2)
            inner = eye_to_world(*rim[0], side)
            k0 = min(range(len(loop)), key=lambda k: (loop[k].co - inner).length)
            top1 = eye_to_world(*rim[1], side)
            fwd = (loop[(k0 + 1) % len(loop)].co - top1).length < (loop[(k0 - 1) % len(loop)].co - top1).length
            layers = align_layers(layers, k0, fwd)
            features["eye"][side] = [[v.index for v in L] for L in layers]
            for L, (su, sv, inset) in zip(layers, EYE_LAYERS):
                for v, (u, w) in zip(L, rim):
                    v.co = eye_to_world(u * su, w * sv, side) + V((0, inset, 0))
                    fixed.add(v.index)
        else:                                               # the mouth
            layers = concentric_layers(loop, 5)
            n_half = len(loop) // 2
            left = V((-MOUTH_HALF_W, 0, MOUTH_C.z))
            k0 = min(range(len(loop)), key=lambda k: (V((loop[k].co.x, 0, loop[k].co.z)) - left).length)
            fwd = loop[(k0 + 2) % len(loop)].co.z > loop[(k0 - 2) % len(loop)].co.z
            layers = align_layers(layers, k0, fwd)
            features["mouth"] = [[v.index for v in L] for L in layers]
            for L, (s, h, inset) in zip(layers, MOUTH_LAYERS):
                for v, (x, z) in zip(L, mouth_shape(s, h, n_half)):
                    v.co = project_front(x, MOUTH_C.z + z, head_sdf) + V((0, inset, 0))
                    fixed.add(v.index)
    nose = min((v for v in bm.verts if abs(v.co.x) < 1e-4 and head_w[v.index] > 0.99),
               key=lambda v: v.co.y)
    fixed.add(nose.index)

    movable = [v for v in bm.verts if head_w[v.index] > 0.999 and v.index not in fixed]
    for _ in range(25):
        new = []
        for v in movable:
            nb = [e.other_vert(v).co for e in v.link_edges]
            new.append(v.co.lerp(sum(nb, V()) / len(nb), 0.5))
        for v, p in zip(movable, new):
            v.co = project_ray(HEAD_C, p - HEAD_C, head_proj_sdf, fallback=p)
    features["nose"] = nose.index
    bm.to_mesh(ob.data)
    bm.free()
    return features


# =============================================================================
#  Face details: eye sockets, eyeballs, lashes, brows, mouth bag
# =============================================================================

MAT = {"skin": 0, "top": 1, "collar": 2, "socks": 3, "shoes": 4, "eye": 5, "line": 6, "mouth": 7}
EYE_R = 0.040          # eyeball sphere radius: big and flat, like anime eyes
EYE_GAP = 0.0025       # how far the eyeball sits behind the lid rim
EYE_SPAN = (math.radians(42), math.radians(36))   # half-angles of the eyeball cap


def sdf_normal(p, sdf=None, h=0.0004):
    sdf = sdf or head_sdf
    g = V((sdf(p + V((h, 0, 0))) - sdf(p - V((h, 0, 0))),
           sdf(p + V((0, h, 0))) - sdf(p - V((0, h, 0))),
           sdf(p + V((0, 0, h))) - sdf(p - V((0, 0, h)))))
    return g.normalized()


def ray_sphere(o, d, c, r):
    d = d.normalized()
    oc = o - c
    b = oc.dot(d)
    disc = b * b - (oc.dot(oc) - r * r)
    t = -b - math.sqrt(max(disc, 0.0))
    return o + d * t


def eye_frame(rim_pts):
    """Eyeball pivot, iris axis and a (side, up) frame for one eye."""
    c = sum(rim_pts, V()) / len(rim_pts)
    c_s = project_front(c.x, c.z, head_sdf)
    n = sdf_normal(c_s)
    pivot = c_s - n * (EYE_R + EYE_GAP)
    hit = ray_sphere(V((c.x, -0.4, c.z)), V((0, 1, 0)), pivot, EYE_R)
    f = (hit - pivot).normalized()
    side = ortho(V((1, 0, 0)), f)
    up = f.cross(side).normalized()
    return pivot, f, side, up


def eye_point(pivot, f, side, up, a, b, r=EYE_R):
    return pivot + (f * (math.cos(a) * math.cos(b)) + side * (math.sin(a) * math.cos(b)) + up * math.sin(b)) * r


def add_quad(bm, vs, mat, outward=None, uvs=None, uv_layer=None):
    f = bm.faces.new(vs)
    f.material_index = mat
    if outward is not None:
        f.normal_update()
        if f.normal.dot(outward) < 0:
            f.normal_flip()
            if uvs is not None:
                uvs = list(reversed(uvs))
                # reversed winding: realign so uv k matches loop k
                vs = list(reversed(vs))
    if uvs is not None and uv_layer is not None:
        for loop in f.loops:
            loop[uv_layer].uv = uvs[vs.index(loop.vert)]
    return f


def strip(bm, rows, mat, outward):
    """Quads between consecutive rows of vertices (lists of BMVerts)."""
    for A, B in zip(rows, rows[1:]):
        for k in range(len(A) - 1):
            add_quad(bm, [A[k], A[k + 1], B[k + 1], B[k]], mat, outward)


def add_face_details(ob, features):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bm.verts.ensure_lookup_table()
    uv = bm.loops.layers.uv.verify()
    body_faces = list(bm.faces)
    out = {"eyes": {}}

    for side, layers in features["eye"].items():
        bm.verts.ensure_lookup_table()
        rim = [bm.verts[i] for i in layers[0]]
        pivot, f, sdv, up = eye_frame([v.co.copy() for v in rim])
        out["eyes"][side] = {"pivot": pivot, "axis": f}
        # socket wall: from the lid rim back to just inside the eyeball
        prev = rim
        walls = []
        for t in (0.45, 1.0):
            ring = []
            for v in rim:
                target = pivot + (v.co - pivot).normalized() * (EYE_R - 0.0012)
                ring.append(bm.verts.new(v.co.lerp(target, t)))
            for k in range(len(rim)):
                k2 = (k + 1) % len(rim)
                fc = bm.faces.new([prev[k], prev[k2], ring[k2], ring[k]])
                fc.material_index = MAT["skin"]
                body_faces.append(fc)
            prev = ring
            walls.append(ring)
        out["eyes"][side]["wall"] = walls
        # eyeball cap (a piece of a sphere around the eye pivot, UV'd for the iris texture)
        A, B = EYE_SPAN
        na, nb = 18, 14
        grid = [[bm.verts.new(eye_point(pivot, f, sdv, up, -A + 2 * A * i / na, -B + 2 * B * j / nb))
                 for j in range(nb + 1)] for i in range(na + 1)]
        eye_verts = [v for col in grid for v in col]
        for i in range(na):
            for j in range(nb):
                vs = [grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1]]
                uvs = [V((i / na, j / nb)), V(((i + 1) / na, j / nb)), V(((i + 1) / na, (j + 1) / nb)),
                       V((i / na, (j + 1) / nb))]
                add_quad(bm, vs, MAT["eye"], f, uvs, uv)
        out["eyes"][side]["verts"] = eye_verts

        # lashes: a dark strip along the upper lid, with a flick at the outer corner
        n_half = len(rim) // 2
        rim_uv = eye_curve(n_half)
        cols = []
        for k in range(n_half + 1):
            s = k / n_half
            u, w = rim_uv[k] if k < n_half else rim_uv[n_half]
            t = 0.0009 + 0.0034 * smoothstep(0.0, 0.55, s) - 0.0008 * smoothstep(0.88, 1.0, s)
            d = V((-0.3 + 0.9 * s, 1.0, 0)).normalized()
            cols.append((u, w, d, t))
        for (du, dw, tt) in ((0.0022, 0.0016, 0.0016), (0.0040, 0.0034, 0.0007)):
            u, w = rim_uv[n_half]
            cols.append((u + du, w + dw, V((0.4, 1.0, 0)).normalized(), tt))
        rows = []
        for r_f, fwd in ((-0.25, 0.0003), (0.55, 0.0007), (1.0, 0.0005)):
            row = []
            for (u, w, d, t) in cols:
                p = eye_to_world(u + d.x * t * r_f, w + d.y * t * r_f, side) + V((0, -fwd, 0))
                row.append(bm.verts.new(p))
            rows.append(row)
        strip(bm, rows, MAT["line"], V((0, -1, 0)))
        lash_verts = [v for row in rows for v in row]
        # lower lash: a thin line under the outer half of the eye
        lo = []
        for k in range(n_half, n_half + 4):
            u, w = rim_uv[k]
            lo.append((u, w, 0.0008 * (1.0 - (k - n_half) / 4.0) + 0.0003))
        rows = []
        for r_f, fwd in ((-0.6, 0.0003), (0.7, 0.0004)):
            rows.append([bm.verts.new(eye_to_world(u + 0.15 * t * r_f, w - t * r_f, side) + V((0, -fwd, 0)))
                         for (u, w, t) in lo])
        strip(bm, rows, MAT["line"], V((0, -1, 0)))
        lash_verts += [v for row in rows for v in row]
        # brow
        rows = []
        for edge in (0.0, 1.0):
            row = []
            for k in range(9):
                s = k / 8
                u = lerp(-0.0135, 0.0205, s)
                w = 0.0262 + 0.0032 * math.sin(math.pi * min(s * 1.1, 1.0)) - 0.0020 * s
                t = 0.0019 * (1 - s) + 0.0006
                row.append(bm.verts.new(eye_to_world(u, w + t * edge, side) + V((0, -0.0004, 0))))
            rows.append(row)
        strip(bm, rows, MAT["line"], V((0, -1, 0)))
        out["eyes"][side]["lash_verts"] = lash_verts + [v for row in rows for v in row]

    # mouth bag behind the lips
    bm.verts.ensure_lookup_table()
    rim = [bm.verts[i] for i in features["mouth"][0]]
    n = len(rim)
    nh = n // 2
    mz = MOUTH_C.z
    rings = [rim]
    for sx, h, depth in ((0.95, 0.0016, 0.002), (1.05, 0.0058, 0.007), (0.85, 0.0045, 0.012)):
        ring = []
        for k, v in enumerate(rim):
            x = v.co.x * sx
            wx = max(abs(rim[0].co.x) * sx, 1e-6)
            e = math.sqrt(max(0.0, 1.0 - (x / wx) ** 2))
            sgn = 0.0 if k in (0, nh) else (1.0 if k < nh else -1.0)
            ring.append(bm.verts.new(V((x, v.co.y + depth, mz + sgn * h * e))))
        rings.append(ring)
    for A_, B_ in zip(rings, rings[1:]):
        for k in range(n):
            k2 = (k + 1) % n
            fc = bm.faces.new([A_[k], A_[k2], B_[k2], B_[k]])
            fc.material_index = MAT["mouth"]
            body_faces.append(fc)
    out["mouth_bag"] = rings[1:]
    last = rings[-1]
    top = last[1:nh]                              # left -> right
    bot = [last[k] for k in range(n - 1, nh, -1)]  # left -> right
    mid = [last[0]]
    for k in range(1, nh - 2):
        a, b = top[k], bot[k]
        mid.append(bm.verts.new((a.co + b.co) * 0.5 + V((0, 0.001, 0))))
    mid.append(last[nh])
    for row_a, row_b in ((top, mid), (mid, bot)):
        for k in range(len(row_a) - 1):
            fc = bm.faces.new([row_a[k], row_a[k + 1], row_b[k + 1], row_b[k]])
            fc.material_index = MAT["mouth"]
            body_faces.append(fc)

    bmesh.ops.recalc_face_normals(bm, faces=[fc for fc in body_faces if fc.is_valid])
    bm.to_mesh(ob.data)
    for side, e in out["eyes"].items():
        e["verts"] = [v.index for v in e["verts"]]
        e["lash_verts"] = [v.index for v in e["lash_verts"]]
        e["wall"] = [[v.index for v in r] for r in e["wall"]]
    out["mouth_bag"] = [[v.index for v in r] for r in out["mouth_bag"]]
    bm.free()
    ob.data.shade_smooth()
    return out


# =============================================================================
#  Hair: a scalp cap (copied from the head's own quads) + swept ribbon locks
# =============================================================================

def cap_offset(p):
    """How far the hair surface sits above the scalp (more volume on top and at the back)."""
    return 0.004 + 0.010 * smoothstep(1.44, 1.585, p.z) + 0.004 * smoothstep(0.0, 0.09, p.y)


def hairline_z(psi):
    """Lowest scalp height covered by the cap, by angle around the head (0 = front, 180 = back)."""
    keys = [(0, 1.522), (40, 1.515), (70, 1.488), (90, 1.474), (104, 1.452), (118, 1.425),
            (145, 1.398), (180, 1.390)]
    a = abs(psi)
    for (a0, z0), (a1, z1) in zip(keys, keys[1:]):
        if a <= a1:
            return lerp(z0, z1, smoothstep(0.0, 1.0, (a - a0) / (a1 - a0)))
    return keys[-1][1]


def head_psi(p):
    return math.degrees(math.atan2(p.x, -(p.y - 0.02)))


def body_collider(p):
    d = sd_capsule(p, (0.0, 0.022, 1.28), (0.0, 0.015, 1.42), 0.040)                 # neck
    d = min(d, sd_ellipsoid(p, (0.0, 0.006, 1.16), (0.135, 0.090, 0.17)))            # chest / back
    d = min(d, sd_capsule(p, (-0.15, 0.006, 1.285), (0.15, 0.006, 1.285), 0.048))     # shoulders
    return d


def hair_collider(p):
    """Negative inside the space the hair may not enter (head + cap volume + upper body)."""
    return min(head_proj_sdf(p) - cap_offset(p) - 0.0015, body_collider(p) - 0.006)


def scalp_point(psi_deg, elev_deg, extra=0.0):
    ps, el = math.radians(psi_deg), math.radians(elev_deg)
    d = V((math.sin(ps) * math.cos(el), -math.cos(ps) * math.cos(el), math.sin(el)))
    p = project_ray(HEAD_C, d, head_proj_sdf)
    n = sdf_normal(p, head_proj_sdf)
    return p + n * (cap_offset(p) + extra), n


def drape(root, n_root, d0, length, tip_z=None, step=0.004, hug=0.25, push=None, hug_z=1.40):
    """Grow a lock from root: start tangent to the scalp along d0, fall under gravity,
    hug the head while above the jaw, and slide over the head/body colliders."""
    d = ortho(d0, n_root)
    p = root.copy()
    pts = [root - n_root * 0.007, root.copy()]      # bury the base in the cap
    travelled = 0.0
    while travelled < length:
        d = d + V((0, 0, -1)) * (0.10 + 0.25 * smoothstep(0.0, 0.08, travelled))
        if push is not None:
            d = d + push * 0.12
        c0 = hair_collider(p)
        if p.z > hug_z and c0 > 0.0015:
            d = d - sdf_normal(p, hair_collider) * hug * min(c0 / 0.01, 1.0)
        d = d.normalized()
        q = p + d * step
        c = hair_collider(q)
        if c < 0:
            q = q - sdf_normal(q, hair_collider) * c
        d = (q - p).normalized()
        p = q
        pts.append(p.copy())
        travelled += step
        if tip_z is not None and p.z <= tip_z:
            break
    return pts


def resample(pts, n):
    acc = [0.0]
    for a, b in zip(pts, pts[1:]):
        acc.append(acc[-1] + (b - a).length)
    L = acc[-1]
    out = []
    j = 0
    for k in range(n + 1):
        t = L * (k / n) ** 0.9            # a little denser toward the tip
        while j < len(acc) - 2 and acc[j + 1] < t:
            j += 1
        seg = max(acc[j + 1] - acc[j], 1e-9)
        out.append(pts[j].lerp(pts[j + 1], clamp((t - acc[j]) / seg, 0.0, 1.0)))
    return out


LOCK_SECTION = [(1.0, 0.0), (0.45, 1.0), (-0.45, 1.0), (-1.0, 0.0), (-0.45, -0.55), (0.45, -0.55)]


def build_lock(bm, uv, path, w0, h0, mat, s_layer, g_layer, group, curl=0.0, taper=True):
    """A tapered ribbon with a lens-shaped 6-sided section; returns its vertices."""
    n = len(path) - 1
    rings = []
    prev_n = None
    for k, p in enumerate(path):
        s = k / n
        t = (path[min(k + 1, n)] - path[max(k - 1, 0)]).normalized()
        # rotation-minimising frame: carry the previous normal along, and only lean toward
        # the collider's surface normal while the lock lies on the head (no twisting)
        g = sdf_normal(p, hair_collider)
        if prev_n is None:
            nn = g
        else:
            near = 1.0 - smoothstep(0.004, 0.015, hair_collider(p))
            nn = prev_n.lerp(g, 0.35 * near) if g.dot(prev_n) > 0 else prev_n
        nn = ortho(nn, t)
        prev_n = nn
        sd = t.cross(nn).normalized()
        if taper:
            w = w0 * (0.80 + 0.40 * math.sin(math.pi * min(s * 1.25, 1.0))) * (1.0 - s) ** 0.55 + 0.0004
            h = h0 * (1.0 - 0.7 * s) + 0.0003
        else:
            w, h = w0, h0
        ring = []
        for (a, b) in LOCK_SECTION:
            q = p + sd * (a * w) + nn * (b * h + curl * s * s * w * (1 - abs(a)))
            v = bm.verts.new(q)
            if s_layer is not None:
                v[s_layer] = s
                v[g_layer] = group
            ring.append(v)
        rings.append(ring)
    m = len(LOCK_SECTION)
    for k in range(n):
        for j in range(m):
            j2 = (j + 1) % m
            f = bm.faces.new([rings[k][j], rings[k][j2], rings[k + 1][j2], rings[k + 1][j]])
            f.material_index = mat
            for loop, (uu, vv) in zip(f.loops, ((j / m, k / n), (j2 / m if j2 else 1.0, k / n),
                                                (j2 / m if j2 else 1.0, (k + 1) / n), (j / m, (k + 1) / n))):
                loop[uv].uv = (uu, vv)
    for ring in (rings[0], rings[-1]):        # close both ends with two quads
        for quad in ((ring[0], ring[1], ring[2], ring[3]), (ring[3], ring[4], ring[5], ring[0])):
            f = bm.faces.new(quad)
            f.material_index = mat
    return [v for r in rings for v in r]


# lock recipes: (group, psi, elev, d0, tip_z, max_len, width, thickness, segments, curl)
def hair_recipes():
    R = []
    # bangs across the forehead
    for k, psi in enumerate([-52, -38, -25, -12, 0, 12, 25, 38, 52]):
        tip = {0: 1.481, 12: 1.478, 25: 1.476, 38: 1.468, 52: 1.452}[abs(psi)]
        sgn = 1 if psi > 0 else -1 if psi < 0 else 0
        d0 = V((0.25 * sgn + math.sin(math.radians(psi)) * 0.4, -1.0, -0.15))
        R.append(("bangs", psi, 64, d0, tip, 0.16, 0.0165 if abs(psi) < 45 else 0.0175, 0.0036, 10, -0.15))
    # face-framing side locks, in front of the ears
    for sgn in (-1, 1):
        R.append(("side_" + ("L" if sgn > 0 else "R"), sgn * 76, 42, V((sgn * 0.15, -0.35, -1.0)),
                  1.285, 0.30, 0.0175, 0.0045, 16, 0.3))
        for psi, el, tip in ((100, 46, 1.235), (116, 42, 1.200)):
            R.append(("side_" + ("L" if sgn > 0 else "R"), sgn * psi, el, V((sgn * 0.25, 0.25, -1.0)),
                      tip, 0.38, 0.0215, 0.0055, 16, 0.2))
    # long back hair: two staggered rows from the crown, falling to the mid back
    for k, psi in enumerate((122, 134, 146, 158, 169, 180)):
        el = 58 if k % 2 == 0 else 44
        tip = 1.12 - 0.012 * k
        for sgn in ((1,) if psi == 180 else (-1, 1)):
            R.append(("back", sgn * psi, el, V((sgn * 0.2 * (psi != 180), 1.0, -0.5)),
                      tip, 0.66, 0.0290, 0.0062, 18, 0.25))
    for psi in (-163, -140, 140, 163):
        R.append(("back", psi, 68, V((0.0, 1.0, -0.4)), 1.14, 0.62, 0.0270, 0.0060, 18, 0.25))
    return R


HAIR_GROUPS = {"cap": 0, "bangs": 1, "top": 2, "side_L": 3, "side_R": 4, "back": 5}


def build_hair(body):
    me = bpy.data.meshes.new("Hair")
    bm = bmesh.new()
    uv = bm.loops.layers.uv.verify()
    s_layer = bm.verts.layers.float.new("hair_s")
    g_layer = bm.verts.layers.int.new("hair_group")
    # ---- the cap: the scalp quads of the head, lifted off the skin
    src = bmesh.new()
    src.from_mesh(body.data)
    dl = src.verts.layers.deform.verify()
    gi = {g.name: g.index for g in body.vertex_groups}
    hidx, eidx = gi["head"], gi.get("ear")

    def is_scalp(v):
        if v[dl].get(hidx, 0.0) < 0.999 or (eidx is not None and v[dl].get(eidx, 0.0) > 0.0):
            return False
        return v.co.z > hairline_z(head_psi(v.co)) - 0.004

    faces = [f for f in src.faces if f.material_index == MAT["skin"] and all(is_scalp(v) for v in f.verts)]
    vmap = {}
    for f in faces:
        for v in f.verts:
            if v.index not in vmap:
                n = sdf_normal(v.co, head_proj_sdf)
                nv = bm.verts.new(v.co + n * cap_offset(v.co))
                nv[s_layer] = 0.0
                nv[g_layer] = HAIR_GROUPS["cap"]
                vmap[v.index] = nv
    for f in faces:
        nf = bm.faces.new([vmap[v.index] for v in f.verts])
        nf.material_index = 0
    src.free()
    bmesh.ops.subdivide_edges(bm, edges=list(bm.edges), cuts=1, use_grid_fill=True)
    for v in bm.verts:
        base = project_ray(HEAD_C, v.co - HEAD_C, head_proj_sdf, fallback=v.co.copy())
        v.co = base + sdf_normal(base, head_proj_sdf) * cap_offset(base)
    # a thickness band around the cap's edge, tucked down to the skin
    edge_loops = []
    bnd = {e for e in bm.edges if e.is_boundary}
    while bnd:
        e = bnd.pop()
        start, cur, prev_e = e.verts[0], e.verts[1], e
        loop = [start]
        while cur is not start:
            loop.append(cur)
            nxt = next(x for x in cur.link_edges if x.is_boundary and x is not prev_e)
            bnd.discard(nxt)
            prev_e, cur = nxt, nxt.other_vert(cur)
        edge_loops.append(loop)
    for loop in edge_loops:                      # smooth the staircase border of the cap
        for _ in range(6):
            new = [(loop[k - 1].co + loop[k].co * 2 + loop[(k + 1) % len(loop)].co) / 4 for k in range(len(loop))]
            for v, q in zip(loop, new):
                base = project_ray(HEAD_C, q - HEAD_C, head_proj_sdf, fallback=q)
                v.co = base + sdf_normal(base, head_proj_sdf) * cap_offset(base)
    for loop in edge_loops:
        inner = []
        for v in loop:
            n = sdf_normal(v.co, head_proj_sdf)
            iv = bm.verts.new(v.co - n * (cap_offset(v.co) - 0.0008))
            iv[g_layer] = HAIR_GROUPS["cap"]
            inner.append(iv)
        for k in range(len(loop)):
            k2 = (k + 1) % len(loop)
            bm.faces.new([loop[k], loop[k2], inner[k2], inner[k]])
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    # ---- locks
    locks = []
    for (grp, psi, el, d0, tip, L, w, h, nseg, curl) in hair_recipes():
        root, n = scalp_point(psi, el, extra=0.0018)
        push, hug_z = None, 1.40
        if grp == "back":       # fall free behind the shoulders, spreading a little
            push = V((math.sin(math.radians(psi)) * 0.35, 0.45, 0.0))
            hug_z = 1.47
        path = drape(root, n, d0, L, tip_z=tip, push=push, hug_z=hug_z)
        path = resample(path, nseg)
        vs = build_lock(bm, uv, path, w, h, 0, s_layer, g_layer, HAIR_GROUPS[grp], curl)
        locks.append((grp, psi, path))
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))   # every lock is a closed tube
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new("Hair", me)
    bpy.context.scene.collection.objects.link(ob)
    me.shade_smooth()
    assign_materials(ob, ["hair"])
    return ob, locks


# =============================================================================
#  Pleated skirt and outfit zones
# =============================================================================

SKIRT_PLEATS = 24
SKIRT_RINGS = [
    # (z, half width, front, back, pleat depth)   waistband -> hem
    (1.012, 0.110, 0.080, 0.078, 0.000),
    (0.985, 0.114, 0.083, 0.082, 0.000),
    (0.930, 0.137, 0.088, 0.098, 0.004),
    (0.860, 0.156, 0.097, 0.120, 0.007),
    (0.790, 0.170, 0.110, 0.133, 0.010),
    (0.725, 0.183, 0.122, 0.145, 0.012),
    (0.668, 0.194, 0.132, 0.156, 0.013),
]


def skirt_ring(z, a, f, b, depth, inset=0.0):
    pts = []
    n = 2 * SKIRT_PLEATS
    for k in range(n):
        th = 2 * math.pi * k / n - math.pi / 2          # k = 0 at the front centre
        c, s_ = math.cos(th), math.sin(th)
        r = 1.0 + (depth if k % 2 == 0 else -depth) / max(a, f)
        x = (a - inset) * c * r
        y = 0.008 + (f if s_ < 0 else b) * s_ * r - inset * s_
        pts.append(V((x, y, z)))
    return pts


def build_skirt():
    me = bpy.data.meshes.new("Skirt")
    bm = bmesh.new()
    uv = bm.loops.layers.uv.verify()
    n = 2 * SKIRT_PLEATS
    outer = [[bm.verts.new(p) for p in skirt_ring(*r)] for r in SKIRT_RINGS]
    inner = [[bm.verts.new(p) for p in skirt_ring(*r, inset=0.003)] for r in SKIRT_RINGS]
    nr = len(SKIRT_RINGS)

    def band(A, B, va, vb, flip=False):
        for k in range(n):
            k2 = (k + 1) % n
            vs = [A[k], A[k2], B[k2], B[k]]
            uvs = [(k / n, va), ((k + 1) / n, va), ((k + 1) / n, vb), (k / n, vb)]
            if flip:
                vs, uvs = vs[::-1], uvs[::-1]
            f = bm.faces.new(vs)
            f.material_index = 0 if 0 < max(va, vb) else 1
            for loop, t in zip(f.loops, uvs):
                loop[uv].uv = t
    for i in range(nr - 1):
        band(outer[i], outer[i + 1], 1 - i / (nr - 1), 1 - (i + 1) / (nr - 1))
        band(inner[i], inner[i + 1], 1 - i / (nr - 1), 1 - (i + 1) / (nr - 1), flip=True)
    band(outer[-1], inner[-1], 0.0, 0.0)                 # hem
    band(inner[0], outer[0], 1.0, 1.0)                   # top of the waistband
    for f in bm.faces:                                   # waistband ring in the ribbon red? no: navy
        f.material_index = 0
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new("Skirt", me)
    bpy.context.scene.collection.objects.link(ob)
    me.shade_smooth()
    assign_materials(ob, ["skirt"])
    return ob


# sailor collar spokes for the left half, front V tip -> back centre: (inner edge, outer edge)
COLLAR_SPOKES = [
    ((0.004, -0.080, 1.200), (0.012, -0.083, 1.188)),
    ((0.012, -0.072, 1.236), (0.046, -0.084, 1.214)),
    ((0.022, -0.060, 1.272), (0.082, -0.070, 1.250)),
    ((0.032, -0.045, 1.306), (0.112, -0.048, 1.282)),
    ((0.042, -0.020, 1.330), (0.138, -0.014, 1.300)),
    ((0.045, 0.010, 1.344), (0.138, 0.026, 1.298)),
    ((0.040, 0.040, 1.345), (0.128, 0.066, 1.250)),
    ((0.030, 0.055, 1.343), (0.122, 0.082, 1.166)),
    ((0.015, 0.062, 1.342), (0.062, 0.090, 1.162)),
    ((0.000, 0.064, 1.342), (0.000, 0.092, 1.160)),
]


def build_collar(body):
    from mathutils.bvhtree import BVHTree
    dg = bpy.context.evaluated_depsgraph_get()
    bvh = BVHTree.FromObject(body, dg)
    left = COLLAR_SPOKES
    right = [((-a[0], a[1], a[2]), (-b[0], b[1], b[2])) for a, b in reversed(COLLAR_SPOKES[:-1])]
    spokes = left + right
    inner_c = [V(a) for a, b in spokes]
    outer_c = [V(b) for a, b in spokes]
    nu, rows = 40, (0.0, 0.3, 0.62, 0.80, 0.88, 1.0)
    me = bpy.data.meshes.new("Collar")
    bm = bmesh.new()
    uv = bm.loops.layers.uv.verify()

    def surf(p, off):
        loc, nrm, _, _ = bvh.find_nearest(p)
        return loc + nrm * off

    layers = []
    for off in (0.0045, 0.0018):
        grid = []
        for i in range(nu + 1):
            t = i / nu
            a, b = catmull_rom(inner_c, t), catmull_rom(outer_c, t)
            grid.append([bm.verts.new(surf(a.lerp(b, r), off + 0.0015 * r)) for r in rows])
        layers.append(grid)
    top, bot = layers
    nr = len(rows)

    def face(vs, mat, uvs=None):
        f = bm.faces.new(vs)
        f.material_index = mat
        if uvs:
            for loop, q in zip(f.loops, uvs):
                loop[uv].uv = q
    for i in range(nu):
        for j in range(nr - 1):
            stripe = 1 if j == 3 else 0
            uvs = [(i / nu, rows[j]), ((i + 1) / nu, rows[j]), ((i + 1) / nu, rows[j + 1]), (i / nu, rows[j + 1])]
            face([top[i][j], top[i + 1][j], top[i + 1][j + 1], top[i][j + 1]], stripe, uvs)
            face([bot[i][j + 1], bot[i + 1][j + 1], bot[i + 1][j], bot[i][j]], 0, uvs[::-1])
    for i in range(nu):                         # inner and outer edges
        for j in (0, nr - 1):
            face([top[i][j], bot[i][j], bot[i + 1][j], top[i + 1][j]], 0)
    for i in (0, nu):                           # the two ends of the V
        for j in range(nr - 1):
            face([top[i][j], top[i][j + 1], bot[i][j + 1], bot[i][j]], 0)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    # the bow: a knot, two loops and two tails (constant-width ribbons)
    knot_c = surf(V((0.0, -0.085, 1.196)), 0.007)
    paths = []
    for sgn in (-1, 1):
        loop_pts = [knot_c + V((sgn * dx, -dy, dz)) for dx, dy, dz in
                    ((0.002, 0.000, 0.000), (0.016, 0.004, 0.010), (0.034, 0.003, 0.009),
                     (0.040, 0.002, -0.002), (0.030, 0.003, -0.010), (0.012, 0.003, -0.004), (0.003, 0.0, -0.001))]
        paths.append(([catmull_rom(loop_pts, k / 12) for k in range(13)], 0.0075, False))
        tail = [knot_c + V((sgn * dx, -dy, dz)) for dx, dy, dz in
                ((0.002, 0.000, -0.003), (0.010, 0.004, -0.025), (0.016, 0.006, -0.050), (0.020, 0.006, -0.068))]
        paths.append(([catmull_rom(tail, k / 8) for k in range(9)], 0.0085, True))
    for pts, w, tap in paths:
        build_lock(bm, uv, pts, w, 0.0016, 2, None, None, 0, taper=False)
    k0 = len(bm.verts)
    bmesh.ops.create_cube(bm, size=1.0, matrix=Matrix.Translation(knot_c) @ Matrix.Diagonal((0.013, 0.008, 0.013, 1)))
    bm.faces.ensure_lookup_table()
    for f in bm.faces[-6:]:
        f.material_index = 2
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    ob = bpy.data.objects.new("Collar", me)
    bpy.context.scene.collection.objects.link(ob)
    assign_materials(ob, ["collar", "top", "ribbon"])     # slots first: indices are clamped
    bm.to_mesh(me)
    bm.free()
    me.shade_smooth()
    return ob


def assign_outfit(body, parts):
    """Paint the body into outfit zones: sailor blouse, collar, socks, shoes."""
    me = body.data
    gi = body.vertex_groups["torso"].index
    in_torso = [any(g.group == gi and g.weight > 0.999 for g in v.groups) for v in me.vertices]
    torso_faces = {p.index for p in me.polygons if all(in_torso[i] for i in p.vertices)}
    ga = body.vertex_groups["arm"].index
    in_arm = [any(g.group == ga and g.weight > 0.0 for g in v.groups) for v in me.vertices]
    S = {s: parts["arms"][s]["S"] for s in (-1, 1)}
    E = {s: parts["arms"][s]["E"] for s in (-1, 1)}
    for poly in me.polygons:
        if poly.material_index != MAT["skin"]:
            continue
        c = poly.center
        side = 1 if c.x > 0 else -1
        z = c.z
        mat = "skin"
        on_arm = any(in_arm[i] for i in poly.vertices)
        if on_arm:
            d = (E[side] - S[side]).normalized()
            t = (c - S[side]).dot(d)
            if t < 0.105:
                mat = "top"
            elif t < 0.125:
                mat = "collar"                         # sleeve cuff
            else:
                mat = "skin"
            poly.material_index = MAT[mat]
            continue
            if t < 0.105:
                mat = "top"
            elif t < 0.125:
                mat = "collar"                         # sleeve cuff
        elif 0.70 < z < 1.36 and abs(c.x) < 0.20 and poly.index in torso_faces:
            mat = "top" if z > 0.985 else "collar"     # blouse; navy shorts under the skirt
        if z < 0.565 and abs(c.x) < 0.13:
            mat = "socks"
        if z < 0.105 and abs(c.x) < 0.13:
            mat = "shoes"
        poly.material_index = MAT[mat]


# =============================================================================
#  UVs
# =============================================================================

REGION_OF_GROUP = {"torso": "torso", "arm": "arm", "hand": "hand", "leg": "leg", "foot": "foot",
                   "neck": "neck", "neckring": "neck", "head": "head", "eye_region": "head",
                   "mouth_region": "head", "ear": "ear"}


def body_regions(ob):
    """Label every face with a body region (+ side) from the construction vertex groups."""
    names = {g.index: g.name for g in ob.vertex_groups}
    vlab = []
    for v in ob.data.vertices:
        best, bw = "detail", 0.0
        for g in v.groups:
            n = names[g.group]
            if n.startswith("finger_"):
                n = "hand"
            r = REGION_OF_GROUP.get(n)
            if r and g.weight > bw:
                best, bw = r, g.weight
        vlab.append(best)
    labels = []
    for p in ob.data.polygons:
        counts = {}
        for i in p.vertices:
            counts[vlab[i]] = counts.get(vlab[i], 0) + 1
        r = max(counts, key=counts.get)
        if r in ("arm", "hand", "leg", "foot", "ear"):
            r += ".L" if p.center.x > 0 else ".R"
        labels.append(r)
    return labels


def limb_frame(p, path, side):
    """Closest point on a limb polyline, its direction and the limb frame there."""
    best = None
    for a, b in zip(path, path[1:]):
        ab = b - a
        u = clamp((p - a).dot(ab) / ab.length_squared, 0.0, 1.0)
        c = a + ab * u
        d2 = (p - c).length_squared
        if best is None or d2 < best[0]:
            best = (d2, c, ab.normalized())
    _, c, d = best
    uf, uu = frame_at(d, side)
    return c, d, uf, uu


def unwrap_body(body, parts):
    labels = body_regions(body)
    bm = bmesh.new()
    bm.from_mesh(body.data)
    bm.faces.ensure_lookup_table()
    eye_mat = MAT["eye"]
    hands = {s: parts["arms"][s]["hand"] for s in (-1, 1)}
    for e in bm.edges:
        if len(e.link_faces) != 2:
            continue
        f0, f1 = e.link_faces
        l0, l1 = labels[f0.index], labels[f1.index]
        m = (e.verts[0].co + e.verts[1].co) / 2
        dvec = (e.verts[1].co - e.verts[0].co).normalized()
        seam = l0 != l1
        on_center = abs(e.verts[0].co.x) < 1e-5 and abs(e.verts[1].co.x) < 1e-5
        if not seam and l0 in ("head", "neck", "torso") and on_center and m.y > 0.0:
            seam = True                                     # cut down the back
        if not seam and l0 == "head" and (f0.calc_center_median().y < 0.012) != (f1.calc_center_median().y < 0.012):
            seam = True                                     # face island | back of the head
        if not seam and l0.startswith("arm"):
            side = 1 if l0.endswith(".L") else -1
            a = parts["arms"][side]
            c, d, uf, uu = limb_frame(m, [a["S"], a["E"], a["W"]], side)
            r = m - c
            ang = math.degrees(math.atan2(r.dot(uu), r.dot(uf)))
            seam = abs(dvec.dot(d)) > 0.7 and abs(angle_wrap(math.radians(ang + 90))) < math.radians(11.5)
        if not seam and l0.startswith("leg"):
            side = 1 if l0.endswith(".L") else -1
            path = [V((side * q.x, q.y, q.z)) for q in (HIP, KNEE, ANKLE)]
            c, d, uf, uu = limb_frame(m, path, side)
            r = m - c
            ang = math.degrees(math.atan2(-side * r.x, -r.y))     # 90 = inner side
            seam = abs(dvec.dot(d)) > 0.7 and abs(ang - 90) < 11.5
        if not seam and l0.startswith("hand"):                  # back of the hand | palm
            uu = hands[1 if l0.endswith(".L") else -1]["uu"]
            seam = (f0.normal.dot(uu) > 0) != (f1.normal.dot(uu) > 0)
        if not seam and l0.startswith("foot"):                  # sole | upper
            seam = (f0.normal.z < -0.6) != (f1.normal.z < -0.6)
        e.seam = seam
    bm.to_mesh(body.data)
    bm.free()
    # unwrap everything except the eyeballs, which keep their own UVs for the iris texture
    for p in body.data.polygons:
        p.select = p.material_index != eye_mat
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    body.select_set(True)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_mode(type="FACE")
    bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=0.003)
    bpy.ops.uv.select_all(action="SELECT")
    bpy.ops.uv.pack_islands(rotate=True, margin=0.004)
    bpy.ops.object.mode_set(mode="OBJECT")
    for p in body.data.polygons:
        p.select = False


def unwrap_hair_cap(hair):
    """The locks already have strand UVs; give the cap a top-down projection in the same space."""
    me = hair.data
    g = me.attributes["hair_group"].data
    uv = me.uv_layers.active.data
    for p in me.polygons:
        if all(g[i].value == HAIR_GROUPS["cap"] for i in p.vertices):
            for li in p.loop_indices:
                co = me.vertices[me.loops[li].vertex_index].co
                ang = math.atan2(co.x, -(co.y - 0.02))
                uv[li].uv = (0.5 + ang / (2 * math.pi), clamp((co.z - 1.38) / 0.22, 0.0, 1.0))


# =============================================================================
#  Rig: a game skeleton with IK controls, skin weights and face shape keys
# =============================================================================

FWD = V((0, -1, 0))
SKIRT_CHAINS = [("Front", 0), ("FrontSide", 45), ("Side", 90), ("BackSide", 135), ("Back", 180)]
SKIRT_JOINT_Z = (0.950, 0.810, 0.665)


def side_name(name, side):
    return name + (".L" if side > 0 else ".R")


def skirt_point(z, ang_deg, inset=0.0015):
    """Point on the skirt (between its two layers) at height z and angle (0 = front, +X = 90)."""
    rs = SKIRT_RINGS
    for (z0, a0, f0, b0, _), (z1, a1, f1, b1, _) in zip(rs, rs[1:]):
        if z1 <= z <= z0:
            u = (z0 - z) / (z0 - z1)
            a, f, b = lerp(a0, a1, u), lerp(f0, f1, u), lerp(b0, b1, u)
            break
    else:
        a, f, b = rs[-1][1:4]
    th = math.radians(ang_deg)
    sx, cy = math.sin(th), -math.cos(th)
    return V(((a - inset) * sx, 0.008 + ((f if cy < 0 else b) - inset) * cy, z))


def lock_chain(path, z_start, n_bones):
    """Joints for a bone chain that follows a hair lock from z_start down to its tip."""
    pts = [p for p in path if p.z <= z_start]
    if len(pts) < 2:
        pts = path[-2:]
    acc = [0.0]
    for a, b in zip(pts, pts[1:]):
        acc.append(acc[-1] + (b - a).length)
    joints = []
    j = 0
    for k in range(n_bones + 1):
        t = acc[-1] * k / n_bones
        while j < len(acc) - 2 and acc[j + 1] < t:
            j += 1
        u = (t - acc[j]) / max(acc[j + 1] - acc[j], 1e-9)
        joints.append(pts[j].lerp(pts[j + 1], clamp(u, 0, 1)))
    return joints


def signed_angle(u, v, normal):
    a = u.angle(v)
    return -a if u.cross(v).dot(normal) > 0 else a


def build_armature(parts, face, locks):
    data = bpy.data.armatures.new("Armature")
    arm = bpy.data.objects.new("Armature", data)
    bpy.context.scene.collection.objects.link(arm)
    data.display_type = "OCTAHEDRAL"
    arm.show_in_front = True
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    eb = data.edit_bones
    colls = {n: data.collections.new(n) for n in ("Deform", "Physics", "Controls")}
    chains = {}

    def bone(name, head, tail, parent=None, roll=FWD, deform=True, connect=False, coll="Deform"):
        b = eb.new(name)
        b.head, b.tail = V(head), V(tail)
        b.align_roll(V(roll))
        if parent:
            b.parent = eb[parent]
            b.use_connect = connect
        b.use_deform = deform
        colls[coll].assign(b)
        return b

    bone("Root", (0, 0, 0), (0, 0, 0.15))
    bone("Hips", (0, 0.012, 0.880), (0, 0.008, 0.970), "Root")
    bone("Spine", (0, 0.008, 0.970), (0, 0.004, 1.060), "Hips", connect=True)
    bone("Chest", (0, 0.004, 1.060), (0, 0.002, 1.150), "Spine", connect=True)
    bone("UpperChest", (0, 0.002, 1.150), (0, 0.012, 1.310), "Chest", connect=True)
    bone("Neck", (0, 0.012, 1.310), (0, 0.016, 1.395), "UpperChest", connect=True)
    bone("Head", (0, 0.016, 1.395), (0, 0.016, 1.585), "Neck", connect=True)
    for side, e in face["eyes"].items():
        bone(side_name("Eye", side), e["pivot"], e["pivot"] + e["axis"] * 0.03, "Head", roll=(0, 0, 1))
    for side in (-1, 1):
        a = parts["arms"][side]
        h = a["hand"]
        S, E, W = a["S"], a["E"], a["W"]
        bone(side_name("Shoulder", side), (side * 0.018, -0.004, 1.292), S, "UpperChest")
        bone(side_name("UpperArm", side), S, E, side_name("Shoulder", side), connect=True)
        bone(side_name("LowerArm", side), E, W, side_name("UpperArm", side), connect=True)
        bone(side_name("Hand", side), W, h["knuckles"], side_name("LowerArm", side), roll=h["uu"], connect=True)
        for name, j in h["digits"].items():
            nm = name.capitalize()
            segs = [(j[0], j[2]), (j[2], j[3]), (j[3], j[4])]
            prev = side_name("Hand", side)
            for k, (p0, p1) in enumerate(segs):
                bn = side_name("%s_%d" % (nm, k + 1), side)
                bone(bn, p0, p1, prev, roll=h["uu"], connect=k > 0)
                prev = bn
        hip = V((side * HIP.x, HIP.y, HIP.z))
        knee = V((side * KNEE.x, KNEE.y, KNEE.z))
        ankle = V((side * ANKLE.x, ANKLE.y, ANKLE.z))
        ball = V((side * (ANKLE.x + 0.008), -0.092, 0.022))
        toe = V((side * (ANKLE.x + 0.012), -0.160, 0.020))
        bone(side_name("UpperLeg", side), hip, knee, "Hips")
        bone(side_name("LowerLeg", side), knee, ankle, side_name("UpperLeg", side), connect=True)
        bone(side_name("Foot", side), ankle, ball, side_name("LowerLeg", side), roll=(0, 0, 1), connect=True)
        bone(side_name("Toes", side), ball, toe, side_name("Foot", side), roll=(0, 0, 1), connect=True)
        # IK controls (not exported: use_deform is off)
        bone(side_name("IK_Foot", side), ankle, ball, "Root", roll=(0, 0, 1), deform=False, coll="Controls")
        bone(side_name("Pole_Knee", side), knee + V((0, -0.40, 0)), knee + V((0, -0.45, 0)), "Root",
             deform=False, coll="Controls")
        bone(side_name("IK_Hand", side), W, h["knuckles"], "Root", roll=h["uu"], deform=False, coll="Controls")
        bone(side_name("Pole_Elbow", side), E + V((0, 0.35, 0)), E + V((0, 0.40, 0)), "Root",
             deform=False, coll="Controls")
    # hair chains (spring-bone / physics friendly)
    face_lock = {1: None, -1: None}
    back = {}
    for grp, psi, path in locks:
        if grp in ("side_L", "side_R") and abs(psi) == 76:
            face_lock[1 if psi > 0 else -1] = path
        if grp == "back" and abs(psi) in (122, 146, 180) and path[-1].z < 1.15:
            back[psi] = path
    for side, path in face_lock.items():
        js = lock_chain(path, 1.47, 3)
        names = [side_name("HairSide_%d" % (k + 1), side) for k in range(3)]
        prev = "Head"
        for k in range(3):
            bone(names[k], js[k], js[k + 1], prev, roll=FWD, connect=k > 0, coll="Physics")
            prev = names[k]
        chains[side_name("HairSide", side)] = (js, names, "Head")
    labels = {180: "C", 146: "In", 122: "Out"}
    for psi, path in sorted(back.items()):
        js = lock_chain(path, 1.43, 4)
        side = 0 if psi == 180 else (1 if psi > 0 else -1)
        base = "HairBack_%s" % labels[abs(psi)]
        names = [(base + "_%d" % (k + 1)) + ("" if side == 0 else (".L" if side > 0 else ".R")) for k in range(4)]
        prev = "Head"
        for k in range(4):
            bone(names[k], js[k], js[k + 1], prev, roll=(0, 1, 0), connect=k > 0, coll="Physics")
            prev = names[k]
        chains[base + str(side)] = (js, names, "Head")
    # skirt chains
    for label, ang in SKIRT_CHAINS:
        for side in ((1,) if ang in (0, 180) else (1, -1)):
            a = ang * side
            js = [skirt_point(z, a) for z in SKIRT_JOINT_Z]
            suffix = "" if ang in (0, 180) else (".L" if side > 0 else ".R")
            names = ["Skirt_%s_%d%s" % (label, k + 1, suffix) for k in range(2)]
            bone(names[0], js[0], js[1], "Hips", roll=(0, 0, 1) if ang == 90 else FWD, coll="Physics")
            bone(names[1], js[1], js[2], names[0], roll=(0, 0, 1) if ang == 90 else FWD, connect=True,
                 coll="Physics")
            chains["Skirt%d" % a] = (js, names, "Hips")
    # pole angles need edit-bone axes
    poles = {}
    for side in (-1, 1):
        for up, lo, pole in (("UpperLeg", "LowerLeg", "Pole_Knee"), ("UpperArm", "LowerArm", "Pole_Elbow")):
            b0, b1, pb = eb[side_name(up, side)], eb[side_name(lo, side)], eb[side_name(pole, side)]
            normal = (b1.tail - b0.head).cross(pb.head - b0.head)
            proj = normal.cross(b0.tail - b0.head)
            poles[side_name(lo, side)] = signed_angle(b0.x_axis, proj, b0.tail - b0.head)
    bpy.ops.object.mode_set(mode="OBJECT")

    # IK constraints with IK/FK switches (custom properties on the armature object)
    for side in (-1, 1):
        for limb, lo, end, tgt, pole, default in (("Leg", "LowerLeg", "Foot", "IK_Foot", "Pole_Knee", 1.0),
                                                  ("Arm", "LowerArm", "Hand", "IK_Hand", "Pole_Elbow", 0.0)):
            prop = side_name("IK_" + limb, side)
            arm[prop] = default
            arm.id_properties_ui(prop).update(min=0.0, max=1.0, description="0 = FK, 1 = IK")
            pb = arm.pose.bones[side_name(lo, side)]
            c = pb.constraints.new("IK")
            c.target, c.subtarget = arm, side_name(tgt, side)
            c.pole_target, c.pole_subtarget = arm, side_name(pole, side)
            c.pole_angle = poles[side_name(lo, side)]
            c.chain_count = 2
            cr = arm.pose.bones[side_name(end, side)].constraints.new("COPY_ROTATION")
            cr.target, cr.subtarget = arm, side_name(tgt, side)
            for con in (c, cr):
                fc = con.driver_add("influence")
                fc.driver.type = "AVERAGE"
                var = fc.driver.variables.new()
                var.type = "SINGLE_PROP"
                var.targets[0].id = arm
                var.targets[0].data_path = '["%s"]' % prop
    for b in data.bones:
        if b.name.startswith(("Hair", "Skirt")):
            data.bones[b.name].color.palette = "THEME04"
        elif not b.use_deform:
            data.bones[b.name].color.palette = "THEME09"
    return arm, chains


# ---------------------------------------------------------------------------- weights
def chain_param(p, joints):
    best_d, best_t = 1e9, 0.0
    n = len(joints) - 1
    for i in range(n):
        a, b = joints[i], joints[i + 1]
        ab = b - a
        u = (p - a).dot(ab) / ab.length_squared
        uc = clamp(u, 0.0, 1.0)
        d = (p - (a + ab * uc)).length
        if d < best_d:
            best_d = d
            best_t = i + (u if (i == 0 and u < 0) or (i == n - 1 and u > 1) else uc)
    return best_d, best_t


def hat_weights(t, names, parent, out, scale=1.0):
    centers = [(-0.5, parent)] + [(i + 0.5, n) for i, n in enumerate(names)]
    t = clamp(t, -0.5, len(names) - 0.5)
    for c, n in centers:
        x = 1.0 - abs(t - c)
        if x > 0:
            out[n] = out.get(n, 0.0) + x * scale
    return out


def set_weights(ob, weights_per_vertex):
    """weights_per_vertex: {vertex index: {group: weight}} (replaces existing weights)."""
    groups = {g.name: g for g in ob.vertex_groups}
    for i, wd in weights_per_vertex.items():
        for g in ob.vertex_groups:
            g.remove([i])
        tot = sum(wd.values()) or 1.0
        for name, w in wd.items():
            if name not in groups:
                groups[name] = ob.vertex_groups.new(name=name)
            groups[name].add([i], w / tot, "REPLACE")


def clean_weights(ob, limit=4, floor=0.01):
    """Game-engine friendly skinning: at most `limit` bones per vertex, tiny weights dropped,
    weights normalized to 1."""
    groups = list(ob.vertex_groups)
    for v in ob.data.vertices:
        ws = sorted(((g.weight, g.group) for g in v.groups if g.weight > 0.0), reverse=True)
        keep = [(w, gi) for w, gi in ws[:limit] if w >= floor] or ws[:1]
        tot = sum(w for w, _ in keep) or 1.0
        keep_ids = {gi for _, gi in keep}
        for w, gi in ws:
            if gi not in keep_ids:
                groups[gi].remove([v.index])
        for w, gi in keep:
            groups[gi].add([v.index], w / tot, "REPLACE")


def bind(ob, arm):
    ob.parent = arm
    mod = ob.modifiers.new("Armature", "ARMATURE")
    mod.object = arm


def weight_body(body, arm, face):
    helper = [g.name for g in body.vertex_groups]
    skip = ("Root", "Eye", "Hair", "Skirt")
    saved = {}
    for b in arm.data.bones:
        saved[b.name] = b.use_deform
        if b.name.startswith(skip):
            b.use_deform = False
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    body.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    for name, val in saved.items():
        arm.data.bones[name].use_deform = val
    for g in [body.vertex_groups[n] for n in helper]:
        g.lock_weight = False
    # eyes, lashes, brows, eye sockets and the mouth bag: rigid
    me = body.data
    fixed = {}
    for side, e in face["eyes"].items():
        for i in e["verts"]:
            fixed[i] = {side_name("Eye", side): 1.0}
        for i in e["lash_verts"]:
            fixed[i] = {"Head": 1.0}
        for ring in e["wall"]:
            for i in ring:
                fixed[i] = {"Head": 1.0}
    for ring in face["mouth_bag"]:
        for i in ring:
            fixed[i] = {"Head": 1.0}
    hg = body.vertex_groups["head"].index
    for v in me.vertices:
        if v.index not in fixed and any(g.group == hg and g.weight > 0.999 for g in v.groups):
            fixed[v.index] = {"Head": 1.0}
    set_weights(body, fixed)
    # the construction groups are not bones: move them out of the way
    for n in helper:
        body.vertex_groups.remove(body.vertex_groups[n])
    clean_weights(body)
    body.modifiers[-1].name = "Armature"


def weight_hair(hair, arm, chains):
    me = hair.data
    s_attr = me.attributes["hair_s"].data
    g_attr = me.attributes["hair_group"].data
    back = [c for k, c in chains.items() if k.startswith("HairBack")]
    W = {}
    for v in me.vertices:
        g = g_attr[v.index].value
        p = v.co
        if g in (HAIR_GROUPS["side_L"], HAIR_GROUPS["side_R"]):
            js, names, parent = chains["HairSide.L" if g == HAIR_GROUPS["side_L"] else "HairSide.R"]
            W[v.index] = hat_weights(chain_param(p, js)[1], names, parent, {})
        elif g == HAIR_GROUPS["back"]:
            ds = sorted((chain_param(p, js)[0], chain_param(p, js)[1], names, parent) for js, names, parent in back)
            (da, ta, na, pa), (db, tb, nb, pb) = ds[0], ds[1]
            wa = db / (da + db + 1e-9)
            wd = hat_weights(ta, na, pa, {}, wa)
            hat_weights(tb, nb, pb, wd, 1.0 - wa)
            W[v.index] = wd
        else:
            W[v.index] = {"Head": 1.0}
    set_weights(hair, W)
    clean_weights(hair)
    bind(hair, arm)


def weight_skirt(skirt, arm, chains):
    W = {}
    angs = sorted(int(k[5:]) for k in chains if k.startswith("Skirt"))
    for v in skirt.data.vertices:
        p = v.co
        a = math.degrees(math.atan2(p.x, -(p.y - 0.008)))
        # two neighbouring chains around the body
        best = sorted(angs, key=lambda c: abs(angle_wrap(math.radians(a - c))))[:2]
        d0 = abs(math.degrees(angle_wrap(math.radians(a - best[0]))))
        d1 = abs(math.degrees(angle_wrap(math.radians(a - best[1]))))
        w0 = d1 / (d0 + d1 + 1e-9)
        z0, z1, z2 = SKIRT_JOINT_Z
        t = (z0 - p.z) / (z0 - z1) if p.z >= z1 else 1.0 + (z1 - p.z) / (z1 - z2)
        wd = {}
        # a little thigh influence keeps the skirt out of a raised leg without cloth physics
        leg = 0.35 * smoothstep(0.93, 0.78, p.z) * clamp(abs(p.x) / 0.10, 0.0, 1.0) * clamp(-(p.y - 0.06) / 0.10, 0.0, 1.0)
        for c, w in ((best[0], w0), (best[1], 1.0 - w0)):
            js, names, parent = chains["Skirt%d" % c]
            hat_weights(t, names, parent, wd, w * (1.0 - leg))
        if leg > 0:
            wd[side_name("UpperLeg", 1 if p.x > 0 else -1)] = leg
        W[v.index] = wd
    set_weights(skirt, W)
    clean_weights(skirt)
    bind(skirt, arm)


def weight_from_body(ob, body, arm, rigid=None):
    """Copy skin weights from the nearest body vertex (clothes that hug the body)."""
    me = body.data
    kd = KDTree(len(me.vertices))
    for v in me.vertices:
        kd.insert(v.co, v.index)
    kd.balance()
    names = {g.index: g.name for g in body.vertex_groups}
    W = {}
    for v in ob.data.vertices:
        if rigid and rigid(v):
            W[v.index] = dict(rigid(v))
            continue
        acc = {}
        for co, j, d in kd.find_n(v.co, 3):
            wgt = 1.0 / (d + 1e-4)
            for g in me.vertices[j].groups:
                acc[names[g.group]] = acc.get(names[g.group], 0.0) + g.weight * wgt
        W[v.index] = acc
    set_weights(ob, W)
    clean_weights(ob)
    bind(ob, arm)


# ---------------------------------------------------------------------------- shape keys
JAW_PIVOT = V((0.0, 0.035, 1.430))


def add_shape_keys(body, features, face):
    me = body.data
    body.shape_key_add(name="Basis", from_mix=False)
    co = [v.co.copy() for v in me.vertices]

    def key(name, deltas):
        k = body.shape_key_add(name=name, from_mix=False)
        for i, d in deltas.items():
            k.data[i].co = co[i] + d
        return k

    # ---- blinks: the upper lid closes down onto a line just above the lower lid
    blink = {}
    for side, layers in features["eye"].items():
        n = len(layers[0])
        nh = n // 2
        e = face["eyes"][side]
        rim_delta = {}
        for k in range(1, nh):
            top, bot = layers[0][k], layers[0][n - k]
            dz = co[bot] - co[top]
            for L, f in zip(layers, (1.0, 0.93, 0.62, 0.25)):
                blink.setdefault(side, {})[L[k]] = dz * 0.86 * f
            for L, f in zip(layers, (1.0, 0.8, 0.4, 0.1)):
                blink[side][L[n - k]] = -dz * 0.14 * f
            rim_delta[top] = dz * 0.86
            rim_delta[bot] = -dz * 0.14
        for ring in e["wall"]:
            for k in range(1, n):
                blink[side][ring[k]] = blink[side].get(layers[0][k], V())
        rim = layers[0]
        for i in e["lash_verts"]:
            j = min(rim, key=lambda r: (co[r] - co[i]).length_squared)
            blink[side][i] = rim_delta.get(j, V())
    key("Blink_L", blink[1])
    key("Blink_R", blink[-1])
    both = dict(blink[1])
    both.update(blink[-1])
    key("Blink", both)

    # ---- mouth: jaw rotation + lip shaping
    mouth = features["mouth"]
    n = len(mouth[0])
    nh = n // 2
    lower, corner = set(), set()
    for L in mouth[:4]:
        lower.update(L[k] for k in range(nh + 1, n))
        corner.update((L[0], L[nh]))
    for ring in face["mouth_bag"]:
        lower.update(ring[k] for k in range(nh + 1, n))
        corner.update((ring[0], ring[nh]))
    upper_set = set()
    for L in mouth[:4]:
        upper_set.update(L[k] for k in range(1, nh))
    for ring in face["mouth_bag"]:
        upper_set.update(ring[k] for k in range(1, nh))
    head_skin = set()
    eye_like = set()
    for e in face["eyes"].values():
        eye_like.update(e["verts"])
        eye_like.update(e["lash_verts"])
    zm = MOUTH_C.z
    for v in me.vertices:
        p = co[v.index]
        if p.z > 1.33 and p.z < 1.47 and abs(p.x) < 0.085 and v.index not in eye_like:
            head_skin.add(v.index)

    def jaw_w(i):
        if i in lower:
            return 1.0
        if i in corner:
            return 0.5
        if i in upper_set:
            return 0.0
        p = co[i]
        if p.y > 0.02 or p.z < 1.352:
            return 0.0
        wz = smoothstep(zm + 0.003, zm - 0.010, p.z)
        wx = 1.0 - smoothstep(0.032, 0.070, abs(p.x))
        wy = 1.0 - smoothstep(-0.02, 0.02, p.y)
        return wz * wx * wy

    def mouth_w(i):
        p = co[i]
        d = math.hypot(p.x, (p.z - zm) * 1.3)
        return (1.0 - smoothstep(0.010, 0.026, d)) * (1.0 - smoothstep(-0.045, -0.035, p.y))

    def shape(jaw_deg, sx=1.0, fwd=0.0, corners_up=0.0):
        R = Matrix.Rotation(math.radians(jaw_deg), 3, "X")
        out = {}
        for i in head_skin:
            p = co[i]
            d = V()
            w = jaw_w(i)
            if w > 0:
                d += ((R @ (p - JAW_PIVOT)) + JAW_PIVOT - p) * w
            m = mouth_w(i)
            if m > 0:
                d.x += (sx - 1.0) * p.x * m
                d.y -= fwd * m
            if corners_up:
                for cx in (-MOUTH_HALF_W, MOUTH_HALF_W):
                    dc = math.hypot(p.x - cx, (p.z - zm) * 1.2)
                    wc = (1.0 - smoothstep(0.003, 0.018, dc)) * (1.0 - smoothstep(-0.045, -0.035, p.y))
                    d.z += corners_up * wc
                    d.x += math.copysign(corners_up * 0.4, cx) * wc
            if d.length_squared > 0:
                out[i] = d
        return out

    key("Mouth_A", shape(6.0, 1.04))
    key("Mouth_I", shape(2.0, 1.16, corners_up=0.0006))
    key("Mouth_U", shape(2.5, 0.72, fwd=0.0016))
    key("Mouth_E", shape(4.5, 1.10))
    key("Mouth_O", shape(6.0, 0.80, fwd=0.0010))
    key("Smile", shape(0.0, 1.04, corners_up=0.0022))


# =============================================================================
#  Materials and the eye texture
# =============================================================================

PALETTE = {
    "skin": (0.98, 0.84, 0.76),
    "top": (0.95, 0.95, 0.97),       # white sailor blouse
    "collar": (0.10, 0.13, 0.30),    # navy collar / cuffs
    "socks": (0.05, 0.05, 0.07),     # black thigh-highs
    "shoes": (0.25, 0.13, 0.07),     # brown loafers
    "eye": (1.0, 1.0, 1.0),
    "line": (0.10, 0.06, 0.06),      # lashes and brows
    "mouth": (0.45, 0.12, 0.12),
    "hair": (0.17, 0.17, 0.36),      # midnight indigo
    "skirt": (0.12, 0.15, 0.34),
    "ribbon": (0.80, 0.10, 0.14),
}


def srgb_to_linear(c):
    return tuple(((x + 0.055) / 1.055) ** 2.4 if x > 0.04045 else x / 12.92 for x in c)


def make_eye_image(size=512):
    import numpy as np
    u, v = np.meshgrid((np.arange(size) + 0.5) / size, (np.arange(size) + 0.5) / size)
    px = 1.5 / size

    def edge(d):                       # anti-aliased inside test for a signed distance
        return np.clip(0.5 - d / px, 0.0, 1.0)

    ru, rv = 0.165, 0.235
    r = np.sqrt(((u - 0.5) / ru) ** 2 + ((v - 0.5) / rv) ** 2)
    col = np.zeros((size, size, 3))
    col[:] = (0.97, 0.96, 0.98)
    sh = np.clip((v - 0.60) / 0.22, 0, 1)[..., None] ** 1.2          # lid shadow on the white
    col = col * (1 - sh * 0.35) + np.array((0.62, 0.66, 0.86)) * sh * 0.35
    t = np.clip((v - (0.5 - rv)) / (2 * rv), 0, 1)[..., None]
    light, dark = np.array((1.00, 0.78, 0.30)), np.array((0.50, 0.20, 0.06))
    iris = light * (1 - t) + dark * t
    iris *= (1.0 + 0.07 * np.sin(np.arctan2(v - 0.5, u - 0.5) * 26))[..., None]
    ring = np.clip((r - 0.78) / 0.22, 0, 1)[..., None]
    iris = iris * (1 - ring) + np.array((0.28, 0.10, 0.04)) * ring
    glow = (np.clip(1 - np.abs(r - 0.68) / 0.22, 0, 1) * np.clip((0.5 - v) / rv + 0.2, 0, 1))[..., None]
    iris = np.clip(iris + glow * np.array((0.35, 0.25, 0.08)), 0, 1)
    rp = np.sqrt(((u - 0.5) / (ru * 0.40)) ** 2 + ((v - 0.505) / (rv * 0.44)) ** 2)
    iris = iris * (1 - edge((rp - 1) * ru * 0.4)[..., None]) + np.array((0.12, 0.04, 0.02)) * edge((rp - 1) * ru * 0.4)[..., None]
    m = edge((r - 1.0) * ru)[..., None]
    col = col * (1 - m) + iris * m
    for (cx, cy, ax, ay) in ((0.5 - 0.36 * ru, 0.5 + 0.42 * rv, 0.30 * ru, 0.20 * rv),
                             (0.5 + 0.38 * ru, 0.5 - 0.48 * rv, 0.12 * ru, 0.09 * rv)):
        d = np.sqrt(((u - cx) / ax) ** 2 + ((v - cy) / ay) ** 2)
        hm = edge((d - 1) * ax)[..., None]
        col = col * (1 - hm) + hm
    rgba = np.concatenate([np.clip(col, 0, 1), np.ones((size, size, 1))], axis=2).astype(np.float32)
    img = bpy.data.images.new("EyeIris", size, size, alpha=False)
    img.pixels.foreach_set(rgba.ravel())      # byte image: values are sRGB
    img.pack()
    return img


def make_material(name, color, image=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    lin = srgb_to_linear(color)
    bsdf.inputs["Base Color"].default_value = (*lin, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.6
    m.diffuse_color = (*lin, 1.0)
    if image is not None:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = image
        tex.location = (-400, 200)
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        nt.nodes.active = tex
    return m


def assign_materials(ob, names):
    ob.data.materials.clear()
    for n in names:
        mat = bpy.data.materials.get("M_" + n)
        if mat is None:
            img = make_eye_image() if n == "eye" else None
            mat = make_material("M_" + n, PALETTE[n], img)
        ob.data.materials.append(mat)


def symmetrize(ob, tol=0.004):
    me = ob.data
    kd = KDTree(len(me.vertices))
    for v in me.vertices:
        kd.insert(v.co, v.index)
    kd.balance()
    worst = 0.0
    done = set()
    for v in me.vertices:
        if v.index in done:
            continue
        if abs(v.co.x) < 1e-5:
            worst = max(worst, abs(v.co.x))
            v.co.x = 0.0
            continue
        m = V((-v.co.x, v.co.y, v.co.z))
        co, j, d = kd.find(m)
        if j == v.index and abs(v.co.x) < 0.002:       # a centre-line vertex that drifted
            v.co.x = 0.0
            continue
        if d > tol or j == v.index:
            print("symmetry: no partner for", tuple(round(c, 4) for c in v.co), d)
            continue
        worst = max(worst, d)
        w = me.vertices[j]
        avg = V(((abs(v.co.x) + abs(w.co.x)) / 2, (v.co.y + w.co.y) / 2, (v.co.z + w.co.z) / 2))
        v.co = V((math.copysign(avg.x, v.co.x), avg.y, avg.z))
        w.co = V((math.copysign(avg.x, w.co.x), avg.y, avg.z))
        done.update((v.index, j))
    print("symmetrize: worst mismatch %.6f m" % worst)


def cage_to_object(cage, name):
    bm = bmesh.new()
    bv = [bm.verts.new(p) for p in cage.co]
    bm.verts.ensure_lookup_table()
    used = set()
    for f in cage.faces:
        used.update(f)
        try:
            bm.faces.new([bv[i] for i in f])
        except ValueError as e:
            print("face error", f, e)
    unused = [bv[i] for i in range(len(bv)) if i not in used]
    if unused:
        print("WARNING: %d unused cage verts" % len(unused))
    bm.edges.ensure_lookup_table()
    crease = bm.edges.layers.float.get("crease_edge") or bm.edges.layers.float.new("crease_edge")
    emap = {}
    for e in bm.edges:
        a, b = e.verts[0].index, e.verts[1].index
        emap[(min(a, b), max(a, b))] = e
    for key, val in cage.creases.items():
        if key in emap:
            emap[key][crease] = val
    for key in cage.seams:
        if key in emap:
            emap[key].seam = True
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    for gname, idx in cage.groups.items():
        vg = ob.vertex_groups.new(name=gname)
        vg.add([i for i in idx if i in used], 1.0, "REPLACE")
    return ob


def report_mesh(ob):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    nonman = [e for e in bm.edges if not e.is_manifold and not e.is_boundary]
    boundary = [e for e in bm.edges if e.is_boundary]
    ngons = [f for f in bm.faces if len(f.verts) != 4]
    poles = {}
    for v in bm.verts:
        if not v.is_boundary:
            poles[len(v.link_edges)] = poles.get(len(v.link_edges), 0) + 1
    print("[%s] verts %d faces %d tris %d | non-manifold %d boundary-edges %d non-quads %d | valence %s"
          % (ob.name, len(bm.verts), len(bm.faces), sum(len(f.verts) - 2 for f in bm.faces),
             len(nonman), len(boundary), len(ngons), dict(sorted(poles.items()))))
    bm.free()


def subdivide(ob, levels=1):
    mod = ob.modifiers.new("Subdivision", "SUBSURF")
    mod.levels = levels
    mod.render_levels = levels
    mod.boundary_smooth = "ALL"
    mod.use_creases = True
    mod.quality = 4
    with bpy.context.temp_override(object=ob, active_object=ob, selected_objects=[ob]):
        bpy.ops.object.modifier_apply(modifier=mod.name)
    ob.data.shade_smooth()


# =============================================================================
#  Scene
# =============================================================================

def clear_scene():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.armatures, bpy.data.images):
        for item in list(coll):
            coll.remove(item)


def export_fbx(path):
    bpy.ops.export_scene.fbx(filepath=path, use_selection=False, object_types={"ARMATURE", "MESH"},
                             use_mesh_modifiers=False, mesh_smooth_type="FACE", add_leaf_bones=False,
                             use_armature_deform_only=True, bake_anim=False, path_mode="COPY",
                             embed_textures=True, apply_scale_options="FBX_SCALE_ALL",
                             axis_forward="-Z", axis_up="Y")


def export_glb(path):
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", export_skins=True, export_morph=True,
                              export_animations=False, export_def_bones=True, export_apply=False)


def summary(objs):
    total = 0
    for ob in objs:
        tris = sum(len(p.vertices) - 2 for p in ob.data.polygons)
        total += tris
        print("  %-8s %6d tris" % (ob.name, tris))
    print("  %-8s %6d tris" % ("total", total))


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default=None)
    ap.add_argument("--fbx", default=None)
    ap.add_argument("--glb", default=None)
    ap.add_argument("--cage", action="store_true", help="keep the unsubdivided cage as an extra object")
    return ap.parse_args(argv)


def main():
    args = parse_args()
    clear_scene()
    cage, parts = build_cage()
    if args.cage:
        c_ob = cage_to_object(cage, "Cage")
        report_mesh(c_ob)
    body = cage_to_object(cage, "Body")
    report_mesh(body)
    subdivide(body)
    features = finish_head(body)
    symmetrize(body)
    report_mesh(body)
    assign_materials(body, sorted(MAT, key=MAT.get))
    body.data.uv_layers.new(name="UVMap")
    face = add_face_details(body, features)
    hair, locks = build_hair(body)
    report_mesh(hair)
    skirt = build_skirt()
    report_mesh(skirt)
    assign_outfit(body, parts)
    collar = build_collar(body)
    report_mesh(collar)
    unwrap_body(body, parts)
    unwrap_hair_cap(hair)
    arm, chains = build_armature(parts, face, locks)
    add_shape_keys(body, features, face)
    weight_body(body, arm, face)
    weight_hair(hair, arm, chains)
    weight_skirt(skirt, arm, chains)
    bow = {i for p in collar.data.polygons if p.material_index == 2 for i in p.vertices}
    weight_from_body(collar, body, arm, rigid=lambda v: {"UpperChest": 1.0} if v.index in bow else None)
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = arm
    summary([body, hair, skirt, collar])
    print("bones: %d deform / %d total" % (sum(b.use_deform for b in arm.data.bones), len(arm.data.bones)))
    if args.save:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save), compress=True)
        print("saved", args.save)
    if args.fbx:
        export_fbx(os.path.abspath(args.fbx))
        print("exported", args.fbx)
    if args.glb:
        export_glb(os.path.abspath(args.glb))
        print("exported", args.glb)


if __name__ == "__main__":
    main()

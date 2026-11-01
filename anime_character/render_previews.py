#!/usr/bin/env python3
"""
render_previews.py - toon-shaded and wireframe preview renders of the anime character.

    blender -b anime_character.blend -P render_previews.py -- OUTDIR [--pose] [--views a,b,c]

Toon look: emission-based cel shading (two tones + a rim light, light direction fixed in world
space) and inverted-hull outlines, so it renders the same in Cycles on a CPU-only machine.
Wireframe views use Workbench with a Wireframe-modifier copy of each mesh (shows real quads).
Nothing here is saved back into the .blend: the materials in the file stay plain Principled BSDF
so FBX/glTF exports carry clean base colours.
"""

import math
import os
import sys

import bpy
from mathutils import Matrix, Vector as V

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT = os.path.abspath(argv[0] if argv else "renders")
POSE = "--pose" in argv
VIEWS = None
if "--views" in argv:
    VIEWS = argv[argv.index("--views") + 1].split(",")
os.makedirs(OUT, exist_ok=True)

sc = bpy.context.scene
vl = bpy.context.view_layer
MESHES = [o for o in sc.objects if o.type == "MESH"]
arm = bpy.data.objects.get("Armature")
LIGHT = V((-0.45, -0.75, 0.55)).normalized()    # key light from the front-left, above


# ---------------------------------------------------------------------------- pose
def rot_world(name, axis, deg):
    pb = arm.pose.bones[name]
    M = pb.matrix.copy()
    c = M.translation.copy()
    pb.matrix = Matrix.Translation(c) @ Matrix.Rotation(math.radians(deg), 4, V(axis)) @ Matrix.Translation(-c) @ M
    vl.update()


def rot_local(name, axis_idx, deg):
    pb = arm.pose.bones[name]
    rot_world(name, pb.matrix.to_3x3().col[axis_idx].normalized(), deg)


def move(name, delta):
    pb = arm.pose.bones[name]
    pb.matrix = Matrix.Translation(V(delta)) @ pb.matrix
    vl.update()


def stress_pose():
    """Arm raised above the shoulder, elbow and knee bent hard, fist, twist, wink."""
    rot_world("Spine", (0, 0, 1), 6)
    rot_world("Chest", (0, 0, 1), 6)
    rot_world("Neck", (0, 0, 1), -8)
    rot_world("Head", (0, 0, 1), -10)
    rot_world("Head", (1, 0, 0), 6)
    rot_world("UpperArm.L", (0, 1, 0), 42)
    rot_world("UpperArm.L", (1, 0, 0), -18)
    rot_local("LowerArm.L", 0, 75)
    rot_world("Shoulder.R", (0, 1, 0), 12)
    rot_world("UpperArm.R", (0, 1, 0), 62)
    rot_local("LowerArm.R", 0, 35)
    for f in ("Index", "Middle", "Ring", "Pinky"):
        for k in (1, 2, 3):
            rot_local("%s_%d.R" % (f, k), 0, 70 if k == 1 else 80)
    rot_local("Thumb_2.R", 0, 30)
    rot_local("Thumb_3.R", 0, 40)
    move("IK_Foot.L", (0.0, -0.20, 0.32))
    rot_world("IK_Foot.L", (1, 0, 0), 25)
    kb = bpy.data.objects["Body"].data.shape_keys.key_blocks
    kb["Blink_L"].value = 1.0
    kb["Mouth_A"].value = 0.55


# ---------------------------------------------------------------------------- toon materials
def toon(mat):
    """Turn a Principled material into an emission cel shader (keeps its image texture)."""
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    if bsdf is None or mat.get("toon"):
        return
    mat["toon"] = True
    base = bsdf.inputs["Base Color"]
    src = base.links[0].from_socket if base.links else None
    col = tuple(base.default_value)
    N = nt.nodes
    L = nt.links
    out = N.get("Material Output")
    geo = N.new("ShaderNodeNewGeometry")
    dot = N.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    dot.inputs[1].default_value = LIGHT
    L.new(geo.outputs["Normal"], dot.inputs[0])
    ramp = N.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
    ramp.color_ramp.elements[1].position = 0.43
    ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
    remap = N.new("ShaderNodeMapRange")
    remap.inputs["From Min"].default_value = -1.0
    remap.inputs["From Max"].default_value = 1.0
    L.new(dot.outputs["Value"], remap.inputs["Value"])
    L.new(remap.outputs["Result"], ramp.inputs["Fac"])
    shadow = N.new("ShaderNodeMix")
    shadow.data_type = "RGBA"
    shadow.blend_type = "MULTIPLY"
    shadow.inputs["B"].default_value = (0.70, 0.64, 0.86, 1.0)
    if src is not None:
        L.new(src, shadow.inputs["A"])
    else:
        shadow.inputs["A"].default_value = col
    lit = N.new("ShaderNodeMix")
    lit.data_type = "RGBA"
    L.new(ramp.outputs["Color"], lit.inputs["Factor"])
    L.new(shadow.outputs["Result"], lit.inputs["A"])
    if src is not None:
        L.new(src, lit.inputs["B"])
    else:
        lit.inputs["B"].default_value = col
    # rim light on the lit side
    lw = N.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.18
    rim = N.new("ShaderNodeMath")
    rim.operation = "GREATER_THAN"
    rim.inputs[1].default_value = 0.55
    L.new(lw.outputs["Facing"], rim.inputs[0])
    rimmix = N.new("ShaderNodeMix")
    rimmix.data_type = "RGBA"
    rimmix.blend_type = "SCREEN"
    rimf = N.new("ShaderNodeMath")
    rimf.operation = "MULTIPLY"
    rimf.inputs[1].default_value = 0.10
    L.new(rim.outputs["Value"], rimf.inputs[0])
    L.new(rimf.outputs["Value"], rimmix.inputs["Factor"])
    L.new(lit.outputs["Result"], rimmix.inputs["A"])
    rimmix.inputs["B"].default_value = (1.0, 0.97, 0.92, 1.0)
    em = N.new("ShaderNodeEmission")
    if mat.name in ("M_eye", "M_line", "M_mouth"):          # unshaded
        if src is not None:
            L.new(src, em.inputs["Color"])
        else:
            em.inputs["Color"].default_value = col
    else:
        L.new(rimmix.outputs["Result"], em.inputs["Color"])
    L.new(em.outputs["Emission"], out.inputs["Surface"])


def outline_material():
    m = bpy.data.materials.get("Outline") or bpy.data.materials.new("Outline")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (0.045, 0.035, 0.06, 1)
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    # the hull is the mesh turned inside out and pushed outward: its near side is seen from
    # behind (transparent), its far side peeks out around the silhouette (the outline)
    nt.links.new(geo.outputs["Backfacing"], mix.inputs["Fac"])
    nt.links.new(em.outputs["Emission"], mix.inputs[1])
    nt.links.new(tr.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return m


def add_outlines():
    """Inverted-hull outlines: an inside-out copy of each mesh pushed outward after skinning
    (no surface coincides with the real mesh). Eyes, lashes and the mouth are left out."""
    import bmesh
    ol = outline_material()
    for ob in MESHES:
        cp = ob.copy()
        cp.data = ob.data.copy()
        cp.name = ob.name + "_outline"
        sc.collection.objects.link(cp)
        bm = bmesh.new()
        bm.from_mesh(cp.data)
        skip = {i for i, m in enumerate(ob.data.materials) if m and m.name in ("M_eye", "M_line", "M_mouth")}
        bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.material_index in skip], context="FACES")
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
        for f in bm.faces:
            f.material_index = 0
        bm.to_mesh(cp.data)
        bm.free()
        cp.data.materials.clear()
        cp.data.materials.append(ol)
        if cp.data.shape_keys and ob.data.shape_keys:
            for kb in ob.data.shape_keys.key_blocks:
                cp.data.shape_keys.key_blocks[kb.name].value = kb.value
        d = cp.modifiers.new("Outline", "DISPLACE")      # after the Armature modifier
        d.direction = "NORMAL"
        d.mid_level = 0.0
        d.strength = -(0.0016 if ob.name == "Hair" else 0.0011)


# ---------------------------------------------------------------------------- cameras
def camera(loc, target, lens=None, ortho=None):
    cam = sc.camera
    if cam is None:
        cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
        sc.collection.objects.link(cam)
        sc.camera = cam
    cam.location = V(loc)
    cam.rotation_euler = (V(target) - V(loc)).to_track_quat("-Z", "Y").to_euler()
    cam.data.clip_start = 0.01
    if ortho:
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = ortho
    else:
        cam.data.type = "PERSP"
        cam.data.lens = lens or 50
    return cam


TOON_VIEWS = {
    "hero_front": ((0.0, -4.2, 1.05), (0.0, 0.0, 0.84), 60, (1100, 1600)),
    "hero_34": ((1.9, -3.6, 1.25), (0.0, 0.0, 0.84), 60, (1100, 1600)),
    "hero_back": ((-1.2, 3.9, 1.20), (0.0, 0.0, 0.84), 60, (1100, 1600)),
    "face": ((0.24, -0.62, 1.50), (0.0, 0.0, 1.455), 80, (1200, 1200)),
    "posed": ((1.6, -3.4, 1.20), (0.0, 0.0, 0.88), 55, (1200, 1600)),
    "posed_face": ((0.12, -0.62, 1.47), (-0.01, 0.0, 1.445), 80, (1200, 1200)),
}
WIRE_VIEWS = {
    "wire_face": ((0.0, -3.0, 1.445), (0.0, 0.0, 1.445), 0.20, (1400, 1400)),
    "wire_head_side": ((3.0, 0.0, 1.45), (0.0, 0.0, 1.45), 0.30, (1400, 1400)),
    "wire_body_front": ((0.0, -3.0, 0.82), (0.0, 0.0, 0.82), 1.72, (1100, 1600)),
    "wire_body_back": ((0.0, 3.0, 0.82), (0.0, 0.0, 0.82), 1.72, (1100, 1600)),
    "wire_hand": ((0.80, -0.32, 1.12), (0.60, -0.02, 0.90), 45.0, (1400, 1400)),
    "wire_posed": ((1.6, -3.4, 1.20), (0.0, 0.0, 0.88), 55.0, (1200, 1600)),
}


def render(name, path_name=None):
    sc.render.filepath = os.path.join(OUT, (path_name or name) + ".png")
    bpy.ops.render.render(write_still=True)
    print("rendered", sc.render.filepath)


def toon_setup():
    sc.render.engine = "CYCLES"
    sc.cycles.samples = 24
    sc.cycles.use_denoising = False
    sc.cycles.transparent_max_bounces = 16
    sc.cycles.max_bounces = 2
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    w = sc.world or bpy.data.worlds.new("World")
    sc.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (0.86, 0.88, 0.93, 1)
    bg.inputs["Strength"].default_value = 1.0
    for ob in MESHES:
        for m in ob.data.materials:
            if m:
                toon(m)
    add_outlines()


def wire_setup():
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.world = sc.world or bpy.data.worlds.new("World")
    sc.world.color = (0.86, 0.88, 0.93)
    sc.display.shading.background_type = "WORLD"
    sh = sc.display.shading
    sh.light = "STUDIO"
    sh.color_type = "SINGLE"
    sh.single_color = (0.80, 0.78, 0.76)
    sh.show_cavity = False
    sc.display.render_aa = "8"
    sc.view_settings.view_transform = "Standard"
    for ob in list(sc.objects):
        if ob.name.endswith("_outline"):
            ob.hide_render = True
    wires = []
    for ob in MESHES:
        w = ob.copy()
        w.name = ob.name + "_wire"
        sc.collection.objects.link(w)
        wm = w.modifiers.new("wf", "WIREFRAME")
        wm.use_even_offset = True
        wm.offset = 1.0
        wm.use_replace = True
        wm.material_offset = 0
        wires.append(w)
    return wires


def main():
    if POSE:
        stress_pose()
    if VIEWS is None or any(v in TOON_VIEWS for v in VIEWS):
        toon_setup()
        for name, (loc, tgt, lens, res) in TOON_VIEWS.items():
            if VIEWS and name not in VIEWS:
                continue
            if name.startswith("posed") != POSE:
                continue
            camera(loc, tgt, lens=lens)
            sc.render.resolution_x, sc.render.resolution_y = res
            render(name)
    if VIEWS is None or any(v in WIRE_VIEWS for v in VIEWS):
        wires = wire_setup()
        # Workbench colours wires by the object's viewport colour: dark lines over the clay
        sc.display.shading.color_type = "OBJECT"
        for ob in MESHES:
            ob.color = (0.80, 0.78, 0.76, 1)
        for w in wires:
            w.color = (0.10, 0.12, 0.20, 1)
        for name, (loc, tgt, scale, res) in WIRE_VIEWS.items():
            if VIEWS and name not in VIEWS:
                continue
            if name.startswith("wire_posed") != POSE:
                continue
            persp = name in ("wire_hand", "wire_posed")
            camera(loc, tgt, lens=scale if persp else None, ortho=None if persp else scale)
            sc.render.resolution_x, sc.render.resolution_y = res
            px = ((V(tgt) - V(loc)).length * 36 / scale if persp else scale) / res[0]
            for w in wires:
                w.modifiers["wf"].thickness = px * 1.5
            render(name)


main()

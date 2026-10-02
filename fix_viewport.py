"""Patch for forest .blend files made before the viewport fix.

Those files show almost nothing in the 3D viewport (only the terrain, rocks and the one big
tree in the foreground), although renders were fine. Cause: the "viewport subset" switch in
each scatter node group was linked to the wrong output of its Random Value node.

How to use: open the .blend in Blender, go to the Scripting workspace, Text > Open... (or
New and paste), press Run Script, then save the file (Ctrl+S). Nothing is re-generated, so
it takes a second.
"""
import bpy

for ng in bpy.data.node_groups:
    if not ng.name.endswith("_Instancer"):
        continue
    rnd = next((n for n in ng.nodes if n.bl_idname == "FunctionNodeRandomValue"), None)
    orr = next((n for n in ng.nodes if n.bl_idname == "FunctionNodeBooleanMath" and n.operation == "OR"), None)
    if rnd is None or orr is None:      # fully visible in the viewport (e.g. rocks)
        continue
    for link in list(ng.links):
        if link.to_socket == orr.inputs[1]:
            ng.links.remove(link)
    ng.links.new(next(s for s in rnd.outputs if s.enabled and s.type == "BOOLEAN"), orr.inputs[1])
    if ng.name.startswith("Trees_"):
        rnd.inputs["Probability"].default_value = 0.08     # share of trees shown in the viewport
    print("fixed", ng.name)

for screen in bpy.data.screens:         # open on the camera's view of the shot
    for area in screen.areas:
        if area.type == "VIEW_3D":
            for space in area.spaces:
                if space.type == "VIEW_3D" and space.region_3d is not None:
                    space.region_3d.view_perspective = "CAMERA"

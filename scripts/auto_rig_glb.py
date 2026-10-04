"""Create a simple, skinned humanoid-style rig for a GLB inside Blender.

Run with:
    blender --background --factory-startup --python auto_rig_glb.py -- \
        --input source.glb --output rigged.glb

This is deliberately a lightweight template rig. It estimates an upright axis
from the longest dimension of the imported mesh and uses Blender's automatic
armature weights, with a nearest-bone fallback for problematic geometry.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import bpy
from mathutils import Vector


RIG_NAME = "TRELLIS_Rig"


def _parse_args():
    blender_args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Rig a generated TRELLIS GLB")
    parser.add_argument("--input", required=True, help="Input GLB file")
    parser.add_argument("--output", required=True, help="Rigged output GLB file")
    return parser.parse_args(blender_args)


def _get_mesh_objects():
    meshes = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and len(obj.data.vertices) > 0
    ]
    if not meshes:
        raise RuntimeError("El GLB no contiene objetos de malla con vértices.")
    return meshes


def _world_bounds(meshes):
    points = [obj.matrix_world @ vertex.co for obj in meshes for vertex in obj.data.vertices]
    if not points:
        raise RuntimeError("No se pudieron leer vértices del GLB.")

    low = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    high = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    size = high - low
    vertical_axis = max(range(3), key=lambda axis: size[axis])
    if size[vertical_axis] <= 1e-8:
        raise RuntimeError("La malla tiene dimensiones insuficientes para crear un rig.")

    remaining_axes = [axis for axis in range(3) if axis != vertical_axis]
    side_axis, depth_axis = sorted(remaining_axes, key=lambda axis: size[axis], reverse=True)
    center = (low + high) * 0.5
    return low, high, size, center, vertical_axis, side_axis, depth_axis


def _bone_specs(bounds):
    low, _high, size, center, vertical_axis, side_axis, depth_axis = bounds
    height = size[vertical_axis]
    side_scale = max(size[side_axis], height * 0.04)
    depth_scale = max(size[depth_axis], height * 0.04)

    def point(height_ratio, side_ratio=0.0, depth_ratio=0.0):
        coords = list(center)
        coords[vertical_axis] = low[vertical_axis] + height_ratio * height
        coords[side_axis] = center[side_axis] + side_ratio * side_scale
        coords[depth_axis] = center[depth_axis] + depth_ratio * depth_scale
        return Vector(coords)

    specs = []

    def add(name, head, tail, parent):
        specs.append((name, point(*head), point(*tail), parent))

    # Central chain. Fractions run from the lower to the upper end of the
    # longest bounding-box axis; these serve as stable default proportions.
    add("root", (0.02,), (0.47,), None)
    add("pelvis", (0.46,), (0.56,), "root")
    add("spine", (0.55,), (0.69,), "pelvis")
    add("chest", (0.69,), (0.79,), "spine")
    add("neck", (0.79,), (0.88,), "chest")
    add("head", (0.88,), (0.99,), "neck")

    for side_name, sign in (("L", 1.0), ("R", -1.0)):
        # Arms branch from the upper torso.
        add("clavicle." + side_name, (0.76,), (0.76, sign * 0.12), "chest")
        add(
            "upper_arm." + side_name,
            (0.76, sign * 0.12),
            (0.63, sign * 0.25),
            "clavicle." + side_name,
        )
        add(
            "forearm." + side_name,
            (0.63, sign * 0.25),
            (0.51, sign * 0.34),
            "upper_arm." + side_name,
        )
        add(
            "hand." + side_name,
            (0.51, sign * 0.34),
            (0.48, sign * 0.40),
            "forearm." + side_name,
        )

        # Legs branch from the pelvis. The foot points along the smaller
        # horizontal dimension; its sign is intentionally just a default.
        add(
            "thigh." + side_name,
            (0.50, sign * 0.11),
            (0.29, sign * 0.12),
            "pelvis",
        )
        add(
            "shin." + side_name,
            (0.29, sign * 0.12),
            (0.08, sign * 0.12),
            "thigh." + side_name,
        )
        add(
            "foot." + side_name,
            (0.08, sign * 0.12),
            (0.04, sign * 0.12, 0.36),
            "shin." + side_name,
        )

    return specs, height


def _create_armature(bounds):
    specs, height = _bone_specs(bounds)
    armature_data = bpy.data.armatures.new(RIG_NAME)
    armature = bpy.data.objects.new(RIG_NAME, armature_data)
    bpy.context.scene.collection.objects.link(armature)
    armature.show_in_front = True

    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT")

    edit_bones = {}
    for name, head, tail, parent_name in specs:
        bone = armature_data.edit_bones.new(name)
        bone.head = head
        bone.tail = tail
        if (bone.tail - bone.head).length < height * 1e-4:
            # Degenerate dimensions can occur for flat or stylized models.
            bone.tail = bone.head + Vector((0.0, 0.0, height * 0.01))
        bone.use_deform = True
        edit_bones[name] = bone
        if parent_name is not None:
            bone.parent = edit_bones[parent_name]
            bone.use_connect = False

    bpy.ops.object.mode_set(mode="OBJECT")
    return armature


def _clear_selection():
    for obj in bpy.context.selected_objects:
        obj.select_set(False)


def _has_complete_weights(mesh):
    return bool(mesh.data.vertices) and all(vertex.groups for vertex in mesh.data.vertices)


def _reset_mesh_skinning(mesh, armature):
    world_matrix = mesh.matrix_world.copy()
    mesh.parent = None
    mesh.matrix_world = world_matrix
    for modifier in list(mesh.modifiers):
        if modifier.type == "ARMATURE" and modifier.object == armature:
            mesh.modifiers.remove(modifier)
    for group in list(mesh.vertex_groups):
        mesh.vertex_groups.remove(group)


def _bone_segments_world(armature):
    matrix = armature.matrix_world
    return [
        (bone.name, matrix @ bone.head_local, matrix @ bone.tail_local)
        for bone in armature.data.bones
        if bone.use_deform
    ]


def _point_segment_distance(point, start, end):
    segment = end - start
    length_squared = segment.length_squared
    if length_squared <= 1e-12:
        return (point - start).length
    factor = max(0.0, min(1.0, (point - start).dot(segment) / length_squared))
    return (point - (start + factor * segment)).length


def _add_fallback_weights(mesh, armature, height):
    """Assign smooth-ish normalized weights to the four nearest bone segments."""
    _reset_mesh_skinning(mesh, armature)
    world_matrix = mesh.matrix_world.copy()
    mesh.parent = armature
    mesh.matrix_world = world_matrix
    modifier = mesh.modifiers.new(name="TRELLIS_Armature", type="ARMATURE")
    modifier.object = armature
    modifier.use_vertex_groups = True

    segments = _bone_segments_world(armature)
    groups = {name: mesh.vertex_groups.new(name=name) for name, _head, _tail in segments}
    # Quantize to 8-bit weights so Blender receives a few batched group.add calls
    # instead of one Python-to-Blender call per vertex and bone.
    buckets = {name: {} for name, _head, _tail in segments}
    epsilon = max(height * 0.003, 1e-6)

    for vertex in mesh.data.vertices:
        world_point = mesh.matrix_world @ vertex.co
        nearest = sorted(
            (
                _point_segment_distance(world_point, start, end),
                name,
            )
            for name, start, end in segments
        )[:4]
        inverse_distances = [1.0 / max(distance, epsilon) ** 2 for distance, _name in nearest]
        total = sum(inverse_distances) or 1.0
        for (distance, name), inverse_distance in zip(nearest, inverse_distances):
            weight = inverse_distance / total
            quantized = max(1, min(255, int(round(weight * 255))))
            buckets[name].setdefault(quantized, []).append(vertex.index)

    for name, weight_buckets in buckets.items():
        group = groups[name]
        for quantized, indices in weight_buckets.items():
            group.add(indices, quantized / 255.0, "REPLACE")

    if not _has_complete_weights(mesh):
        raise RuntimeError("El cálculo de respaldo dejó vértices sin pesos.")


def _skin_meshes(meshes, armature, height):
    for mesh in meshes:
        _clear_selection()
        armature.select_set(True)
        mesh.select_set(True)
        bpy.context.view_layer.objects.active = armature

        try:
            result = bpy.ops.object.parent_set(type="ARMATURE_AUTO")
            success = "FINISHED" in result and mesh.parent == armature and _has_complete_weights(mesh)
        except Exception as exc:  # Blender can fail on non-manifold / dense meshes.
            print(f"[auto-rig] Pesos automáticos fallaron en {mesh.name}: {exc}")
            success = False

        if success:
            print(f"[auto-rig] Pesos automáticos asignados a {mesh.name}.")
            continue

        print(f"[auto-rig] Usando pesos de respaldo para {mesh.name}.")
        _add_fallback_weights(mesh, armature, height)


def _export_rigged_glb(meshes, armature, output_path):
    _clear_selection()
    for mesh in meshes:
        mesh.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.context.view_layer.update()

    result = bpy.ops.export_scene.gltf(
        filepath=str(output_path),
        export_format="GLB",
        use_selection=True,
        export_skins=True,
        export_animations=False,
    )
    if "FINISHED" not in result:
        raise RuntimeError(f"El exportador glTF de Blender no terminó correctamente: {result}")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("El exportador de Blender no generó el archivo GLB de salida.")


def run(input_path, output_path):
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".glb":
        raise ValueError(f"El archivo de entrada no es un GLB existente: {source}")
    if destination.suffix.lower() != ".glb" or source == destination:
        raise ValueError("La salida debe ser un archivo .glb distinto de la entrada.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    imported = bpy.ops.import_scene.gltf(filepath=str(source))
    if "FINISHED" not in imported:
        raise RuntimeError(f"No se pudo importar {source}: {imported}")

    meshes = _get_mesh_objects()
    bounds = _world_bounds(meshes)
    _low, _high, _size, _center, vertical_axis, _side_axis, _depth_axis = bounds
    axis_names = ("X", "Y", "Z")
    print(f"[auto-rig] Eje vertical estimado: {axis_names[vertical_axis]}.")

    armature = _create_armature(bounds)
    _specs, height = _bone_specs(bounds)
    _skin_meshes(meshes, armature, height)
    _export_rigged_glb(meshes, armature, destination)
    print(f"[auto-rig] GLB riggeado guardado en: {destination}")


def main():
    args = _parse_args()
    try:
        run(args.input, args.output)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()

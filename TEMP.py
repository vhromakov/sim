#!/usr/bin/env python3
"""
Read an input mesh, split it into connected components, make each component
watertight, then boolean-union all components into a single watertight mesh.

Dependencies:
    pip install trimesh point-cloud-utils pyvista

Usage:
    python mesh_heal_and_union.py INPUT.stl OUTPUT.stl
"""

import argparse
import numpy as np
import trimesh
import point_cloud_utils as pcu
import pyvista as pv


# ----------------- Helpers -----------------


def make_component_watertight(tm: trimesh.Trimesh,
                              resolution: int = 50_000) -> trimesh.Trimesh:
    """
    Make a single component watertight using point_cloud_utils.make_mesh_watertight.
    """
    v = np.asarray(tm.vertices, dtype=float)
    f = np.asarray(tm.faces, dtype=np.int64)

    if len(f) == 0:
        # Nothing to heal
        return tm.copy()

    v = np.ascontiguousarray(v)
    f = np.ascontiguousarray(f)

    vw, fw = pcu.make_mesh_watertight(v, f, resolution)
    healed = trimesh.Trimesh(vw, fw, process=False)
    return healed


def trimesh_to_pv(tm: trimesh.Trimesh) -> pv.PolyData:
    """
    Convert a trimesh.Trimesh to pyvista.PolyData for boolean operations.
    """
    verts = np.asarray(tm.vertices, dtype=float)
    faces = np.asarray(tm.faces, dtype=np.int64)

    if faces.size == 0:
        return pv.PolyData(verts)

    n_faces = faces.shape[0]
    # PyVista: [3, i0, i1, i2, 3, j0, j1, j2, ...]
    faces_pv = np.hstack(
        [np.full((n_faces, 1), 3, dtype=np.int64), faces]
    ).ravel()

    return pv.PolyData(verts, faces_pv)


def pv_to_trimesh(poly: pv.PolyData) -> trimesh.Trimesh:
    """
    Convert pyvista.PolyData back to trimesh.Trimesh.
    """
    verts = np.asarray(poly.points, dtype=float)
    faces = np.asarray(poly.faces, dtype=np.int64)

    if faces.size == 0:
        return trimesh.Trimesh(
            vertices=verts,
            faces=np.zeros((0, 3), dtype=np.int64),
            process=False,
        )

    faces = faces.reshape(-1, 4)[:, 1:]
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def boolean_union_with_pyvista(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    """
    Boolean-union a list of trimesh meshes using PyVista's boolean_union.
    """
    if not meshes:
        raise ValueError("boolean_union_with_pyvista: no meshes provided")

    print(f"[STEP] Boolean union of {len(meshes)} components with PyVista...")

    pv_union = trimesh_to_pv(meshes[0])

    for i, tm in enumerate(meshes[1:], start=1):
        pv_m = trimesh_to_pv(tm)
        print(
            f"   - Union with component {i}: "
            f"verts={len(tm.vertices)}, faces={len(tm.faces)}"
        )

        # boolean_union returns a new PolyData
        pv_union = pv_union.boolean_union(pv_m, progress_bar=False)

        if pv_union is None or pv_union.n_points == 0:
            raise RuntimeError(f"PyVista boolean_union failed at step {i}")

    union_tm = pv_to_trimesh(pv_union)
    union_tm.remove_unreferenced_vertices()
    return union_tm


# ----------------- Main pipeline -----------------


def process_mesh(input_path: str,
                 output_path: str,
                 watertight_resolution: int = 50_000) -> None:
    print(f"[INFO] Loading mesh: {input_path}")

    # Load without processing so we can see original stats
    mesh = trimesh.load(input_path, process=False)

    if isinstance(mesh, trimesh.Scene):
        print("[INFO] Input is a Scene, merging geometry into a single mesh...")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    print(
        f"[INFO] Loaded mesh: "
        f"verts={len(mesh.vertices)}, faces={len(mesh.faces)}, "
        f"watertight={mesh.is_watertight}"
    )

    # Build a processed version to weld duplicate vertices
    print("[STEP] Merging duplicate vertices to build connectivity graph...")
    mesh_proc = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        process=True,  # welds vertices & builds adjacency
    )

    print(
        f"[INFO] After merge: "
        f"verts={len(mesh_proc.vertices)}, faces={len(mesh_proc.faces)}, "
        f"watertight={mesh_proc.is_watertight}"
    )

    print("[STEP] Splitting into connected components...")
    components = mesh_proc.split(only_watertight=False)
    print(f"[INFO] Found {len(components)} components")

    healed_components: list[trimesh.Trimesh] = []

    for idx, comp in enumerate(components):
        print(
            f"[STEP] Component {idx}: "
            f"verts={len(comp.vertices)}, faces={len(comp.faces)}, "
            f"watertight={comp.is_watertight}"
        )

        try:
            healed = make_component_watertight(
                comp,
                resolution=watertight_resolution,
            )
            print(
                f"   -> Healed component {idx}: "
                f"verts={len(healed.vertices)}, faces={len(healed.faces)}, "
                f"watertight={healed.is_watertight}"
            )
        except Exception as e:
            print(
                f"   !! Failed to make component {idx} watertight, "
                f"using original. Error: {e}"
            )
            healed = comp

        healed_components.append(healed)

    if len(healed_components) == 1:
        print("[STEP] Only one component; skipping boolean union.")
        union_tm = healed_components[0]
    else:
        print("[STEP] Performing boolean union of all healed components...")
        union_tm = boolean_union_with_pyvista(healed_components)

    union_tm.remove_unreferenced_vertices()

    print(
        f"[RESULT] Union mesh: "
        f"verts={len(union_tm.vertices)}, faces={len(union_tm.faces)}, "
        f"watertight={union_tm.is_watertight}"
    )

    print(f"[STEP] Exporting union mesh to: {output_path}")
    union_tm.export(output_path)
    print("[DONE]")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Split mesh into components, make each watertight, "
            "and boolean-union them into a single mesh."
        )
    )
    parser.add_argument("input", help="Input mesh file (STL/OBJ/PLY/...)")
    parser.add_argument("output", help="Output mesh file (STL/OBJ/PLY/...)")
    parser.add_argument(
        "--resolution",
        type=int,
        default=50_000,
        help=(
            "Resolution for point_cloud_utils.make_mesh_watertight "
            "(default: 50000)"
        ),
    )

    args = parser.parse_args()
    process_mesh(args.input, args.output, watertight_resolution=args.resolution)


if __name__ == "__main__":
    main()

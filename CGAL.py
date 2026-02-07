#!/usr/bin/env python3
"""
Hardcoded STL -> two outputs:
1) point_cloud_utils.make_mesh_watertight (saved via trimesh)
2) PyMeshLab Alpha Wrap (generate_alpha_wrap)
"""

import os
import numpy as np

import point_cloud_utils as pcu
import trimesh
import pymeshlab
import pymeshfix as pfix


# ---- hardcoded paths ----
IN_STL = r"MODELS/CSC16_U00P_.stl"
OUT_PCU = r"OUT_watertight_pcu.stl"
OUT_ALPHAWRAP = r"OUT_watertight_alphawrap.stl"


def watertight_with_pcu(in_stl: str, out_stl: str, resolution: int = 50_000) -> None:
    mesh = trimesh.load(in_stl)
    v = mesh.vertices.astype(np.float64)
    f = mesh.faces.astype(np.int64)
    vw, fw = pcu.make_mesh_watertight(v, f, resolution)

    m = trimesh.Trimesh(vertices=vw, faces=fw, process=False)

    # Convert to pymeshfix format
    v, f = m.vertices, m.faces
    tin = pfix.MeshFix(v, f)

    # Repair the mesh
    # This operation removes singularities, self-intersections, and degenerate elements
    tin.repair()

    # Get the cleaned vertices and faces
    v_out, f_out = tin.v, tin.f

    # Optionally, save the result
    cleaned_mesh = trimesh.Trimesh(vertices=v_out, faces=f_out)
    cleaned_mesh.export('cleaned_mesh.stl')


    m.export(out_stl)
    print(f"[PCU] wrote: {out_stl}  (verts={len(vw)}, faces={len(fw)})")


def main() -> None:
    if not os.path.isfile(IN_STL):
        raise FileNotFoundError(f"Input STL not found: {IN_STL}")

    watertight_with_pcu(IN_STL, OUT_PCU, resolution=50_000)


if __name__ == "__main__":
    main()

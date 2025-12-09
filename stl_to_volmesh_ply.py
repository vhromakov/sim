#!/usr/bin/env python3
"""
STL -> TetGen tetra mesh -> split into 3 vertical regions by Z
using *geometric clipping with boxes*, so region boundaries are
exactly planar.

Uses: trimesh, tetgen, pyvista
"""

import numpy as np
import trimesh
import tetgen
import pyvista as pv


def main():
    # 1) Load STL as surface mesh
    stl_path = "MODELS/CSC16_U00P_0.2_remesh_.stl"
    # stl_path = "MODELS/sphere.stl"
    # stl_path = "MODELS/torus.stl"
    mesh = trimesh.load(stl_path)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump().sum()

    print(f"[INFO] Loaded STL: {stl_path}")
    print(f"[INFO] Vertices: {mesh.vertices.shape[0]}, Faces: {mesh.faces.shape[0]}")

    # 2) Tetrahedralize with TetGen
    t = tetgen.TetGen(mesh.vertices, mesh.faces)
    t.tetrahedralize(
        plc=True,
        quality=True,
        steinerleft=-1,   # you found this helpful for nicer tets
    )
    grid = t.grid  # pyvista.UnstructuredGrid
    print(grid)

    # 3) Compute Z-split positions from the *tet mesh* bounds
    xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds
    dz = (zmax - zmin) / 3.0
    z1 = zmin + dz
    z2 = zmin + 2.0 * dz

    print(f"[INFO] Z bounds: {zmin:.3f} .. {zmax:.3f}")
    print(f"[INFO] Split planes at z1={z1:.3f}, z2={z2:.3f}")

    # 4) Clip with axis-aligned boxes to get 3 slabs with planar boundaries

    # bottom:  z in [zmin, z1]
    bounds_bottom = (xmin, xmax, ymin, ymax, zmin, z1)
    bottom = grid.clip_box(bounds_bottom, invert=False)

    # middle: z in [z1, z2]
    bounds_middle = (xmin, xmax, ymin, ymax, z1, z2)
    middle = grid.clip_box(bounds_middle, invert=False)

    # top:    z in [z2, zmax]
    bounds_top = (xmin, xmax, ymin, ymax, z2, zmax)
    top = grid.clip_box(    , invert=False)

    print("[INFO] Cells per slab:",
          bottom.n_cells, middle.n_cells, top.n_cells)

    # 5) Visualize: also clip in X just to expose internals
    # (optional but helps to see inside)

    bottom_clip = bottom.clip(normal="x", origin=bottom.center)
    middle_clip = middle.clip(normal="x", origin=middle.center)
    top_clip = top.clip(normal="x", origin=top.center)

    p = pv.Plotter()
    p.add_mesh(bottom_clip, color="red",   show_edges=True, opacity=0.9)
    p.add_mesh(middle_clip, color="green", show_edges=True, opacity=0.9)
    p.add_mesh(top_clip,    color="blue",  show_edges=True, opacity=0.9)
    p.add_axes()
    p.show(title="Three Z-slabs with planar boundaries")


if __name__ == "__main__":
    main()

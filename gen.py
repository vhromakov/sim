#!/usr/bin/env python3
"""
Simplest STL -> voxelized STL converter.

Pipeline:
1. Load input STL
2. Move mesh so its min corner = (0,0,0)
3. Voxelize
4. Convert voxels to box-mesh
5. Move voxelized mesh back to original position
6. Export STL
"""

import argparse
import trimesh
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Voxelize an STL and export the voxelized STL."
    )
    parser.add_argument("input_stl", help="Path to input STL file")
    parser.add_argument("output_stl", help="Path to output voxelized STL")
    parser.add_argument(
        "--pitch", "-p",
        type=float,
        required=True,
        help="Voxel cube size (same units as STL)"
    )
    args = parser.parse_args()

    # -----------------------------
    # Load mesh
    # -----------------------------
    mesh = trimesh.load(args.input_stl)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    print(f"[LOAD] Loaded {args.input_stl}")
    print(f"[LOAD] BBox before: {mesh.bounds}")

    # -----------------------------
    # Move mesh so min-corner becomes (0,0,0)
    # -----------------------------
    min_corner = mesh.bounds[0].copy()
    min_corner[2] += args.pitch / 2
    mesh.apply_translation(-min_corner)

    print(f"[MOVE] Translation applied: {-min_corner}")
    print(f"[MOVE] BBox now: {mesh.bounds}")

    # -----------------------------
    # Voxelize
    # -----------------------------
    print(f"[VOX] Voxelizing with pitch = {args.pitch} ...")
    vox = mesh.voxelized(pitch=args.pitch)
    vox.fill()

    print(f"[VOX] Grid shape: {vox.shape}")
    print(f"[VOX] Filled cells: {len(vox.sparse_indices)}")

    # -----------------------------
    # Convert voxel grid into box mesh
    # -----------------------------
    vox_mesh = vox.as_boxes()
    print(f"[VOX] Voxel-mesh: {len(vox_mesh.vertices)} vertices, {len(vox_mesh.faces)} faces")

    # -----------------------------
    # Move voxel mesh back to original place
    # -----------------------------
    vox_mesh.apply_translation(min_corner)

    print(f"[MOVE] Voxel mesh moved back by: {min_corner}")
    print(f"[MOVE] Output bbox: {vox_mesh.bounds}")

    # -----------------------------
    # Export
    # -----------------------------
    vox_mesh.export(args.output_stl)
    print(f"[SAVE] Voxelized STL saved to {args.output_stl}")


if __name__ == "__main__":
    main()

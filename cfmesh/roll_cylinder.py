#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import trimesh


def _axis_indices(axis: str) -> tuple[int, tuple[int, int]]:
    axis = axis.lower()
    if axis == "x":
        return 0, (1, 2)  # radial plane YZ
    if axis == "y":
        return 1, (0, 2)  # radial plane XZ
    if axis == "z":
        return 2, (0, 1)  # radial plane XY
    raise ValueError("axis must be one of: x, y, z")


def roll_cylinder(
    *,
    input_unrolled_stl: str,
    meta_json: str,
    output_stl: str,
) -> None:
    meta = json.loads(Path(meta_json).read_text(encoding="utf-8"))

    # Validate minimally
    for k in ["R", "axis", "bottom_axis", "center_radial_plane", "phi_ref", "theta_range", "embedding"]:
        if k not in meta:
            raise ValueError(f"Missing key in metadata: {k}")

    R = float(meta["R"])
    axis = str(meta["axis"]).lower()
    phi_ref = float(meta["phi_ref"])
    center = meta["center_radial_plane"]
    c0, c1 = float(center[0]), float(center[1])

    if R <= 0:
        raise ValueError("Metadata R must be > 0")

    axis_i, (p0_i, p1_i) = _axis_indices(axis)

    mesh = trimesh.load(input_unrolled_stl, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Failed to load a triangle mesh from input STL")

    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)

    # The unrolled embedding is always:
    #   X = s
    #   Y = h (axis coordinate)
    #   Z = offset = R - r
    s = V[:, 0]
    h = V[:, 1]
    offset = V[:, 2]

    theta = s / R
    r = R - offset

    # Reconstruct radial-plane coordinates
    ang = theta + phi_ref
    u = r * np.cos(ang)
    v = r * np.sin(ang)

    Vrec = np.zeros_like(V)
    Vrec[:, axis_i] = h
    Vrec[:, p0_i] = c0 + u
    Vrec[:, p1_i] = c1 + v

    out = trimesh.Trimesh(vertices=Vrec, faces=F, process=False)
    out.export(output_stl)

    print("[OK] wrote:", output_stl)
    print(f"  R={R} axis={axis.upper()} center(radial plane)=({c0:.6g},{c1:.6g}) phi_ref={phi_ref:.12g}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_unrolled", help="unrolled STL produced by unroll_cylinder.py")
    ap.add_argument("meta_json", help="JSON produced by unroll_cylinder.py")
    ap.add_argument("output", help="output STL (rolled back)")
    args = ap.parse_args()

    roll_cylinder(
        input_unrolled_stl=args.input_unrolled,
        meta_json=args.meta_json,
        output_stl=args.output,
    )


if __name__ == "__main__":
    main()

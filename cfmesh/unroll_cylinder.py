#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Tuple, Optional

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


def _idx(name: str) -> int:
    name = name.lower()
    if name == "x":
        return 0
    if name == "y":
        return 1
    if name == "z":
        return 2
    raise ValueError("bottom-axis must be one of: x, y, z")


def _wrap_signed(theta: np.ndarray) -> np.ndarray:
    # (-pi, pi]
    return np.arctan2(np.sin(theta), np.cos(theta))


def unroll_cylinder(
    *,
    input_stl: str,
    output_stl: str,
    R: float,
    axis: str = "y",
    bottom_axis: str = "z",
    center: Optional[Tuple[float, float]] = None,
    theta_range: str = "signed",  # signed | 0_2pi | raw
    meta_path: Optional[str] = None,
) -> None:
    if R <= 0:
        raise ValueError("R must be > 0")

    mesh = trimesh.load(input_stl, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Failed to load a triangle mesh from input STL")

    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)

    axis_i, (p0_i, p1_i) = _axis_indices(axis)
    bottom_i = _idx(bottom_axis)
    if bottom_i == axis_i:
        raise ValueError("bottom-axis must be perpendicular to cylinder axis")

    # anchor = bottom-most vertex along bottom_axis
    anchor = V[np.argmin(V[:, bottom_i])]
    a0, a1 = float(anchor[p0_i]), float(anchor[p1_i])

    # Determine cylinder center in the radial plane.
    # If center not provided, enforce the diagram rule:
    # "vertical arrow up from bottom point hits cylinder axis"
    # => center = anchor + R * (+bottom_axis direction) in the radial plane.
    if center is None:
        if bottom_i == p0_i:
            c0, c1 = a0 + R, a1
        elif bottom_i == p1_i:
            c0, c1 = a0, a1 + R
        else:
            raise ValueError("bottom-axis must lie in the cylinder radial plane")
    else:
        c0, c1 = float(center[0]), float(center[1])

    # phi_ref: angle of vector (center -> anchor); makes anchor map to theta=0 (s=0)
    du0 = a0 - c0
    du1 = a1 - c1
    if math.hypot(du0, du1) <= 1e-12:
        raise ValueError("Anchor coincides with cylinder center (bad center/R or mesh)")
    phi_ref = math.atan2(du1, du0)

    # Cylindrical coords
    u = V[:, p0_i] - c0
    v = V[:, p1_i] - c1
    r = np.sqrt(u * u + v * v)

    theta = np.arctan2(v, u) - phi_ref
    if theta_range == "signed":
        theta = _wrap_signed(theta)
    elif theta_range == "0_2pi":
        theta = np.mod(theta, 2.0 * np.pi)
    elif theta_range == "raw":
        pass
    else:
        raise ValueError("theta_range must be: signed | 0_2pi | raw")

    h = V[:, axis_i]
    s = theta * R
    offset = (R - r)

    # Embed into XYZ for STL: X'=s, Y'=h, Z'=offset
    Vout = np.zeros_like(V)
    Vout[:, 0] = s
    Vout[:, 1] = h
    Vout[:, 2] = offset

    out = trimesh.Trimesh(vertices=Vout, faces=F, process=False)
    out.export(output_stl)

    # Write metadata JSON for perfect inversion
    if meta_path is None:
        meta_path = str(Path(output_stl).with_suffix(".json"))

    meta = {
        "version": 1,
        "R": float(R),
        "axis": axis.lower(),
        "bottom_axis": bottom_axis.lower(),
        "center_radial_plane": [float(c0), float(c1)],  # (cx,cz) if axis=y; (cx,cy) if axis=z; (cy,cz) if axis=x
        "phi_ref": float(phi_ref),
        "theta_range": theta_range,
        "embedding": {"s_axis": 0, "h_axis": 1, "offset_axis": 2, "meaning": "X=s, Y=h(axis coord), Z=R-r"},
        "debug": {
            "anchor_xyz": [float(anchor[0]), float(anchor[1]), float(anchor[2])],
            "anchor_radial_distance": float(math.hypot(a0 - c0, a1 - c1)),
            "radial_plane_indices": [p0_i, p1_i],
            "axis_index": axis_i,
        },
    }

    Path(meta_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[OK] wrote:", output_stl)
    print("[OK] wrote:", meta_path)
    print(f"  R={R} axis={axis.upper()} bottom_axis={bottom_axis.upper()} theta_range={theta_range}")
    print(f"  center(radial plane) = ({c0:.6g}, {c1:.6g}), phi_ref={phi_ref:.12g}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="input STL")
    ap.add_argument("output", help="output unrolled STL")
    ap.add_argument("--R", type=float, required=True, help="cylinder radius")
    ap.add_argument("--axis", choices=["x", "y", "z"], default="y", help="cylinder axis (default: y)")
    ap.add_argument("--bottom-axis", choices=["x", "y", "z"], default="z",
                    help="axis used for bottom-most point & 'up to center' rule (default: z)")
    ap.add_argument("--center", nargs=2, type=float, default=None, metavar=("C0", "C1"),
                    help="override cylinder center in radial plane. For axis=y pass (cx, cz).")
    ap.add_argument("--theta-range", choices=["signed", "0_2pi", "raw"], default="signed",
                    help="theta wrapping used during unroll (default: signed)")
    ap.add_argument("--meta", default=None, help="output metadata JSON path (default: output.stl -> output.json)")
    args = ap.parse_args()

    unroll_cylinder(
        input_stl=args.input,
        output_stl=args.output,
        R=args.R,
        axis=args.axis,
        bottom_axis=args.bottom_axis,
        center=tuple(args.center) if args.center is not None else None,
        theta_range=args.theta_range,
        meta_path=args.meta,
    )


if __name__ == "__main__":
    main()

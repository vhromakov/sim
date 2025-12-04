#!/usr/bin/env python3
import argparse
import json
import numpy as np
import trimesh
from pygem import FFD


def load_lattice_json(path: str) -> np.ndarray:
    """
    Load a lattice JSON written by export_three_bbox_ffd_lattices_json.

    Returns:
        ctrl_points: np.ndarray, shape (nx, ny, nz, 3)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nx = int(data["nx"])
    ny = int(data["ny"])
    nz = int(data["nz"])
    pts = np.array(data["points"], dtype=float)

    if pts.shape != (nx, ny, nz, 3):
        raise ValueError(
            f"{path}: points shape {pts.shape} does not match "
            f"({nx},{ny},{nz},3)"
        )

    return pts


def build_ffd_from_two_lattices(
    ctrl_src: np.ndarray,
    ctrl_dst: np.ndarray,
):
    """
    Build an FFD that maps ctrl_src -> ctrl_dst.

    Both lattices:
        - shape (nx, ny, nz, 3)
        - same dimensions
        - ctrl_src must represent a rectangular grid in a box
          (as produced by export_three_bbox_ffd_lattices_json).
    """
    if ctrl_src.shape != ctrl_dst.shape:
        raise ValueError(
            f"Source and destination lattices must have same shape, "
            f"got {ctrl_src.shape} vs {ctrl_dst.shape}"
        )

    if ctrl_src.ndim != 4 or ctrl_src.shape[3] != 3:
        raise ValueError(
            f"ctrl_src must have shape (nx, ny, nz, 3), got {ctrl_src.shape}"
        )

    nx, ny, nz, _ = ctrl_src.shape

    xs = ctrl_src[..., 0]
    ys = ctrl_src[..., 1]
    zs = ctrl_src[..., 2]

    x_min = float(xs.min())
    x_max = float(xs.max())
    y_min = float(ys.min())
    y_max = float(ys.max())
    z_min = float(zs.min())
    z_max = float(zs.max())

    Lx = x_max - x_min
    Ly = y_max - y_min
    Lz = z_max - z_min

    eps = 1e-12
    if Lx < eps: Lx = eps
    if Ly < eps: Ly = eps
    if Lz < eps: Lz = eps

    ffd = FFD(n_control_points=[nx, ny, nz])
    ffd.box_origin[:] = np.array([x_min, y_min, z_min], dtype=float)
    ffd.box_length[:] = np.array([Lx, Ly, Lz], dtype=float)
    ffd.rot_angle[:] = np.array([0.0, 0.0, 0.0], dtype=float)

    if hasattr(ffd, "reset_weights"):
        ffd.reset_weights()
    else:
        ffd.array_mu_x[:] = 0.0
        ffd.array_mu_y[:] = 0.0
        ffd.array_mu_z[:] = 0.0

    # Parameters in [0,1] along each axis
    if nx > 1:
        alphas = np.linspace(0.0, 1.0, nx)
    else:
        alphas = np.zeros(1)
    if ny > 1:
        betas = np.linspace(0.0, 1.0, ny)
    else:
        betas = np.zeros(1)
    if nz > 1:
        gammas = np.linspace(0.0, 1.0, nz)
    else:
        gammas = np.zeros(1)

    # Fill mu so that base + mu*L = ctrl_dst
    for i in range(nx):
        alpha = alphas[i]
        base_x = x_min + alpha * Lx
        for j in range(ny):
            beta = betas[j]
            base_y = y_min + beta * Ly
            for k in range(nz):
                gamma = gammas[k]
                base_z = z_min + gamma * Lz

                tx, ty, tz = ctrl_dst[i, j, k, :]

                ffd.array_mu_x[i, j, k] = (tx - base_x) / Lx
                ffd.array_mu_y[i, j, k] = (ty - base_y) / Ly
                ffd.array_mu_z[i, j, k] = (tz - base_z) / Lz

    return ffd


def apply_ffd_to_points(ffd: FFD, points: np.ndarray) -> np.ndarray:
    """
    Apply FFD to a (N,3) array of points.

    Adjust this if your PyGeM version uses a different API
    (e.g. ffd(points) vs ffd.apply(points)).
    """
    # Most PyGeM versions overload __call__ for FFD:
    # new_points = ffd(points)
    new_points = ffd(points)
    return new_points


def main():
    parser = argparse.ArgumentParser(
        description="Apply three FFD lattices to an STL: "
                    "base -> curved -> deformed."
    )
    parser.add_argument("--input-stl", required=True, help="Input STL file")
    parser.add_argument("--output-stl", required=True, help="Output STL file")
    parser.add_argument("--lat-base", required=True,
                        help="JSON lattice 1 (base) path")
    parser.add_argument("--lat-curved", required=True,
                        help="JSON lattice 2 (curved) path")
    parser.add_argument("--lat-deformed", required=True,
                        help="JSON lattice 3 (deformed) path")

    args = parser.parse_args()

    # 1) Load lattices
    ctrl_base     = load_lattice_json(args.lat_base)
    ctrl_curved   = load_lattice_json(args.lat_curved)
    ctrl_deformed = load_lattice_json(args.lat_deformed)

    # 2) Build FFDs
    ffd_curve = build_ffd_from_two_lattices(ctrl_base,   ctrl_curved)
    ffd_def   = build_ffd_from_two_lattices(ctrl_curved, ctrl_deformed)

    # 3) Load STL
    mesh = trimesh.load(args.input_stl)
    pts = mesh.vertices.copy()

    # 4) Apply base->curved FFD
    pts = apply_ffd_to_points(ffd_curve, pts)

    # 5) Apply curved->deformed FFD
    pts = apply_ffd_to_points(ffd_def, pts)

    # 6) Save
    mesh.vertices = pts
    mesh.export(args.output_stl)
    print(f"Deformed STL written to: {args.output_stl}")


if __name__ == "__main__":
    main()

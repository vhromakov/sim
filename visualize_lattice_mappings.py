#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np


def load_lattice_json(path: str) -> np.ndarray:
    """
    Load an FFD lattice JSON written by export_ffd_lattice_json.

    JSON format:
      {
        "nx": int,
        "ny": int,
        "nz": int,
        "box_origin": [ox, oy, oz],
        "box_length": [lx, ly, lz],
        "mu_x": [[[...]], ...],   # (nx, ny, nz)
        "mu_y": [[[...]], ...],
        "mu_z": [[[...]], ...]
      }

    Returns:
        ctrl_points: np.ndarray, shape (nx, ny, nz, 3)
            Full control point coordinates in world space:
                base = box_origin + t * box_length
                ctrl = base + mu
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nx = int(data["nx"])
    ny = int(data["ny"])
    nz = int(data["nz"])

    # Read box origin and length
    ox, oy, oz = map(float, data["box_origin"])
    lx, ly, lz = map(float, data["box_length"])

    # Read mu arrays and check shapes
    mu_x = np.array(data["mu_x"], dtype=float)
    mu_y = np.array(data["mu_y"], dtype=float)
    mu_z = np.array(data["mu_z"], dtype=float)

    expected_shape = (nx, ny, nz)
    if mu_x.shape != expected_shape:
        raise ValueError(
            f"{path}: mu_x shape {mu_x.shape} does not match {expected_shape}"
        )
    if mu_y.shape != expected_shape:
        raise ValueError(
            f"{path}: mu_y shape {mu_y.shape} does not match {expected_shape}"
        )
    if mu_z.shape != expected_shape:
        raise ValueError(
            f"{path}: mu_z shape {mu_z.shape} does not match {expected_shape}"
        )

    # Allocate control point array (nx, ny, nz, 3)
    ctrl_points = np.zeros((nx, ny, nz, 3), dtype=float)

    # Parameters along each axis in [0,1]
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

    # Reconstruct world-space control points: base + mu
    for i in range(nx):
        alpha = alphas[i]
        base_x = ox + alpha * lx
        for j in range(ny):
            beta = betas[j]
            base_y = oy + beta * ly
            for k in range(nz):
                gamma = gammas[k]
                base_z = oz + gamma * lz

                dx = mu_x[i, j, k]
                dy = mu_y[i, j, k]
                dz = mu_z[i, j, k]

                ctrl_points[i, j, k, 0] = base_x + dx
                ctrl_points[i, j, k, 1] = base_y + dy
                ctrl_points[i, j, k, 2] = base_z + dz

    return ctrl_points


def write_correspondence_ply(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    output_path: str,
):
    """
    Write a PLY file that shows:
      - points from pts_a
      - points from pts_b
      - edges connecting each pts_a[i] to pts_b[i]

    pts_a, pts_b: shape (nx, ny, nz, 3), must have the same shape.
    """
    if pts_a.shape != pts_b.shape:
        raise ValueError(
            f"Shape mismatch in correspondence: {pts_a.shape} vs {pts_b.shape}"
        )

    if pts_a.ndim != 4 or pts_a.shape[3] != 3:
        raise ValueError(
            f"Expected (nx, ny, nz, 3), got {pts_a.shape}"
        )

    # Flatten
    a_flat = pts_a.reshape(-1, 3)
    b_flat = pts_b.reshape(-1, 3)
    n = a_flat.shape[0]

    # Vertices: first all A points, then all B points
    vertices = np.vstack([a_flat, b_flat])  # (2N, 3)

    # Edges: connect i (A) -> i+N (B)
    edges = [(i, i + n) for i in range(n)]

    # Make sure directory exists
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # ASCII PLY with vertices + edges
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {vertices.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("end_header\n")

        # vertices
        for x, y, z in vertices:
            f.write(f"{x:.9f} {y:.9f} {z:.9f}\n")

        # edges
        for v1, v2 in edges:
            f.write(f"{v1} {v2}\n")

    print(
        f"[PLY] Wrote correspondence file {output_path} "
        f"(verts={vertices.shape[0]}, edges={len(edges)})"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize lattice updates by writing PLY files with "
            "base↔curved and curved↔deformed correspondences."
        )
    )
    parser.add_argument("--lat-base", required=True, help="JSON lattice 1 (base FFD)")
    parser.add_argument("--lat-curved", required=True, help="JSON lattice 2 (curved FFD)")
    parser.add_argument("--lat-deformed", required=True, help="JSON lattice 3 (deformed FFD)")
    parser.add_argument(
        "--out-prefix",
        required=True,
        help="Output prefix for generated PLY files",
    )

    args = parser.parse_args()

    print("[INFO] Loading lattices...")
    ctrl_base     = load_lattice_json(args.lat_base)
    ctrl_curved   = load_lattice_json(args.lat_curved)
    ctrl_deformed = load_lattice_json(args.lat_deformed)

    # 1) base vs curved
    out1 = args.out_prefix + "_base_vs_curved.ply"
    print("[INFO] Writing base↔curved PLY...")
    write_correspondence_ply(ctrl_base, ctrl_curved, out1)

    # 2) curved vs deformed
    out2 = args.out_prefix + "_curved_vs_deformed.ply"
    print("[INFO] Writing curved↔deformed PLY...")
    write_correspondence_ply(ctrl_curved, ctrl_deformed, out2)

    print("[DONE] All correspondence PLYs written.")


if __name__ == "__main__":
    main()

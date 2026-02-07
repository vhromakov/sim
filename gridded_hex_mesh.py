#!/usr/bin/env python3
"""
Gridded (voxel) HEX mesh from a closed triangle surface (STL/OBJ/etc).

- Builds a regular grid of cuboids (dx,dy,dz).
- Keeps voxels that lie inside the solid (filled).
- Converts voxels -> shared vertices + HEX8 connectivity.
- Exports with meshio to .vtu/.msh/.inp/etc.

pip install trimesh meshio numpy
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import trimesh
import meshio


@dataclass
class HexMesh:
    points: np.ndarray  # (N,3) float
    hexes: np.ndarray   # (M,8) int (0-based)


def _as_single_trimesh(loaded) -> trimesh.Trimesh:
    if isinstance(loaded, trimesh.Scene):
        geoms = list(loaded.geometry.values())
        if not geoms:
            raise ValueError("Empty scene")
        return trimesh.util.concatenate(geoms)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    raise TypeError(f"Unsupported type: {type(loaded)}")


def gridded_hex_mesh(
    mesh_path: str,
    *,
    cell_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    fill_interior: bool = True,
    align_shift_frac: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> HexMesh:
    """
    align_shift_frac: shift the grid in fractions of cell size (each component typically in [-0.5, 0.5]).
                     This mimics Carbon’s “alignment” knob: you slide the grid relative to the part.
    """
    dx, dy, dz = map(float, cell_size)
    if dx <= 0 or dy <= 0 or dz <= 0:
        raise ValueError("cell_size must be positive")

    m0 = _as_single_trimesh(trimesh.load_mesh(mesh_path, force="mesh"))
    m0 = m0.process(validate=True)

    if not m0.is_watertight:
        raise ValueError(
            "Input mesh must be watertight/closed for solid voxelization. "
            "Repair it first (e.g. watertight/boolean repair) or use a different inside-test."
        )

    # --- Scale trick: convert anisotropic (dx,dy,dz) voxels into unit cubes in scaled space.
    S = np.eye(4, dtype=np.float64)
    S[0, 0] = 1.0 / dx
    S[1, 1] = 1.0 / dy
    S[2, 2] = 1.0 / dz

    m = m0.copy()
    m.apply_transform(S)

    # Optional: shift mesh by fractional voxel steps to adjust grid alignment
    shift = np.array(align_shift_frac, dtype=np.float64)
    Tshift = np.eye(4, dtype=np.float64)
    Tshift[:3, 3] = shift  # in scaled-space voxel units (pitch=1)
    m.apply_transform(Tshift)

    # Voxelize with unit pitch in scaled space
    vox = m.voxelized(pitch=1.0)

    # Fill interior so we get a solid voxel grid (not just a shell)
    if fill_interior:
        vox = vox.fill()  # in-place mutation, returns self

    filled = vox.sparse_indices  # (M,3) int indices of occupied voxels

    if filled.shape[0] == 0:
        raise ValueError("No voxels filled. Try larger part, smaller voxels, or check units.")

    # Corner offsets for a HEX8 (common convention)
    # (-,-,-), (+,-,-), (+,+,-), (-,+,-), (-,-,+), (+,-,+), (+,+,+), (-,+,+)
    corner_offsets = np.array(
        [
            [-1, -1, -1],
            [ 1, -1, -1],
            [ 1,  1, -1],
            [-1,  1, -1],
            [-1, -1,  1],
            [ 1, -1,  1],
            [ 1,  1,  1],
            [-1,  1,  1],
        ],
        dtype=np.int64,
    )

    # Build a half-step integer lattice for corners to avoid float dedupe issues:
    # voxel index (i,j,k) is the cell center in index-space; corners are at (i±0.5, ...)
    corner_keys = filled[:, None, :] * 2 + corner_offsets[None, :, :]  # (M,8,3) int
    flat_keys = corner_keys.reshape(-1, 3)

    uniq_keys, inv = np.unique(flat_keys, axis=0, return_inverse=True)
    hexes = inv.reshape(-1, 8).astype(np.int64)

    # Convert corner lattice coords (key/2) -> scaled-space coords using voxel transform
    idx_float = uniq_keys.astype(np.float64) / 2.0
    ones = np.ones((idx_float.shape[0], 1), dtype=np.float64)
    h = np.hstack([idx_float, ones])
    pts_scaled = (h @ vox.transform.T)[:, :3]

    # Undo the earlier mesh shift + scaling to go back to original world space
    # We applied: world -> scaled via S, then shifted by align; so reverse:
    # reverse shift (in scaled units), then scale back by (dx,dy,dz).
    pts_scaled_unshift = pts_scaled - shift[None, :]
    points = pts_scaled_unshift * np.array([dx, dy, dz], dtype=np.float64)[None, :]

    return HexMesh(points=points, hexes=hexes)


import numpy as np
import trimesh

def boundary_vertex_indices(hexes: np.ndarray) -> np.ndarray:
    faces4 = np.array([
        [0,1,2,3], [4,5,6,7], [0,1,5,4],
        [2,3,7,6], [1,2,6,5], [3,0,4,7],
    ], dtype=np.int64)

    quads = hexes[:, faces4].reshape(-1, 4)
    keys = np.sort(quads, axis=1)

    uniq, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    boundary_quads = quads[counts[inv] == 1]
    return np.unique(boundary_quads.reshape(-1))


def build_hex_edge_adjacency(hexes: np.ndarray, n_verts: int):
    # 12 edges of HEX8
    edges = np.array([
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7],
    ], dtype=np.int64)

    E = hexes[:, edges].reshape(-1, 2)
    E = np.sort(E, axis=1)
    E = np.unique(E, axis=0)

    # Build sparse Laplacian
    from scipy import sparse
    rows = np.concatenate([E[:,0], E[:,1]])
    cols = np.concatenate([E[:,1], E[:,0]])
    data = np.ones(len(rows), dtype=np.float64)
    A = sparse.coo_matrix((data, (rows, cols)), shape=(n_verts, n_verts)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    L = sparse.diags(deg) - A
    return L


def morph_hex_mesh_to_surface(
    *,
    surface_mesh: trimesh.Trimesh,
    points: np.ndarray,
    hexes: np.ndarray,
    alpha: float = 0.35,       # move boundary by this fraction toward the surface
    max_snap: float | None = None,
    smooth_interior: bool = True,
) -> np.ndarray:
    pts = points.copy()

    b = boundary_vertex_indices(hexes)
    pq = trimesh.proximity.ProximityQuery(surface_mesh)
    closest, dist, _tri = pq.on_surface(pts[b])

    disp = closest - pts[b]
    if max_snap is not None:
        d = np.linalg.norm(disp, axis=1)
        scale = np.ones_like(d)
        m = d > max_snap
        scale[m] = max_snap / (d[m] + 1e-12)
        disp *= scale[:, None]

    pts[b] = pts[b] + alpha * disp

    if not smooth_interior:
        return pts

    # Harmonic smoothing of interior with boundary fixed (Dirichlet)
    from scipy.sparse.linalg import spsolve

    n = len(pts)
    is_b = np.zeros(n, dtype=bool)
    is_b[b] = True
    I = np.where(~is_b)[0]
    B = np.where(is_b)[0]

    L = build_hex_edge_adjacency(hexes, n)
    Lii = L[I][:, I]
    Lib = L[I][:, B]

    for k in range(3):
        rhs = -Lib @ pts[B, k]
        pts[I, k] = spsolve(Lii, rhs)

    return pts


def hexmesh_to_surface_stl(points: np.ndarray, hexes: np.ndarray, out_stl: str):
    """
    Convert voxel hex mesh to a triangle surface by emitting faces that belong to only one hex.
    Assumes axis-aligned voxel topology (which we have).
    """
    # Hex faces (as 4-cycles) in terms of the 8 hex corner indices
    faces4 = np.array([
        [0,1,2,3],  # -Z
        [4,5,6,7],  # +Z
        [0,1,5,4],  # -Y
        [2,3,7,6],  # +Y
        [1,2,6,5],  # +X
        [3,0,4,7],  # -X
    ], dtype=np.int64)

    # Collect all quad faces from all hexes
    all_quads = hexes[:, faces4].reshape(-1, 4)  # (M*6,4)

    # Canonical key for "same face" independent of winding
    keys = np.sort(all_quads, axis=1)

    # Count duplicates: internal faces appear twice, boundary faces once
    uniq, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    boundary_mask = counts[inv] == 1
    boundary_quads = all_quads[boundary_mask]

    # Triangulate quads (0,1,2) and (0,2,3)
    t1 = boundary_quads[:, [0,1,2]]
    t2 = boundary_quads[:, [0,2,3]]
    tris = np.vstack([t1, t2]).astype(np.int64)

    tm = trimesh.Trimesh(vertices=points, faces=tris, process=False)
    tm.export(out_stl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh", help="Input STL/OBJ/etc (watertight recommended)")
    ap.add_argument("--dx", type=float, default=1.0)
    ap.add_argument("--dy", type=float, default=1.0)
    ap.add_argument("--dz", type=float, default=1.0)
    ap.add_argument("--out", default="hex_mesh.vtu", help="Output file (.vtu/.msh/.inp/...)")
    ap.add_argument("--no-fill", action="store_true", help="Don't fill interior (surface voxels only)")
    ap.add_argument("--ax", type=float, default=0.0, help="Grid shift fraction in X (scaled voxel units)")
    ap.add_argument("--ay", type=float, default=0.0)
    ap.add_argument("--az", type=float, default=0.0)
    args = ap.parse_args()

    hm = gridded_hex_mesh(
        args.mesh,
        cell_size=(args.dx, args.dy, args.dz),
        fill_interior=not args.no_fill,
        align_shift_frac=(args.ax, args.ay, args.az),
    )

    surf = _as_single_trimesh(trimesh.load_mesh(args.mesh, force="mesh"))

    hm.points = morph_hex_mesh_to_surface(
        surface_mesh=surf,
        points=hm.points,
        hexes=hm.hexes,
        alpha=0.35,
        max_snap=0.10,          # mm (example)
        smooth_interior=True,
    )

    # Example: export surface STL
    hexmesh_to_surface_stl(hm.points, hm.hexes, "OUTPUT/hex_surface.stl")
    print("Wrote hex_surface.stl (triangle skin of the voxel solid)")


if __name__ == "__main__":
    main()

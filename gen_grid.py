#!/usr/bin/env python3
"""
Cylindrical voxelization debug script.

Takes an input STL, voxelizes it in cylindrical param space, maps voxels
to curved hexahedral nodes on a cylinder (axis = global +Y), and exports
the voxel-node lattice as a single PLY point cloud:

    <job_name>_lattice_orig.ply

No CalculiX, no FFD, no FRD. Intended purely to debug how voxelization
and cylindrical mapping behave.
"""

import argparse
import os
from typing import List, Tuple, Dict, Set, Optional

import math
import numpy as np
import trimesh


# ============================================================
#  Cylindrical mapping helpers
# ============================================================

def world_to_param_cyl(point, cx, cz, R0):
    """
    Map world coordinates (x, y, z) to cylindrical param space (u, v, w).

    Cylinder axis is along global Y and centered at (cx, cz) in XZ plane.
    R0 is the base cylinder radius.

    Returns:
        (u, v, w)
        u: coordinate along cylinder axis (Y)
        v: arc length along circumference (R0 * angle)
        w: radial offset from base cylinder (r - R0)
    """
    x, y, z = point
    dx = x - cx
    dz = z - cz

    u = y
    r = math.sqrt(dx * dx + dz * dz)

    if r < 1e-12:
        theta = 0.0
    else:
        theta = math.atan2(dz, dx)

    v = R0 * theta
    w = r - R0

    return (u, v, w)


def param_to_world_cyl(param_point, cx, cz, R0, theta_offset: float = 0.0):
    """
    Map param space point (u, v, w) back to world (x, y, z)
    using the same cylinder definition.

    Args:
        param_point: (u, v, w)
        cx, cz: cylinder center in XZ plane
        R0: base radius used for angular mapping
        theta_offset: global angular offset applied to all points
                      (used to align lowest point with voxel center)

    Returns:
        (x, y, z)
    """
    u, v, w = param_point

    theta = v / R0 + theta_offset
    r = R0 + w

    x = cx + r * math.cos(theta)
    y = u
    z = cz + r * math.sin(theta)

    return (x, y, z)


def _unwrap_angle_array(thetas: np.ndarray) -> np.ndarray:
    """
    Unwrap a sequence of angles so that they form a continuous curve
    (no jumps > pi). Used so we can get a clean [theta_min, theta_max]
    interval for the STL.
    """
    if thetas.size == 0:
        return thetas
    out = np.zeros_like(thetas)
    out[0] = thetas[0]
    for i in range(1, len(thetas)):
        diff = thetas[i] - thetas[i - 1]
        # wrap diff into [-pi, pi]
        diff_wrapped = (diff + math.pi) % (2.0 * math.pi) - math.pi
        out[i] = out[i - 1] + diff_wrapped
    return out


# ============================================================
#  Cylindrical voxelization -> voxel node lattice
# ============================================================

def generate_cylindrical_voxel_lattice(
    input_stl: str,
    cube_size: float,
    cyl_center_xz: Optional[Tuple[float, float]] = None,
    cyl_radius: Optional[float] = None,
):
    """
    Voxelize input_stl in cylindrical param space and build the voxel-node
    lattice (global coordinates of all curved hexahedral nodes).

    Returns:
        vertices: List[(x, y, z)]  unique voxel node positions
        lowest_point_world: (x, y, z) of lowest-Z STL vertex
        axis_point_world:   point on cylinder axis near lowest point (same Y)
        edge_left_world:    radial edge direction (angular left)
        edge_right_world:   radial edge direction (angular right)
        cyl_params:         (cx, cz, R0_world, theta_offset)
    """
    if cube_size <= 0:
        raise ValueError("cube_size must be positive")

    mesh = trimesh.load(input_stl)
    if not isinstance(mesh, trimesh.Trimesh):
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
        else:
            raise ValueError("Failed to load STL as a valid 3D mesh")

    print(f"[VOXEL] Loaded mesh from {input_stl}")
    print(f"[VOXEL] Watertight: {mesh.is_watertight}, bbox extents: {mesh.extents}")

    # Bounds in world space (original STL)
    bounds = mesh.bounds  # [[xmin,ymin,zmin], [xmax,ymax,zmax]]
    x_min_w, y_min_w, z_min_w = bounds[0]
    x_max_w, y_max_w, z_max_w = bounds[1]

    cyl_params: Optional[Tuple[float, float, float, float]] = None  # (cx, cz, R0_world, theta_offset)

    r_lowest: Optional[float] = None
    theta_lowest: Optional[float] = None
    cx_cyl: Optional[float] = None
    cz_cyl: Optional[float] = None
    R0_param: Optional[float] = None  # radius used for world->param mapping
    lowest_point_world: Optional[Tuple[float, float, float]] = None
    axis_point_world: Optional[Tuple[float, float, float]] = None
    edge_left_world: Optional[Tuple[float, float, float]] = None
    edge_right_world: Optional[Tuple[float, float, float]] = None

    # --- cylindrical preprocessing ---
    verts_world = mesh.vertices.copy()

    # --- Find lowest point (min Z) in world space ---
    z_coords = verts_world[:, 2]
    min_z_idx = int(np.argmin(z_coords))
    x_low, y_low, z_low = verts_world[min_z_idx]
    lowest_point_world = (float(x_low), float(y_low), float(z_low))

    # Start from this X as cylinder center X
    cx_cyl = float(x_low)

    # Decide CZ and initial radius R0_param for world->param mapping
    if cyl_radius is not None and cyl_radius > 0.0:
        # User-provided radius (for mapping)
        R0_param = float(cyl_radius)

        z_center_bbox = 0.5 * (z_min_w + z_max_w)
        cz_plus = z_low + R0_param
        cz_minus = z_low - R0_param

        if abs(cz_plus - z_center_bbox) < abs(cz_minus - z_center_bbox):
            cz_cyl = cz_plus
        else:
            cz_cyl = cz_minus

        if cyl_center_xz is not None:
            print(
                "[VOXEL] Note: --cyl-center-xz provided, but CZ is "
                "overridden to enforce lowest point on cylinder."
            )

        print(
            f"[VOXEL] Using user radius R0_param={R0_param:.3f}; "
            f"lowest point ({x_low:.3f}, {y_low:.3f}, {z_low:.3f}) "
            f"lies on cylinder (by CZ selection)."
        )
    else:
        # Estimate center/radius from geometry
        if cyl_center_xz is not None:
            cx_cli, cz_cli = cyl_center_xz
            cx_cyl = cx_cli
            cz_cyl = cz_cli
            print(
                "[VOXEL] No radius given; using explicit cyl-center-xz "
                f"({cx_cyl:.3f}, {cz_cyl:.3f})"
            )
        else:
            cz_cyl = 0.5 * (z_min_w + z_max_w)
            print(
                "[VOXEL] No radius given; using cx from lowest-Z point "
                f"and cz=bbox center: cx={cx_cyl:.3f}, cz={cz_cyl:.3f}"
            )

    dx_all = verts_world[:, 0] - cx_cyl
    dz_all = verts_world[:, 2] - cz_cyl
    r_all = np.sqrt(dx_all * dx_all + dz_all * dz_all)
    R0_param = float(np.mean(r_all))

    # Radial/angle of the lowest point (with chosen center)
    dx_low = x_low - cx_cyl
    dz_low = z_low - cz_cyl
    r_lowest = math.sqrt(dx_low * dx_low + dz_low * dz_low)
    theta_lowest = math.atan2(dz_low, dx_low)  # angle of lowest point

    # --- Angular edges of the STL around the lowest direction ---------
    # Angles of all STL vertices around the cylinder axis
    theta_all = np.arctan2(dz_all, dx_all)

    # Differences w.r.t. theta_lowest, wrapped to [-pi, pi]
    dtheta_all = theta_all - theta_lowest
    dtheta_all = (dtheta_all + math.pi) % (2.0 * math.pi) - math.pi

    # Left edge = most negative delta, right edge = most positive delta
    idx_left = int(np.argmin(dtheta_all))
    idx_right = int(np.argmax(dtheta_all))

    theta_left = theta_lowest + float(dtheta_all[idx_left])
    theta_right = theta_lowest + float(dtheta_all[idx_right])

    # Use max radial extent for nice long edge rays
    r_edge = float(r_all.max())
    y_ref = float(y_low)  # keep all rays in the same horizontal plane

    axis_point_world = (float(cx_cyl), y_ref, float(cz_cyl))
    edge_left_world = (
        float(cx_cyl + r_edge * math.cos(theta_left)),
        y_ref,
        float(cz_cyl + r_edge * math.sin(theta_left)),
    )
    edge_right_world = (
        float(cx_cyl + r_edge * math.cos(theta_right)),
        y_ref,
        float(cz_cyl + r_edge * math.sin(theta_right)),
    )

    print(
        "[VOXEL] Angular edges: dtheta_left="
        f"{float(dtheta_all[idx_left]):.6f}, "
        f"dtheta_right={float(dtheta_all[idx_right]):.6f}"
    )

    print(
        f"[VOXEL] Cyl center from lowest Z vertex: "
        f"min_z={z_low:.3f}, lowest_pt=({x_low:.3f}, {y_low:.3f}, {z_low:.3f}), "
        f"r_low={r_lowest:.3f}, theta_low={theta_lowest:.3f} rad"
    )
    print(
        f"[VOXEL] Cylindrical voxel mode: axis=+Y, "
        f"center=({cx_cyl:.3f}, {cz_cyl:.3f}), R0_param={R0_param:.3f}"
    )

    # Map mesh into param space (u,v,w) using R0_param
    verts_param = np.zeros_like(verts_world)
    for i, (x, y, z) in enumerate(verts_world):
        verts_param[i] = world_to_param_cyl((x, y, z), cx_cyl, cz_cyl, R0_param)
    mesh.vertices = verts_param

    # --- Tangential (v) extents of the STL in param space -------------
    v_min_mesh = float(verts_param[:, 1].min())
    v_max_mesh = float(verts_param[:, 1].max())
    print(
        f"[VOXEL] Param v extents of STL: v_min={v_min_mesh:.6f}, "
        f"v_max={v_max_mesh:.6f}"
    )

    # --- Voxelization in param coordinates ---
    print(f"[VOXEL] Voxelizing with cube size = {cube_size} ...")
    vox = mesh.voxelized(pitch=cube_size)
    vox.fill()

    indices = vox.sparse_indices  # (N,3) with (ix,iy,iz) in param grid
    if indices.size == 0:
        print("[VOXEL] No voxels found – check cube size or input mesh.")
        return [], None, None, None, None, None

    total_voxels = indices.shape[0]
    print(f"[VOXEL] Total filled voxels (cubes): {total_voxels}")

    # --- Angular trimming in param v: keep only bands that intersect STL in v ---
    half = cube_size / 2.0
    eps = 1e-9

    unique_iy = np.unique(indices[:, 1])

    # Compute v_center for each angular band using voxel centers
    centers_param = vox.indices_to_points(
        np.array([[0, int(iy), 0] for iy in unique_iy], dtype=float)
    )
    v_centers = centers_param[:, 1]

    # Map iy -> (v_lo, v_hi)
    v_interval_by_iy: Dict[int, Tuple[float, float]] = {}
    for iy_val, v_c in zip(unique_iy, v_centers):
        v_lo = float(v_c - half)
        v_hi = float(v_c + half)
        v_interval_by_iy[int(iy_val)] = (v_lo, v_hi)

    # Build mask: keep cells whose v-interval intersects [v_min_mesh, v_max_mesh]
    mask = np.zeros(indices.shape[0], dtype=bool)
    for row_idx, (ix, iy, iz) in enumerate(indices):
        v_lo, v_hi = v_interval_by_iy[int(iy)]
        if (v_hi >= v_min_mesh - eps) and (v_lo <= v_max_mesh + eps):
            mask[row_idx] = True

    indices = indices[mask]

    if indices.size == 0:
        print("[VOXEL] After angular trimming (param v), no voxels left – aborting.")
        return [], None, None, None, None, None

    total_voxels = indices.shape[0]
    print(f"[VOXEL] Total voxels after angular trim (param v): {total_voxels}")

    # sort by iz, iy, ix
    order = np.lexsort((indices[:, 0], indices[:, 1], indices[:, 2]))
    indices_sorted = indices[order]

    # Map iz -> slice "position" in param space: w_center
    unique_iz = np.unique(indices_sorted[:, 2])
    layer_info = []
    for iz in unique_iz:
        idx_arr = np.array([[0, 0, iz]], dtype=float)
        pt = vox.indices_to_points(idx_arr)[0]  # in param coordinates
        slice_coord = float(pt[2])  # w in cylindrical param space
        layer_info.append((iz, slice_coord))

    # Sorting slice order: high w first (outer radial band as base slice)
    layer_info.sort(key=lambda x: x[1], reverse=True)

    z_slices: List[float] = [pos for (iz, pos) in layer_info]

    vertex_index_map: Dict[Tuple[int, int, int], int] = {}
    vertices: List[Tuple[float, float, float]] = []

    def get_vertex_index(
        key: Tuple[int, int, int],
        coord: Tuple[float, float, float],
    ) -> int:
        if key in vertex_index_map:
            return vertex_index_map[key]
        idx = len(vertices) + 1
        vertex_index_map[key] = idx
        vertices.append(coord)
        return idx

    half = cube_size / 2.0

    # --- For curved voxels: compute R0_world (radial alignment) and theta_offset (tangential alignment)
    R0_world: Optional[float] = None
    theta_offset: float = 0.0

    if (
        r_lowest is None or R0_param is None or
        cx_cyl is None or cz_cyl is None or theta_lowest is None
    ):
        raise RuntimeError("[VOXEL] Internal error: cylindrical parameters not initialized.")

    # --- Radial: force lowest point onto outer face of base slice ---
    if z_slices:
        # Base slice is slice_idx 0 (outermost radial band) after sort(reverse=True)
        w_center_base = z_slices[0]           # param w of slice center
        w_face_outer = w_center_base + half   # param w of outer face

        # Want: R0_world + w_face_outer = r_lowest  ->  R0_world = r_lowest - w_face_outer
        R0_world = r_lowest - w_face_outer

        if R0_world <= 0.0:
            print(
                f"[VOXEL] WARNING: computed R0_world={R0_world:.6f} <= 0; "
                f"falling back to R0_param={R0_param:.6f}"
            )
            R0_world = max(R0_param, cube_size)

        print(
            "[VOXEL] Aligning outer face of base radial layer with lowest point:\n"
            f"        r_low={r_lowest:.6f}, w_center_base={w_center_base:.6f}, "
            f"w_face_outer={w_face_outer:.6f} -> R0_world={R0_world:.6f}"
        )
    else:
        R0_world = R0_param
        print(
            "[VOXEL] No slices? Using R0_param for R0_world "
            f"(R0_world={R0_world:.6f})."
        )

    # --- Tangential: find voxel in base slice whose center angle is closest to theta_lowest ---
    base_slice_iz = layer_info[0][0] if layer_info else None
    best_dtheta = None
    best_theta_center = None

    if base_slice_iz is not None:
        for (ix, iy, iz) in indices_sorted:
            if iz != base_slice_iz:
                continue

            center_param = vox.indices_to_points(
                np.array([[ix, iy, iz]], dtype=float)
            )[0]
            u_c, v_c, w_c = center_param

            # Use R0_world with zero offset to get preliminary center
            x_c, y_c, z_c = param_to_world_cyl((u_c, v_c, w_c), cx_cyl, cz_cyl, R0_world, theta_offset=0.0)
            theta_center = math.atan2(z_c - cz_cyl, x_c - cx_cyl)

            # Smallest angular difference mod 2π
            dtheta = abs(((theta_center - theta_lowest + math.pi) % (2.0 * math.pi)) - math.pi)

            if best_dtheta is None or dtheta < best_dtheta:
                best_dtheta = dtheta
                best_theta_center = theta_center

    if best_theta_center is not None:
        theta_offset = theta_lowest - best_theta_center
        print(
            "[VOXEL] Tangential alignment: base-slice voxel center angle "
            f"{best_theta_center:.6f} -> theta_low={theta_lowest:.6f}, "
            f"theta_offset={theta_offset:.6f}"
        )
    else:
        theta_offset = 0.0
        print("[VOXEL] WARNING: could not find base slice voxel for tangential alignment.")

    # This is the cylinder definition used by the voxel mesh
    cyl_params = (cx_cyl, cz_cyl, R0_world, theta_offset)

    # --- Build voxel-node lattice (curved hex nodes) -------------------------
    print("[VOXEL] Building voxel-node lattice (curved hexa nodes) ...")

    for (ix, iy, iz) in indices_sorted:
        center_param = vox.indices_to_points(
            np.array([[ix, iy, iz]], dtype=float)
        )[0]
        u_c, v_c, w_c = center_param

        u0, u1 = u_c - half, u_c + half
        v0, v1 = v_c - half, v_c + half
        w0, w1 = w_c - half, w_c + half

        # param-space corners of this voxel
        p0 = (u0, v0, w0)
        p1 = (u1, v0, w0)
        p2 = (u1, v1, w0)
        p3 = (u0, v1, w0)
        p4 = (u0, v0, w1)
        p5 = (u1, v0, w1)
        p6 = (u1, v1, w1)
        p7 = (u0, v1, w1)

        # map to world (curved hexa)
        x0, y0, z0 = param_to_world_cyl(p0, cx_cyl, cz_cyl, R0_world, theta_offset)
        x1, y1, z1 = param_to_world_cyl(p1, cx_cyl, cz_cyl, R0_world, theta_offset)
        x2, y2, z2 = param_to_world_cyl(p2, cx_cyl, cz_cyl, R0_world, theta_offset)
        x3, y3, z3 = param_to_world_cyl(p3, cx_cyl, cz_cyl, R0_world, theta_offset)
        x4, y4, z4 = param_to_world_cyl(p4, cx_cyl, cz_cyl, R0_world, theta_offset)
        x5, y5, z5 = param_to_world_cyl(p5, cx_cyl, cz_cyl, R0_world, theta_offset)
        x6, y6, z6 = param_to_world_cyl(p6, cx_cyl, cz_cyl, R0_world, theta_offset)
        x7, y7, z7 = param_to_world_cyl(p7, cx_cyl, cz_cyl, R0_world, theta_offset)

        # Deduplicate nodes by (ix,iy,iz-based) key
        get_vertex_index((ix,   iy,   iz),   (x0, y0, z0))
        get_vertex_index((ix+1, iy,   iz),   (x1, y1, z1))
        get_vertex_index((ix+1, iy+1, iz),   (x2, y2, z2))
        get_vertex_index((ix,   iy+1, iz),   (x3, y3, z3))
        get_vertex_index((ix,   iy,   iz+1), (x4, y4, z4))
        get_vertex_index((ix+1, iy,   iz+1), (x5, y5, z5))
        get_vertex_index((ix+1, iy+1, iz+1), (x6, y6, z6))
        get_vertex_index((ix,   iy+1, iz+1), (x7, y7, z7))

    print(f"[VOXEL] Lattice nodes: {len(vertices)}")
    return vertices, lowest_point_world, axis_point_world, edge_left_world, edge_right_world, cyl_params


# ============================================================
#  Lattice PLY export (orig only, no deformation)
# ============================================================

def export_lattice_orig_ply(
    vertices: List[Tuple[float, float, float]],
    out_path: str,
    max_points: int = 20000,
    lowest_point: Optional[Tuple[float, float, float]] = None,
    cyl_params: Optional[Tuple[float, ...]] = None,
    axis_point: Optional[Tuple[float, float, float]] = None,
    edge_left_point: Optional[Tuple[float, float, float]] = None,
    edge_right_point: Optional[Tuple[float, float, float]] = None,
    marker_segments_vertical: int = 64,
    marker_segments_radial: int = 32,
):
    """
    Export lattice as a single PLY point cloud (original nodes only).

    Markers:
      - vertical line through lowest_point (if provided),
      - radial line from cylinder axis to lowest_point (if cyl_params+lowest_point),
      - radial line from axis to left angular edge,
      - radial line from axis to right angular edge.
    """
    if not vertices:
        print("[PLY] No vertices, skipping lattice export.")
        return

    if not out_path:
        return

    verts = np.array(vertices, dtype=float)
    n_nodes = verts.shape[0]
    print(f"[PLY] Preparing lattice_orig with {n_nodes} nodes")

    points = verts

    def maybe_subsample(points_arr: np.ndarray) -> np.ndarray:
        total = points_arr.shape[0]
        if total > max_points:
            step = int(np.ceil(total / max_points))
            idx = np.arange(0, total, step, dtype=int)
            return points_arr[idx]
        return points_arr

    points = maybe_subsample(points)

    # ------------------ Vertical marker line through lowest point ------------------
    if lowest_point is not None:
        x0, y0, z0 = lowest_point
        # Use lattice Y extents for line length
        y_min = float(verts[:, 1].min())
        y_max = float(verts[:, 1].max())

        ys = np.linspace(y_min, y_max, marker_segments_vertical, dtype=float)
        xs = np.full_like(ys, x0)
        zs = np.full_like(ys, z0)

        marker_vert = np.column_stack([xs, ys, zs])
        points = np.vstack([points, marker_vert])

        print(
            f"[PLY] Added vertical marker line through lowest point "
            f"({x0:.6f}, {y0:.6f}, {z0:.6f}) with {marker_segments_vertical} points."
        )

    # ------------------ Radial line (axis -> lowest point) ------------------
    if lowest_point is not None and cyl_params is not None:
        x0, y0, z0 = lowest_point

        # Cylinder axis center in XZ plane
        if len(cyl_params) >= 2:
            cx = float(cyl_params[0])
            cz = float(cyl_params[1])
        else:
            cx = cz = 0.0

        xs = np.linspace(cx, x0, marker_segments_radial, dtype=float)
        ys = np.full_like(xs, y0)
        zs = np.linspace(cz, z0, marker_segments_radial, dtype=float)

        marker_radial = np.column_stack([xs, ys, zs])
        points = np.vstack([points, marker_radial])

        print(
            f"[PLY] Added radial marker line axis->lowest "
            f"({cx:.6f}, {y0:.6f}, {cz:.6f}) -> ({x0:.6f}, {y0:.6f}, {z0:.6f})."
        )

    # ------------------ Left/right edge radial lines ------------------
    if axis_point is not None and edge_left_point is not None:
        ax, ay, az = axis_point
        lx, ly, lz = edge_left_point

        xs = np.linspace(ax, lx, marker_segments_radial, dtype=float)
        ys = np.linspace(ay, ly, marker_segments_radial, dtype=float)
        zs = np.linspace(az, lz, marker_segments_radial, dtype=float)

        marker_left = np.column_stack([xs, ys, zs])
        points = np.vstack([points, marker_left])

        print(
            "[PLY] Added LEFT edge radial line "
            f"axis({ax:.6f},{ay:.6f},{az:.6f}) -> "
            f"edge_left({lx:.6f},{ly:.6f},{lz:.6f})."
        )

    if axis_point is not None and edge_right_point is not None:
        ax, ay, az = axis_point
        rx, ry, rz = edge_right_point

        xs = np.linspace(ax, rx, marker_segments_radial, dtype=float)
        ys = np.linspace(ay, ry, marker_segments_radial, dtype=float)
        zs = np.linspace(az, rz, marker_segments_radial, dtype=float)

        marker_right = np.column_stack([xs, ys, zs])
        points = np.vstack([points, marker_right])

        print(
            "[PLY] Added RIGHT edge radial line "
            f"axis({ax:.6f},{ay:.6f},{az:.6f}) -> "
            f"edge_right({rx:.6f},{ry:.6f},{rz:.6f})."
        )

    # ------------------ Write PLY ------------------
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in points:
            f.write(f"{x} {y} {z}\n")

    print(f"[PLY] lattice_orig written to: {out_path}")


# ============================================================
#  CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Cylindrically voxelize an STL and export the voxel-node lattice "
            "as <job_name>_lattice_orig.ply for debugging."
        )
    )
    parser.add_argument("input_stl", help="Path to input STL file")
    parser.add_argument(
        "job_name",
        help="Base name; output will be '<job_name>_lattice_orig.ply'.",
    )
    parser.add_argument(
        "--cube-size",
        "-s",
        type=float,
        required=True,
        help="Edge length of each voxel cube in param space (same units as STL, e.g. mm)",
    )
    parser.add_argument(
        "--cyl-center-xz",
        type=float,
        nargs=2,
        metavar=("CX", "CZ"),
        help="Cylinder axis center in XZ plane (optional).",
    )
    parser.add_argument(
        "--cyl-radius",
        type=float,
        help=(
            "Base cylinder radius R0 for param mapping (optional). "
            "If omitted, estimated from STL geometry."
        ),
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input_stl):
        print(f"Input STL not found: {args.input_stl}")
        raise SystemExit(1)

    (
        vertices,
        lowest_point,
        axis_point,
        edge_left_point,
        edge_right_point,
        cyl_params,
    ) = generate_cylindrical_voxel_lattice(
        args.input_stl,
        args.cube_size,
        cyl_center_xz=tuple(args.cyl_center_xz) if args.cyl_center_xz else None,
        cyl_radius=args.cyl_radius,
    )

    if not vertices:
        print("No lattice generated, aborting.")
        raise SystemExit(1)

    lattice_path = args.job_name + "_lattice_orig.ply"
    export_lattice_orig_ply(
        vertices,
        lattice_path,
        lowest_point=lowest_point,
        cyl_params=cyl_params,
        axis_point=axis_point,
        edge_left_point=edge_left_point,
        edge_right_point=edge_right_point,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
STL -> cylindrical voxel C3D8R hex mesh -> uncoupled thermo-mechanical CalculiX job
+ optional Gmsh per-layer tet remesh
+ optional voxel debug STL.

Pipeline (cylindrical workflow only):

1. Voxelize input STL into cubes of size cube_size in cylindrical param space.
2. Map cubes to curved C3D8R hexes on a cylinder (axis = global +Y).
3. Optionally: for each voxel slice, build a surface mesh (triangulated cubes)
   and let Gmsh volumetrically remesh that slice into tetrahedra, then stack
   all slices into a global C3D4 mesh.
4. Single CalculiX job:
   - Procedure: *UNCOUPLED TEMPERATURE-DISPLACEMENT
   - One step per radial slice:
       * Apply curing (via TEMP DOF) to that slice's NSET.
       * Base nodes (first slice) mechanically fixed.
       * Temperatures and displacements carry over between steps.
"""

import argparse
import os
from typing import List, Tuple, Dict, Set, Optional

import math
import numpy as np
import trimesh
import subprocess
from pygem import FFD  # unused here but kept for compatibility with other variants
from datetime import datetime


def log(msg):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")


# ============================================================
#  Cylindrical mapping helpers
# ============================================================

def world_to_param_cyl(
    point,
    cx: float,
    cz: float,
    R0: float,
):
    x, y, z = point
    dx = x - cx
    dz = z - cz

    u = y
    r = math.sqrt(dx * dx + dz * dz)
    theta = math.atan2(dz, dx)
    v = R0 * theta
    w = r - R0
    return (u, v, w)


def param_to_world_cyl(
    param_point,
    cx: float,
    cz: float,
    R0: float,
):
    u, v, w = param_point
    theta = v / R0
    r = R0 + w

    x = cx + r * math.cos(theta)
    y = u
    z = cz + r * math.sin(theta)
    return (x, y, z)


# ============================================================
#  Voxel mesh -> CalculiX job (cylindrical only)
# ============================================================

def generate_global_cubic_hex_mesh(
    input_stl: str,
    cube_size: float,
    cyl_radius: float,
):
    mesh = trimesh.load(input_stl)

    log(f"[VOXEL] Loaded mesh from {input_stl}")
    log(f"[VOXEL] Watertight: {mesh.is_watertight}, bbox extents: {mesh.extents}")

    # vh: Move mesh on Y axis so its in the middle of Y voxels
    bounds_orig = mesh.bounds  # [[xmin,ymin,zmin], [xmax,ymax,zmax]]
    x_min_o, y_min_o, z_min_o = bounds_orig[0]
    x_max_o, y_max_o, z_max_o = bounds_orig[1]
    y_length = y_max_o - y_min_o
    float_cubes = y_length / cube_size
    ceil_cubes = math.ceil(float_cubes)
    reminder = (ceil_cubes - float_cubes) / 2
    y_offset = -float(y_min_o) - cube_size / 2 + (reminder * cube_size)
    mesh.apply_translation([0.0, y_offset, 0.0])

    # Bounds in voxelization world space (shifted STL)
    bounds = mesh.bounds  # [[xmin,ymin,zmin], [xmax,ymax,zmax]]
    x_min_w, y_min_w, z_min_w = bounds[0]
    x_max_w, y_max_w, z_max_w = bounds[1]

    cyl_params: Optional[Tuple[float, ...]] = None  # (cx, cz, R0, v_offset, y_offset)
    r_lowest: Optional[float] = None
    theta_lowest: Optional[float] = None
    cx_cyl: Optional[float] = None
    cz_cyl: Optional[float] = None
    R0_param: Optional[float] = None  # base radius used for world->param mapping
    lowest_point_world: Optional[Tuple[float, float, float]] = None
    axis_point_world: Optional[Tuple[float, float, float]] = None
    edge_left_world: Optional[Tuple[float, float, float]] = None
    edge_right_world: Optional[Tuple[float, float, float]] = None
    theta_leftmost: Optional[float] = None

    verts_world = mesh.vertices.copy()

    # --- Find lowest point (min Z) in voxelization world space ---
    z_coords = verts_world[:, 2]
    min_z_idx = int(np.argmin(z_coords))
    x_low, y_low, z_low = verts_world[min_z_idx]
    lowest_point_world = (float(x_low), float(y_low), float(z_low))

    # --- Find left-most point (min X) in voxelization world space ---
    x_coords = verts_world[:, 0]
    min_x_idx = int(np.argmin(x_coords))
    x_left, y_left, z_left = verts_world[min_x_idx]

    # Use lowest point X as cylinder center X
    cx_cyl = float(x_low)

    # Decide CZ to place the axis so that lowest point lies on given circle
    base_radius: Optional[float] = None
    base_radius = float(cyl_radius)
    z_center_bbox = 0.5 * (z_min_w + z_max_w)
    cz_plus = z_low + base_radius
    cz_minus = z_low - base_radius

    if abs(cz_plus - z_center_bbox) < abs(cz_minus - z_center_bbox):
        cz_cyl = cz_plus
    else:
        cz_cyl = cz_minus

    # --- compute radii & angles of all vertices wrt chosen axis ---
    dx_all = verts_world[:, 0] - cx_cyl
    dz_all = verts_world[:, 2] - cz_cyl
    r_all = np.sqrt(dx_all * dx_all + dz_all * dz_all)

    # Angle of the left-most bbox vertex relative to cylinder axis
    dx_left = x_left - cx_cyl
    dz_left = z_left - cz_cyl
    theta_leftmost = math.atan2(dz_left, dx_left)

    log(
        f"[VOXEL] Left-most bbox vertex: "
        f"({x_left:.3f}, {y_left:.3f}, {z_left:.3f}), "
        f"theta_left={theta_leftmost:.6f} rad"
    )

    # Radial/angle of the lowest point (with chosen center)
    dx_low = x_low - cx_cyl
    dz_low = z_low - cz_cyl
    r_lowest = math.sqrt(dx_low * dx_low + dz_low * dz_low)
    theta_lowest = math.atan2(dz_low, dx_low)  # angle of lowest point

    # --- Angular edges of the STL around the lowest direction ---------
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

    log(
        "[VOXEL] Angular edges: dtheta_left="
        f"{float(dtheta_all[idx_left]):.6f}, "
        f"dtheta_right={float(dtheta_all[idx_right]):.6f}"
    )

    log(
        f"[VOXEL] Cyl center from lowest Z vertex: "
        f"min_z={z_low:.3f}, lowest_pt=({x_low:.3f}, {y_low:.3f}, {z_low:.3f}), "
        f"r_low={r_lowest:.6f}, theta_low={theta_lowest:.6f} rad"
    )

    # ------------------------------------------------------------
    # Compute effective mapping radius R0
    # ------------------------------------------------------------
    R_true = r_lowest  # should equal base_radius numerically
    half = cube_size * 0.5

    inward_offset = R_true - math.sqrt(R_true * R_true - half * half)
    total_offset = cube_size / 2 + cube_size / 5  # vh: or inward_offset
    R0_param = R_true + total_offset

    # Final cylinder mapping radius
    R0 = float(R0_param)
    log(
        f"[VOXEL] Cylindrical voxel mode: axis=+Y, "
        f"center=({cx_cyl:.3f}, {cz_cyl:.3f}), R0={R0:.6f}"
    )

    # vh: tangential voxel alignment
    theta_cube_size = cube_size / R0
    theta_length = float(dtheta_all[idx_right] - dtheta_all[idx_left])
    float_cubes = theta_length / theta_cube_size
    ceil_cubes = math.ceil(float_cubes)
    reminder = (ceil_cubes - float_cubes) / 2
    log(f"vh2 {theta_cube_size} {theta_length} {float_cubes} {ceil_cubes} {reminder} {theta_leftmost}")
    theta_leftmost = theta_leftmost

    theta_ref = theta_leftmost
    v_ref = R0 * theta_ref
    v_offset = -v_ref
    angle_offset = -theta_ref

    log(
        f"[VOXEL] Rotating so left-most bbox point is at angle 0: "
        f"theta_ref={theta_ref:.6f} rad, "
        f"v_ref={v_ref:.6f}, "
        f"v_offset={v_offset:.6f}, "
        f"angle_offset={angle_offset:.6f} rad"
    )

    # Map mesh into param space (u,v,w) using final (R0)
    verts_param = np.zeros_like(verts_world)
    for i, (x, y, z) in enumerate(verts_world):
        verts_param[i] = world_to_param_cyl(
            (x, y, z),
            cx_cyl,
            cz_cyl,
            R0,
        )

    # Apply tangential rotation so left-most point gets v=0 in param space
    verts_param[:, 1] += v_offset
    mesh.vertices = verts_param

    # ----------------------------------------------------------
    # Create bounding-box mesh of the transformed STL in param space
    # ----------------------------------------------------------
    bounds_param = mesh.bounds  # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    min_x, min_y, min_w = bounds_param[0]
    max_x, max_y, max_w = bounds_param[1]

    bbox_corners = np.array([
        [min_x, min_y, min_w],
        [max_x, min_y, min_w],
        [max_x, max_y, min_w],
        [min_x, max_y, min_w],
        [min_x, min_y, max_w],
        [max_x, min_y, max_w],
        [max_x, max_y, max_w],
        [min_x, max_y, max_w],
    ])

    bbox_faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ])

    bbox_mesh_param = trimesh.Trimesh(
        vertices=bbox_corners,
        faces=bbox_faces,
        process=False
    )

    log(f"[VOXEL] Created param-space bounding box mesh: {bbox_mesh_param.bounds}")

    # --- Tangential (v) extents of the STL in param space -------------
    v_min_mesh = float(verts_param[:, 1].min())
    v_max_mesh = float(verts_param[:, 1].max())
    log(
        f"[VOXEL] Param v extents of STL: v_min={v_min_mesh:.6f}, "
        f"v_max={v_max_mesh:.6f}"
    )

    # --- Voxelization in param coordinates (STL) ---
    log(f"[VOXEL] Voxelizing STL with cube size = {cube_size} ...")
    vox = mesh.voxelized(pitch=cube_size)
    vox.fill()

    indices = vox.sparse_indices  # (N,3) with (ix,iy,iz)
    total_voxels = indices.shape[0]
    log(f"[VOXEL] Total filled voxels (STL cubes): {total_voxels}")

    order = np.lexsort((indices[:, 0], indices[:, 1], indices[:, 2]))
    indices_sorted = indices[order]

    # --- Voxelization in param coordinates (BBOX) ---
    log(f"[VOXEL] Voxelizing bounding-box with cube size = {cube_size} ...")
    vox_bbox = bbox_mesh_param.voxelized(pitch=cube_size)
    vox_bbox.fill()

    indices_bbox = vox_bbox.sparse_indices
    log(f"[VOXEL] Total filled bbox voxels: {indices_bbox.shape[0]}")

    order_bbox = np.lexsort((indices_bbox[:, 0], indices_bbox[:, 1], indices_bbox[:, 2]))
    indices_bbox_sorted = indices_bbox[order_bbox]

    # --- Map iz -> slice "position" in param space: w_center (STL) ---
    unique_iz = np.unique(indices_sorted[:, 2])
    layer_info: List[Tuple[int, float]] = []
    for iz in unique_iz:
        idx_arr = np.array([[0, 0, iz]], dtype=float)
        pt = vox.indices_to_points(idx_arr)[0]
        slice_coord = float(pt[2])
        layer_info.append((int(iz), slice_coord))

    layer_info.sort(key=lambda x: x[1], reverse=True)

    iz_to_slice: Dict[int, int] = {}
    z_slices: List[float] = []
    for slice_idx, (iz, pos) in enumerate(layer_info):
        iz_to_slice[int(iz)] = slice_idx
        z_slices.append(pos)

    # --- Map iz -> slice "position" in param space: w_center (BBOX) ---
    unique_iz_bbox = np.unique(indices_bbox_sorted[:, 2])
    layer_info_bbox: List[Tuple[int, float]] = []
    for iz in unique_iz_bbox:
        idx_arr = np.array([[0, 0, iz]], dtype=float)
        pt = vox_bbox.indices_to_points(idx_arr)[0]
        slice_coord = float(pt[2])
        layer_info_bbox.append((int(iz), slice_coord))

    layer_info_bbox.sort(key=lambda x: x[1], reverse=True)

    iz_to_slice_bbox: Dict[int, int] = {}
    z_slices_bbox: List[float] = []
    for slice_idx, (iz, pos) in enumerate(layer_info_bbox):
        iz_to_slice_bbox[int(iz)] = slice_idx
        z_slices_bbox.append(pos)

    # --- Build voxel-node lattice & hex connectivity (STL) ---
    vertex_index_map: Dict[Tuple[int, int, int], int] = {}
    vertices: List[Tuple[float, float, float]] = []
    hexes: List[Tuple[int, int, int, int, int, int, int, int]] = []
    slice_to_eids: Dict[int, List[int]] = {i: [] for i in range(len(z_slices))}

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

    # --- Build voxel-node lattice & hex connectivity (BBOX) ---
    vertex_index_map_bbox: Dict[Tuple[int, int, int], int] = {}
    vertices_bbox: List[Tuple[float, float, float]] = []
    hexes_bbox: List[Tuple[int, int, int, int, int, int, int, int]] = []
    slice_to_eids_bbox: Dict[int, List[int]] = {
        i: [] for i in range(len(z_slices_bbox))
    }

    def get_vertex_index_bbox(
        key: Tuple[int, int, int],
        coord: Tuple[float, float, float],
    ) -> int:
        if key in vertex_index_map_bbox:
            return vertex_index_map_bbox[key]
        idx = len(vertices_bbox) + 1
        vertex_index_map_bbox[key] = idx
        vertices_bbox.append(coord)
        return idx

    log("[VOXEL] Building voxel-node lattice (curved hexa nodes) for STL ...")

    half = cube_size * 0.5

    for (ix, iy, iz) in indices_sorted:
        center_param = vox.indices_to_points(
            np.array([[ix, iy, iz]], dtype=float)
        )[0]
        u_c, v_c, w_c = center_param

        v_c -= v_offset

        u0, u1 = u_c - half, u_c + half
        v0, v1 = v_c - half, v_c + half
        w0, w1 = w_c - half, w_c + half

        p0 = (u0, v0, w0)
        p1 = (u1, v0, w0)
        p2 = (u1, v1, w0)
        p3 = (u0, v1, w0)
        p4 = (u0, v0, w1)
        p5 = (u1, v0, w1)
        p6 = (u1, v1, w1)
        p7 = (u0, v1, w1)

        x0, y0, z0 = param_to_world_cyl(p0, cx_cyl, cz_cyl, R0)
        x1, y1, z1 = param_to_world_cyl(p1, cx_cyl, cz_cyl, R0)
        x2, y2, z2 = param_to_world_cyl(p2, cx_cyl, cz_cyl, R0)
        x3, y3, z3 = param_to_world_cyl(p3, cx_cyl, cz_cyl, R0)
        x4, y4, z4 = param_to_world_cyl(p4, cx_cyl, cz_cyl, R0)
        x5, y5, z5 = param_to_world_cyl(p5, cx_cyl, cz_cyl, R0)
        x6, y6, z6 = param_to_world_cyl(p6, cx_cyl, cz_cyl, R0)
        x7, y7, z7 = param_to_world_cyl(p7, cx_cyl, cz_cyl, R0)

        v0_idx = get_vertex_index((ix,   iy,   iz),   (x0, y0, z0))
        v1_idx = get_vertex_index((ix+1, iy,   iz),   (x1, y1, z1))
        v2_idx = get_vertex_index((ix+1, iy+1, iz),   (x2, y2, z2))
        v3_idx = get_vertex_index((ix,   iy+1, iz),   (x3, y3, z3))
        v4_idx = get_vertex_index((ix,   iy,   iz+1), (x4, y4, z4))
        v5_idx = get_vertex_index((ix+1, iy,   iz+1), (x5, y5, z5))
        v6_idx = get_vertex_index((ix+1, iy+1, iz+1), (x6, y6, z6))
        v7_idx = get_vertex_index((ix,   iy+1, iz+1), (x7, y7, z7))

        hexes.append((v0_idx, v1_idx, v2_idx, v3_idx,
                      v4_idx, v5_idx, v6_idx, v7_idx))

        eid = len(hexes)
        slice_idx = iz_to_slice[int(iz)]
        slice_to_eids[slice_idx].append(eid)

    log("[VOXEL] Building voxel-node lattice (curved hexa nodes) for BBOX ...")

    for (ix, iy, iz) in indices_bbox_sorted:
        center_param = vox_bbox.indices_to_points(
            np.array([[ix, iy, iz]], dtype=float)
        )[0]
        u_c, v_c, w_c = center_param

        v_c -= v_offset

        u0, u1 = u_c - half, u_c + half
        v0, v1 = v_c - half, v_c + half
        w0, w1 = w_c - half, w_c + half

        p0 = (u0, v0, w0)
        p1 = (u1, v0, w0)
        p2 = (u1, v1, w0)
        p3 = (u0, v1, w0)
        p4 = (u0, v0, w1)
        p5 = (u1, v0, w1)
        p6 = (u1, v1, w1)
        p7 = (u0, v1, w1)

        x0, y0, z0 = param_to_world_cyl(p0, cx_cyl, cz_cyl, R0)
        x1, y1, z1 = param_to_world_cyl(p1, cx_cyl, cz_cyl, R0)
        x2, y2, z2 = param_to_world_cyl(p2, cx_cyl, cz_cyl, R0)
        x3, y3, z3 = param_to_world_cyl(p3, cx_cyl, cz_cyl, R0)
        x4, y4, z4 = param_to_world_cyl(p4, cx_cyl, cz_cyl, R0)
        x5, y5, z5 = param_to_world_cyl(p5, cx_cyl, cz_cyl, R0)
        x6, y6, z6 = param_to_world_cyl(p6, cx_cyl, cz_cyl, R0)
        x7, y7, z7 = param_to_world_cyl(p7, cx_cyl, cz_cyl, R0)

        v0_idx = get_vertex_index_bbox((ix,   iy,   iz),   (x0, y0, z0))
        v1_idx = get_vertex_index_bbox((ix+1, iy,   iz),   (x1, y1, z1))
        v2_idx = get_vertex_index_bbox((ix+1, iy+1, iz),   (x2, y2, z2))
        v3_idx = get_vertex_index_bbox((ix,   iy+1, iz),   (x3, y3, z3))
        v4_idx = get_vertex_index_bbox((ix,   iy,   iz+1), (x4, y4, z4))
        v5_idx = get_vertex_index_bbox((ix+1, iy,   iz+1), (x5, y5, z5))
        v6_idx = get_vertex_index_bbox((ix+1, iy+1, iz+1), (x6, y6, z6))
        v7_idx = get_vertex_index_bbox((ix,   iy+1, iz+1), (x7, y7, z7))

        hexes_bbox.append((v0_idx, v1_idx, v2_idx, v3_idx,
                           v4_idx, v5_idx, v6_idx, v7_idx))

        eid = len(hexes_bbox)
        slice_idx = iz_to_slice_bbox[int(iz)]
        slice_to_eids_bbox[slice_idx].append(eid)

    cyl_params = (cx_cyl, cz_cyl, R0, v_offset, y_offset)

    log(
        f"[VOXEL] Built STL mesh (voxelization frame): {len(vertices)} nodes, "
        f"{len(hexes)} hex elements, {len(z_slices)} radial slices."
    )
    log(
        f"[VOXEL] Built BBOX mesh (voxelization frame): {len(vertices_bbox)} nodes, "
        f"{len(hexes_bbox)} hex elements, {len(z_slices_bbox)} radial slices."
    )

    # --- Undo Y-shift for returned data so voxels sit back where STL was ---
    vertices = [(x, y - y_offset, z) for (x, y, z) in vertices]
    vertices_bbox = [(x, y - y_offset, z) for (x, y, z) in vertices_bbox]

    if lowest_point_world is not None:
        lx, ly, lz = lowest_point_world
        lowest_point_world = (lx, ly - y_offset, lz)

    if axis_point_world is not None:
        ax, ay, az = axis_point_world
        axis_point_world = (ax, ay - y_offset, az)

    if edge_left_world is not None:
        ex, ey, ez = edge_left_world
        edge_left_world = (ex, ey - y_offset, ez)

    if edge_right_world is not None:
        rx, ry, rz = edge_right_world
        edge_right_world = (rx, ry - y_offset, rz)

    log(
        f"[VOXEL] Shifted voxel nodes and marker points back by {-y_offset:.6f} in Y "
        "(original STL frame)"
    )

    indices_bbox_sorted = indices_bbox_sorted[:, [1, 0, 2]]
    vertex_index_map_bbox = {
        (iy, ix, iz): idx
        for (ix, iy, iz), idx in vertex_index_map_bbox.items()
    }

    return (
        # STL mesh results
        vertices,
        hexes,
        slice_to_eids,
        z_slices,
        cyl_params,
        lowest_point_world,
        axis_point_world,
        edge_left_world,
        edge_right_world,
        indices_sorted,      # for STL debug export
        vox,                 # for STL debug export

        # BBOX mesh results
        vertices_bbox,
        hexes_bbox,
        slice_to_eids_bbox,
        z_slices_bbox,
        indices_bbox_sorted, # for BBOX debug export
        vox_bbox,            # for BBOX debug export
        vertex_index_map_bbox,
    )


# ============================================================
#  Gmsh per-layer remesh helpers
# ============================================================

def build_layer_trimesh(
    slice_voxels: List[Tuple[int, int, int]],
    vox,
    cube_size: float,
    cyl_params: Tuple[float, float, float, float, float],
) -> trimesh.Trimesh:
    """
    Build ONLY the external surface of a voxel layer:
    - For each voxel, check 6 neighbors.
    - Emit a face only if the neighbor in that direction is absent.
    => no internal faces, so no overlapping facets in the boundary mesh.
    """
    cx, cz, R0, v_offset, y_offset = cyl_params
    half = cube_size * 0.5

    slice_set: Set[Tuple[int, int, int]] = set(
        (int(ix), int(iy), int(iz)) for (ix, iy, iz) in slice_voxels
    )

    vertices: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []

    # Neighbor directions in index-space
    neighbor_dirs = {
        "xm": (-1, 0, 0),
        "xp": (1, 0, 0),
        "ym": (0, -1, 0),
        "yp": (0, 1, 0),
        "zm": (0, 0, -1),
        "zp": (0, 0, 1),
    }

    # For each face, which 4 corner indices (in the 0..7 cube corner list) form that quad
    face_quads = {
        "zm": (0, 1, 2, 3),  # bottom in w-
        "zp": (4, 5, 6, 7),  # top in w+
        "xm": (0, 3, 7, 4),
        "xp": (1, 2, 6, 5),
        "ym": (0, 1, 5, 4),
        "yp": (3, 2, 6, 7),
    }

    for (ix, iy, iz) in slice_voxels:
        ix = int(ix)
        iy = int(iy)
        iz = int(iz)

        center_param = vox.indices_to_points(
            np.array([[ix, iy, iz]], dtype=float)
        )[0]
        u_c, v_c, w_c = center_param

        v_c -= v_offset

        u0, u1 = u_c - half, u_c + half
        v0, v1 = v_c - half, v_c + half
        w0, w1 = w_c - half, w_c + half

        corners_param = [
            (u0, v0, w0),  # 0
            (u1, v0, w0),  # 1
            (u1, v1, w0),  # 2
            (u0, v1, w0),  # 3
            (u0, v0, w1),  # 4
            (u1, v0, w1),  # 5
            (u1, v1, w1),  # 6
            (u0, v1, w1),  # 7
        ]

        corners_world = []
        for p in corners_param:
            x, y, z = param_to_world_cyl(p, cx, cz, R0)
            y -= y_offset
            corners_world.append((x, y, z))

        for face_key, (dx, dy, dz) in neighbor_dirs.items():
            neigh = (ix + dx, iy + dy, iz + dz)
            if neigh in slice_set:
                # Internal face → skip
                continue

            q = face_quads[face_key]
            base = len(vertices)
            # append 4 vertices of this quad
            for idx in q:
                vertices.append(corners_world[idx])

            # two triangles: (0,1,2) and (0,2,3) in local quad coords
            faces.append((base + 0, base + 1, base + 2))
            faces.append((base + 0, base + 2, base + 3))

    if not vertices or not faces:
        return trimesh.Trimesh(vertices=[], faces=[],
                               process=False)

    mesh = trimesh.Trimesh(
        vertices=np.array(vertices, dtype=float),
        faces=np.array(faces, dtype=np.int32),
        process=False,
    )

    # Cleanup to avoid any lingering duplicates/degenerates
    try:
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    return mesh


def gmsh_remesh_layer(
    layer_mesh: trimesh.Trimesh,
    target_size: Optional[float],
) -> Tuple[Optional[List[Tuple[float, float, float]]],
           Optional[List[Tuple[int, int, int, int]]]]:
    """
    Use Gmsh to volumetrically remesh a single layer surface mesh into tets.

    Supports multiple disconnected components:
      - Split the layer mesh into connected components.
      - Remesh each component separately in Gmsh.
      - Concatenate nodes/tets with proper index offsets.

    Returns:
      (vertices_all, tets_all)
      where node indices in tets_all are 1..len(vertices_all),
      or (None, None) on failure.
    """
    if layer_mesh.is_empty:
        log("[GMSH] Layer mesh is empty; skipping.")
        return None, None

    try:
        import gmsh
    except ImportError:
        log("[GMSH] gmsh module not found; cannot remesh layer.")
        return None, None

    import tempfile

    def _remesh_one_component(comp: trimesh.Trimesh):
        """Remesh a single connected component with Gmsh (returns local verts/tets)."""
        if comp.vertices.size == 0 or comp.faces.size == 0:
            return None, None

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".stl")
        os.close(tmp_fd)

        try:
            comp.export(tmp_path)

            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("layer_comp")

            gmsh.merge(tmp_path)

            if target_size is not None and target_size > 0.0:
                gmsh.option.setNumber("Mesh.MeshSizeMin", target_size)
                gmsh.option.setNumber("Mesh.MeshSizeMax", target_size)

            angle = 40.0
            forceParametrizablePatches = False
            includeBoundary = True
            curveAngle = 180.0

            gmsh.model.mesh.classifySurfaces(
                angle * math.pi / 180.0,
                includeBoundary,
                forceParametrizablePatches,
                curveAngle * math.pi / 180.0,
            )
            gmsh.model.mesh.createGeometry()

            surfs = gmsh.model.getEntities(2)
            if surfs is None or len(surfs) == 0:
                log("[GMSH] Component remesh: no surfaces after classifySurfaces.")
                return None, None

            surf_tags = [s[1] for s in surfs]
            sl = gmsh.model.geo.addSurfaceLoop(surf_tags)
            vol = gmsh.model.geo.addVolume([sl])
            gmsh.model.geo.synchronize()

            gmsh.model.mesh.generate(3)

            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            if node_tags.size == 0:
                log("[GMSH] Component remesh: no nodes after generate.")
                return None, None

            tag_to_local: Dict[int, int] = {}
            vertices_local: List[Tuple[float, float, float]] = [None] * len(node_tags)  # type: ignore

            for i, tag in enumerate(node_tags):
                x = float(node_coords[3 * i + 0])
                y = float(node_coords[3 * i + 1])
                z = float(node_coords[3 * i + 2])
                idx = i + 1
                tag_to_local[int(tag)] = idx
                vertices_local[i] = (x, y, z)

            tet_type = gmsh.model.mesh.getElementType("Tetrahedron", 1)
            types, elem_tags_list, elem_nodes_list = gmsh.model.mesh.getElements(3, vol)

            if not types:
                log("[GMSH] Component remesh: no volume elements.")
                return None, None

            tet_idx = None
            for i, t in enumerate(types):
                if t == tet_type:
                    tet_idx = i
                    break

            if tet_idx is None:
                log("[GMSH] Component remesh: no tetrahedron elements.")
                return None, None

            elem_nodes_flat = elem_nodes_list[tet_idx]
            if len(elem_nodes_flat) % 4 != 0:
                log("[GMSH] Component remesh: bad tetra connectivity.")
                return None, None

            tets_local: List[Tuple[int, int, int, int]] = []
            for i in range(0, len(elem_nodes_flat), 4):
                conn_tags = elem_nodes_flat[i:i+4]
                try:
                    conn = tuple(tag_to_local[int(t)] for t in conn_tags)
                except KeyError:
                    log("[GMSH] Component remesh: unknown node tag in connectivity.")
                    return None, None
                tets_local.append(conn)

            return vertices_local, tets_local

        finally:
            try:
                gmsh.finalize()
            except Exception:
                pass
            try:
                os.remove(tmp_path)
            except OSError:
                pass


    try:
        return _remesh_one_component(layer_mesh)
    except Exception as e:
        log(f"[GMSH]   Exception during component remesh: {e}")
        return None, None


# ============================================================
#  CalculiX runner
# ============================================================

def run_calculix(job_name: str, ccx_cmd: str = "ccx"):
    log(f"[RUN] Launching CalculiX: {ccx_cmd} {job_name}")

    log_path = f"{job_name}_ccx_output.txt"
    try:
        my_env = os.environ.copy()
        my_env["OMP_NUM_THREADS"] = "6"
        my_env["OMP_DYNAMIC"] = "FALSE"

        with open(log_path, "w", encoding="utf-8") as logfile:
            proc = subprocess.Popen(
                [ccx_cmd, job_name],
                stdout=logfile,
                stderr=subprocess.STDOUT,
                text=True,
                env=my_env
            )
    except FileNotFoundError:
        log(f"[RUN] ERROR: CalculiX command not found: {ccx_cmd}")
        return False

    rc = proc.wait()
    log(f"[RUN] CalculiX completed with return code {rc}")
    log(f"[RUN] Full output written to: {log_path}")

    return rc == 0


def export_voxel_debug_stl(
    indices_sorted,
    vox,
    cube_size,
    cyl_params,
    out_path
):
    cx, cz, R0, v_offset, y_offset = cyl_params

    half = cube_size / 2.0
    vertices = []
    faces = []

    for (ix, iy, iz) in indices_sorted:
        center_param = vox.indices_to_points(
            np.array([[ix, iy, iz]], dtype=float)
        )[0]
        u_c, v_c, w_c = center_param

        v_c -= v_offset

        u0, u1 = u_c - half, u_c + half
        v0, v1 = v_c - half, v_c + half
        w0, w1 = w_c - half, w_c + half

        corners = [
            (u0, v0, w0),
            (u1, v0, w0),
            (u1, v1, w0),
            (u0, v1, w0),
            (u0, v0, w1),
            (u1, v0, w1),
            (u1, v1, w1),
            (u0, v1, w1),
        ]

        world = []
        for p in corners:
            x, y, z = param_to_world_cyl(p, cx, cz, R0)
            y = y - y_offset
            world.append((x, y, z))

        base = len(vertices)
        vertices.extend(world)

        cube_faces = [
            (0, 1, 2), (0, 2, 3),
            (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4),
            (1, 2, 6), (1, 6, 5),
            (2, 3, 7), (2, 7, 6),
            (3, 0, 4), (3, 4, 7),
        ]
        for (a, b, c) in cube_faces:
            faces.append((base + a, base + b, base + c))

    vertices_np = np.array(vertices, dtype=float)
    faces_np = np.array(faces, dtype=np.int32)
    mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np, process=False)
    mesh.export(out_path)
    log(f"[VOXDBG] Voxel debug STL written: {out_path}")


# ============================================================
#  CalculiX job writer (supports C3D8R and C3D4)
# ============================================================

def write_calculix_job(
    path: str,
    vertices: List[Tuple[float, float, float]],
    elements: List[Tuple[int, ...]],
    slice_to_eids: Dict[int, List[int]],
    z_slices: List[float],
    shrinkage_curve: List[float],
    cure_shrink_per_unit: float,
    output_stride: int = 1,
):
    n_nodes = len(vertices)
    n_elems = len(elements)
    n_slices = len(z_slices)

    time_per_layer = 1.0
    time_per_layer_step = 1.0
    total_weight = float(sum(shrinkage_curve))
    shrinkage_curve = [float(w) / total_weight for w in shrinkage_curve]

    if output_stride < 1:
        output_stride = 1

    nen = len(elements[0]) if elements else 0
    is_tet = nen == 4

    with open(path, "w", encoding="utf-8") as f:
        f.write("**\n** Auto-generated incremental-cure shrink job\n**\n")
        f.write("*HEADING\n")
        f.write(
            "Voxel cylindrical mesh uncoupled temperature-displacement "
            "(layer-wise MODEL CHANGE + shrinkage-curve-driven curing)\n"
        )

        # NODES
        f.write("** Nodes +++++++++++++++++++++++++++++++++++++++++++++++++++\n")
        f.write("*NODE\n")
        for i, (x, y, z) in enumerate(vertices, start=1):
            f.write(f"{i}, {x}, {y}, {z}\n")

        # ELEMENTS
        f.write("** Elements ++++++++++++++++++++++++++++++++++++++++++++++++\n")
        if is_tet:
            f.write("*ELEMENT, TYPE=C3D4, ELSET=ALL\n")
            for eid, (n0, n1, n2, n3) in enumerate(elements, start=1):
                f.write(f"{eid}, {n0}, {n1}, {n2}, {n3}\n")
        else:
            f.write("*ELEMENT, TYPE=C3D8R, ELSET=ALL\n")
            for eid, (n0, n1, n2, n3, n4, n5, n6, n7) in enumerate(elements, start=1):
                f.write(
                    f"{eid}, {n0}, {n1}, {n2}, {n3}, "
                    f"{n4}, {n5}, {n6}, {n7}\n"
                )

        # NODE SETS
        f.write("** Node sets +++++++++++++++++++++++++++++++++++++++++++++++\n")
        f.write("*NSET, NSET=ALLNODES, GENERATE\n")
        f.write(f"1, {n_nodes}, 1\n")

        base_nodes: List[int] = []

        f.write("** Element + node sets (per slice) +++++++++++++++++++++++++\n")
        slice_names: List[str] = []
        slice_node_ids: Dict[int, List[int]] = {}

        def _write_id_list_lines(ff, ids: List[int], per_line: int = 16):
            for i in range(0, len(ids), per_line):
                chunk = ids[i:i+per_line]
                ff.write(", ".join(str(x) for x in chunk) + "\n")

        for slice_idx in range(n_slices):
            eids = slice_to_eids.get(slice_idx, [])
            if not eids:
                continue
            valid_eids = [eid for eid in eids if 1 <= eid <= n_elems]
            if not valid_eids:
                log(
                    f"[WARN] Slice {slice_idx} has no valid element "
                    f"IDs within 1..{n_elems}"
                )
                continue

            name = f"SLICE_{slice_idx:03d}"
            slice_names.append(name)
            f.write(f"*ELSET, ELSET={name}\n")
            _write_id_list_lines(f, valid_eids)

            nodes_in_slice: Set[int] = set()
            for eid in valid_eids:
                conn = elements[eid - 1]
                nodes_in_slice.update(conn)

            node_list = sorted(nodes_in_slice)
            slice_node_ids[slice_idx] = node_list

            nset_name = f"{name}_NODES"
            f.write(f"*NSET, NSET={nset_name}\n")
            _write_id_list_lines(f, node_list)

        # BASE node set: first slice, all its nodes (for tets and hex)
        existing_slice_idxs = sorted(slice_node_ids.keys())
        if existing_slice_idxs:
            base_slice = existing_slice_idxs[0]
            base_nodes = slice_node_ids[base_slice]

            f.write("** Base node set = first slice nodes ++++++++++++++++++++++\n")
            f.write("*NSET, NSET=BASE\n")
            _write_id_list_lines(f, base_nodes)
        else:
            f.write("** Warning: no slices -> no BASE node set.\n")
            base_nodes = []

        # MATERIAL
        f.write("** Materials +++++++++++++++++++++++++++++++++++++++++++++++\n")
        f.write("*MATERIAL, NAME=ABS\n")
        f.write("*DENSITY\n")
        f.write("1.12e-9\n")
        f.write("*ELASTIC\n")
        f.write("2800., 0.35\n")

        alpha = -float(cure_shrink_per_unit)
        f.write("*EXPANSION, ZERO=0.\n")
        f.write(f"{alpha:.6E}\n")

        f.write("*CONDUCTIVITY\n")
        f.write("0.20\n")
        f.write("*SPECIFIC HEAT\n")
        f.write("1.30e+9\n")
        f.write("*SOLID SECTION, ELSET=ALL, MATERIAL=ABS\n")

        # INITIAL CONDITIONS
        f.write("** Initial conditions (cure variable) ++++++++++++++++++++++\n")
        f.write("*INITIAL CONDITIONS, TYPE=TEMPERATURE\n")
        f.write("ALLNODES, 0.0\n")

        # STEPS
        f.write("** Steps +++++++++++++++++++++++++++++++++++++++++++++++++++\n")
        if n_slices == 0 or not slice_names:
            f.write("** No slices -> no steps.\n")
        else:
            existing_slice_idxs = [int(name.split("_")[1]) for name in slice_names]
            existing_slice_idxs.sort()

            cure_state: Dict[int, float] = {idx: 0.0 for idx in existing_slice_idxs}
            applied_count: Dict[int, int] = {idx: 0 for idx in existing_slice_idxs}
            printed: Dict[int, bool] = {idx: False for idx in existing_slice_idxs}

            step_counter = 1

            f.write("** --------------------------------------------------------\n")
            f.write("** Step 1: initial dummy step with full model (no curing)\n")
            f.write("** --------------------------------------------------------\n")
            f.write("*STEP\n")
            f.write("*UNCOUPLED TEMPERATURE-DISPLACEMENT, SOLVER=PASTIX\n")
            f.write(f"{time_per_layer_step:.6f}, {time_per_layer:.6f}\n")

            if base_nodes:
                f.write("** Boundary conditions (mechanical) +++++++++++++++++++++++\n")
                f.write("*BOUNDARY\n")
                f.write("BASE, 1, 6, 0.\n")

            f.write("*END STEP\n")
            step_counter += 1

            curve_len = len(shrinkage_curve)
            total_cure_steps = len(existing_slice_idxs) + curve_len - 1

            for global_k in range(total_cure_steps):
                slice_to_add: Optional[int] = None
                if global_k < len(existing_slice_idxs):
                    slice_to_add = existing_slice_idxs[global_k]

                if slice_to_add is not None:
                    printed[slice_to_add] = True

                prev_cure_state = cure_state.copy()

                for j in existing_slice_idxs:
                    if not printed[j]:
                        continue
                    k_applied = applied_count[j]
                    if k_applied >= curve_len:
                        continue
                    increment = shrinkage_curve[k_applied]
                    if increment != 0.0:
                        cure_state[j] = min(
                            1.0,
                            cure_state[j] + increment,
                        )
                    applied_count[j] = k_applied + 1

                f.write("** --------------------------------------------------------\n")
                if slice_to_add is not None:
                    name = f"SLICE_{slice_to_add:03d}"
                    z_val = z_slices[slice_to_add]
                    f.write(
                        f"** Step {step_counter}: add slice {name} at param w = {z_val} "
                        f"and advance shrinkage curve\n"
                    )
                else:
                    f.write(
                        f"** Step {step_counter}: post-cure step "
                        f"(no new slices, advance shrinkage curve)\n"
                    )
                f.write("** --------------------------------------------------------\n")
                f.write("*STEP\n")
                f.write("*UNCOUPLED TEMPERATURE-DISPLACEMENT, SOLVER=PASTIX\n")
                f.write(f"{time_per_layer_step:.6f}, {time_per_layer:.6f}\n")

                if slice_to_add is not None:
                    name = f"SLICE_{slice_to_add:03d}"
                    if slice_to_add == existing_slice_idxs[0]:
                        remove = [
                            f"SLICE_{other:03d}"
                            for other in existing_slice_idxs
                            if other != slice_to_add
                        ]
                        if remove:
                            f.write("** Model change: keep only the first slice active\n")
                            f.write("*MODEL CHANGE, TYPE=ELEMENT, REMOVE\n")
                            for nm in remove:
                                f.write(f"{nm}\n")
                    else:
                        f.write("** Model change: add new slice\n")
                        f.write("*MODEL CHANGE, TYPE=ELEMENT, ADD\n")
                        f.write(f"{name}\n")

                write_outputs = (
                    output_stride <= 1
                    or (global_k + 1) % output_stride == 0
                    or global_k == total_cure_steps - 1
                )

                if write_outputs:
                    f.write("** Field outputs +++++++++++++++++++++++++++++++++++++++++++\n")
                    f.write("*NODE FILE\n")
                    f.write("U\n")
                else:
                    f.write(
                        "** Field outputs disabled for this step "
                        f"(output_stride = {output_stride})\n"
                    )
                    f.write("*NODE FILE\n")

                f.write("** Boundary conditions (base + shrinkage-curve cure) +++++\n")
                f.write("*BOUNDARY\n")

                for j in existing_slice_idxs:
                    if not printed[j]:
                        continue
                    cure_val = cure_state[j]
                    if cure_val == 0.0:
                        continue
                    if cure_val == prev_cure_state.get(j, 0.0):
                        continue
                    nset_j = f"SLICE_{j:03d}_NODES"
                    f.write(f"{nset_j}, 11, 11, {cure_val:.6f}\n")

                f.write("*END STEP\n")
                step_counter += 1

    log(f"[CCX] Wrote incremental-cure UT-D job to: {path}")
    log(
        f"[CCX] Nodes: {n_nodes}, elements: {n_elems}, "
        f"slices: {n_slices}, shrinkage_curve={shrinkage_curve}, "
        f"cure_shrink_per_unit={cure_shrink_per_unit}"
    )


# ============================================================
#  CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Voxelize an STL in cylindrical coordinates and generate an "
            "uncoupled thermo-mechanical CalculiX job with C3D8R hexahedra "
            "and layer-by-layer *UNCOUPLED TEMPERATURE-DISPLACEMENT steps. "
            "Optionally remesh each voxel slice with Gmsh into tetrahedra."
        )
    )
    parser.add_argument("input_stl", help="Path to input STL file")
    parser.add_argument(
        "--out-dir",
        "-o",
        default=".",
        help="Directory for all output files (default: current directory).",
    )
    parser.add_argument(
        "--cube-size",
        "-s",
        type=float,
        required=True,
        help="Edge length of each voxel cube (same units as STL, e.g. mm)",
    )
    parser.add_argument(
        "--run-ccx",
        action="store_true",
        help="If set, run CalculiX on the generated job.",
    )
    parser.add_argument(
        "--ccx-cmd",
        default="ccx",
        help="CalculiX executable (default 'ccx').",
    )
    parser.add_argument(
        "--output-stride",
        type=int,
        default=1,
        help=(
            "Request nodal outputs only every Nth curing step (>=1). "
            "The very last curing step always produces output."
        ),
    )
    parser.add_argument(
        "--export-lattice",
        action="store_true",
        help=(
            "Unused here; placeholder for other variants."
        ),
    )
    parser.add_argument(
        "--cyl-radius",
        type=float,
        required=True,
        help="Base cylinder radius R0 for param mapping.",
    )
    parser.add_argument(
        "--gmsh-size",
        type=float,
        default=None,
        help=(
            "If set, remesh each voxel slice with Gmsh into tetrahedra with "
            "approximate target size. If omitted, CalculiX uses the original "
            "voxel C3D8R mesh."
        ),
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input_stl):
        log(f"Input STL not found: {args.input_stl}")
        raise SystemExit(1)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    job_base = os.path.splitext(os.path.basename(args.input_stl))[0]
    basepath = os.path.join(out_dir, job_base)

    (
        vertices,
        hexes,
        slice_to_eids,
        z_slices,
        cyl_params,
        lowest_point,
        axis_point,
        edge_left_point,
        edge_right_point,
        indices_sorted,
        vox,

        vertices_bbox,
        hexes_bbox,
        slice_to_eids_bbox,
        z_slices_bbox,
        indices_bbox_sorted,
        vox_bbox,
        vertex_index_map_bbox,
    ) = generate_global_cubic_hex_mesh(
        args.input_stl,
        args.cube_size,
        cyl_radius=args.cyl_radius,
    )

    # Optional Gmsh per-layer remesh into tets
    if args.gmsh_size is not None:
        log(
            f"[GMSH] Starting per-layer volumetric TET remesh with target size "
            f"{args.gmsh_size} ..."
        )
        try:
            import gmsh  # just to check availability
        except ImportError:
            log("[GMSH] gmsh module not found; skipping remesh.")
        else:
            # Build mapping slice_idx -> list of voxel indices (ix,iy,iz)
            izs = indices_sorted[:, 2]
            unique_iz = np.unique(izs)
            z_arr = np.array(z_slices, dtype=float)

            iz_to_slice = {}
            for iz in unique_iz:
                idx_arr = np.array([[0, 0, iz]], dtype=float)
                pt = vox.indices_to_points(idx_arr)[0]
                w = float(pt[2])
                slice_idx = int(np.argmin(np.abs(z_arr - w)))
                iz_to_slice[int(iz)] = slice_idx

            slice_voxel_map: Dict[int, List[Tuple[int, int, int]]] = {
                i: [] for i in range(len(z_slices))
            }
            for (ix, iy, iz) in indices_sorted:
                sidx = iz_to_slice[int(iz)]
                slice_voxel_map[sidx].append((int(ix), int(iy), int(iz)))

            vertices_tet: List[Tuple[float, float, float]] = []
            elements_tet: List[Tuple[int, int, int, int]] = []
            slice_to_eids_tet: Dict[int, List[int]] = {i: [] for i in range(len(z_slices))}
            global_node_offset = 0
            global_elem_offset = 0
            remesh_ok = True

            for slice_idx in range(len(z_slices)):
                voxels = slice_voxel_map.get(slice_idx, [])
                if not voxels:
                    continue

                log(f"[GMSH] Remeshing slice {slice_idx} with {len(voxels)} voxels...")
                layer_mesh = build_layer_trimesh(
                    voxels,
                    vox,
                    args.cube_size,
                    cyl_params,
                )
                verts_local, tets_local = gmsh_remesh_layer(
                    layer_mesh,
                    args.gmsh_size,
                )

                if verts_local is None or tets_local is None:
                    log(f"[GMSH] Slice {slice_idx} remesh failed; aborting Gmsh remesh.")
                    remesh_ok = False
                    break

                node_map: Dict[int, int] = {}
                for local_id, (x, y, z) in enumerate(verts_local, start=1):
                    global_id = global_node_offset + local_id
                    vertices_tet.append((x, y, z))
                    node_map[local_id] = global_id
                global_node_offset += len(verts_local)

                eids_slice: List[int] = []
                for i, tet in enumerate(tets_local):
                    conn_global = tuple(node_map[n] for n in tet)
                    elements_tet.append(conn_global)
                    eid_global = global_elem_offset + i + 1
                    eids_slice.append(eid_global)
                global_elem_offset += len(tets_local)

                slice_to_eids_tet[slice_idx].extend(eids_slice)

            if remesh_ok and elements_tet:
                log(
                    f"[GMSH] Per-layer tet remesh OK: "
                    f"{len(vertices)} -> {len(vertices_tet)} nodes, "
                    f"{len(hexes)} -> {len(elements_tet)} elements."
                )
                vertices = vertices_tet
                hexes = elements_tet  # now actually C3D4
                slice_to_eids = slice_to_eids_tet
            else:
                log("[GMSH] Using original voxel C3D8R mesh (tet remesh skipped).")

    # Export voxel debug STL (always based on voxel grid)
    export_voxel_debug_stl(
        indices_sorted,
        vox,
        args.cube_size,
        cyl_params,
        basepath + "_voxels.stl"
    )

    # Generate CalculiX job
    utd_job = basepath + "_utd"
    utd_inp = utd_job + ".inp"

    write_calculix_job(
        utd_inp,
        vertices,
        hexes,
        slice_to_eids,
        z_slices,
        shrinkage_curve=[5, 4, 3, 2, 1],
        cure_shrink_per_unit=0.1,
        output_stride=args.output_stride,
    )

    if args.run_ccx:
        ok = run_calculix(utd_job, ccx_cmd=args.ccx_cmd)
        if not ok:
            log("[RUN] UTD job failed.")


if __name__ == "__main__":
    main()

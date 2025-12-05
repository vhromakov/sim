#!/usr/bin/env python3
"""
STL -> cylindrical voxel C3D8R hex mesh -> uncoupled thermo-mechanical CalculiX job
+ FFD deformation of the original STL using PyGeM (forward + pre-deformed)
+ optional lattice visualization as separate PLY point clouds.

Pipeline (cylindrical workflow only):

1. Voxelize input STL into cubes of size cube_size in cylindrical param space.
2. Map cubes to curved C3D8R hexes on a cylinder (axis = global +Y).
3. Single CalculiX job:
   - Procedure: *UNCOUPLED TEMPERATURE-DISPLACEMENT
   - One step per radial slice:
       * Apply curing (via TEMP DOF) to that slice's NSET.
       * Base nodes (outer radial face of first radial slice) mechanically fixed.
       * Temperatures and displacements carry over between steps.
4. If --run-ccx:
   - Run the thermo-mechanical job -> <job_name>_utd.frd
   - Read nodal displacements from that .frd
   - Build PyGeM FFD lattice matching voxel grid
   - forward:  deform input STL -> <job_name>_deformed.stl  (PyGeM)
   - inverse: pre-deform STL   -> <job_name>_deformed_pre.stl  (by flipping FFD control weights)
   - if --export-lattice:
        * <job_name>_lattice_orig.ply  (original lattice nodes)
        * <job_name>_lattice_def.ply   (deformed lattice nodes)
"""

import argparse
import os
from typing import List, Tuple, Dict, Set, Optional

import math
import numpy as np
import trimesh
import subprocess
from pygem import FFD
from datetime import datetime

import os
import json
import numpy as np


def log(msg):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")


# ============================================================
#  Cylindrical mapping helpers (copied from gen_grid.py)
# ============================================================

def world_to_param_cyl(
    point,
    cx: float,
    cz: float,
    R0: float,
):
    """
    Map world coordinates (x, y, z) to cylindrical param space (u, v, w).

    Cylinder axis is along global Y and centered at (cx, cz) in XZ plane.
    R0 is the base cylinder radius used for angular mapping.

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
    """
    Map param space point (u, v, w) back to world (x, y, z)
    using the same cylinder definition.

    Args:
        param_point: (u, v, w)
        cx, cz: cylinder center in XZ plane
        R0: base radius used for angular mapping

    Returns:
        (x, y, z)
    """
    u, v, w = param_point

    theta = v / R0
    r = R0 + w

    x = cx + r * math.cos(theta)
    y = u
    z = cz + r * math.sin(theta)

    return (x, y, z)


# ============================================================
#  Voxel mesh -> CalculiX job (cylindrical only, rewritten
#  around gen_grid.py voxelization pipeline)
# ============================================================

def generate_global_cubic_hex_mesh(
    input_stl: str,
    cube_size: float,
    cyl_radius: float,
):
    """
    Voxelize input_stl in cylindrical param space and build a global C3D8R brick mesh
    mapped onto a cylinder (curved voxels).

    - Cylinder axis is global +Y.
    - Cylinder center (cx, cz) chosen so that the lowest-Z vertex defines cx,
      and cz is either derived from cyl_radius (lowest point lies on that circle)
      or from bbox center if radius is not given.
    - The effective mapping radius R0 is:
          R0 = R_true + cube_size / 2 + inward_offset
      where:
          R_true = distance from cylinder center to lowest point
          inward_offset = sagitta so that midpoint of a 1-voxel chord lies on R0.
    - We rotate the mesh in param space so that the lowest point ends up at angle 0
      (v=0), then undo this rotation when mapping voxels back to world.
    """

    mesh = trimesh.load(input_stl)

    log(f"[VOXEL] Loaded mesh from {input_stl}")
    log(f"[VOXEL] Watertight: {mesh.is_watertight}, bbox extents: {mesh.extents}")

    # vh: Move mesh on Y axis (goes from us to depth of cylinder) so its in the middle of Y voxels
    bounds_orig = mesh.bounds  # [[xmin,ymin,zmin], [xmax,ymax,zmax]]
    x_min_o, y_min_o, z_min_o = bounds_orig[0]
    x_max_o, y_max_o, z_max_o = bounds_orig[1]
    y_length = y_max_o - y_min_o
    float_cubes = y_length / cube_size
    ceil_cubes = math.ceil(float_cubes)
    reminder = (ceil_cubes - float_cubes) / 2
    y_offset = -float(y_min_o) - cube_size / 2 + (reminder * cube_size)
    mesh.apply_translation([0.0, y_offset, 0.0])
    # vh: ---

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

    # Decide CZ (and possibly initial "design" radius) to place the axis
    base_radius: Optional[float] = None  # design radius tied to lowest point

    # User-provided radius: we'll place the axis so that the lowest point
    # lies exactly on this circle.
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

    # vh: Rize model up, because having it on radius will make bottom layer voxels be on the middle of bottom plane
    inward_offset = R_true - math.sqrt(R_true * R_true - half * half)
    total_offset = cube_size / 2 + cube_size / 5  # vh: or inward_offset
    R0_param = R_true + total_offset
    # vh: ---

    # Final cylinder mapping radius
    R0 = float(R0_param)
    log(
        f"[VOXEL] Cylindrical voxel mode: axis=+Y, "
        f"center=({cx_cyl:.3f}, {cz_cyl:.3f}), R0={R0:.6f}"
    )

    # vh: Here we rotate the model on X axis to make it be in the middle of voxels on X
    theta_cube_size = cube_size / R0
    theta_length = float(dtheta_all[idx_right] - dtheta_all[idx_left])
    float_cubes = theta_length / theta_cube_size
    ceil_cubes = math.ceil(float_cubes)
    reminder = (ceil_cubes - float_cubes) / 2
    log(f"vh2 {theta_cube_size} {theta_length} {float_cubes} {ceil_cubes} {reminder} {theta_leftmost}")
    theta_leftmost = theta_leftmost  # ??? + theta_cube_size / 2 - (reminder * theta_cube_size)
    # vh: ---

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

    # Apply tangential rotation so the lowest point gets v=0 in param space
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
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],  # side
        [1, 2, 6], [1, 6, 5],  # side
        [2, 3, 7], [2, 7, 6],  # side
        [3, 0, 4], [3, 4, 7],  # side
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

    indices = vox.sparse_indices  # (N,3) with (ix,iy,iz) in param grid
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
        pt = vox.indices_to_points(idx_arr)[0]  # in param coordinates
        slice_coord = float(pt[2])  # w in cylindrical param space
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

    # --- Prepare lattice structures (STL) ---
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

    # --- Prepare lattice structures (BBOX) ---
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

    # --- Build voxel-node lattice & hex connectivity (STL) ---
    log("[VOXEL] Building voxel-node lattice (curved hexa nodes) for STL ...")

    for (ix, iy, iz) in indices_sorted:
        center_param = vox.indices_to_points(
            np.array([[ix, iy, iz]], dtype=float)
        )[0]
        u_c, v_c, w_c = center_param

        # Undo the tangential rotation before mapping back to world
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

    # --- Build voxel-node lattice & hex connectivity (BBOX) ---
    log("[VOXEL] Building voxel-node lattice (curved hexa nodes) for BBOX ...")

    for (ix, iy, iz) in indices_bbox_sorted:
        center_param = vox_bbox.indices_to_points(
            np.array([[ix, iy, iz]], dtype=float)
        )[0]
        u_c, v_c, w_c = center_param

        # Undo the tangential rotation before mapping back to world
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

    # Store cylindrical parameters + v_offset + y_offset
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
    # indices_bbox_sorted = indices_bbox_sorted[::-1]
    vertex_index_map_bbox = {
        (iy, ix, iz): idx
        for (ix, iy, iz), idx in vertex_index_map_bbox.items()
    }
    # vertex_index_map_bbox = {
    #     k: vertex_index_map_bbox[k]
    #     for k in reversed(list(vertex_index_map_bbox.keys()))
    # }

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


from typing import Dict, Tuple, List, Iterable
from typing import Dict, Tuple, List

def retarget_ffd_to_bbox_vertices(
    vertices_bbox: List[Tuple[float, float, float]],
    vertex_index_map_bbox: Dict[Tuple[int, int, int], int],
    ffd,
) -> None:
    """
    Modify FFD in-place so that each FFD control point position matches
    the corresponding bbox vertex position.

    After this, for every node:
        bbox_vert == box_origin + t * box_length + mu_new

    where mu_new is stored in ffd.array_mu_x/y/z.
    """

    num_nodes = len(vertices_bbox)

    # 1) Recover (ix, iy, iz) for each node index, in vertices_bbox order
    node_keys_in_vertex_order: List[Tuple[int, int, int]] = [None] * num_nodes
    for (ix, iy, iz), idx in vertex_index_map_bbox.items():
        node_keys_in_vertex_order[idx - 1] = (ix, iy, iz)

    if any(k is None for k in node_keys_in_vertex_order):
        raise RuntimeError("Some node indices are missing in vertex_index_map_bbox")

    # 2) Build mapping from global integer coords (ix,iy,iz) to local FFD indices
    ix_vals = sorted({ix for (ix, _, _) in node_keys_in_vertex_order})
    iy_vals = sorted({iy for (_, iy, _) in node_keys_in_vertex_order})
    # IMPORTANT: reverse iz so that "top" (largest iz) comes first,
    # matching the way cube vertices are traversed (top 4, then bottom 4).
    iz_vals = sorted({iz for (_, _, iz) in node_keys_in_vertex_order}, reverse=True)

    ix_to_local = {ix: i for i, ix in enumerate(ix_vals)}
    iy_to_local = {iy: j for j, iy in enumerate(iy_vals)}
    iz_to_local = {iz: k for k, iz in enumerate(iz_vals)}

    nx = len(ix_vals)
    ny = len(iy_vals)
    nz = len(iz_vals)

    # 3) Sanity-check against FFD resolution
    if hasattr(ffd, "n_control_points"):
        nx_ffd, ny_ffd, nz_ffd = ffd.n_control_points
        assert (nx, ny, nz) == (nx_ffd, ny_ffd, nz_ffd), (
            f"BBox grid ({nx},{ny},{nz}) != FFD grid ({nx_ffd},{ny_ffd},{nz_ffd})"
        )

    # 4) Pre-read box origin and length
    ox, oy, oz = map(float, ffd.box_origin)
    lx, ly, lz = map(float, ffd.box_length)

    # 5) For each node, compute the base position and set mu so that
    #    control point lands exactly on bbox_vert
    for i_node, (ix, iy, iz) in enumerate(node_keys_in_vertex_order):
        # Node data
        bx, by, bz = vertices_bbox[i_node]

        # Local indices in FFD grid
        i_local = ix_to_local[ix]
        j_local = iy_to_local[iy]
        k_local = iz_to_local[iz]

        # Parametric coordinates t in [0,1] along each axis
        tx = 0.0 if nx == 1 else i_local / (nx - 1)
        ty = 0.0 if ny == 1 else j_local / (ny - 1)
        tz = 0.0 if nz == 1 else k_local / (nz - 1)

        # Base (undeformed) control point position from the box
        base_x = ox + tx * lx
        base_y = oy + ty * ly
        base_z = oz + tz * lz

        # Desired displacement so that control point == bbox_vert
        dx = bx - base_x
        dy = by - base_y
        dz = bz - base_z

        # Write into FFD displacement arrays
        ffd.array_mu_x[i_local, j_local, k_local] = dx
        ffd.array_mu_y[i_local, j_local, k_local] = dy
        ffd.array_mu_z[i_local, j_local, k_local] = dz


def traverse_bbox_and_ffd_in_same_order(
    vertices_bbox: List[Tuple[float, float, float]],
    vertex_index_map_bbox: Dict[Tuple[int, int, int], int],
    ffd,
):
    """
    Yield (node_index, bbox_vertex, ffd_pos, ffd_disp) where:

    - node_index: 1-based node index (same as in hex connectivity)
    - bbox_vertex: coordinates from vertices_bbox in mesh order
    - ffd_pos:     FFD control point world position (box + mu)
    - ffd_disp:    FFD displacement vector (mu_x, mu_y, mu_z)
    """

    num_nodes = len(vertices_bbox)

    # 1) Recover (ix, iy, iz) for each node index, in vertices_bbox order
    node_keys_in_vertex_order: List[Tuple[int, int, int]] = [None] * num_nodes
    for (ix, iy, iz), idx in vertex_index_map_bbox.items():
        node_keys_in_vertex_order[idx - 1] = (ix, iy, iz)

    if any(k is None for k in node_keys_in_vertex_order):
        raise RuntimeError("Some node indices are missing in vertex_index_map_bbox")

    # 2) Build mapping from global integer coords (ix,iy,iz) to local FFD indices
    ix_vals = sorted({ix for (ix, _, _) in node_keys_in_vertex_order})
    iy_vals = sorted({iy for (_, iy, _) in node_keys_in_vertex_order})

    # IMPORTANT: reverse iz so that "top" (largest iz) comes first,
    # matching the way cube vertices are traversed (top 4, then bottom 4).
    iz_vals = sorted({iz for (_, _, iz) in node_keys_in_vertex_order}, reverse=True)

    ix_to_local = {ix: i for i, ix in enumerate(ix_vals)}
    iy_to_local = {iy: j for j, iy in enumerate(iy_vals)}
    iz_to_local = {iz: k for k, iz in enumerate(iz_vals)}

    nx = len(ix_vals)
    ny = len(iy_vals)
    nz = len(iz_vals)

    # 3) Sanity-check against FFD resolution (PyGeM: ffd.n_control_points = [nx,ny,nz])
    if hasattr(ffd, "n_control_points"):
        nx_ffd, ny_ffd, nz_ffd = ffd.n_control_points
        assert (nx, ny, nz) == (nx_ffd, ny_ffd, nz_ffd), (
            f"BBox grid ({nx},{ny},{nz}) != FFD grid ({nx_ffd},{ny_ffd},{nz_ffd})"
        )

    # 4) Helpers to get FFD displacement and position
    def get_ffd_disp(i: int, j: int, k: int):
        dx = float(ffd.array_mu_x[i, j, k])
        dy = float(ffd.array_mu_y[i, j, k])
        dz = float(ffd.array_mu_z[i, j, k])
        return (dx, dy, dz)

    def get_ffd_pos(i: int, j: int, k: int):
        # parametric coordinates in [0,1] along each axis
        tx = 0.0 if nx == 1 else i / (nx - 1)
        ty = 0.0 if ny == 1 else j / (ny - 1)
        tz = 0.0 if nz == 1 else k / (nz - 1)

        ox, oy, oz = map(float, ffd.box_origin)
        lx, ly, lz = map(float, ffd.box_length)

        base_x = ox + tx * lx
        base_y = oy + ty * ly
        base_z = oz + tz * lz

        dx, dy, dz = get_ffd_disp(i, j, k)

        return (base_x + dx, base_y + dy, base_z + dz)

    # 5) Traverse in the same order as vertices_bbox
    for i_node, (ix, iy, iz) in enumerate(node_keys_in_vertex_order):
        node_index = i_node + 1  # 1-based
        bbox_vert = vertices_bbox[i_node]

        i_local = ix_to_local[ix]
        j_local = iy_to_local[iy]
        k_local = iz_to_local[iz]

        ffd_disp = get_ffd_disp(i_local, j_local, k_local)
        ffd_pos  = get_ffd_pos(i_local, j_local, k_local)

        yield node_index, bbox_vert, ffd_pos, ffd_disp


def write_calculix_job(
    path: str,
    vertices: List[Tuple[float, float, float]],
    hexes: List[Tuple[int, int, int, int, int, int, int, int]],
    slice_to_eids: Dict[int, List[int]],
    z_slices: List[float],
    shrinkage_curve: List[float],
    cure_shrink_per_unit: float,
    output_stride: int = 1,
):
    """
    Additive-style uncoupled temperature-displacement job with MODEL CHANGE
    and incremental curing driven by a per-layer shrinkage curve.

    This version assumes cylindrical voxels: the BASE node set is taken from
    the outer radial faces (nodes 4..7) of the first radial slice.
    """

    n_nodes = len(vertices)
    n_elems = len(hexes)
    n_slices = len(z_slices)

    # simple time scheme: 1.0 per step
    time_per_layer = 1.0
    time_per_layer_step = 1.0
    total_weight = float(sum(shrinkage_curve))
    shrinkage_curve = [float(w) / total_weight for w in shrinkage_curve]

    # ensure sane stride
    if output_stride < 1:
        output_stride = 1

    with open(path, "w", encoding="utf-8") as f:
        f.write("**\n** Auto-generated incremental-cure shrink job\n**\n")
        f.write("*HEADING\n")
        f.write(
            "Voxel C3D8R uncoupled temperature-displacement "
            "(layer-wise MODEL CHANGE + shrinkage-curve-driven curing, cylindrical)\n"
        )

        # -------------------- NODES --------------------
        f.write("** Nodes +++++++++++++++++++++++++++++++++++++++++++++++++++\n")
        f.write("*NODE\n")
        for i, (x, y, z) in enumerate(vertices, start=1):
            f.write(f"{i}, {x}, {y}, {z}\n")

        # -------------------- ELEMENTS --------------------
        f.write("** Elements ++++++++++++++++++++++++++++++++++++++++++++++++\n")
        f.write("*ELEMENT, TYPE=C3D8R, ELSET=ALL\n")
        for eid, (n0, n1, n2, n3, n4, n5, n6, n7) in enumerate(hexes, start=1):
            f.write(
                f"{eid}, {n0}, {n1}, {n2}, {n3}, "
                f"{n4}, {n5}, {n6}, {n7}\n"
            )

        # -------------------- SETS --------------------
        f.write("** Node sets +++++++++++++++++++++++++++++++++++++++++++++++\n")
        f.write("*NSET, NSET=ALLNODES, GENERATE\n")
        f.write(f"1, {n_nodes}, 1\n")

        # We'll define BASE after we build per-slice node sets
        base_nodes: List[int] = []

        f.write("** Element + node sets (per slice) +++++++++++++++++++++++++\n")
        slice_names: List[str] = []
        slice_node_ids: Dict[int, List[int]] = {}

        def _write_id_list_lines(f, ids: List[int], per_line: int = 16):
            for i in range(0, len(ids), per_line):
                chunk = ids[i:i + per_line]
                f.write(", ".join(str(x) for x in chunk) + "\n")

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

            # Element set
            name = f"SLICE_{slice_idx:03d}"
            slice_names.append(name)
            f.write(f"*ELSET, ELSET={name}\n")
            _write_id_list_lines(f, valid_eids)

            # Node set from element connectivity
            nodes_in_slice: Set[int] = set()
            for eid in valid_eids:
                n0, n1, n2, n3, n4, n5, n6, n7 = hexes[eid - 1]
                nodes_in_slice.update([n0, n1, n2, n3, n4, n5, n6, n7])

            node_list = sorted(nodes_in_slice)
            slice_node_ids[slice_idx] = node_list

            nset_name = f"{name}_NODES"
            f.write(f"*NSET, NSET={nset_name}\n")
            _write_id_list_lines(f, node_list)

        # -------------------- BASE from outer radial faces of first slice ---------------
        existing_slice_idxs = sorted(slice_node_ids.keys())
        if existing_slice_idxs:
            base_slice = existing_slice_idxs[0]
            base_eids = slice_to_eids.get(base_slice, [])

            base_node_set: Set[int] = set()
            for eid in base_eids:
                n0, n1, n2, n3, n4, n5, n6, n7 = hexes[eid - 1]

                # Cylindrical case: "bottom" = outer radial face (w1) = nodes 4..7
                base_node_set.update([n4, n5, n6, n7])

            base_nodes = sorted(base_node_set)

            f.write("** Base node set = outer radial faces of first slice +++++++++++++++\n")
            f.write("*NSET, NSET=BASE\n")
            _write_id_list_lines(f, base_nodes)
        else:
            f.write("** Warning: no slices -> no BASE node set.\n")
            base_nodes = []

        # -------------------- MATERIAL (ABS with cure-shrink) ----------------
        f.write("** Materials +++++++++++++++++++++++++++++++++++++++++++++++\n")
        f.write("*MATERIAL, NAME=ABS\n")
        f.write("*DENSITY\n")
        f.write("1.12e-9\n")
        f.write("*ELASTIC\n")
        f.write("2800., 0.35\n")

        alpha = -float(cure_shrink_per_unit)  # higher T -> shrink
        f.write("*EXPANSION, ZERO=0.\n")
        f.write(f"{alpha:.6E}\n")

        f.write("*CONDUCTIVITY\n")
        f.write("0.20\n")
        f.write("*SPECIFIC HEAT\n")
        f.write("1.30e+9\n")
        f.write("*SOLID SECTION, ELSET=ALL, MATERIAL=ABS\n")

        # -------------------- INITIAL "TEMPERATURE" (CURE) -------------------
        f.write("** Initial conditions (cure variable) ++++++++++++++++++++++\n")
        f.write("*INITIAL CONDITIONS, TYPE=TEMPERATURE\n")
        f.write(f"ALLNODES, {0.0}\n")  # typically 0.0 or 293

        # -------------------- STEPS --------------------
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

            # Step 1: dummy, full model, uncured
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

                # remember cure state before this step
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

                # Decide whether to request field outputs for this curing step:
                # - every Nth step according to output_stride
                # - always for the very last curing step
                write_outputs = (
                    output_stride <= 1
                    or (global_k + 1) % output_stride == 0
                    or global_k == total_cure_steps - 1
                )

                if write_outputs:
                    f.write("** Field outputs +++++++++++++++++++++++++++++++++++++++++++\n")
                    f.write("*NODE FILE\n")
                    f.write("U\n")
                    # f.write("*EL FILE\n")
                    # f.write("S, E, HFL, NOE\n")
                else:
                    f.write(
                        "** Field outputs disabled for this step "
                        f"(output_stride = {output_stride})\n"
                    )
                    # This wipes previous node-file selections so no results are written
                    f.write("*NODE FILE\n")

                f.write("** Boundary conditions (base + shrinkage-curve cure) +++++\n")
                f.write("*BOUNDARY\n")

                for j in existing_slice_idxs:
                    if not printed[j]:
                        continue
                    cure_val = cure_state[j]
                    if cure_val == 0.0:
                        continue
                    # skip slices whose cure value did not change in this step
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
#  CalculiX runner
# ============================================================

def run_calculix(job_name: str, ccx_cmd: str = "ccx"):
    log(f"[RUN] Launching CalculiX: {ccx_cmd} {job_name}")

    log_path = f"{job_name}_ccx_output.txt"
    try:
        my_env = os.environ.copy()
        # my_env["PASTIX_GPU"] = "1"
        my_env["OMP_NUM_THREADS"] = "6"
        my_env["OMP_DYNAMIC"] = "FALSE"
        my_env["ЬЛД_NUM_THREADS"] = "6"
        my_env["ЬЛД_DYNAMIC"] = "FALSE"

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


# ============================================================
#  FRD parsing
# ============================================================

def read_frd_displacements(frd_path: str) -> Dict[int, np.ndarray]:
    if not os.path.isfile(frd_path):
        log(f"[FRD] File not found: {frd_path}")
        return {}

    disp: Dict[int, np.ndarray] = {}
    in_disp = False

    log(f"[FRD] Parsing displacements from: {frd_path}")

    with open(frd_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            s = line.lstrip()
            if not s:
                continue

            if s.startswith("-4") and "DISP" in s:
                disp = {}
                in_disp = True
                continue

            if not in_disp:
                continue

            if s.startswith("-3"):
                in_disp = False
                continue

            if s.startswith("-1"):
                try:
                    nid_str = line[3:13]
                    v1_str = line[13:25]
                    v2_str = line[25:37]
                    v3_str = line[37:49]

                    nid = int(nid_str)
                    ux = float(v1_str)
                    uy = float(v2_str)
                    uz = float(v3_str)

                    disp[nid] = np.array([ux, uy, uz], dtype=float)
                except ValueError:
                    continue

    log(f"[FRD] Parsed displacement data for {len(disp)} nodes.")
    return disp


# ============================================================
#  PyGeM FFD lattice from voxel mesh
#  (resolution & param extents aligned with gen_grid.py)
# ============================================================

def build_ffd_from_lattice(
    vertices: List[Tuple[float, float, float]],
    cube_size: float,
    displacements: Dict[int, np.ndarray],
    cyl_params: Optional[Tuple[float, ...]] = None,   # <<< CHANGED (variadic)
):
    """
    Build a PyGeM FFD lattice from voxel nodes, entirely in *input model
    coordinates* (world space).

    - FFD box_origin / box_length are the axis-aligned bbox of voxel nodes.
    - Logical indices (ix,iy,iz) come from world coordinates with spacing cube_size.
    - Displacements from FRD are in world coords and stored directly as mu_x/y/z.

    Cylindrical mapping is ONLY used earlier to create the curved voxel mesh;
    it is NOT used for FFD itself. This keeps STL, voxels and FFD all in the
    same coordinate system.
    """
    if FFD is None:
        raise RuntimeError("PyGeM FFD is not available. Please install 'pygem'.")

    if not vertices:
        raise ValueError("build_ffd_from_lattice: no vertices provided")

    coords = np.array(vertices, dtype=float)

    # World-space bounds of voxel nodes (input-model coordinates)
    x_min = float(coords[:, 0].min())
    x_max = float(coords[:, 0].max())
    y_min = float(coords[:, 1].min())
    y_max = float(coords[:, 1].max())
    z_min = float(coords[:, 2].min())
    z_max = float(coords[:, 2].max())

    Lx = max(x_max - x_min, 1e-12)
    Ly = max(y_max - y_min, 1e-12)
    Lz = max(z_max - z_min, 1e-12)

    def compute_nodes(span: float) -> int:
        # N_cells ≈ span / cube_size → nodes = N_cells + 1
        if span <= 0.0:
            return 2
        return max(2, int(round(span / cube_size)) + 1)

    nx = compute_nodes(Lx)
    ny = compute_nodes(Ly)
    nz = compute_nodes(Lz)

    log(
        "[FFD] World-space extents (voxel nodes):\n"
        f"      x=[{x_min:.6f}, {x_max:.6f}] (Lx={Lx:.6f})\n"
        f"      y=[{y_min:.6f}, {y_max:.6f}] (Ly={Ly:.6f})\n"
        f"      z=[{z_min:.6f}, {z_max:.6f}] (Lz={Lz:.6f})"
    )
    log(f"[FFD] FFD resolution (nodes): ({nx}, {ny}, {nz})")

    # ------------------------------------------------------------
    # Construct FFD object in world coordinates
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Fill control weights from nodal displacements
    # using world-coordinate logical indices
    # ------------------------------------------------------------
    logical_idx: List[Tuple[int, int, int]] = []
    inv_h = 1.0 / float(cube_size)

    for nid, (x, y, z) in enumerate(vertices, start=1):
        ix = int(round((x - x_min) * inv_h))
        iy = int(round((y - y_min) * inv_h))
        iz = int(round((z - z_min) * inv_h))
        logical_idx.append((ix, iy, iz))

    for nid, (ix, iy, iz) in enumerate(logical_idx, start=1):
        if not (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
            continue

        u = displacements.get(nid)
        if u is None:
            continue

        # u is a world displacement (ux,uy,uz)
        ffd.array_mu_x[ix, iy, iz] = u[0] / Lx
        ffd.array_mu_y[ix, iy, iz] = u[1] / Ly
        ffd.array_mu_z[ix, iy, iz] = u[2] / Lz

    # Mark that this FFD is world-based; cylindrical info is only
    # kept for visualization helpers if you want it.
    setattr(ffd, "_curved_voxels", False)
    setattr(ffd, "_cyl_params", cyl_params)
    setattr(ffd, "_param_bounds", None)

    return ffd


# ============================================================
#  Lattice visualization (PLYs)
# ============================================================

def export_lattice_ply_split( 
    vertices: List[Tuple[float, float, float]],
    displacements: Dict[int, np.ndarray],
    orig_path: str,
    def_path: str,
    max_points: int = 20000,
    lowest_point: Optional[Tuple[float, float, float]] = None,
    cyl_params: Optional[Tuple[float, ...]] = None,   # already variadic
    axis_point: Optional[Tuple[float, float, float]] = None,
    edge_left_point: Optional[Tuple[float, float, float]] = None,
    edge_right_point: Optional[Tuple[float, float, float]] = None,
    marker_segments_vertical: int = 64,
    marker_segments_radial: int = 32,
):
    """
    Export lattice as two PLY point clouds (original + deformed).

    Markers:
      - vertical line through lowest_point (if provided),
      - radial line from cylinder axis to lowest_point (if cyl_params+lowest_point),
      - radial line from axis to left angular edge,
      - radial line from axis to right angular edge.
    """
    if not vertices:
        log("[PLY] No vertices, skipping lattice export.")
        return

    if not orig_path and not def_path:
        return

    verts = np.array(vertices, dtype=float)
    n_nodes = verts.shape[0]

    disp_arr = np.zeros_like(verts)
    for nid, u in displacements.items():
        if 1 <= nid <= n_nodes:
            disp_arr[nid - 1] = u

    original_points = verts
    deformed_points = verts + disp_arr

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

        original_points = np.vstack([original_points, marker_vert])
        deformed_points = np.vstack([deformed_points, marker_vert])

        log(
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

        original_points = np.vstack([original_points, marker_radial])
        deformed_points = np.vstack([deformed_points, marker_radial])

        log(
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
        original_points = np.vstack([original_points, marker_left])
        deformed_points = np.vstack([deformed_points, marker_left])

        log(
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
        original_points = np.vstack([original_points, marker_right])
        deformed_points = np.vstack([deformed_points, marker_right])

        log(
            "[PLY] Added RIGHT edge radial line "
            f"axis({ax:.6f},{ay:.6f},{az:.6f}) -> "
            f"edge_right({rx:.6f},{ry:.6f},{rz:.6f})."
        )

    # ------------------ Write PLYs ------------------
    if orig_path:
        with open(orig_path, "w", encoding="utf-8") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {original_points.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for x, y, z in original_points:
                f.write(f"{x} {y} {z}\n")
        log(f"[PLY] Original lattice written to: {orig_path}")

    if def_path:
        with open(def_path, "w", encoding="utf-8") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {deformed_points.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for x, y, z in deformed_points:
                f.write(f"{x} {y} {z}\n")
        log(f"[PLY] Deformed lattice written to: {def_path}")


def export_ffd_control_points(ffd, basepath: str):
    """
    Export FFD control points as a single PLY with edges (arrows)
    from original positions to deformed positions, in *FFD box
    coordinates* (no cylindrical transforms).

    Layout:
      - First N vertices  : original control points
      - Next  N vertices  : deformed control points
      - N edges           : each edge connects (i -> i+N)
    """
    # Need these attributes from PyGeM's FFD
    if not hasattr(ffd, "array_mu_x") or not hasattr(ffd, "box_origin") or not hasattr(ffd, "box_length"):
        log("[PLY] FFD control-point export not supported by this PyGeM FFD object.")
        return

    mu_x = np.asarray(ffd.array_mu_x, dtype=float)
    mu_y = np.asarray(ffd.array_mu_y, dtype=float)
    mu_z = np.asarray(ffd.array_mu_z, dtype=float)

    nx, ny, nz = mu_x.shape

    origin = np.asarray(ffd.box_origin, dtype=float).reshape(3)
    length = np.asarray(ffd.box_length, dtype=float).reshape(3)
    Lx, Ly, Lz = length

    log(
        "[PLY] Exporting FFD control arrows in FFD box coords "
        f"(origin={origin}, length={length}, "
        f"shape=({nx},{ny},{nz}))"
    )

    pts_orig = []
    pts_def = []

    # Build original + deformed control points (rectangular box only)
    for i in range(nx):
        s = i / (nx - 1) if nx > 1 else 0.0
        for j in range(ny):
            t = j / (ny - 1) if ny > 1 else 0.0
            for k in range(nz):
                u = k / (nz - 1) if nz > 1 else 0.0

                # Base position in FFD box coordinates
                base = origin + np.array([s * Lx, t * Ly, u * Lz], dtype=float)

                # Displacement of control point in world units mapped to box
                disp = np.array(
                    [
                        mu_x[i, j, k] * Lx,
                        mu_y[i, j, k] * Ly,
                        mu_z[i, j, k] * Lz,
                    ],
                    dtype=float,
                )

                pts_orig.append(base)
                pts_def.append(base + disp)

    pts_orig = np.asarray(pts_orig, dtype=float)
    pts_def = np.asarray(pts_def, dtype=float)

    n = pts_orig.shape[0]
    if n != pts_def.shape[0]:
        log("[PLY] ERROR: mismatch between original and deformed control point counts.")
        return

    # Combine vertices: first all originals, then all deformed
    all_verts = np.vstack([pts_orig, pts_def])

    # Edges: from i -> i+n
    edges = np.array([[i, i + n] for i in range(n)], dtype=int)

    out_path = basepath + "_ffd_ctrl_arrows.ply"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {all_verts.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element edge {edges.shape[0]}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("end_header\n")

        # vertices
        for x, y, z in all_verts:
            f.write(f"{x} {y} {z}\n")

        # edges
        for v1, v2 in edges:
            f.write(f"{v1} {v2}\n")

    log(f"[PLY] FFD control arrows written to: {out_path}")


def export_deformed_voxels_stl(
    vertices: List[Tuple[float, float, float]],
    displacements: Dict[int, np.ndarray],
    hexes: List[Tuple[int, int, int, int, int, int, int, int]],
    out_path: str,
):
    """
    Export a deformed voxel mesh as STL by:
      - taking the original hex connectivity,
      - moving each node by its FRD displacement,
      - tesselating each hex into 12 triangles.

    This does NOT use cylindrical mapping, it just deforms the already-curved
    hex nodes by their nodal displacements.
    """
    if not vertices or not hexes:
        log("[VOXDBG] No vertices/hexes, skipping deformed voxel export.")
        return

    verts_orig = np.array(vertices, dtype=float)
    n_nodes = verts_orig.shape[0]

    # Build deformed nodal coords
    disp_arr = np.zeros_like(verts_orig)
    for nid, u in displacements.items():
        if 1 <= nid <= n_nodes:
            disp_arr[nid - 1] = u

    verts_def = verts_orig + disp_arr

    all_tri_vertices = []
    all_tri_faces = []

    cube_faces = [
        (0, 1, 2), (0, 2, 3),  # bottom
        (4, 5, 6), (4, 6, 7),  # top
        (0, 1, 5), (0, 5, 4),  # front
        (1, 2, 6), (1, 6, 5),  # right
        (2, 3, 7), (2, 7, 6),  # back
        (3, 0, 4), (3, 4, 7),  # left
    ]

    for (n0, n1, n2, n3, n4, n5, n6, n7) in hexes:
        # node IDs are 1-based
        p0 = verts_def[n0 - 1]
        p1 = verts_def[n1 - 1]
        p2 = verts_def[n2 - 1]
        p3 = verts_def[n3 - 1]
        p4 = verts_def[n4 - 1]
        p5 = verts_def[n5 - 1]
        p6 = verts_def[n6 - 1]
        p7 = verts_def[n7 - 1]

        base_idx = len(all_tri_vertices)
        all_tri_vertices.extend([p0, p1, p2, p3, p4, p5, p6, p7])

        for (a, b, c) in cube_faces:
            all_tri_faces.append((base_idx + a, base_idx + b, base_idx + c))

    if not all_tri_vertices:
        log("[VOXDBG] No triangles produced for deformed voxels.")
        return

    vertices_np = np.array(all_tri_vertices, dtype=float)
    faces_np = np.array(all_tri_faces, dtype=np.int32)

    mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np, process=False)
    mesh.export(out_path)
    log(f"[VOXDBG] Deformed voxel STL written: {out_path}")

# ============================================================
#  STL deformation with PyGeM (forward + pre-deformed)
# ============================================================

def ffd_apply_points(ffd, pts: np.ndarray) -> np.ndarray:
    try:
        return ffd(pts)
    except TypeError:
        return ffd.deform(pts)

def save_ffd_lattice(ffd, path):
    import json

    mu_x = ffd.array_mu_x.tolist()
    mu_y = ffd.array_mu_y.tolist()
    mu_z = ffd.array_mu_z.tolist()

    data = {
        "nx": ffd.array_mu_x.shape[0],
        "ny": ffd.array_mu_x.shape[1],
        "nz": ffd.array_mu_x.shape[2],
        "box_origin": ffd.box_origin.tolist(),
        "box_length": ffd.box_length.tolist(),
        "mu_x": mu_x,
        "mu_y": mu_y,
        "mu_z": mu_z
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    log(f"[FFD] Saved FFD lattice to: {path}")


def deform_input_stl_with_frd_pygem(
    input_stl: str,
    mech_frd_path: str,
    vertices: List[Tuple[float, float, float]],
    cube_size: float,
    output_stl: str,
    lattice_basepath: str = None,
    cyl_params: Optional[Tuple[float, ...]] = None,  # <<< CHANGED (variadic)
    lowest_point: Optional[Tuple[float, float, float]] = None,
    axis_point: Optional[Tuple[float, float, float]] = None,
    edge_left_point: Optional[Tuple[float, float, float]] = None,
    edge_right_point: Optional[Tuple[float, float, float]] = None,
    hexes: Optional[List[Tuple[int, int, int, int, int, int, int, int]]] = None,
    deformed_voxels_path: Optional[str] = None,
):

    if FFD is None:
        log("[FFD] PyGeM FFD not available; skipping STL deformation.")
        return

    displacements = read_frd_displacements(mech_frd_path)
    if not displacements:
        log("[FFD] No displacements found, skipping STL deformation.")
        return

    log(f"[FFD] Displacements available for {len(displacements)} nodes "
        f"out of {len(vertices)} total.")
    
    # Deformed voxel STL export (if requested)
    if hexes is not None and deformed_voxels_path:
        export_deformed_voxels_stl(
            vertices,
            displacements,
            hexes,
            deformed_voxels_path,
        )

    # Lattice of voxel nodes (orig+def)
    if lattice_basepath:
        orig_ply = lattice_basepath + "_orig.ply"
        def_ply = lattice_basepath + "_def.ply"
        export_lattice_ply_split(
            vertices,
            displacements,
            orig_ply,
            def_ply,
            lowest_point=lowest_point,
            cyl_params=cyl_params,
            axis_point=axis_point,
            edge_left_point=edge_left_point,
            edge_right_point=edge_right_point,
        )

    # Build FFD (marking cylindrical info if applicable)
    ffd = build_ffd_from_lattice(
        vertices,
        cube_size,
        displacements,
        cyl_params=cyl_params,
    )

    # Export FFD control points (now cylindrical-shaped when cyl_params != None)
    if lattice_basepath:
        save_ffd_lattice(ffd, lattice_basepath + "_ffd.json")
        export_ffd_control_points(ffd, lattice_basepath)

    mesh = trimesh.load(input_stl)
    if not isinstance(mesh, trimesh.Trimesh):
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
        else:
            log("[FFD] Could not load input STL mesh, aborting deformation.")
            return

    orig_verts = mesh.vertices.copy()
    n_verts = orig_verts.shape[0]
    log(f"[FFD] Deforming input STL with {n_verts} vertices using PyGeM FFD...")

    deformed_verts = ffd_apply_points(ffd, orig_verts)
    mesh.vertices = deformed_verts
    mesh.export(output_stl)
    log(f"[FFD] Deformed STL (PyGeM) written to: {output_stl}")

    log("[FFD] Computing pre-deformed STL by reversing FFD control displacements...")

    mesh.vertices = orig_verts

    try:
        if hasattr(ffd, "array_mu_x") and hasattr(ffd, "array_mu_y") and hasattr(ffd, "array_mu_z"):
            ffd.array_mu_x *= -1.0
            ffd.array_mu_y *= -1.0
            ffd.array_mu_z *= -1.0
        else:
            log("[FFD] WARNING: FFD inversion via weight flipping not supported by this PyGeM API.")
            return
    except Exception as e:
        log(f"[FFD] Error while flipping FFD control weights for pre-deformation: {e}")
        return

    predeformed_verts = ffd_apply_points(ffd, orig_verts)
    predeformed_path = os.path.splitext(output_stl)[0] + "_pre.stl"
    mesh.vertices = predeformed_verts
    mesh.export(predeformed_path)
    log(f"[FFD] Pre-deformed STL written to: {predeformed_path}")


def export_voxel_debug_stl(
    indices_sorted,
    vox,
    cube_size,
    cyl_params,
    out_path
):
    """
    Export all voxels as an STL mesh for visualization.
    Each voxel is drawn as 12 triangles (6 faces).

    Uses cylindrical mapping:
      - voxel centers/corners are in param (u,v,w)
      - corners are mapped through param_to_world_cyl().
    """

    cx, cz, R0, v_offset, y_offset = cyl_params

    half = cube_size / 2.0
    vertices = []
    faces = []

    for (ix, iy, iz) in indices_sorted:
        # center in param coords
        center_param = vox.indices_to_points(
            np.array([[ix, iy, iz]], dtype=float)
        )[0]
        u_c, v_c, w_c = center_param

        # Undo the tangential rotation for debug mesh
        v_c -= v_offset

        # axis-aligned corners in param coords
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
            y = y - y_offset # move voxels back to original Y frame
            world.append((x, y, z))

        base = len(vertices)
        vertices.extend(world)

        # add 12 triangles of the cube
        cube_faces = [
            (0, 1, 2), (0, 2, 3),  # bottom
            (4, 5, 6), (4, 6, 7),  # top
            (0, 1, 5), (0, 5, 4),  # front
            (1, 2, 6), (1, 6, 5),  # right
            (2, 3, 7), (2, 7, 6),  # back
            (3, 0, 4), (3, 4, 7),  # left
        ]
        for (a, b, c) in cube_faces:
            faces.append((base + a, base + b, base + c))

    # Export STL
    vertices_np = np.array(vertices, dtype=float)
    faces_np = np.array(faces, dtype=np.int32)
    mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np, process=False)
    mesh.export(out_path)
    log(f"[VOXDBG] Voxel debug STL written: {out_path}")


def build_curved_ffd_from_bbox_voxels(
    indices_bbox_sorted: np.ndarray,
    vox_bbox: "trimesh.voxel.VoxelGrid",
    cube_size: float,
    cyl_params: Tuple[float, float, float, float, float],
    vertices_bbox,
    vertex_index_map_bbox,
):
    # --- 1) Grid ranges & lattice dimensions (in voxel index space) ---
    ix_vals = indices_bbox_sorted[:, 0]
    iy_vals = indices_bbox_sorted[:, 1]
    iz_vals = indices_bbox_sorted[:, 2]

    print(indices_bbox_sorted)

    ix_min, ix_max = int(ix_vals.min()), int(ix_vals.max())
    iy_min, iy_max = int(iy_vals.min()), int(iy_vals.max())
    iz_min, iz_max = int(iz_vals.min()), int(iz_vals.max())

    # Nodes are voxel corners: range ix_min..ix_max+1 etc.
    nx = (ix_max - ix_min + 1) + 1
    ny = (iy_max - iy_min + 1) + 1
    nz = (iz_max - iz_min + 1) + 1

    x_min = min(v[0] for v in vertices_bbox)
    x_max = max(v[0] for v in vertices_bbox)
    y_min = min(v[1] for v in vertices_bbox)
    y_max = max(v[1] for v in vertices_bbox)
    z_min = min(v[2] for v in vertices_bbox)
    z_max = max(v[2] for v in vertices_bbox)

    Lx = x_max - x_min
    Ly = y_max - y_min
    Lz = z_max - z_min

    # --- 4) Create PyGeM FFD with classic interface ---
    ffd = FFD(n_control_points=[nx, ny, nz])

    ffd.box_origin[:] = np.array([x_min, y_min, z_min], dtype=float)
    ffd.box_length[:] = np.array([Lx, Ly, Lz], dtype=float)
    ffd.rot_angle[:] = np.array([0.0, 0.0, 0.0], dtype=float)

    export_ffd_to_ply(
        ffd=ffd,
        output_ply="OUTPUT\CSC16_U00P_" + "_lattice1_base.ply",
    )
    export_ffd_lattice_json(ffd, "OUTPUT\CSC16_U00P_" + "_json_lattice1_base.json")

    for v in vertices_bbox:
        print(f"({v[0]},{v[1]},{v[2]})")
    
    print("-------------")

    retarget_ffd_to_bbox_vertices(vertices_bbox, vertex_index_map_bbox, ffd)
    for nid, bbox_v, ffd_pos, ffd_disp in traverse_bbox_and_ffd_in_same_order(
        vertices_bbox,
        vertex_index_map_bbox,
        ffd,
    ):
        # log(f"vh: {nid} {bbox_v} {ffd_pos} {ffd_disp}")
        print(f"({ffd_pos[0]},{ffd_pos[1]},{ffd_pos[2]})")
        # print(f"({ffd_disp[0]},{ffd_disp[1]},{ffd_disp[2]})")

    return ffd


def export_ffd_to_ply(
    ffd,
    output_ply: str,
):
    """
    Export PyGeM FFD lattice control points as a PLY point cloud.

    - Computes full control point world coordinates:
          pos = box_origin + t * box_length + mu
    - Works with FFD objects containing:
          n_control_points
          box_origin
          box_length
          array_mu_x/y/z

    Output:
        A PLY file containing one point per control lattice vertex.
    """

    import numpy as np
    import os
    import trimesh

    # Extract FFD grid resolution
    nx, ny, nz = ffd.n_control_points
    ox, oy, oz = map(float, ffd.box_origin)
    lx, ly, lz = map(float, ffd.box_length)

    # Allocate array for full control point positions
    ctrl_points = np.zeros((nx, ny, nz, 3), dtype=float)

    for i in range(nx):
        tx = 0.0 if nx == 1 else i / (nx - 1)
        for j in range(ny):
            ty = 0.0 if ny == 1 else j / (ny - 1)
            for k in range(nz):
                tz = 0.0 if nz == 1 else k / (nz - 1)

                # Base position from the FFD box
                base_x = ox + tx * lx
                base_y = oy + ty * ly
                base_z = oz + tz * lz

                # Add FFD displacement
                dx = float(ffd.array_mu_x[i, j, k])
                dy = float(ffd.array_mu_y[i, j, k])
                dz = float(ffd.array_mu_z[i, j, k])

                ctrl_points[i, j, k, 0] = base_x + dx
                ctrl_points[i, j, k, 1] = base_y + dy
                ctrl_points[i, j, k, 2] = base_z + dz

    # Flatten (nx, ny, nz, 3) -> (N, 3)
    verts = ctrl_points.reshape(-1, 3)

    # Create point cloud
    pc = trimesh.points.PointCloud(verts)

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(output_ply))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Export PLY
    pc.export(output_ply)

    log(
        f"[PLY] FFD lattice points exported: {output_ply} "
        f"(nx,ny,nz)=({nx},{ny},{nz}), total={verts.shape[0]}"
    )


def export_ffd_lattice_points_to_ply(
    ctrl_points: np.ndarray,
    output_ply: str,
):
    """
    Export FFD control points (nx, ny, nz, 3) as a PLY point cloud.

    Args:
        ctrl_points : np.ndarray of shape (nx, ny, nz, 3)
            The control point coordinates in world space.
        output_ply  : str
            Path to output .ply file.
    """
    if ctrl_points.ndim != 4 or ctrl_points.shape[3] != 3:
        raise ValueError(
            f"ctrl_points must have shape (nx, ny, nz, 3), got {ctrl_points.shape}"
        )

    nx, ny, nz, _ = ctrl_points.shape

    # Flatten to (N, 3)
    verts = ctrl_points.reshape(-1, 3)

    # Create point cloud
    pc = trimesh.points.PointCloud(verts)

    # Make sure directory exists
    out_dir = os.path.dirname(os.path.abspath(output_ply))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    pc.export(output_ply)

    log(
        f"[PLY] FFD lattice points exported: {output_ply} "
        f"(nx,ny,nz)=({nx},{ny},{nz}), total points={verts.shape[0]}"
    )


def deform_bbox_ffd_from_voxel_displacements(
    ffd,
    vertices: List[Tuple[float, float, float]],
    displacements: Dict[int, Tuple[float, float, float]],
    tol: float = 1e-6,
) -> None:
    """
    Given:
      - ffd: PyGeM FFD object with fields:
            n_control_points (nx, ny, nz)
            box_origin (3,)
            box_length (3,)
            array_mu_x, array_mu_y, array_mu_z (nx, ny, nz)
      - vertices: list of voxel-mesh node coords (world coords), index = nid-1
      - displacements: dict {node_id -> (ux, uy, uz)} from FRD
        (node_id starts from 1, as in your existing code)

    Effect:
      - Modifies ffd.array_mu_x / array_mu_y / array_mu_z in-place so that
        control points that coincide with voxel nodes get the same displacement.
      - Control points which do not coincide with voxel nodes are left unchanged.
    """

    if tol <= 0.0:
        raise ValueError(f"tol must be > 0, got {tol}")

    inv_tol = 1.0 / float(tol)

    # ------------------------------------------------------------
    # 1) Build a map from quantized node position -> displacement
    #    Only store nodes that actually have displacement data.
    # ------------------------------------------------------------
    node_map: Dict[Tuple[int, int, int], Tuple[float, float, float]] = {}

    for nid, (x, y, z) in enumerate(vertices, start=1):
        u = displacements.get(nid)
        if u is None:
            continue
        key = (
            int(round(x * inv_tol)),
            int(round(y * inv_tol)),
            int(round(z * inv_tol)),
        )
        node_map[key] = u  # if duplicates, last wins (fine here)

    log(
        f"[FFD-DEF] Built voxel-node displacement map: {len(node_map)} entries "
        f"(tol={tol})"
    )

    # ------------------------------------------------------------
    # 2) Loop over all FFD control points, match them to nodes and
    #    update array_mu_* in-place.
    # ------------------------------------------------------------
    nx, ny, nz = ffd.n_control_points
    ox, oy, oz = map(float, ffd.box_origin)
    lx, ly, lz = map(float, ffd.box_length)

    matched = 0
    total_ctrl = nx * ny * nz

    for i in range(nx):
        tx = 0.0 if nx == 1 else i / (nx - 1)
        base_x = ox + tx * lx

        for j in range(ny):
            ty = 0.0 if ny == 1 else j / (ny - 1)
            base_y = oy + ty * ly

            for k in range(nz):
                tz = 0.0 if nz == 1 else k / (nz - 1)
                base_z = oz + tz * lz

                # Current displacement at this control point
                dx = float(ffd.array_mu_x[i, j, k])
                dy = float(ffd.array_mu_y[i, j, k])
                dz = float(ffd.array_mu_z[i, j, k])

                # Current world position of control point
                x = base_x + dx
                y = base_y + dy
                z = base_z + dz

                key = (
                    int(round(x * inv_tol)),
                    int(round(y * inv_tol)),
                    int(round(z * inv_tol)),
                )
                u = node_map.get(key)
                if u is None:
                    continue

                ux, uy, uz = u

                # Apply voxel-node displacement to this control point:
                #   new_pos = old_pos + u
                #   => (base + mu_new) = (base + mu_old) + u
                #   => mu_new = mu_old + u
                ffd.array_mu_x[i, j, k] = dx + ux
                ffd.array_mu_y[i, j, k] = dy + uy
                ffd.array_mu_z[i, j, k] = dz + uz

                matched += 1

    log(
        f"[FFD-DEF] Deformed bbox FFD (in-place): "
        f"matched {matched} / {total_ctrl} control points to voxel nodes."
    )


import os
import json
import numpy as np

def export_ffd_lattice_json(
    ffd,
    out_path: str,
):
    """
    Export a single PyGeM FFD object as JSON.

    JSON format (matches load_ffd helper):

      {
        "nx": int,
        "ny": int,
        "nz": int,
        "box_origin": [ox, oy, oz],
        "box_length": [lx, ly, lz],
        "mu_x": [[[...]], ...],   # shape (nx, ny, nz)
        "mu_y": [[[...]], ...],
        "mu_z": [[[...]], ...]
      }
    """

    # Extract sizes
    nx, ny, nz = ffd.n_control_points

    data = {
        "nx": int(nx),
        "ny": int(ny),
        "nz": int(nz),
        "box_origin": [float(v) for v in ffd.box_origin],
        "box_length": [float(v) for v in ffd.box_length],
        "mu_x": np.array(ffd.array_mu_x, dtype=float).tolist(),
        "mu_y": np.array(ffd.array_mu_y, dtype=float).tolist(),
        "mu_z": np.array(ffd.array_mu_z, dtype=float).tolist(),
    }

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    log(
        f"[FFD-EXPORT] FFD lattice exported to {out_path} "
        f"(nx,ny,nz)=({nx},{ny},{nz})"
    )

    return out_path


# ============================================================
#  CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Voxelize an STL in cylindrical coordinates and generate an "
            "uncoupled thermo-mechanical CalculiX job with C3D8R hexahedra "
            "and layer-by-layer *UNCOUPLED TEMPERATURE-DISPLACEMENT steps, "
            "then deform the input STL using PyGeM FFD (forward + pre-deformed) "
            "and optionally export the lattice as separate PLY point clouds."
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
        help="CalculiX executable (default 'ccx'). Example: "
             "'ccx', 'ccx_static', 'C:\\path\\to\\ccx.exe'",
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
            "Export FFD lattice as '<base_name>_lattice_orig.ply' and "
            "'<base_name>_lattice_def.ply' point clouds, where <base_name> is "
            "derived from the input STL filename."
        ),
    )
    # Cylindrical options
    parser.add_argument(
        "--cyl-radius",
        type=float,
        help=(
            "Base cylinder radius R0 for param mapping. "
            "If omitted, estimated from STL geometry."
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

    # 1) Mesh (cylindrical curved voxels only)
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

        # BBOX mesh results
        vertices_bbox,
        hexes_bbox,
        slice_to_eids_bbox,
        z_slices_bbox,
        indices_bbox_sorted, # for BBOX debug export
        vox_bbox,            # for BBOX debug export
        vertex_index_map_bbox,
    ) = generate_global_cubic_hex_mesh(
        args.input_stl,
        args.cube_size,
        cyl_radius=args.cyl_radius,
    )

    ffd_bbox = build_curved_ffd_from_bbox_voxels(
        indices_bbox_sorted=indices_bbox_sorted,
        vox_bbox=vox_bbox,
        cube_size=args.cube_size,
        cyl_params=cyl_params,
        vertices_bbox=vertices_bbox,
        vertex_index_map_bbox=vertex_index_map_bbox,
    )
    export_ffd_to_ply(
        ffd=ffd_bbox,
        output_ply=basepath + "_lattice2_curved.ply",
    )
    export_ffd_lattice_json(ffd_bbox, basepath + "_json_lattice2_curved.json")

    # --- Export voxel debug STL ---
    export_voxel_debug_stl(
        indices_sorted,
        vox,
        args.cube_size,
        cyl_params,
        basepath + "_voxels.stl"
    )

    # 2) Single uncoupled thermo-mechanical job
    utd_job = basepath + "_utd"
    utd_inp = utd_job + ".inp"

    write_calculix_job(
        utd_inp,
        vertices,
        hexes,
        slice_to_eids,
        z_slices,
        shrinkage_curve=[5, 4, 3, 2, 1],
        cure_shrink_per_unit=0.003,  # 3%
        output_stride=args.output_stride,
    )

    # 3) Optional run + PyGeM FFD deformation + lattice export
    if args.run_ccx:
        ok = run_calculix(utd_job, ccx_cmd=args.ccx_cmd)
        if not ok:
            log("[RUN] UTD job failed, skipping FFD.")
            return

        utd_frd = utd_job + ".frd"
        displacements = read_frd_displacements(utd_frd)
        deform_bbox_ffd_from_voxel_displacements(
            ffd=ffd_bbox,
            vertices=vertices,
            displacements=displacements,
            tol=1e-6,
        )
        export_ffd_to_ply(
            ffd=ffd_bbox,
            output_ply=basepath + "_lattice3_deformed.ply",
        )
        export_ffd_lattice_json(ffd_bbox, basepath + "_json_lattice3_deformed.json")

        # if os.path.isfile(utd_frd):
        #     deformed_stl = basepath + "_deformed.stl"
        #     lattice_basepath = basepath + "_lattice" if args.export_lattice else None
        #     deformed_voxels_stl = basepath + "_voxels_def.stl"

        #     deform_input_stl_with_frd_pygem(
        #         args.input_stl,
        #         utd_frd,
        #         vertices,
        #         args.cube_size,
        #         deformed_stl,
        #         lattice_basepath=lattice_basepath,
        #         cyl_params=cyl_params,
        #         lowest_point=lowest_point,
        #         axis_point=axis_point,
        #         edge_left_point=edge_left_point,
        #         edge_right_point=edge_right_point,
        #         hexes=hexes,
        #         deformed_voxels_path=deformed_voxels_stl,
        #     )

        # else:
        #     log(f"[FFD] Thermo-mechanical FRD '{utd_frd}' not found, skipping STL deformation.")


if __name__ == "__main__":
    main()

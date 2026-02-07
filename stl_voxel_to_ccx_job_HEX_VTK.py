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

from vtkmodules.vtkCommonCore import vtkPoints, vtkDoubleArray
from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid, vtkGenericCell
from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator

import numpy as np

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

import os
import json
import numpy as np

import os
import re
from typing import Dict, Tuple, Optional, List

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


def _pick_columns_hex(
    available_cols: dict[tuple[int, int], int],  # (ix,iy) -> bottom iz
    *,
    spacing: int,
    snap_radius: int = 0,
) -> list[tuple[int, int]]:
    """
    Choose (ix,iy) columns on a staggered/hex-like pattern inside the available footprint.

    spacing: approximate center-to-center spacing in ix direction (in voxel columns)
    snap_radius: if a hex node doesn't exist in available_cols, try to snap to a nearby column
                 within +-snap_radius (Chebyshev neighborhood), picking the closest by Euclidean.
    """
    if spacing <= 0:
        raise ValueError("spacing must be > 0")

    cols = set(available_cols.keys())
    if not cols:
        return []

    ix_min = min(ix for ix, _ in cols)
    ix_max = max(ix for ix, _ in cols)
    iy_min = min(iy for _, iy in cols)
    iy_max = max(iy for _, iy in cols)

    dx = spacing
    # Hex vertical spacing ~ sqrt(3)/2 * dx
    dy = max(1, int(round(dx * 0.8660254037844386)))
    x_shift = dx // 2

    chosen: set[tuple[int, int]] = set()

    row = 0
    for iy in range(iy_min, iy_max + 1, dy):
        x0 = ix_min + (x_shift if (row & 1) else 0)
        for ix in range(x0, ix_max + 1, dx):
            p = (ix, iy)
            if p in available_cols:
                chosen.add(p)
                continue

            if snap_radius > 0:
                best = None
                best_d2 = None
                for oy in range(-snap_radius, snap_radius + 1):
                    for ox in range(-snap_radius, snap_radius + 1):
                        q = (ix + ox, iy + oy)
                        if q not in available_cols:
                            continue
                        d2 = ox * ox + oy * oy
                        if best is None or d2 < best_d2:
                            best, best_d2 = q, d2
                if best is not None:
                    chosen.add(best)

        row += 1

    return sorted(chosen)


def add_pillars_and_base(
    indices: np.ndarray,
    *,
    pillar_layers: int,
    grid: str = "square",   # "square" or "hex"
    spacing: int = 4,
    snap_radius: int = 0,   # used for grid="hex"
    down_dir: int = +1,     # in your setup: DOWN = iz+1
    add_base: bool = True,
    base_pad: int = 1,
) -> np.ndarray:
    """
    Adds pillars going DOWN from bottom and a 1-layer base plate.

    - Pillars: all end at same iz_target; shortest pillar length = pillar_layers.
    - grid="hex": place pillars on a staggered/hex-like pattern.
    """
    idx = indices.astype(np.int64, copy=False)
    occ = set(map(tuple, idx.tolist()))

    if pillar_layers <= 0 or not occ:
        return idx

    # --- bottom voxel per (ix,iy) column ---
    col_bottom: dict[tuple[int, int], int] = {}
    for (ix, iy, iz) in occ:
        k = (ix, iy)
        prev = col_bottom.get(k)
        if prev is None:
            col_bottom[k] = iz
        else:
            # down_dir=+1 => bottommost is MAX iz; down_dir=-1 => bottommost is MIN iz
            col_bottom[k] = max(prev, iz) if down_dir == +1 else min(prev, iz)

    # --- choose which columns to support ---
    if grid == "square":
        chosen_cols = [(ix, iy) for (ix, iy) in col_bottom.keys()
                       if (ix % spacing == 0 and iy % spacing == 0)]
    elif grid == "hex":
        chosen_cols = _pick_columns_hex(col_bottom, spacing=spacing, snap_radius=snap_radius)
    else:
        raise ValueError(f"Unknown grid={grid!r}")

    if not chosen_cols:
        return idx

    seeds = [(ix, iy, col_bottom[(ix, iy)]) for (ix, iy) in chosen_cols]

    # --- shared target base height ---
    seed_iz = np.array([s[2] for s in seeds], dtype=np.int64)
    lowest_seed_iz = int(seed_iz.max() if down_dir == +1 else seed_iz.min())
    iz_target = lowest_seed_iz + down_dir * pillar_layers

    # --- add pillar voxels down to iz_target ---
    for (ix, iy, iz0) in seeds:
        iz = iz0 + down_dir
        while (iz - iz_target) * down_dir <= 0:
            occ.add((ix, iy, int(iz)))
            iz += down_dir

    # --- add rectangular 1-layer base plate at iz_target ---
    if add_base:
        ix_all = idx[:, 0]
        iy_all = idx[:, 1]
        ix_min = int(ix_all.min()) - base_pad
        ix_max = int(ix_all.max()) + base_pad
        iy_min = int(iy_all.min()) - base_pad
        iy_max = int(iy_all.max()) + base_pad

        for ix in range(ix_min, ix_max + 1):
            for iy in range(iy_min, iy_max + 1):
                occ.add((ix, iy, iz_target))

    out = np.array(list(occ), dtype=np.int64)
    order = np.lexsort((out[:, 0], out[:, 1], out[:, 2]))
    return out[order]


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

    # --- add pillars + base ---
    indices = add_pillars_and_base(
        indices,
        pillar_layers=7,     # shortest pillar length
        grid="hex",
        spacing=6,            # bigger => fewer pillars
        snap_radius=1,        # helps keep coverage on irregular footprints
        down_dir=+1,
        add_base=True,
        base_pad=1,
    )

    total_voxels = indices.shape[0]
    log(f"[VOXEL] Total filled voxels (STL cubes): {total_voxels}")

    order = np.lexsort((indices[:, 0], indices[:, 1], indices[:, 2]))
    indices_sorted = indices[order]

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

    # Store cylindrical parameters + v_offset + y_offset
    cyl_params = (cx_cyl, cz_cyl, R0, v_offset, y_offset)

    log(
        f"[VOXEL] Built STL mesh (voxelization frame): {len(vertices)} nodes, "
        f"{len(hexes)} hex elements, {len(z_slices)} radial slices."
    )

    # --- Undo Y-shift for returned data so voxels sit back where STL was ---
    vertices = [(x, y - y_offset, z) for (x, y, z) in vertices]

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
    )


from typing import Dict, Tuple, List, Iterable
from typing import Dict, Tuple, List


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
        my_env["MKL_NUM_THREADS"] = "6"
        my_env["MKL_DYNAMIC"] = "FALSE"

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

_FLOAT_RE = re.compile(r'[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][+-]?\d+)?')

def read_ccx_frd_last_displacements(
    frd_path: str,
    expected_nodes: Optional[int] = None,
) -> Dict[int, Tuple[float, float, float]]:
    """
    Read CalculiX .frd and return nodal displacements (U) from the LAST DISP block.

    Robust to fixed-width FRD formatting where values may have no spaces:
      ... 6.32943E-01-5.61525E-03-2.19507E-01

    Returns: {node_id (1-based): (ux, uy, uz)}
    """
    if not os.path.isfile(frd_path):
        raise FileNotFoundError(f"FRD not found: {frd_path}")

    disp_blocks: List[Dict[int, Tuple[float, float, float]]] = []
    current: Optional[Dict[int, Tuple[float, float, float]]] = None
    in_disp = False

    with open(frd_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.rstrip("\n")
            st = s.strip()
            if not st:
                continue

            # Start of a new result block
            if st.startswith("-4"):
                # close previous DISP block if we were in one
                if in_disp and current is not None:
                    disp_blocks.append(current)

                in_disp = False
                current = None

                # detect displacement block
                if "DISP" in st:
                    in_disp = True
                    current = {}
                continue

            if not in_disp or current is None:
                continue

            # End markers
            if st.startswith("-3") or st.startswith("-2"):
                disp_blocks.append(current)
                in_disp = False
                current = None
                continue

            # Nodal record line: "-1 <node> <ux><uy><uz>" (often fixed width)
            if st.startswith("-1"):
                m = re.match(r"\s*-1\s*(\d+)\s*(.*)$", s)
                if not m:
                    # fixed-width fallback for node id (rare)
                    try:
                        node_id = int(s[2:12])
                        rest = s[12:]
                    except Exception:
                        continue
                else:
                    node_id = int(m.group(1))
                    rest = m.group(2)

                nums = [float(x) for x in _FLOAT_RE.findall(rest)]
                if len(nums) >= 3:
                    current[node_id] = (nums[0], nums[1], nums[2])
                else:
                    # last-resort fixed width parse for 3 fields of 12 chars
                    try:
                        fields = [rest[i:i+12].strip() for i in range(0, 36, 12)]
                        vals = [float(v) for v in fields if v]
                        if len(vals) >= 3:
                            current[node_id] = (vals[0], vals[1], vals[2])
                    except Exception:
                        pass

    # if file ended mid-block
    if in_disp and current is not None:
        disp_blocks.append(current)

    if not disp_blocks:
        raise RuntimeError(f"No DISP blocks found in FRD: {frd_path}")

    last = disp_blocks[-1]

    if expected_nodes is not None and len(last) != expected_nodes:
        print(f"[FRD][WARN] Last DISP block has {len(last)} nodes, expected {expected_nodes}. "
              f"(You may be missing outputs or parsing still wrong.)")

    return last


def build_vtk_hex_grid(
    vertices: List[Tuple[float, float, float]],
    hexes: List[Tuple[int, int, int, int, int, int, int, int]],
) -> vtkUnstructuredGrid:
    """
    Build vtkUnstructuredGrid with VTK_HEXAHEDRON cells.
    Input hex connectivity is 1-based (CalculiX style). VTK uses 0-based.
    """
    grid = vtkUnstructuredGrid()

    pts = vtkPoints()
    pts.SetNumberOfPoints(len(vertices))
    for i, (x, y, z) in enumerate(vertices):
        pts.SetPoint(i, float(x), float(y), float(z))
    grid.SetPoints(pts)

    # Insert cells
    for conn in hexes:
        # VTK point ids are 0-based
        ids = [int(n) - 1 for n in conn]
        # InsertNextCell(cellType, npts, ptIds)
        grid.InsertNextCell(12, 8, ids)  # 12 == VTK_HEXAHEDRON

    return grid


def attach_point_displacements(
    grid: vtkUnstructuredGrid,
    disp_by_node_id_1based: Dict[int, Tuple[float, float, float]],
):
    """
    Attach point-data array 'U' (3 components) to grid points.
    Any missing nodes get (0,0,0).
    """
    npts = grid.GetNumberOfPoints()
    arr = vtkDoubleArray()
    arr.SetName("U")
    arr.SetNumberOfComponents(3)
    arr.SetNumberOfTuples(npts)

    for pid in range(npts):
        node_id = pid + 1  # 1-based
        ux, uy, uz = disp_by_node_id_1based.get(node_id, (0.0, 0.0, 0.0))
        arr.SetTuple3(pid, float(ux), float(uy), float(uz))

    grid.GetPointData().AddArray(arr)
    grid.GetPointData().SetActiveVectors("U")


def deform_stl_with_hex_displacements_vtk(
    input_stl: str,
    output_stl: str,
    grid: vtkUnstructuredGrid,
):
    """
    For each STL vertex:
      - find containing hex cell (vtkCellLocator)
      - get interpolation weights
      - interpolate nodal displacement from 'U'
      - apply to STL vertex

    If point is outside and use_closest_fallback=True:
      - use locator.FindClosestPoint() and interpolate at that closest point on the located cell.
    """
    mesh = trimesh.load(input_stl, force="mesh")
    V = np.asarray(mesh.vertices, dtype=np.float64)
    nV = V.shape[0]

    # locator
    locator = vtkStaticCellLocator()
    locator.SetDataSet(grid)
    locator.BuildLocator()

    u_arr = grid.GetPointData().GetArray("U")
    if u_arr is None:
        raise RuntimeError("Grid has no point-data array named 'U'")

    generic = vtkGenericCell()

    out = V.copy()

    # reusable buffers
    pcoords = [0.0, 0.0, 0.0]
    weights = [0.0] * 8  # hex has 8 nodes

    for i in range(nV):
        x = [float(V[i, 0]), float(V[i, 1]), float(V[i, 2])]

        # FindCell returns cellId or -1
        cell_id = locator.FindCell(x, 0, generic, pcoords, weights)

        if cell_id >= 0:
            # interpolate from the found cell using weights
            pid_list = generic.GetPointIds()
            du = np.zeros(3, dtype=np.float64)
            for k in range(8):
                pid = pid_list.GetId(k)  # VTK point id
                ux, uy, uz = u_arr.GetTuple3(pid)
                du[0] += weights[k] * ux
                du[1] += weights[k] * uy
                du[2] += weights[k] * uz
            out[i, :] += du
            continue
        else:
            raise RuntimeError("Couldn't find cell for point")

    mesh.vertices = out
    mesh.export(output_stl)

    log(f"[VTK] Deformed STL written: {output_stl}")
    log(f"[VTK] Vertices: {nV}")


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
    ) = generate_global_cubic_hex_mesh(
        args.input_stl,
        args.cube_size,
        cyl_radius=args.cyl_radius,
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
        cure_shrink_per_unit=0.05,  # 3%
        output_stride=args.output_stride,
    )

    # 3) Optional run + PyGeM FFD deformation + lattice export
    if args.run_ccx:
        ok = run_calculix(utd_job, ccx_cmd=args.ccx_cmd)
        if not ok:
            log("[RUN] UTD job failed, skipping FFD.")
            return

        # Read last-step displacements from FRD
        frd_path = utd_job + ".frd"
        disp = read_ccx_frd_last_displacements(frd_path, expected_nodes=len(vertices))
        log(f"[FRD] Using last DISP block: {len(disp)} nodes")

        # Build VTK unstructured grid from your C3D8R hex mesh
        grid = build_vtk_hex_grid(vertices, hexes)
        attach_point_displacements(grid, disp)

        # Deform the original STL directly (no FFD)
        out_deformed = basepath + "_deformed_vtk.stl"
        deform_stl_with_hex_displacements_vtk(
            input_stl=args.input_stl,
            output_stl=out_deformed,
            grid=grid,
        )


if __name__ == "__main__":
    main()

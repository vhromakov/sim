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

import triangle as tr
import tetgen  # <--- ADD THIS

import argparse
import os
from typing import List, Tuple, Dict, Set, Optional
import triangle as tr

import math
import numpy as np
import trimesh
import subprocess
from pygem import FFD
from datetime import datetime

from pathlib import Path
import numpy as np
import math
import trimesh
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


def write_ply_edges(path: Path, points: np.ndarray, edges: list[tuple[int, int]]) -> None:
    """
    Write an ASCII PLY with vertices + edges (no faces).
    Many viewers (Meshlab, CloudCompare) can show edges.
    """
    points = np.asarray(points, dtype=float)
    n_verts = points.shape[0]
    edges = list(edges)
    n_edges = len(edges)

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_verts}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element edge {n_edges}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("end_header\n")
        # vertices
        for (x, y, z) in points:
            f.write(f"{x} {y} {z}\n")
        # edges
        for (i0, i1) in edges:
            f.write(f"{i0} {i1}\n")


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

    Additionally:
    - Build a marching-cubes surface from the voxel grid in param space,
      map it back to world space on the cylinder and return it as mc_mesh_world.
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

    # -------------------------------------------------
    # Marching-cubes surface from voxel grid
    #   (index space -> param space -> world)
    # -------------------------------------------------
    log("[VOXEL] Running marching cubes via vox.marching_cubes ...")

    # 1) marching_cubes returns vertices in *index* space (like [ix+0.5, iy, iz+0.25])
    mc_param = vox.marching_cubes  # Trimesh in voxel index coords

    # 2) convert those index coords to param-space (u, v, w)
    #    using the same transform as for voxel centers
    mc_vertices_param = vox.indices_to_points(mc_param.vertices)  # (N, 3) in param coords
    mc_faces = mc_param.faces

    # 3) map param-space vertices to world on cylinder,
    #    applying the same v_offset and y_offset logic as for hexes
    mc_vertices_world = np.zeros_like(mc_vertices_param)
    for i, (u, v, w) in enumerate(mc_vertices_param):
        # undo tangential rotation we added before voxelization
        v_adj = v - v_offset

        # param -> world on cylinder
        x, y, z = param_to_world_cyl((u, v_adj, w), cx_cyl, cz_cyl, R0)

        # undo initial Y shift so it matches original STL frame
        y -= y_offset

        mc_vertices_world[i] = (x, y, z)

    mc_mesh_world = trimesh.Trimesh(
        vertices=mc_vertices_world,
        faces=mc_faces,
        process=False,
    )

    log(
        f"[VOXEL] Marching-cubes surface (world): "
        f"{len(mc_vertices_world)} verts, {len(mc_faces)} faces"
    )

    # Store cylindrical parameters + v_offset + y_offset
    cyl_params = (cx_cyl, cz_cyl, R0, v_offset, y_offset)

    log(
        f"[VOXEL] Built STL mesh (voxelization frame): {len(vertices)} nodes, "
        f"{len(hexes)} hex elements, {len(z_slices)} radial slices."
    )

    # --- Undo Y-shift for returned marker points so they sit back where STL was ---
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
        mc_mesh_world,       # marching-cubes surface in world coords
    )


from typing import Dict, Tuple, List, Iterable
from typing import Dict, Tuple, List


def write_calculix_job_tet(
    path: str,
    vertices: List[Tuple[float, float, float]],
    tet_elements: List[Tuple[int, int, int, int]],
    slice_to_eids: Dict[int, List[int]],
    z_slices: List[float],
    shrinkage_curve: List[float],
    cure_shrink_per_unit: float,
    cyl_params: Optional[Tuple[float, float, float, float, float]],
    cube_size: float,
    output_stride: int = 1,
):
    """
    Same layer-wise uncoupled temperature-displacement job as with hexes,
    but using C3D4 tets instead of C3D8R hexes.
    """

    n_nodes = len(vertices)
    n_elems = len(tet_elements)
    n_slices = len(z_slices)

    time_per_layer = 1.0
    time_per_layer_step = 1.0
    total_weight = float(sum(shrinkage_curve))
    shrinkage_curve = [float(w) / total_weight for w in shrinkage_curve]

    if output_stride < 1:
        output_stride = 1

    with open(path, "w", encoding="utf-8") as f:
        f.write("**\n** Auto-generated incremental-cure shrink job (C3D4 tets)\n**\n")
        f.write("*HEADING\n")
        f.write(
            "Tet C3D4 uncoupled temperature-displacement "
            "(layer-wise MODEL CHANGE + shrinkage-curve-driven curing, cylindrical)\n"
        )

        # -------------------- NODES --------------------
        f.write("** Nodes +++++++++++++++++++++++++++++++++++++++++++++++++++\n")
        f.write("*NODE\n")
        for i, (x, y, z) in enumerate(vertices, start=1):
            f.write(f"{i}, {x}, {y}, {z}\n")

        # -------------------- ELEMENTS --------------------
        f.write("** Elements ++++++++++++++++++++++++++++++++++++++++++++++++\n")
        f.write("*ELEMENT, TYPE=C3D4, ELSET=ALL\n")
        for eid, (n0, n1, n2, n3) in enumerate(tet_elements, start=1):
            f.write(f"{eid}, {n0}, {n1}, {n2}, {n3}\n")

        # -------------------- SETS --------------------
        f.write("** Node sets +++++++++++++++++++++++++++++++++++++++++++++++\n")
        f.write("*NSET, NSET=ALLNODES, GENERATE\n")
        f.write(f"1, {n_nodes}, 1\n")

        base_nodes: List[int] = []

        f.write("** Element + node sets (per slice) +++++++++++++++++++++++++\n")
        slice_names: List[str] = []
        slice_node_ids: Dict[int, List[int]] = {}

        def _write_id_list_lines(fh, ids: List[int], per_line: int = 16):
            for i in range(0, len(ids), per_line):
                chunk = ids[i:i + per_line]
                fh.write(", ".join(str(x) for x in chunk) + "\n")

        # Build per-slice ELSET/NSET from tet elements
        for slice_idx in sorted(slice_to_eids.keys()):
            eids = slice_to_eids.get(slice_idx, [])
            if not eids:
                continue
            valid_eids = [eid for eid in eids if 1 <= eid <= n_elems]
            if not valid_eids:
                log(
                    f"[WARN] Tet slice {slice_idx} has no valid element "
                    f"IDs within 1..{n_elems}"
                )
                continue

            name = f"SLICE_{slice_idx:03d}"
            slice_names.append(name)
            f.write(f"*ELSET, ELSET={name}\n")
            _write_id_list_lines(f, valid_eids)

            nodes_in_slice: Set[int] = set()
            for eid in valid_eids:
                n0, n1, n2, n3 = tet_elements[eid - 1]
                nodes_in_slice.update([n0, n1, n2, n3])

            node_list = sorted(nodes_in_slice)
            slice_node_ids[slice_idx] = node_list

            nset_name = f"{name}_NODES"
            f.write(f"*NSET, NSET={nset_name}\n")
            _write_id_list_lines(f, node_list)

        # -------------------- BASE node set via outer radius of first slice -----
        existing_slice_idxs = sorted(slice_node_ids.keys())
        if existing_slice_idxs and cyl_params is not None:
            base_slice = existing_slice_idxs[0]
            node_ids = slice_node_ids[base_slice]

            cx, cz, R0, v_offset, y_off = cyl_params
            # find nodes on outer cylindrical surface in this slice
            rs = []
            for nid in node_ids:
                x, y, z = vertices[nid - 1]
                r = math.sqrt((x - cx) ** 2 + (z - cz) ** 2)
                rs.append(r)
            if rs:
                r_max = max(rs)
                tol_r = cube_size * 0.5
                base_set: Set[int] = set()
                for nid, r in zip(node_ids, rs):
                    if r_max - r <= tol_r:
                        base_set.add(nid)
                base_nodes = sorted(base_set)

        if base_nodes:
            f.write("** Base node set = outer radial surface of first slice ++++++\n")
            f.write("*NSET, NSET=BASE\n")
            _write_id_list_lines(f, base_nodes)
        else:
            f.write("** Warning: no BASE node set defined.\n")

        # -------------------- MATERIAL (same ABS with cure-shrink) -------------
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

        # -------------------- INITIAL "TEMPERATURE" (CURE) -------------------
        f.write("** Initial conditions (cure variable) ++++++++++++++++++++++\n")
        f.write("*INITIAL CONDITIONS, TYPE=TEMPERATURE\n")
        f.write(f"ALLNODES, {0.0}\n")

        # -------------------- STEPS (same structure as hex version) ----------
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

                prev_cure_state = cure_state.copy()

                for j in existing_slice_idxs:
                    if not printed[j]:
                        continue
                    k_applied = applied_count[j]
                    if k_applied >= curve_len:
                        continue
                    increment = shrinkage_curve[k_applied]
                    if increment != 0.0:
                        cure_state[j] = min(1.0, cure_state[j] + increment)
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

    log(f"[CCX] Wrote incremental-cure UT-D tet job to: {path}")
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


import os
import json
import numpy as np
from pathlib import Path
import numpy as np
import trimesh


def export_slices_as_stl(args, vertices, slice_to_eids, hexes):
    # -----------------------------
    # Export each slice as a STL
    # -----------------------------
    out_dir = Path(getattr(args, "output_dir", "OUTPUT_SLICES"))
    out_dir.mkdir(parents=True, exist_ok=True)

    verts_np = np.asarray(vertices, dtype=float)

    # hexes are 1-based node indices
    def hex_to_faces(hex_nodes):
        v0, v1, v2, v3, v4, v5, v6, v7 = hex_nodes
        # 6 quad faces of a hex (consistent ordering)
        return [
            (v0, v1, v2, v3),  # bottom
            (v4, v5, v6, v7),  # top
            (v0, v1, v5, v4),  # side
            (v1, v2, v6, v5),  # side
            (v2, v3, v7, v6),  # side
            (v3, v0, v4, v7),  # side
        ]

    base_name = Path(args.input_stl).stem

    for slice_idx in sorted(slice_to_eids.keys()):
        eids = slice_to_eids[slice_idx]
        if not eids:
            continue

        # Count faces: internal faces appear twice, boundary faces once
        face_counter = {}
        face_oriented = {}

        for eid in eids:
            h = hexes[eid - 1]  # hexes are 1-based in this list
            for f in hex_to_faces(h):
                key = tuple(sorted(f))  # key without orientation
                face_counter[key] = face_counter.get(key, 0) + 1
                # store one oriented version for triangulation
                if key not in face_oriented:
                    face_oriented[key] = f

        tri_faces = []
        for key, count in face_counter.items():
            if count != 1:
                # internal face, shared by two hexes -> skip
                continue
            v0, v1, v2, v3 = face_oriented[key]
            # convert to 0-based indices
            v0 -= 1
            v1 -= 1
            v2 -= 1
            v3 -= 1
            # split quad into two triangles
            tri_faces.append([v0, v1, v2])
            tri_faces.append([v0, v2, v3])

        if not tri_faces:
            continue

        tri_faces_np = np.asarray(tri_faces, dtype=int)

        slice_mesh = trimesh.Trimesh(
            vertices=verts_np,
            faces=tri_faces_np,
            process=False,
        )

        out_path = out_dir / f"{base_name}_slice_{slice_idx:03d}.stl"
        slice_mesh.export(out_path.as_posix())
        log(f"[SLICES] Wrote slice {slice_idx} STL to: {out_path}")


def build_bottom_cap_faces_from_loops(
    verts_idx_slice: np.ndarray,
    faces_local: np.ndarray,
    z_band: float = 0.5,
) -> np.ndarray:
    """
    Build a radial bottom cap for one MC slice using boundary contour loops.

    Parameters
    ----------
    verts_idx_slice : (N, 3)
        Slice vertices in index coords (ix, iy, iz), local indexing [0..N-1].
    faces_local : (F, 3)
        Slice faces in terms of local vertex indices [0..N-1].
        (All faces of the slice, not pre-filtered by plane.)
    z_band : float
        Max allowed |mean_z(loop) - max_mean_z| in index units for loops
        to be considered "bottom" loops.

    Returns
    -------
    faces_cap_local : (K, 3) int
        Triangle faces (local vertex indices) forming the bottom cap.
        May be empty if no suitable loops are found.
    """

    if verts_idx_slice.shape[0] < 3 or faces_local.shape[0] == 0:
        return np.zeros((0, 3), dtype=int)

    # --- 1) Find boundary edges (edges used exactly once in this slice) ---
    edge_count: dict[tuple[int, int], int] = {}
    for (i0, i1, i2) in faces_local:
        for a, b in ((i0, i1), (i1, i2), (i2, i0)):
            e = (min(a, b), max(a, b))
            edge_count[e] = edge_count.get(e, 0) + 1

    boundary_edges = [e for e, c in edge_count.items() if c == 1]
    if not boundary_edges:
        return np.zeros((0, 3), dtype=int)

    # --- 2) Build adjacency on boundary edges and extract loops ---
    from collections import defaultdict

    adj: dict[int, set[int]] = defaultdict(set)
    for a, b in boundary_edges:
        adj[a].add(b)
        adj[b].add(a)

    loops: list[list[int]] = []
    visited_edges: set[tuple[int, int]] = set()

    def edge_key(a: int, b: int) -> tuple[int, int]:
        return (min(a, b), max(a, b))

    for a, b in boundary_edges:
        if edge_key(a, b) in visited_edges:
            continue

        # start a new loop following edges
        loop: list[int] = [a]
        prev = None
        cur = a

        for _ in range(len(boundary_edges) * 4):  # safety cap
            neigh = adj[cur]
            if prev is None:
                candidates = [b]
            else:
                candidates = [n for n in neigh if n != prev]

            if not candidates:
                break

            nxt = candidates[0]
            visited_edges.add(edge_key(cur, nxt))
            loop.append(nxt)

            prev, cur = cur, nxt
            if cur == a:
                break

        if len(loop) > 2 and loop[0] == loop[-1]:
            loops.append(loop[:-1])  # drop duplicate last

    if not loops:
        return np.zeros((0, 3), dtype=int)

    # --- 3) Compute mean z and area per loop to detect "bottom" loops ---
    loop_stats = []  # (loop_idx, mean_z, signed_area)
    for li, loop in enumerate(loops):
        if len(loop) < 3:
            continue
        zs = verts_idx_slice[loop, 2]
        mean_z = float(zs.mean())

        poly2d = verts_idx_slice[loop, :2].astype(float)
        area = 0.0
        for i in range(len(poly2d)):
            x0, y0 = poly2d[i]
            x1, y1 = poly2d[(i + 1) % len(poly2d)]
            area += x0 * y1 - x1 * y0
        area *= 0.5

        loop_stats.append((li, mean_z, area))

    if not loop_stats:
        return np.zeros((0, 3), dtype=int)

    max_mean_z = max(s[1] for s in loop_stats)
    bottom_loop_ids = [
        li for (li, mean_z, _area) in loop_stats
        if abs(mean_z - max_mean_z) <= z_band
    ]
    if not bottom_loop_ids:
        return np.zeros((0, 3), dtype=int)

    # ================== 4) DEPTH (how many loops fully contain it) =====
    poly2d_by_loop: dict[int, np.ndarray] = {}
    for li in bottom_loop_ids:
        loop = loops[li]
        poly2d_by_loop[li] = verts_idx_slice[loop, :2].astype(float)

    def _point_in_poly(pt: np.ndarray, poly: np.ndarray) -> bool:
        """Ray casting point-in-polygon test in 2D."""
        x, y = float(pt[0]), float(pt[1])
        inside = False
        n = poly.shape[0]
        for i in range(n):
            x0, y0 = poly[i]
            x1, y1 = poly[(i + 1) % n]
            cond = ((y0 > y) != (y1 > y))
            if cond:
                x_int = x0 + (y - y0) * (x1 - x0) / (y1 - y0 + 1e-16)
                if x_int > x:
                    inside = not inside
        return inside

    def _poly_fully_inside(inner: np.ndarray, outer: np.ndarray) -> bool:
        """Return True if **all points** of `inner` lie inside `outer`."""
        for pt in inner:
            if not _point_in_poly(pt, outer):
                return False
        return True

    depth: dict[int, int] = {li: 0 for li in bottom_loop_ids}
    for li in bottom_loop_ids:
        inner_poly = poly2d_by_loop[li]
        d = 0
        for lj in bottom_loop_ids:
            if lj == li:
                continue
            outer_poly = poly2d_by_loop[lj]
            if _poly_fully_inside(inner_poly, outer_poly):
                d += 1
        depth[li] = d

    # ============ 5) Helper: interior point for a polygon (no centroids) ==
    def _polygon_signed_area(poly: np.ndarray) -> float:
        """Signed area of a simple polygon (positive if CCW)."""
        x = poly[:, 0]
        y = poly[:, 1]
        return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))


    def _tri_area2(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Twice the signed area of triangle abc (positive if CCW)."""
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


    def _point_in_triangle(p: np.ndarray,
                        a: np.ndarray,
                        b: np.ndarray,
                        c: np.ndarray,
                        eps: float = 1e-12) -> bool:
        """Check if p is inside triangle abc (including edges)."""
        v0 = c - a
        v1 = b - a
        v2 = p - a

        den = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(den) < eps:
            return False  # degenerate

        # barycentric coordinates
        u = (v2[0] * v1[1] - v1[0] * v2[1]) / den
        v = (v0[0] * v2[1] - v2[0] * v0[1]) / den
        w = 1.0 - u - v

        return (u >= -eps) and (v >= -eps) and (w >= -eps)


    def _triangulate_polygon_ear_clip(poly: np.ndarray):
        """
        Ear-clipping triangulation for a simple polygon (no self-intersections, no holes).
        Returns a list of triangles as triplets of vertex indices (into `poly_ccw`).
        """
        poly = np.asarray(poly, dtype=float)
        n = len(poly)
        if n < 3:
            return [], poly

        # Use a CCW copy for triangulation; keep the original intact for _point_in_poly.
        poly_ccw = poly.copy()
        area = _polygon_signed_area(poly_ccw)
        if abs(area) < 1e-14:
            return [], poly  # degenerate polygon

        if area < 0.0:
            poly_ccw = poly_ccw[::-1].copy()

        indices = list(range(n))
        triangles = []

        max_iters = 10_000
        iters = 0

        while len(indices) > 3 and iters < max_iters:
            ear_found = False

            for k in range(len(indices)):
                i_prev = indices[k - 1]
                i = indices[k]
                i_next = indices[(k + 1) % len(indices)]

                a = poly_ccw[i_prev]
                b = poly_ccw[i]
                c = poly_ccw[i_next]

                area2 = _tri_area2(a, b, c)
                if area2 <= 1e-14:
                    # Not a strictly convex corner; skip
                    continue

                # Check that no other vertex lies inside this candidate ear
                is_ear = True
                for j in indices:
                    if j in (i_prev, i, i_next):
                        continue
                    if _point_in_triangle(poly_ccw[j], a, b, c):
                        is_ear = False
                        break

                if not is_ear:
                    continue

                # Found an ear
                triangles.append((i_prev, i, i_next))
                del indices[k]
                ear_found = True
                break

            if not ear_found:
                # Probably non-simple or numerically tricky polygon
                break

            iters += 1

        if len(indices) == 3:
            triangles.append(tuple(indices))

        return triangles, poly_ccw


    def _find_interior_point(poly: np.ndarray) -> np.ndarray:
        """
        Find a point strictly inside `poly`.
        Does NOT use polygon area centroid.

        Strategy:
        - Ear-clip triangulate the polygon.
        - Take the largest-area triangle.
        - Return its centroid (guaranteed inside that triangle).
        - If triangulation fails, fall back to bbox-based heuristics.
        """
        poly = np.asarray(poly, dtype=float)
        if poly.ndim != 2 or poly.shape[1] != 2:
            raise ValueError("poly must be (N, 2)")

        n = len(poly)
        if n == 0:
            raise ValueError("Empty polygon")
        if n == 1:
            return poly[0].copy()
        if n == 2:
            # Segment: return midpoint
            return 0.5 * (poly[0] + poly[1])

        # ---- Main method: triangulation + largest triangle centroid ----
        tris, poly_ccw = _triangulate_polygon_ear_clip(poly)

        best_centroid = None
        best_area2 = 0.0

        for (i, j, k) in tris:
            a = poly_ccw[i]
            b = poly_ccw[j]
            c = poly_ccw[k]
            area2 = abs(_tri_area2(a, b, c))
            if area2 <= 1e-16:
                continue
            if area2 > best_area2:
                best_area2 = area2
                best_centroid = (a + b + c) / 3.0

        if best_centroid is not None:
            # Extra safety: verify with your point-in-poly (in original vertex order)
            if _point_in_poly(best_centroid, poly):
                return best_centroid

        # ---- Fallback 1: bbox center ----
        xmin, ymin = poly.min(axis=0)
        xmax, ymax = poly.max(axis=0)
        if xmax == xmin and ymax == ymin:
            return poly[0].copy()

        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        center = np.array([cx, cy], dtype=float)

        if _point_in_poly(center, poly):
            return center

        # ---- Fallback 2: nudge vertices toward bbox center ----
        for v in poly:
            cand = v + 0.25 * (center - v)
            if _point_in_poly(cand, poly):
                return cand

        # Last resort: just return center (if your _point_in_poly is very strict,
        # this might rarely be outside; but realistically this should almost never happen)
        return center

    # ================== 6) TRIANGULATE FOR ALL EVEN DEPTHS ==============
    # For each loop with even depth d: it's an "outer" region; all loops
    # with depth d+1 fully inside it are holes for that region.
    faces_cap_local: list[list[int]] = []
    tri_call_idx = 0

    for outer_li in bottom_loop_ids:
        d_outer = depth[outer_li]
        if d_outer % 2 != 0:
            continue  # only even-depth loops define solid regions

        outer_poly = poly2d_by_loop[outer_li]

        # find all depth-(d_outer+1) loops fully inside this outer
        hole_lis: list[int] = []
        for li in bottom_loop_ids:
            if li == outer_li or depth[li] != d_outer + 1:
                continue
            inner_poly = poly2d_by_loop[li]
            if _poly_fully_inside(inner_poly, outer_poly):
                hole_lis.append(li)

        # Collect all vertex ids used by outer + its holes
        vert_ids_set: set[int] = set(loops[outer_li])
        for li in hole_lis:
            vert_ids_set.update(loops[li])
        if len(vert_ids_set) < 3:
            continue

        vert_ids_sorted = sorted(vert_ids_set)
        local_to_pslg: dict[int, int] = {
            v: i for i, v in enumerate(vert_ids_sorted)
        }

        pts2d = np.array(
            [verts_idx_slice[v, :2] for v in vert_ids_sorted],
            dtype=float,
        )

        segments: list[tuple[int, int]] = []

        # segments for outer loop
        outer_loop = loops[outer_li]
        L_out = len(outer_loop)
        for i in range(L_out):
            v0 = outer_loop[i]
            v1 = outer_loop[(i + 1) % L_out]
            s0 = local_to_pslg[v0]
            s1 = local_to_pslg[v1]
            segments.append((s0, s1))

        # segments for each hole loop
        for li in hole_lis:
            loop = loops[li]
            L = len(loop)
            if L < 2:
                continue
            for i in range(L):
                v0 = loop[i]
                v1 = loop[(i + 1) % L]
                s0 = local_to_pslg[v0]
                s1 = local_to_pslg[v1]
                segments.append((s0, s1))

        if not segments:
            continue

        # Hole points: robust interior points for each hole polygon
        holes_pts: list[list[float]] = []
        for li in hole_lis:
            poly_hole = poly2d_by_loop[li]
            p_in = _find_interior_point(poly_hole)
            holes_pts.append([float(p_in[0]), float(p_in[1])])

        data = {
            "vertices": pts2d,
            "segments": np.array(segments, dtype=int),
        }
        if holes_pts:
            data["holes"] = np.array(holes_pts, dtype=float)

        # --- DEBUG Triangle input for this outer+holes -----------------
        tri_call_idx += 1
        pts3d = np.column_stack(
            [pts2d, np.zeros((pts2d.shape[0],), dtype=float)]
        )

        tri_out = tr.triangulate(data, "pQ")
        if "triangles" not in tri_out:
            continue

        tris_pslg = tri_out["triangles"]
        if tris_pslg.shape[0] == 0:
            continue

        # Map PSLG indices back to original local vertex indices
        for a, b, c in tris_pslg:
            ga = vert_ids_sorted[a]
            gb = vert_ids_sorted[b]
            gc = vert_ids_sorted[c]
            faces_cap_local.append([ga, gb, gc])

    if not faces_cap_local:
        return np.zeros((0, 3), dtype=int)

    return np.asarray(faces_cap_local, dtype=int)


def export_mc_slices_as_stl(args, indices_sorted, vox, mc_mesh_world):
    base_name = Path(args.input_stl).stem
    out_dir = Path(getattr(args, "output_dir", "OUTPUT_SLICES"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Directories
    mc_out_dir = out_dir / "mc_slices"
    mc_out_dir.mkdir(parents=True, exist_ok=True)

    vol_out_dir = out_dir / "mc_slice_volumes"
    vol_out_dir.mkdir(parents=True, exist_ok=True)

    tet_out_dir = out_dir / "mc_slice_tets"
    tet_out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------
    # 1) Build iz -> slice_idx mapping from voxel info
    # -------------------------------------------------------
    unique_iz = np.unique(indices_sorted[:, 2])
    unique_iz_desc = np.sort(unique_iz)[::-1]  # slice 0 = outermost / top
    iz_to_slice = {int(iz): int(i) for i, iz in enumerate(unique_iz_desc)}
    num_slices = len(unique_iz_desc)

    # -------------------------------------------------------
    # 2) Marching-cubes mesh in index + world space
    # -------------------------------------------------------
    mc_idx = vox.marching_cubes
    mc_verts_idx = np.asarray(mc_idx.vertices, dtype=float)
    mc_faces_idx = np.asarray(mc_idx.faces, dtype=int)

    mc_verts_world = np.asarray(mc_mesh_world.vertices, dtype=float)

    min_iz = int(unique_iz.min())
    max_iz = int(unique_iz.max())

    # slice_idx -> list of MC face indices
    slice_to_mc_faces: dict[int, list[int]] = {i: [] for i in range(num_slices)}

    # -------------------------------------------------------
    # 3) Assign each MC triangle to iz (-> slice)
    # -------------------------------------------------------
    for f_idx, (i0, i1, i2) in enumerate(mc_faces_idx):
        z0 = mc_verts_idx[i0, 2]
        z1 = mc_verts_idx[i1, 2]
        z2 = mc_verts_idx[i2, 2]
        z_mean = (z0 + z1 + z2) / 3.0
        iz = int(math.floor(z_mean + 1e-6))

        if iz < min_iz:
            iz = min_iz
        elif iz > max_iz:
            iz = max_iz

        slice_idx = iz_to_slice.get(iz)
        if slice_idx is not None:
            slice_to_mc_faces[slice_idx].append(f_idx)

    # -------------------------------------------------------
    # 4) Per-slice data for later volume + tetgen creation
    # -------------------------------------------------------
    # Each entry:
    #   "verts_world": (Nv,3)
    #   "verts_idx":   (Nv,3) in index coords
    #   "side_faces":  (Fs,3) local indices (no original "bottom plane")
    #   "bottom_cap":  (Fc,3) local indices (cap from loops)
    slice_data: dict[int, dict[str, np.ndarray]] = {}

    z_tol = 1e-3

    for slice_idx, f_indices in slice_to_mc_faces.items():
        if not f_indices:
            continue

        f_indices_arr = np.asarray(f_indices, dtype=int)
        sub_faces_idx = mc_faces_idx[f_indices_arr]

        # Collect vertices actually used and remap to [0..N-1]
        unique_v, inverse = np.unique(sub_faces_idx.flatten(), return_inverse=True)
        sub_verts_world = mc_verts_world[unique_v]
        sub_verts_idx = mc_verts_idx[unique_v]
        sub_faces_local = inverse.reshape((-1, 3))  # faces in local [0..Nv-1]

        # ------------ Remove original "bottom" (max Z) triangles ------------
        z_vals_local = sub_verts_idx[:, 2]
        z_bottom = float(z_vals_local.max())
        on_bottom = np.abs(z_vals_local - z_bottom) < z_tol

        bottom_face_mask = np.array(
            [on_bottom[tri].all() for tri in sub_faces_local],
            dtype=bool,
        )

        kept_faces = sub_faces_local[~bottom_face_mask]
        faces_for_loops = sub_faces_local  # loops see full shell

        removed_count = int(bottom_face_mask.sum())

        # ------------ Generate bottom cap from contour loops -----------------
        cap_faces_local = build_bottom_cap_faces_from_loops(
            sub_verts_idx,
            faces_for_loops,
            z_band=0.5,  # roughly half a voxel in index space
        )

        if cap_faces_local.size > 0:
            all_faces_local = np.vstack([kept_faces, cap_faces_local])
            log(
                f"[MC-SLICES] Slice {slice_idx}: "
                f"removed {removed_count} old bottom triangles, "
                f"added {cap_faces_local.shape[0]} loop-based bottom-cap triangles"
            )
        else:
            all_faces_local = kept_faces
            log(
                f"[MC-SLICES] Slice {slice_idx}: "
                f"removed {removed_count} old bottom triangles, "
                f"no new bottom-cap triangles generated from loops"
            )

        # Debug STL of capped MC slice (as before)
        slice_mesh_debug = trimesh.Trimesh(
            vertices=sub_verts_world,
            faces=all_faces_local,
            process=False,
        )
        debug_path = mc_out_dir / f"{base_name}_mc_slice_{slice_idx:03d}.stl"
        slice_mesh_debug.export(debug_path.as_posix())
        log(f"[MC-SLICES] Wrote MC slice {slice_idx} STL to: {debug_path}")

        # Save data for volume + tetgen step
        slice_data[slice_idx] = {
            "verts_world": sub_verts_world,
            "verts_idx": sub_verts_idx,
            "side_faces": kept_faces,
            "bottom_cap": cap_faces_local,
        }

    # -------------------------------------------------------
    # 5) Build per-layer watertight-ish volumes + TetGen
    # -------------------------------------------------------
    # -------------------------------------------------------
    # 5) Build per-layer watertight-ish volumes + TetGen
    #    AND assemble a global tet mesh with layer info
    # -------------------------------------------------------
    if not slice_data:
        log("[MC-SLICES] No slice data to build volumes / tets.")
        return [], [], {}

    max_slice_idx = max(slice_data.keys())

    # Global tet mesh accumulators
    global_vertices: list[tuple[float, float, float]] = []
    global_elements: list[tuple[int, int, int, int]] = []  # 1-based node indices
    slice_to_tet_eids: dict[int, list[int]] = {i: [] for i in range(max_slice_idx + 1)}

    # coord -> global node index (1-based), using rounded coords as key
    vert_map: dict[tuple[float, float, float], int] = {}

    def get_global_vid(x: float, y: float, z: float, ndp: int = 6) -> int:
        key = (round(x, ndp), round(y, ndp), round(z, ndp))
        if key in vert_map:
            return vert_map[key]
        idx = len(global_vertices) + 1
        vert_map[key] = idx
        global_vertices.append((x, y, z))
        return idx

    for slice_idx in range(max_slice_idx + 1):
        if slice_idx not in slice_data:
            continue

        data = slice_data[slice_idx]
        verts_world = data["verts_world"]
        side_faces = data["side_faces"]
        bottom_cap = data["bottom_cap"]

        if verts_world.size == 0 or side_faces.size == 0:
            log(f"[MC-SLICES] Slice {slice_idx} has empty geometry; skipping volume/tetgen.")
            continue

        # Start volume with this slice's own vertices
        vol_verts = verts_world.copy()
        vol_faces_list = []

        # Side walls
        vol_faces_list.append(side_faces.astype(int))

        # Bottom cap (from this slice)
        if bottom_cap.size > 0:
            vol_faces_list.append(bottom_cap.astype(int))

        # Top cap: use bottom cap of slice_idx+1 (or own for last slice)
        if slice_idx < max_slice_idx and (slice_idx + 1) in slice_data:
            next_data = slice_data[slice_idx + 1]
            next_bottom = next_data["bottom_cap"]
            next_verts = next_data["verts_world"]
            if next_bottom.size > 0:
                offset = vol_verts.shape[0]
                vol_verts = np.vstack([vol_verts, next_verts])

                # Use next slice's bottom cap as TOP cap, but flip orientation
                top_cap = next_bottom.astype(int) + offset
                top_cap = top_cap[:, [0, 2, 1]]  # flip winding -> flip normals

                vol_faces_list.append(top_cap)
                log(
                    f"[MC-SLICES] Slice {slice_idx}: using bottom cap of slice "
                    f"{slice_idx + 1} as TOP cap (copied {next_bottom.shape[0]} tris, flipped normals)"
                )
        else:
            # Last slice: reuse its own cap as "top" by duplicating vertices.
            if bottom_cap.size > 0:
                offset = vol_verts.shape[0]
                vol_verts = np.vstack([vol_verts, verts_world])

                # Duplicate own bottom cap as TOP cap, flip orientation
                top_cap = bottom_cap.astype(int) + offset
                top_cap = top_cap[:, [0, 2, 1]]  # flip winding

                vol_faces_list.append(top_cap)
                log(
                    f"[MC-SLICES] Slice {slice_idx}: last slice, reusing its own "
                    f"bottom cap as TOP cap (duplicated vertices, flipped normals)."
                )

        vol_faces = np.vstack(vol_faces_list).astype(int)

        # --------- Export pre-TetGen watertight-ish volume surface ----------
        vol_mesh = trimesh.Trimesh(
            vertices=vol_verts,
            faces=vol_faces,
            process=False,
        )

        try:
            trimesh.repair.fix_normals(vol_mesh, multibody=True)
        except Exception as e:
            log(f"[MC-VOLUME] Warning: fix_normals failed on slice {slice_idx}: {e}")

        vol_path = vol_out_dir / f"{base_name}_mc_slice_{slice_idx:03d}_volume_preTet.stl"
        vol_mesh.export(vol_path.as_posix())
        log(f"[MC-VOLUME] Wrote pre-TetGen volume slice {slice_idx} STL to: {vol_path}")

        # --------- TetGen volumetric mesh for this slice --------------------
        try:
            tet = tetgen.TetGen(vol_mesh.vertices, vol_mesh.faces)
            tet_out = tet.tetrahedralize(
                order=1,
                mindihedral=10,
                minratio=1.5,
                steinerleft=-1,
                quality=True,
            )
            tet_points = np.asarray(tet_out[0], dtype=float)
            tet_elements = np.asarray(tet_out[1], dtype=int)
            log(
                f"[MC-TETGEN] Slice {slice_idx}: "
                f"{tet_points.shape[0]} nodes, {tet_elements.shape[0]} tets"
            )
        except Exception as e:
            log(f"[MC-TETGEN] TetGen tetrahedralize() failed on slice {slice_idx}: {e}")
            continue

        # Build unique edges from local tets (for per-slice PLY)
        edges_set: set[tuple[int, int]] = set()
        if tet_elements.ndim == 1:
            tet_elements = tet_elements.reshape((-1, tet_elements.shape[0]))

        for elem in tet_elements:
            # Use first 4 nodes as linear tet vertices
            v = [int(elem[0]), int(elem[1]), int(elem[2]), int(elem[3])]
            edge_pairs = [
                (v[0], v[1]), (v[0], v[2]), (v[0], v[3]),
                (v[1], v[2]), (v[1], v[3]), (v[2], v[3]),
            ]
            for a, b in edge_pairs:
                if a == b:
                    continue
                if a > b:
                    a, b = b, a
                edges_set.add((a, b))

        edges_list = sorted(edges_set)
        ply_path = tet_out_dir / f"{base_name}_mc_slice_{slice_idx:03d}_tets.ply"
        write_ply_edges(ply_path, tet_points, edges_list)
        log(f"[MC-TETGEN] Wrote TetGen edge PLY for slice {slice_idx} to: {ply_path}")

        # --------- Add this slice's tets to global mesh --------------------
        # Map local tet nodes -> global node indices
        local_to_global: dict[int, int] = {}
        for local_vid, (x, y, z) in enumerate(tet_points):
            gid = get_global_vid(float(x), float(y), float(z))
            local_to_global[local_vid] = gid

        for elem in tet_elements:
            v = [int(elem[0]), int(elem[1]), int(elem[2]), int(elem[3])]
            g = tuple(local_to_global[i] for i in v)
            global_elements.append(g)
            eid = len(global_elements)
            slice_to_tet_eids[slice_idx].append(eid)

    return global_vertices, global_elements, slice_to_tet_eids




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
        mc_mesh_world,   # NEW
    ) = generate_global_cubic_hex_mesh(
        args.input_stl,
        args.cube_size,
        cyl_radius=args.cyl_radius,
    )
    export_slices_as_stl(args, vertices, slice_to_eids, hexes)  # optional hex debug

    tet_vertices, tet_elements, tet_slice_to_eids = export_mc_slices_as_stl(
        args,
        indices_sorted,
        vox,
        mc_mesh_world,
    )

    # 2) Single uncoupled thermo-mechanical job (TETS instead of HEXES)
    utd_job = basepath + "_utd_tet"
    utd_inp = utd_job + ".inp"

    write_calculix_job_tet(
        utd_inp,
        tet_vertices,
        tet_elements,
        tet_slice_to_eids,
        z_slices,
        shrinkage_curve=[5, 4, 3, 2, 1],
        cure_shrink_per_unit=0.003,  # 3%
        cyl_params=cyl_params,
        cube_size=args.cube_size,
        output_stride=args.output_stride,
    )

    # 3) Optional run + PyGeM FFD deformation + lattice export
    if args.run_ccx:
        ok = run_calculix(utd_job, ccx_cmd=args.ccx_cmd)
        if not ok:
            log("[RUN] UTD job failed, skipping FFD.")
            return

if __name__ == "__main__":
    main()

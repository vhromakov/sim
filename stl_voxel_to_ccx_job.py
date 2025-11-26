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

# --- PyGeM import (supports old/new layouts) ---
try:
    from pygem.ffd import FFD
except ImportError:
    try:
        from pygem import FFD
    except ImportError:
        FFD = None


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


def export_cylinder_debug_stl(
    cx: float,
    cz: float,
    R0: float,
    y_min: float,
    y_max: float,
    path: str,
    n_theta: int = 64,
):
    """
    Export a simple cylinder surface as STL for debugging.

    - Axis: global +Y
    - Center in XZ: (cx, cz)
    - Radius: R0
    - Extends from y_min to y_max

    This is just a triangulated tube (no caps).
    """
    if R0 <= 0.0:
        print("[CYL] Radius <= 0, skipping cylinder STL export.")
        return

    thetas = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)

    # Two rings: bottom at y_min and top at y_max
    bottom = []
    top = []
    for theta in thetas:
        x = cx + R0 * math.cos(theta)
        z = cz + R0 * math.sin(theta)
        bottom.append([x, y_min, z])
        top.append([x, y_max, z])

    bottom = np.array(bottom, dtype=float)
    top = np.array(top, dtype=float)

    vertices = np.vstack([bottom, top])
    faces = []

    # Connect triangles between bottom and top rings
    for i in range(n_theta):
        j = (i + 1) % n_theta

        # indices in vertices array
        b0 = i
        b1 = j
        t0 = i + n_theta
        t1 = j + n_theta

        # quad (b0, b1, t1, t0) -> two triangles
        faces.append([b0, b1, t1])
        faces.append([b0, t1, t0])

    faces = np.array(faces, dtype=np.int64)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(path)
    print(f"[CYL] Debug cylinder STL written to: {path}")


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
#  Voxel mesh -> CalculiX job (cylindrical only)
# ============================================================

def generate_global_cubic_hex_mesh(
    input_stl: str,
    cube_size: float,
    cyl_center_xz: Optional[Tuple[float, float]] = None,
    cyl_radius: Optional[float] = None,
):
    """
    Voxelize input_stl in cylindrical param space and build a global C3D8R brick mesh
    mapped onto a cylinder (curved voxels).

    Workflow (always cylindrical):

      - STL vertices are analyzed in world space (x,y,z).
      - Cylinder axis is global +Y.
      - We choose cylinder center (cx, cz) and a mapping radius R0_param.
      - STL is mapped to cylindrical param space (u, v, w) via world_to_param_cyl.
      - Voxelization happens in (u, v, w) using trimesh.voxelized().
      - Angular trimming in param v and world-angle keeps only bands intersecting STL.
      - Voxel corners in param space are mapped back to world via param_to_world_cyl,
        giving curved hexahedra that follow the cylinder.
      - Slices become radial layers (bands in w).
      - The lowest STL point (by Z) is:
          * On the outer radial face of the base slice,
          * Tangentially aligned so that it lies on the center line
            (angle) of its voxel in XZ plane.
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

    # --- cylindrical preprocessing (always) ---
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
        return [], [], {}, [], None, None, None, None, None, None, None

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
        return [], [], {}, [], None, None, None, None, None, None, None

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

    iz_to_slice: Dict[int, int] = {}
    z_slices: List[float] = []
    for slice_idx, (iz, pos) in enumerate(layer_info):
        iz_to_slice[int(iz)] = slice_idx
        z_slices.append(pos)

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

    # This is the cylinder definition used by the *voxel mesh* and FFD
    cyl_params = (cx_cyl, cz_cyl, R0_world, theta_offset)

    # Optional: debug cylinder with the aligned R0_world and theta_offset
    input_base = os.path.splitext(os.path.basename(input_stl))[0]
    cyl_debug_stl = input_base + "_cylinder_debug.stl"

    n_theta = 64
    thetas = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    bottom = []
    top = []
    for th in thetas:
        th_off = th + theta_offset
        x = cx_cyl + R0_world * math.cos(th_off)
        z = cz_cyl + R0_world * math.sin(th_off)
        bottom.append([x, y_min_w, z])
        top.append([x, y_max_w, z])
    bottom = np.array(bottom, dtype=float)
    top = np.array(top, dtype=float)
    vertices_cyl = np.vstack([bottom, top])
    faces_cyl = []
    for i in range(n_theta):
        j = (i + 1) % n_theta
        b0 = i
        b1 = j
        t0 = i + n_theta
        t1 = j + n_theta
        faces_cyl.append([b0, b1, t1])
        faces_cyl.append([b0, t1, t0])
    faces_cyl = np.array(faces_cyl, dtype=np.int64)
    mesh_cyl = trimesh.Trimesh(vertices=vertices_cyl, faces=faces_cyl, process=False)
    mesh_cyl.export(cyl_debug_stl)
    print(f"[CYL] Debug cylinder STL written to: {cyl_debug_stl}")

    # --- Angular trimming in WORLD ANGLE space ------------------------
    # 1) Compute model angular range around cylinder (unwrapped)
    dx_m = verts_world[:, 0] - cx_cyl
    dz_m = verts_world[:, 2] - cz_cyl
    theta_model_raw = np.arctan2(dz_m, dx_m)
    theta_model_unwrapped = _unwrap_angle_array(theta_model_raw)
    theta_min_model = float(theta_model_unwrapped.min())
    theta_max_model = float(theta_model_unwrapped.max())
    theta_mid_model = 0.5 * (theta_min_model + theta_max_model)

    print(
        "[VOXEL] Model angular range (unwrapped): "
        f"[{theta_min_model:.6f}, {theta_max_model:.6f}] rad "
        f"(span={theta_max_model - theta_min_model:.6f})"
    )

    # Angular half-width of a voxel band in v-direction
    dtheta_half = (cube_size * 0.5) / R0_world

    # 2) For each angular index iy, compute its band center angle in world,
    #    then keep it iff its angular interval intersects the model's range.
    unique_iy = np.unique(indices[:, 1])
    if unique_iy.size == 0:
        print("[VOXEL] No angular bands to trim (world angle), aborting.")
        return [], [], {}, [], None, None, None, None, None, None, None

    if len(z_slices) == 0:
        print("[VOXEL] No slices for angular trimming (world angle), aborting.")
        return [], [], {}, [], None, None, None, None, None, None, None

    base_slice_iz = int(layer_info[0][0])

    iy_keep: Set[int] = set()
    for iy_val in unique_iy:
        # Use geometric center of cell (ix=0, this iy, base_slice_iz) in param
        pt_param = vox.indices_to_points(
            np.array([[0, int(iy_val), base_slice_iz]], dtype=float)
        )[0]
        u_c, v_c, w_c = pt_param

        # Map param center to world using final R0_world and theta_offset
        x_c, y_c, z_c = param_to_world_cyl(
            (u_c, v_c, w_c),
            cx_cyl,
            cz_cyl,
            R0_world,
            theta_offset,
        )

        theta_c_raw = math.atan2(z_c - cz_cyl, x_c - cx_cyl)

        # Bring this band angle near the model's range (same branch)
        k = round((theta_mid_model - theta_c_raw) / (2.0 * math.pi))
        theta_c = theta_c_raw + k * 2.0 * math.pi

        band_min = theta_c - dtheta_half
        band_max = theta_c + dtheta_half

        # Interval intersection test with model [theta_min_model, theta_max_model]
        if (band_max >= theta_min_model) and (band_min <= theta_max_model):
            iy_keep.add(int(iy_val))

    if not iy_keep:
        print("[VOXEL] Angular trimming (world angle) would remove all bands, aborting.")
        return [], [], {}, [], None, None, None, None, None, None, None

    mask_ang = np.array([int(iy) in iy_keep for ix, iy, iz in indices], dtype=bool)
    indices = indices[mask_ang]

    if indices.size == 0:
        print("[VOXEL] After angular trimming (world angle), no voxels left – aborting.")
        return [], [], {}, [], None, None, None, None, None, None, None

    total_voxels = indices.shape[0]
    print(f"[VOXEL] Total voxels after angular trim (world angle): {total_voxels}")

    print("[VOXEL] Building global C3D8R mesh (cylindrical) ...")

    # ----------- Curved voxels on cylindrical surface -----------
    if cyl_params is None or R0_world is None:
        raise RuntimeError("[VOXEL] cyl_params missing in cylindrical voxel mode.")
    cx_cyl_use, cz_cyl_use, R0_use, theta_offset_use = cyl_params

    for (ix, iy, iz) in indices_sorted:
        center_param = vox.indices_to_points(
            np.array([[ix, iy, iz]], dtype=float)
        )[0]
        u_c, v_c, w_c = center_param

        u0, u1 = u_c - half, u_c + half
        v0, v1 = v_c - half, v_c + half
        w0, w1 = w_c - half, w_c + half

        # param-space corners of this voxel
        p0 = (u0, v0, w0)  # "bottom" in param space
        p1 = (u1, v0, w0)
        p2 = (u1, v1, w0)
        p3 = (u0, v1, w0)
        p4 = (u0, v0, w1)  # "top" in param space
        p5 = (u1, v0, w1)
        p6 = (u1, v1, w1)
        p7 = (u0, v1, w1)

        # map to world (curved hexa), using aligned R0_world + theta_offset
        x0, y0, z0 = param_to_world_cyl(p0, cx_cyl_use, cz_cyl_use, R0_use, theta_offset_use)
        x1, y1, z1 = param_to_world_cyl(p1, cx_cyl_use, cz_cyl_use, R0_use, theta_offset_use)
        x2, y2, z2 = param_to_world_cyl(p2, cx_cyl_use, cz_cyl_use, R0_use, theta_offset_use)
        x3, y3, z3 = param_to_world_cyl(p3, cx_cyl_use, cz_cyl_use, R0_use, theta_offset_use)
        x4, y4, z4 = param_to_world_cyl(p4, cx_cyl_use, cz_cyl_use, R0_use, theta_offset_use)
        x5, y5, z5 = param_to_world_cyl(p5, cx_cyl_use, cz_cyl_use, R0_use, theta_offset_use)
        x6, y6, z6 = param_to_world_cyl(p6, cx_cyl_use, cz_cyl_use, R0_use, theta_offset_use)
        x7, y7, z7 = param_to_world_cyl(p7, cx_cyl_use, cz_cyl_use, R0_use, theta_offset_use)

        # bottom face (outer side still corresponds to w1)
        v0_idx = get_vertex_index((ix,   iy,   iz),   (x0, y0, z0))
        v1_idx = get_vertex_index((ix+1, iy,   iz),   (x1, y1, z1))
        v2_idx = get_vertex_index((ix+1, iy+1, iz),   (x2, y2, z2))
        v3_idx = get_vertex_index((ix,   iy+1, iz),   (x3, y3, z3))
        # top face
        v4_idx = get_vertex_index((ix,   iy,   iz+1), (x4, y4, z4))
        v5_idx = get_vertex_index((ix+1, iy,   iz+1), (x5, y5, z5))
        v6_idx = get_vertex_index((ix+1, iy+1, iz+1), (x6, y6, z6))
        v7_idx = get_vertex_index((ix,   iy+1, iz+1), (x7, y7, z7))

        hexes.append((v0_idx, v1_idx, v2_idx, v3_idx, v4_idx, v5_idx, v6_idx, v7_idx))

        eid = len(hexes)
        slice_idx = iz_to_slice[int(iz)]
        slice_to_eids[slice_idx].append(eid)

    print(
        f"[VOXEL] Built mesh: {len(vertices)} nodes, "
        f"{len(hexes)} hex elements, {len(z_slices)} slices."
    )
    return (
        vertices,
        hexes,
        slice_to_eids,
        z_slices,
        cyl_params,
        lowest_point_world,
        axis_point_world,
        edge_left_world,
        edge_right_world,
        indices_sorted,   # for debug export
        vox               # for debug export
    )


def _write_id_list_lines(f, ids: List[int], per_line: int = 16):
    for i in range(0, len(ids), per_line):
        chunk = ids[i:i + per_line]
        f.write(", ".join(str(x) for x in chunk) + "\n")


def write_calculix_job(
    path: str,
    vertices: List[Tuple[float, float, float]],
    hexes: List[Tuple[int, int, int, int, int, int, int, int]],
    slice_to_eids: Dict[int, List[int]],
    z_slices: List[float],
    base_temp: float = 0.0,
    shrinkage_curve: List[float] = [5, 4, 3, 2, 1],
    max_cure: float = 1.0,
    cure_shrink_per_unit: float = 0.3,  # 3%
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

    # --- Normalize shrinkage_curve as weights so entries sum to 1.0 ----------
    if not shrinkage_curve:
        shrinkage_curve = [1.0]

    total_weight = float(sum(shrinkage_curve))
    if total_weight <= 0.0:
        shrinkage_curve = [1.0]
        total_weight = 1.0

    shrinkage_curve = [float(w) / total_weight for w in shrinkage_curve]

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

        for slice_idx in range(n_slices):
            eids = slice_to_eids.get(slice_idx, [])
            if not eids:
                continue
            valid_eids = [eid for eid in eids if 1 <= eid <= n_elems]
            if not valid_eids:
                print(
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
        f.write("1.02E-09\n")
        f.write("*ELASTIC\n")
        f.write("2000., 0.394\n")

        alpha = -float(cure_shrink_per_unit)  # higher T -> shrink
        f.write("*EXPANSION, ZERO=0.\n")
        f.write(f"{alpha:.6E}\n")

        f.write("*CONDUCTIVITY\n")
        f.write("0.2256\n")
        f.write("*SPECIFIC HEAT\n")
        f.write("1386000000.\n")
        f.write("*SOLID SECTION, ELSET=ALL, MATERIAL=ABS\n")

        # -------------------- INITIAL "TEMPERATURE" (CURE) -------------------
        f.write("** Initial conditions (cure variable) ++++++++++++++++++++++\n")
        f.write("*INITIAL CONDITIONS, TYPE=TEMPERATURE\n")
        f.write(f"ALLNODES, {base_temp}\n")  # typically 0.0 or 293

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
            f.write("*UNCOUPLED TEMPERATURE-DISPLACEMENT\n")
            f.write(f"{time_per_layer_step:.6f}, {time_per_layer:.6f}\n")

            if base_nodes:
                f.write("** Boundary conditions (mechanical) +++++++++++++++++++++++\n")
                f.write("*BOUNDARY, OP=NEW\n")
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

                for j in existing_slice_idxs:
                    if not printed[j]:
                        continue
                    k_applied = applied_count[j]
                    if k_applied >= curve_len:
                        continue
                    increment = shrinkage_curve[k_applied]
                    if increment != 0.0:
                        cure_state[j] = min(
                            max_cure,
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
                f.write("*UNCOUPLED TEMPERATURE-DISPLACEMENT\n")
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

                f.write("** Field outputs +++++++++++++++++++++++++++++++++++++++++++\n")
                f.write("*NODE FILE\n")
                f.write("RF, U, NT, RFL\n")
                f.write("*EL FILE\n")
                f.write("S, E, HFL, NOE\n")

                f.write("** Boundary conditions (base + shrinkage-curve cure) +++++\n")
                f.write("*BOUNDARY, OP=MOD\n")

                for j in existing_slice_idxs:
                    if not printed[j]:
                        continue
                    cure_val = cure_state[j]
                    if cure_val == 0.0:
                        continue
                    nset_j = f"SLICE_{j:03d}_NODES"
                    f.write(f"{nset_j}, 11, 11, {cure_val:.6f}\n")

                f.write("*END STEP\n")
                step_counter += 1

    print(f"[CCX] Wrote incremental-cure UT-D job to: {path}")
    print(
        f"[CCX] Nodes: {n_nodes}, elements: {n_elems}, "
        f"slices: {n_slices}, shrinkage_curve={shrinkage_curve}, "
        f"max_cure={max_cure}, cure_shrink_per_unit={cure_shrink_per_unit}"
    )


# ============================================================
#  CalculiX runner
# ============================================================

def run_calculix(job_name: str, ccx_cmd: str = "ccx"):
    print(f"[RUN] Launching CalculiX: {ccx_cmd} {job_name}")
    try:
        my_env = os.environ.copy()
        my_env["OMP_NUM_THREADS"] = "12"
        result = subprocess.run(
            [ccx_cmd, job_name],
            check=False,
            capture_output=True,
            text=True,
            env=my_env
        )
    except FileNotFoundError:
        print(f"[RUN] ERROR: CalculiX command not found: {ccx_cmd}")
        return False

    print(f"[RUN] CalculiX return code: {result.returncode}")
    if result.stdout:
        print("----- CalculiX STDOUT -----")
        print(result.stdout)
    if result.stderr:
        print("----- CalculiX STDERR -----")
        print(result.stderr)
    print("[RUN] Done.")
    return result.returncode == 0


# ============================================================
#  FRD parsing
# ============================================================

def read_frd_displacements(frd_path: str) -> Dict[int, np.ndarray]:
    if not os.path.isfile(frd_path):
        print(f"[FRD] File not found: {frd_path}")
        return {}

    disp: Dict[int, np.ndarray] = {}
    in_disp = False

    print(f"[FRD] Parsing displacements from: {frd_path}")

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

    print(f"[FRD] Parsed displacement data for {len(disp)} nodes.")
    return disp


# ============================================================
#  PyGeM FFD lattice from voxel mesh
# ============================================================

def build_ffd_from_lattice(
    vertices: List[Tuple[float, float, float]],
    cube_size: float,
    displacements: Dict[int, np.ndarray],
    cyl_params: Optional[Tuple[float, float, float, float]] = None,
):
    """
    Build a PyGeM FFD lattice from voxel nodes.

    If cyl_params is provided, cylindrical meta-data is attached for
    cylindrical visualization of control points; FFD deformation itself
    is standard rectangular in world (x, y, z).
    """
    if FFD is None:
        raise RuntimeError("PyGeM FFD is not available. Please install 'pygem'.")

    coords = np.array(vertices, dtype=float)

    # World-space bounds (for classic rectangular FFD box)
    x_min = float(coords[:, 0].min())
    x_max = float(coords[:, 0].max())
    y_min = float(coords[:, 1].min())
    y_max = float(coords[:, 1].max())
    z_min = float(coords[:, 2].min())
    z_max = float(coords[:, 2].max())

    inv_h = 1.0 / float(cube_size)

    imax = jmax = kmax = 0
    logical_idx: List[Tuple[int, int, int]] = []

    for nid, (x, y, z) in enumerate(vertices, start=1):
        ix = int(round((x - x_min) * inv_h))
        iy = int(round((y - y_min) * inv_h))
        iz = int(round((z - z_min) * inv_h))
        logical_idx.append((ix, iy, iz))
        if ix > imax:
            imax = ix
        if iy > jmax:
            jmax = iy
        if iz > kmax:
            kmax = iz

    nx = imax + 1
    ny = jmax + 1
    nz = kmax + 1

    print(f"[FFD] Building PyGeM FFD: n_control_points = ({nx}, {ny}, {nz})")

    ffd = FFD(n_control_points=[nx, ny, nz])

    Lx = max(x_max - x_min, 1e-12)
    Ly = max(y_max - y_min, 1e-12)
    Lz = max(z_max - z_min, 1e-12)

    ffd.box_origin[:] = np.array([x_min, y_min, z_min], dtype=float)
    ffd.box_length[:] = np.array([Lx, Ly, Lz], dtype=float)
    ffd.rot_angle[:] = np.array([0.0, 0.0, 0.0], dtype=float)

    if hasattr(ffd, "reset_weights"):
        ffd.reset_weights()
    else:
        ffd.array_mu_x[:] = 0.0
        ffd.array_mu_y[:] = 0.0
        ffd.array_mu_z[:] = 0.0

    # Fill control weights from nodal displacements
    for nid, (ix, iy, iz) in enumerate(logical_idx, start=1):
        if not (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
            continue

        u = displacements.get(nid)
        if u is None:
            continue

        ffd.array_mu_x[ix, iy, iz] = u[0] / Lx
        ffd.array_mu_y[ix, iy, iz] = u[1] / Ly
        ffd.array_mu_z[ix, iy, iz] = u[2] / Lz

    # --- Cylindrical meta-data for exporting curved FFD lattice -------------
    if cyl_params is not None:
        # Support both (cx, cz, R0) and (cx, cz, R0, theta_offset)
        if len(cyl_params) == 4:
            cx_cyl, cz_cyl, R0, theta_offset = cyl_params
        else:
            cx_cyl, cz_cyl, R0 = cyl_params
            theta_offset = 0.0

        # Compute cylindrical param bounds (u,v,w) for all nodes
        uvw = np.zeros_like(coords)
        for i, (x, y, z) in enumerate(coords):
            uvw[i] = world_to_param_cyl((x, y, z), cx_cyl, cz_cyl, R0)

        u_vals = uvw[:, 0]
        v_vals = uvw[:, 1]
        w_vals = uvw[:, 2]

        u_min = float(u_vals.min())
        u_max = float(u_vals.max())
        v_min = float(v_vals.min())
        v_max = float(v_vals.max())
        w_min = float(w_vals.min())
        w_max = float(w_vals.max())

        setattr(ffd, "_curved_voxels", True)
        setattr(ffd, "_cyl_params", cyl_params)
        setattr(ffd, "_param_bounds", (u_min, u_max, v_min, v_max, w_min, w_max))

        print(
            "[FFD] Curved-voxel FFD: cylindrical param bounds "
            f"u=[{u_min:.3f},{u_max:.3f}], "
            f"v=[{v_min:.3f},{v_max:.3f}], "
            f"w=[{w_min:.3f},{w_max:.3f}]"
        )
    else:
        setattr(ffd, "_curved_voxels", False)
        setattr(ffd, "_cyl_params", None)
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
    cyl_params: Optional[Tuple[float, ...]] = None,
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
        print("[PLY] No vertices, skipping lattice export.")
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

    def maybe_subsample(points: np.ndarray) -> np.ndarray:
        total = points.shape[0]
        if total > max_points:
            step = int(np.ceil(total / max_points))
            idx = np.arange(0, total, step, dtype=int)
            return points[idx]
        return points

    # Subsample lattice itself (if needed)
    original_points = maybe_subsample(original_points)
    deformed_points = maybe_subsample(deformed_points)

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

        original_points = np.vstack([original_points, marker_radial])
        deformed_points = np.vstack([deformed_points, marker_radial])

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
        original_points = np.vstack([original_points, marker_left])
        deformed_points = np.vstack([deformed_points, marker_left])

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
        original_points = np.vstack([original_points, marker_right])
        deformed_points = np.vstack([deformed_points, marker_right])

        print(
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
        print(f"[PLY] Original lattice written to: {orig_path}")

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
        print(f"[PLY] Deformed lattice written to: {def_path}")


def export_ffd_control_points(ffd, basepath: str):
    if not hasattr(ffd, "array_mu_x") or not hasattr(ffd, "box_origin") or not hasattr(ffd, "box_length"):
        print("[PLY] FFD control-point export not supported by this PyGeM FFD object.")
        return

    mu_x = np.asarray(ffd.array_mu_x, dtype=float)
    mu_y = np.asarray(ffd.array_mu_y, dtype=float)
    mu_z = np.asarray(ffd.array_mu_z, dtype=float)

    nx, ny, nz = mu_x.shape
    origin = np.asarray(ffd.box_origin, dtype=float).reshape(3)
    length = np.asarray(ffd.box_length, dtype=float).reshape(3)
    Lx, Ly, Lz = length

    # NEW: curved-voxel meta-data (if available)
    curved = bool(getattr(ffd, "_curved_voxels", False))
    cyl_params = getattr(ffd, "_cyl_params", None)
    param_bounds = getattr(ffd, "_param_bounds", None)

    use_cyl = curved and cyl_params is not None and param_bounds is not None

    if use_cyl:
        if len(cyl_params) == 4:
            cx_cyl, cz_cyl, R0, theta_offset = cyl_params
        else:
            cx_cyl, cz_cyl, R0 = cyl_params
            theta_offset = 0.0
        u_min, u_max, v_min, v_max, w_min, w_max = param_bounds
        print(
            "[PLY] Exporting FFD control points in cylindrical form "
            f"(center=({cx_cyl:.3f},{cz_cyl:.3f}), R0={R0:.3f}, "
            f"theta_offset={theta_offset:.3f})"
        )
    else:
        print("[PLY] Exporting FFD control points in rectangular form.")

    pts_orig = []
    pts_def = []

    for i in range(nx):
        s = i / (nx - 1) if nx > 1 else 0.0
        for j in range(ny):
            t = j / (ny - 1) if ny > 1 else 0.0
            for k in range(nz):
                u = k / (nz - 1) if nz > 1 else 0.0

                # Base position in rectangular box
                base_rect = origin + np.array([s * Lx, t * Ly, u * Lz], dtype=float)

                if use_cyl:
                    # Map uniform grid in [0,1]^3 -> cylindrical param bounds
                    u_cyl = u_min + s * (u_max - u_min)   # along axis (Y)
                    v_cyl = v_min + t * (v_max - v_min)   # arc-length
                    w_cyl = w_min + u * (w_max - w_min)   # radial offset

                    base_world = np.array(
                        param_to_world_cyl((u_cyl, v_cyl, w_cyl), cx_cyl, cz_cyl, R0, theta_offset),
                        dtype=float,
                    )
                else:
                    base_world = base_rect

                # Displacement of control point in world units (still based on rectangular box lengths)
                disp = np.array(
                    [
                        mu_x[i, j, k] * Lx,
                        mu_y[i, j, k] * Ly,
                        mu_z[i, j, k] * Lz,
                    ],
                    dtype=float,
                )

                pts_orig.append(base_world)
                pts_def.append(base_world + disp)

    pts_orig = np.asarray(pts_orig, dtype=float)
    pts_def = np.asarray(pts_def, dtype=float)

    orig_path = basepath + "_ffd_ctrl_orig.ply"
    def_path = basepath + "_ffd_ctrl_def.ply"

    def write_ply(points: np.ndarray, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {points.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for x, y, z in points:
                f.write(f"{x} {y} {z}\n")
        print(f"[PLY] FFD control points written to: {path}")

    write_ply(pts_orig, orig_path)
    write_ply(pts_def, def_path)


# ============================================================
#  STL deformation with PyGeM (forward + pre-deformed)
# ============================================================

def ffd_apply_points(ffd, pts: np.ndarray) -> np.ndarray:
    try:
        return ffd(pts)
    except TypeError:
        return ffd.deform(pts)


def deform_input_stl_with_frd_pygem(
    input_stl: str,
    mech_frd_path: str,
    vertices: List[Tuple[float, float, float]],
    cube_size: float,
    output_stl: str,
    lattice_basepath: str = None,
    cyl_params: Optional[Tuple[float, ...]] = None,
    lowest_point: Optional[Tuple[float, float, float]] = None,
    axis_point: Optional[Tuple[float, float, float]] = None,
    edge_left_point: Optional[Tuple[float, float, float]] = None,
    edge_right_point: Optional[Tuple[float, float, float]] = None,
):

    if FFD is None:
        print("[FFD] PyGeM FFD not available; skipping STL deformation.")
        return

    displacements = read_frd_displacements(mech_frd_path)
    if not displacements:
        print("[FFD] No displacements found, skipping STL deformation.")
        return

    print(f"[FFD] Displacements available for {len(displacements)} nodes "
          f"out of {len(vertices)} total.")

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
        export_ffd_control_points(ffd, lattice_basepath)

    mesh = trimesh.load(input_stl)
    if not isinstance(mesh, trimesh.Trimesh):
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
        else:
            print("[FFD] Could not load input STL mesh, aborting deformation.")
            return

    orig_verts = mesh.vertices.copy()
    n_verts = orig_verts.shape[0]
    print(f"[FFD] Deforming input STL with {n_verts} vertices using PyGeM FFD...")

    deformed_verts = ffd_apply_points(ffd, orig_verts)
    mesh.vertices = deformed_verts
    mesh.export(output_stl)
    print(f"[FFD] Deformed STL (PyGeM) written to: {output_stl}")

    print("[FFD] Computing pre-deformed STL by reversing FFD control displacements...")

    mesh.vertices = orig_verts

    try:
        if hasattr(ffd, "array_mu_x") and hasattr(ffd, "array_mu_y") and hasattr(ffd, "array_mu_z"):
            ffd.array_mu_x *= -1.0
            ffd.array_mu_y *= -1.0
            ffd.array_mu_z *= -1.0
        else:
            print("[FFD] WARNING: FFD inversion via weight flipping not supported by this PyGeM API.")
            return
    except Exception as e:
        print(f"[FFD] Error while flipping FFD control weights for pre-deformation: {e}")
        return

    predeformed_verts = ffd_apply_points(ffd, orig_verts)
    predeformed_path = os.path.splitext(output_stl)[0] + "_pre.stl"
    mesh.vertices = predeformed_verts
    mesh.export(predeformed_path)
    print(f"[FFD] Pre-deformed STL written to: {predeformed_path}")


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

    Always uses cylindrical mapping:
      - voxel centers/corners are in param (u,v,w)
      - corners are mapped through param_to_world_cyl().
    """
    if cyl_params is None:
        print("[VOXDBG] cyl_params missing, cannot export curved voxels.")
        return

    if len(cyl_params) == 4:
        cx, cz, R0, theta_offset = cyl_params
    else:
        cx, cz, R0 = cyl_params
        theta_offset = 0.0

    half = cube_size / 2.0
    vertices = []
    faces = []

    for (ix, iy, iz) in indices_sorted:
        # center in param coords
        center = vox.indices_to_points(
            np.array([[ix, iy, iz]], dtype=float)
        )[0]
        cx0, cy0, cz0 = center

        # axis-aligned corners in param coords
        u0, u1 = cx0 - half, cx0 + half
        v0, v1 = cy0 - half, cy0 + half
        w0, w1 = cz0 - half, cz0 + half

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

        world = [
            param_to_world_cyl(p, cx, cz, R0, theta_offset)
            for p in corners
        ]

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
    print(f"[VOXDBG] Voxel debug STL written: {out_path}")


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
        "job_name",
        help=(
            "Base job name (UTD .inp will be '<job_name>_utd.inp'; CalculiX will "
            "produce '<job_name>_utd.frd')."
        ),
    )
    parser.add_argument(
        "--cube-size",
        "-s",
        type=float,
        required=True,
        help="Edge length of each voxel cube (same units as STL, e.g. mm)",
    )
    parser.add_argument(
        "--base-temp",
        type=float,
        default=293.0,
        help="Base/support temperature (K) for initial cure value (default 293)",
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
             "'ccx', 'ccx_static', 'C:\\\\path\\\\to\\\\ccx.exe'",
    )
    parser.add_argument(
        "--export-lattice",
        action="store_true",
        help=(
            "Export FFD lattice as '<job_name>_lattice_orig.ply' and "
            "'<job_name>_lattice_def.ply' point clouds."
        ),
    )
    # Cylindrical options
    parser.add_argument(
        "--cyl-center-xz",
        type=float,
        nargs=2,
        metavar=("CX", "CZ"),
        help="Cylinder axis center in XZ plane.",
    )
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
        print(f"Input STL not found: {args.input_stl}")
        raise SystemExit(1)

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
        cyl_center_xz=tuple(args.cyl_center_xz) if args.cyl_center_xz else None,
        cyl_radius=args.cyl_radius,
    )

    # --- Export voxel debug STL ---
    voxel_debug_stl = args.job_name + "_voxels.stl"
    export_voxel_debug_stl(
        indices_sorted,
        vox,
        args.cube_size,
        cyl_params,
        voxel_debug_stl
    )

    if not vertices or not hexes:
        print("No mesh generated, aborting.")
        raise SystemExit(1)

    n_slices = len(z_slices)
    if n_slices == 0:
        print("No slices generated, aborting.")
        raise SystemExit(1)

    # 2) Single uncoupled thermo-mechanical job
    utd_job = args.job_name + "_utd"
    utd_inp = utd_job + ".inp"

    write_calculix_job(
        utd_inp,
        vertices,
        hexes,
        slice_to_eids,
        z_slices,
        base_temp=args.base_temp,
    )

    # 3) Optional run + PyGeM FFD deformation + lattice export
    if args.run_ccx:
        ok = run_calculix(utd_job, ccx_cmd=args.ccx_cmd)
        if not ok:
            print("[RUN] UTD job failed, skipping FFD.")
            return

        utd_frd = utd_job + ".frd"
        if os.path.isfile(utd_frd):
            deformed_stl = args.job_name + "_deformed.stl"
            lattice_basepath = args.job_name + "_lattice" if args.export_lattice else None
            deform_input_stl_with_frd_pygem(
                args.input_stl,
                utd_frd,
                vertices,
                args.cube_size,
                deformed_stl,
                lattice_basepath=lattice_basepath,
                cyl_params=cyl_params,
                lowest_point=lowest_point,
                axis_point=axis_point,
                edge_left_point=edge_left_point,
                edge_right_point=edge_right_point,
            )

        else:
            print(f"[FFD] Thermo-mechanical FRD '{utd_frd}' not found, skipping STL deformation.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
STL -> voxel C3D8R hex mesh -> uncoupled thermo-mechanical CalculiX job
+ FFD deformation of the original STL using PyGeM (forward + pre-deformed)
+ optional lattice visualization as separate PLY point clouds.

Pipeline:

1. Voxelize input STL into cubes of size cube_size.
2. Build global hex mesh (C3D8R elements).
3. Single CalculiX job:
   - Procedure: *UNCOUPLED TEMPERATURE-DISPLACEMENT
   - One step per slice:
       * Apply curing (via TEMP DOF) to that slice's NSET.
       * Base nodes mechanically fixed (UX, UY, UZ = 0).
       * Base nodes thermally fixed at base_temp.
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

New:
- Optional cylindrical “curved voxel” mode (--curved-voxels) where voxels are
  axis-aligned in cylindrical param space (u, v, w), then mapped to a cylinder
  in world space. Each slice becomes a curved layer following the cylindrical
  surface (radial bands).
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


def param_to_world_cyl(param_point, cx, cz, R0):
    """
    Map param space point (u, v, w) back to world (x, y, z)
    using the same cylinder definition.

    Args:
        param_point: (u, v, w)
        cx, cz: cylinder center in XZ plane
        R0: base radius

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


# ============================================================
#  Voxel mesh -> CalculiX job
# ============================================================

def generate_global_cubic_hex_mesh(
    input_stl: str,
    cube_size: float,
    curved_voxels: bool = False,
    cyl_center_xz: Optional[Tuple[float, float]] = None,
    cyl_radius: Optional[float] = None,
):
    """
    Voxelize input_stl and build a global C3D8R brick mesh.

    If curved_voxels is False (default):
        - Standard axis-aligned voxels in world (x,y,z).
        - Slices are horizontal layers in Z.

    If curved_voxels is True:
        - STL is first mapped to cylindrical param space (u,v,w) using
          world_to_param_cyl (axis along Y).
        - Voxelization happens in (u,v,w).
        - Voxel corners (in param space) are mapped back to world via
          param_to_world_cyl, giving curved hexahedra that follow a cylinder.
        - Slices become radial layers (bands in w).

    Returns:
        vertices: List[(x, y, z)]          world-coordinates of nodes
        hexes:    List[(v1..v8)]           1-based node indices
        slice_to_eids: Dict[slice_index -> List[element_id]]
        z_slices: List[float]              slice "position" (Z or param w)
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

    cyl_params = None
    if curved_voxels:
        # Work in world space to determine cylinder first
        bounds = mesh.bounds  # shape (2,3): [[xmin,ymin,zmin], [xmax,ymax,zmax]]
        x_min, y_min, z_min = bounds[0]
        x_max, y_max, z_max = bounds[1]

        verts_world = mesh.vertices.copy()

        # --- Find lowest point (min Z) ---
        z_coords = verts_world[:, 2]
        min_z_idx = int(np.argmin(z_coords))
        x_low, y_low, z_low = verts_world[min_z_idx]

        # We'll start from this X as cylinder center X
        cx_cyl = float(x_low)

        if cyl_radius is not None and cyl_radius > 0.0:
            # --- User provided radius: enforce that lowest point lies on cylinder ---
            R0 = float(cyl_radius)

            z_center_bbox = 0.5 * (z_min + z_max)
            cz_plus = z_low + R0
            cz_minus = z_low - R0

            # Choose center Z that keeps axis near bbox center
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
                f"[VOXEL] Using user radius R0={R0:.3f}; lowest point "
                f"({x_low:.3f}, {y_low:.3f}, {z_low:.3f}) lies on cylinder."
            )

        else:
            # --- No radius provided: fall back to old behaviour ---
            if cyl_center_xz is not None:
                # Explicit center overrides auto X/Z
                cx_cli, cz_cli = cyl_center_xz
                cx_cyl = cx_cli
                cz_cyl = cz_cli
                print(
                    "[VOXEL] No radius given; using explicit cyl-center-xz "
                    f"({cx_cyl:.3f}, {cz_cyl:.3f})"
                )
            else:
                # X from lowest-Z point, Z from bbox center
                cz_cyl = 0.5 * (z_min + z_max)
                print(
                    "[VOXEL] No radius given; using cx from lowest-Z point "
                    f"and cz=bbox center: cx={cx_cyl:.3f}, cz={cz_cyl:.3f}"
                )

            dx = verts_world[:, 0] - cx_cyl
            dz = verts_world[:, 2] - cz_cyl
            r = np.sqrt(dx * dx + dz * dz)
            R0 = float(np.mean(r))

        print(
            f"[VOXEL] Cyl center from lowest Z vertex: "
            f"min_z={z_low:.3f}, lowest_pt=({x_low:.3f}, {y_low:.3f}, {z_low:.3f})"
        )
        print(
            f"[VOXEL] Cylindrical voxel mode ON: axis=+Y, "
            f"center=({cx_cyl:.3f}, {cz_cyl:.3f}), R0={R0:.3f}"
        )

        # --- Debug cylinder STL (ideal cylinder) ---
        input_base = os.path.splitext(os.path.basename(input_stl))[0]
        cyl_debug_stl = input_base + "_cylinder_debug.stl"
        export_cylinder_debug_stl(cx_cyl, cz_cyl, R0, y_min, y_max, cyl_debug_stl)

        # Map mesh into param space (u,v,w)
        verts_param = np.zeros_like(verts_world)
        for i, (x, y, z) in enumerate(verts_world):
            verts_param[i] = world_to_param_cyl((x, y, z), cx_cyl, cz_cyl, R0)
        mesh.vertices = verts_param
        cyl_params = (cx_cyl, cz_cyl, R0)

    print(f"[VOXEL] Voxelizing with cube size = {cube_size} ...")
    vox = mesh.voxelized(pitch=cube_size)
    vox.fill()

    indices = vox.sparse_indices  # (N,3) with (ix,iy,iz)
    if indices.size == 0:
        print("[VOXEL] No voxels found – check cube size or input mesh.")
        return [], [], {}, []

    total_voxels = indices.shape[0]
    print(f"[VOXEL] Total filled voxels (cubes): {total_voxels}")

    # sort by iz, iy, ix
    order = np.lexsort((indices[:, 0], indices[:, 1], indices[:, 2]))
    indices_sorted = indices[order]

    # map iz -> slice "position"
    # planar: z_center (world Z); curved: w_center (param w)
    unique_iz = np.unique(indices_sorted[:, 2])
    layer_info = []
    for iz in unique_iz:
        idx_arr = np.array([[0, 0, iz]], dtype=float)
        pt = vox.indices_to_points(idx_arr)[0]  # in mesh coordinate system
        slice_coord = float(pt[2])  # Z in planar, w in cylindrical param space
        layer_info.append((iz, slice_coord))
    # Sorting slice order: planar = low Z first; cylindrical = high w first
    if curved_voxels:
        # w = radial offset; larger w is farther from cylinder center.
        # For printing/curing order, slice 0 must be at model bottom → largest w.
        layer_info.sort(key=lambda x: x[1], reverse=True)
    else:
        # planar slicing: bottom Z first
        layer_info.sort(key=lambda x: x[1])

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
    print("[VOXEL] Building global C3D8R mesh ...")

    if not curved_voxels:
        # ----------- Original axis-aligned cubic voxels -----------
        for (ix, iy, iz) in indices_sorted:
            center = vox.indices_to_points(
                np.array([[ix, iy, iz]], dtype=float)
            )[0]
            cx, cy, cz = center
            x0, x1 = cx - half, cx + half
            y0, y1 = cy - half, cy + half
            z0, z1 = cz - half, cz + half

            # bottom face
            v0 = get_vertex_index((ix,   iy,   iz),   (x0, y0, z0))  # x-,y-,z-
            v1 = get_vertex_index((ix+1, iy,   iz),   (x1, y0, z0))  # x+,y-,z-
            v2 = get_vertex_index((ix+1, iy+1, iz),   (x1, y1, z0))  # x+,y+,z-
            v3 = get_vertex_index((ix,   iy+1, iz),   (x0, y1, z0))  # x-,y+,z-
            # top face
            v4 = get_vertex_index((ix,   iy,   iz+1), (x0, y0, z1))  # x-,y-,z+
            v5 = get_vertex_index((ix+1, iy,   iz+1), (x1, y0, z1))  # x+,y-,z+
            v6 = get_vertex_index((ix+1, iy+1, iz+1), (x1, y1, z1))  # x+,y+,z+
            v7 = get_vertex_index((ix,   iy+1, iz+1), (x0, y1, z1))  # x-,y+,z+

            hexes.append((v0, v1, v2, v3, v4, v5, v6, v7))

            eid = len(hexes)
            slice_idx = iz_to_slice[int(iz)]
            slice_to_eids[slice_idx].append(eid)
    else:
        # ----------- Curved voxels on cylindrical surface -----------
        if cyl_params is None:
            raise RuntimeError("cyl_params missing in curved voxel mode")
        cx_cyl, cz_cyl, R0 = cyl_params

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

            # map to world (curved hexa)
            x0, y0, z0 = param_to_world_cyl(p0, cx_cyl, cz_cyl, R0)
            x1, y1, z1 = param_to_world_cyl(p1, cx_cyl, cz_cyl, R0)
            x2, y2, z2 = param_to_world_cyl(p2, cx_cyl, cz_cyl, R0)
            x3, y3, z3 = param_to_world_cyl(p3, cx_cyl, cz_cyl, R0)
            x4, y4, z4 = param_to_world_cyl(p4, cx_cyl, cz_cyl, R0)
            x5, y5, z5 = param_to_world_cyl(p5, cx_cyl, cz_cyl, R0)
            x6, y6, z6 = param_to_world_cyl(p6, cx_cyl, cz_cyl, R0)
            x7, y7, z7 = param_to_world_cyl(p7, cx_cyl, cz_cyl, R0)

            # bottom face
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
    return vertices, hexes, slice_to_eids, z_slices


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
    heat_flux: float = 1.0,          # kept for compatibility, NOT used
    shrinkage_curve: List[float] = [1],
    max_cure: float = 1.0,
    cure_shrink_per_unit: float = 0.2,
    curved_voxels: bool = False,     # <--- NEW
):
    """
    Additive-style uncoupled temperature-displacement job with MODEL CHANGE
    and incremental curing driven by a per-layer shrinkage curve.

    (Unchanged logic – slices may now be curved radial layers when curved_voxels
    mode is used.)
    """

    n_nodes = len(vertices)
    n_elems = len(hexes)

    # detect bottom nodes as "BASE" by minimum world Z
    # z_coords = np.array([v[2] for v in vertices], dtype=float)
    # z_min = float(z_coords.min())
    # tol = (z_coords.max() - z_min + 1e-9) * 1e-3
    # base_nodes = [i + 1 for i, z in enumerate(z_coords) if abs(z - z_min) <= tol]

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
            "(layer-wise MODEL CHANGE + shrinkage-curve-driven curing)\n"
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
        base_nodes = []

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

        # -------------------- BASE from bottom faces of first slice ---------------
        existing_slice_idxs = sorted(slice_node_ids.keys())
        if existing_slice_idxs:
            base_slice = existing_slice_idxs[0]
            base_eids = slice_to_eids.get(base_slice, [])

            base_node_set: Set[int] = set()
            for eid in base_eids:
                n0, n1, n2, n3, n4, n5, n6, n7 = hexes[eid - 1]

                if curved_voxels:
                    # Cylindrical case: "bottom" = outer radial face (w1) = nodes 4..7
                    base_node_set.update([n4, n5, n6, n7])
                else:
                    # Planar case: bottom = lower Z face = nodes 0..3
                    base_node_set.update([n0, n1, n2, n3])

            base_nodes = sorted(base_node_set)

            f.write("** Base node set = bottom faces of first slice +++++++++++++++\n")
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
        f.write(f"ALLNODES, {base_temp}\n")  # typically 0.0

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
                        f"** Step {step_counter}: add slice {name} at z = {z_val} "
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
#  (Old mechanical job writer – now unused, kept only for reference)
# ============================================================

def write_mechanical_job(*args, **kwargs):
    print("[WARN] write_mechanical_job() is deprecated and not used in this script.")


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
):
    if FFD is None:
        raise RuntimeError("PyGeM FFD is not available. Please install 'pygem'.")

    coords = np.array(vertices, dtype=float)
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

    for nid, (ix, iy, iz) in enumerate(logical_idx, start=1):
        if not (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
            continue

        u = displacements.get(nid)
        if u is None:
            continue

        ffd.array_mu_x[ix, iy, iz] = u[0] / Lx
        ffd.array_mu_y[ix, iy, iz] = u[1] / Ly
        ffd.array_mu_z[ix, iy, iz] = u[2] / Lz

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
):
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

    def maybe_subsample(points):
        total = points.shape[0]
        if total > max_points:
            step = int(np.ceil(total / max_points))
            idx = np.arange(0, total, step, dtype=int)
            return points[idx]
        return points

    original_points = maybe_subsample(original_points)
    deformed_points = maybe_subsample(deformed_points)

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

    pts_orig = []
    pts_def = []

    for i in range(nx):
        s = i / (nx - 1) if nx > 1 else 0.0
        for j in range(ny):
            t = j / (ny - 1) if ny > 1 else 0.0
            for k in range(nz):
                u = k / (nz - 1) if nz > 1 else 0.0

                base = origin + np.array([s * Lx, t * Ly, u * Lz], dtype=float)

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

    if lattice_basepath:
        orig_ply = lattice_basepath + "_orig.ply"
        def_ply = lattice_basepath + "_def.ply"
        export_lattice_ply_split(vertices, displacements, orig_ply, def_ply)

    ffd = build_ffd_from_lattice(vertices, cube_size, displacements)

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


# ============================================================
#  CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Voxelize an STL and generate an uncoupled thermo-mechanical CalculiX job "
            "with C3D8R hexahedra and layer-by-layer *UNCOUPLED TEMPERATURE-DISPLACEMENT "
            "steps, then deform the input STL using PyGeM FFD (forward + pre-deformed) "
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
        help="Base/support temperature (K) for boundary condition (default 293)",
    )
    parser.add_argument(
        "--heat-flux",
        type=float,
        default=1.0e3,
        help="Heat flux value used in *DFLUX (default 1.0e3)",
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
    # --- New cylindrical voxel options ---
    parser.add_argument(
        "--curved-voxels",
        action="store_true",
        help=(
            "Use cylindrical curved voxels instead of axis-aligned cubes. "
            "Cylinder axis is global +Y; slices become radial layers."
        ),
    )
    parser.add_argument(
        "--cyl-center-xz",
        type=float,
        nargs=2,
        metavar=("CX", "CZ"),
        help="Cylinder axis center in XZ plane (used with --curved-voxels).",
    )
    parser.add_argument(
        "--cyl-radius",
        type=float,
        help=(
            "Base cylinder radius R0 (used with --curved-voxels). "
            "If omitted, estimated from STL geometry."
        ),
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input_stl):
        print(f"Input STL not found: {args.input_stl}")
        raise SystemExit(1)

    # 1) Mesh
    vertices, hexes, slice_to_eids, z_slices = generate_global_cubic_hex_mesh(
        args.input_stl,
        args.cube_size,
        curved_voxels=args.curved_voxels,
        cyl_center_xz=tuple(args.cyl_center_xz) if args.cyl_center_xz else None,
        cyl_radius=args.cyl_radius,
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
        # base_temp=args.base_temp,
        # heat_flux=args.heat_flux,
        curved_voxels=args.curved_voxels,   # <--- NEW
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
            )
        else:
            print(f"[FFD] Thermo-mechanical FRD '{utd_frd}' not found, skipping STL deformation.")


if __name__ == "__main__":
    main()

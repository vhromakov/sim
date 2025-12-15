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
import point_cloud_utils as pcu
import pymeshfix as pfix


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


import math
import numpy as np
import trimesh


def slice_tet_volume_along_w(
    nodes_param: np.ndarray,          # (N,3) float
    elems_0based: np.ndarray,         # (M,4) int, TetGen t.elem (0-based)
    w_step: float,
    w_min: float | None = None,
    w_max: float | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[float]]:
    """
    Slice a tetrahedral VOLUME mesh (param space) into slabs along +w (axis = z).

    Returns:
      tet_slices_param : list[(nodes_param_slice, elems_local_1based)]
        nodes_param_slice : (Ni,3)
        elems_local_1based : (Mi,4) 1-based indices into nodes_param_slice
      z_slices : slice center values in w
    """
    nodes_param = np.asarray(nodes_param, dtype=np.float64)
    elems_0based = np.asarray(elems_0based, dtype=np.int64)

    if nodes_param.size == 0 or elems_0based.size == 0:
        return [], []

    w_all = nodes_param[:, 2]
    if w_min is None:
        w_min = float(w_all.min())
    if w_max is None:
        w_max = float(w_all.max())

    if w_max <= w_min:
        # single slice, everything
        elems_1based = (elems_0based + 1).astype(np.int32)
        return [(nodes_param.copy(), elems_1based)], [0.5 * (w_min + w_max)]

    eps = 1e-9
    w_min_eps = w_min - eps
    w_max_eps = w_max + eps
    num_slices = int(math.ceil((w_max_eps - w_min_eps) / w_step))

    # Prefer robust cutting via VTK/PyVista (creates new vertices on cut planes)
    try:
        import pyvista as pv

        # Build a VTK UnstructuredGrid with tetra cells
        # VTK cell array format: [4, a,b,c,d, 4, a,b,c,d, ...]
        cells = np.empty((elems_0based.shape[0], 5), dtype=np.int64)
        cells[:, 0] = 4
        cells[:, 1:] = elems_0based
        cells_vtk = cells.ravel()

        celltypes = np.full(elems_0based.shape[0], pv.CellType.TETRA, dtype=np.uint8)
        grid0 = pv.UnstructuredGrid(cells_vtk, celltypes, nodes_param)

        slices: list[tuple[np.ndarray, np.ndarray]] = []
        z_slices: list[float] = []

        for si in range(num_slices):
            w0 = w_min_eps + si * w_step
            w1 = w0 + w_step
            z_slices.append(0.5 * (w0 + w1))

            g = grid0

            # Keep w >= w0
            # (If your result comes out inverted, flip invert=True here.)
            g = g.clip(normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, w0), invert=False)

            if g.n_cells == 0:
                slices.append((np.zeros((0, 3), np.float64), np.zeros((0, 4), np.int32)))
                continue

            # Keep w <= w1
            g = g.clip(normal=(0.0, 0.0, -1.0), origin=(0.0, 0.0, w1), invert=False)

            if g.n_cells == 0:
                slices.append((np.zeros((0, 3), np.float64), np.zeros((0, 4), np.int32)))
                continue

            pts = np.asarray(g.points, dtype=np.float64)

            # Extract only tetra cells from the clipped result.
            # g.cells is a flat array: [npts, id0, id1, ..., npts, id0, ...]
            flat = np.asarray(g.cells, dtype=np.int64)
            ctypes = np.asarray(g.celltypes, dtype=np.uint8)

            tets_local_0 = []
            idx = 0
            dropped = 0

            for ct in ctypes:
                n = int(flat[idx]); idx += 1
                conn = flat[idx:idx + n]; idx += n

                if ct == pv.CellType.TETRA and n == 4:
                    tets_local_0.append(conn.astype(np.int64))
                else:
                    dropped += 1

            if len(tets_local_0) == 0:
                if dropped > 0:
                    log(f"[SLICE-TET] Slice {si}: no tets after clipping (dropped {dropped} non-tet cells)")
                slices.append((np.zeros((0, 3), np.float64), np.zeros((0, 4), np.int32)))
                continue

            tets_local_0 = np.vstack(tets_local_0)
            elems_local_1 = (tets_local_0 + 1).astype(np.int32)

            if dropped > 0:
                log(f"[SLICE-TET] Slice {si}: kept {elems_local_1.shape[0]} tets, dropped {dropped} non-tet cells")

            slices.append((pts, elems_local_1))

        # Your convention: slice 0 is bottom
        slices.reverse()
        z_slices.reverse()
        return slices, z_slices

    except Exception as e:
        # Fallback (no cutting): assign whole tets by centroid w.
        # This DOES NOT create cut cells, but gives you per-slice groups quickly.
        log(f"[SLICE-TET] PyVista slicing unavailable, fallback to centroid binning: {e}")

        cent_w = nodes_param[elems_0based].mean(axis=1)[:, 2]  # (M,)
        slices: list[tuple[np.ndarray, np.ndarray]] = []
        z_slices: list[float] = []

        for si in range(num_slices):
            w0 = w_min_eps + si * w_step
            w1 = w0 + w_step
            z_slices.append(0.5 * (w0 + w1))

            mask = (cent_w >= w0) & (cent_w <= w1)
            elems_here = elems_0based[mask]
            if elems_here.size == 0:
                slices.append((np.zeros((0, 3), np.float64), np.zeros((0, 4), np.int32)))
                continue

            # Build compacted node list for this slice
            used = np.unique(elems_here.ravel())
            map_old_to_new = {int(old): i for i, old in enumerate(used)}
            nodes_slice = nodes_param[used]

            elems_new0 = np.vectorize(lambda x: map_old_to_new[int(x)])(elems_here).astype(np.int64)
            elems_new1 = (elems_new0 + 1).astype(np.int32)
            slices.append((nodes_slice, elems_new1))

        slices.reverse()
        z_slices.reverse()
        return slices, z_slices


def tet_edges_from_elems_1based(elems_1based: np.ndarray) -> list[tuple[int, int]]:
    """
    elems_1based: (M,K) int, 1-based node indices (per-slice local).
    Returns edges as 0-based pairs (i, j) for write_ply_edges.
    Works for K>=4. (If K>4, uses all node pairs.)
    """
    elems = np.asarray(elems_1based, dtype=np.int64)
    if elems.size == 0:
        return []

    edges: set[tuple[int, int]] = set()
    for row in elems:
        ids = [int(x) - 1 for x in row.tolist() if int(x) > 0]
        L = len(ids)
        for i in range(L):
            for j in range(i + 1, L):
                a, b = ids[i], ids[j]
                if a == b:
                    continue
                if a < b:
                    edges.add((a, b))
                else:
                    edges.add((b, a))
    return sorted(edges)


def build_global_tet_from_slices(
    tet_slices: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int, int]], dict[int, list[int]]]:
    vertices: list[tuple[float, float, float]] = []
    elements: list[tuple[int, int, int, int]] = []
    slice_to_eids: dict[int, list[int]] = {}

    node_offset = 0
    elem_offset = 0

    for si, (nodes, elems_local_1based) in enumerate(tet_slices):
        nodes = np.asarray(nodes, dtype=np.float64)
        elems_local_1based = np.asarray(elems_local_1based, dtype=np.int32)

        if nodes.size == 0 or elems_local_1based.size == 0:
            slice_to_eids[si] = []
            continue

        for p in nodes:
            vertices.append((float(p[0]), float(p[1]), float(p[2])))

        eids_here: list[int] = []
        for k in range(elems_local_1based.shape[0]):
            a, b, c, d = elems_local_1based[k]
            elements.append((
                int(a) + node_offset,
                int(b) + node_offset,
                int(c) + node_offset,
                int(d) + node_offset,
            ))
            eids_here.append(elem_offset + k + 1)

        slice_to_eids[si] = eids_here
        node_offset += nodes.shape[0]
        elem_offset += elems_local_1based.shape[0]

    return vertices, elements, slice_to_eids


def generate_global_tet_mesh(
    input_stl: str,
    cube_size: float,
    cyl_radius: float,
    out_dir: str,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[float], tuple[float, float, float, float, float]]:
    mesh = trimesh.load(input_stl)
    verts_world = mesh.vertices.copy()

    # --- cylinder center logic (unchanged) ---
    bounds = mesh.bounds
    z_min_w, z_max_w = bounds[0, 2], bounds[1, 2]

    z_coords = verts_world[:, 2]
    min_z_idx = int(np.argmin(z_coords))
    x_low, y_low, z_low = verts_world[min_z_idx]

    cx_cyl = float(x_low)

    base_radius = float(cyl_radius)
    z_center_bbox = 0.5 * (z_min_w + z_max_w)
    cz_plus = z_low + base_radius
    cz_minus = z_low - base_radius
    cz_cyl = cz_plus if abs(cz_plus - z_center_bbox) < abs(cz_minus - z_center_bbox) else cz_minus

    R0 = base_radius
    cyl_params = (cx_cyl, cz_cyl, float(R0), 0.0, 0.0)

    # --- map to param space ---
    verts_param = np.zeros_like(verts_world)
    for i, (x, y, z) in enumerate(verts_world):
        verts_param[i] = world_to_param_cyl((x, y, z), cx_cyl, cz_cyl, R0)

    mesh_param = trimesh.Trimesh(vertices=verts_param, faces=mesh.faces, process=False)

    # --- watertight + repair in param space ---
    try:
        v_param = mesh_param.vertices.astype(np.float64)
        f_param = mesh_param.faces.astype(np.int64)

        vw, fw = pcu.make_mesh_watertight(v_param, f_param, resolution=50_000)
        tin = pfix.MeshFix(vw, fw)
        tin.repair()
        mesh_param = trimesh.Trimesh(vertices=tin.v, faces=tin.f, process=False)
        log(f"[WT] repaired param mesh: verts={len(mesh_param.vertices)} faces={len(mesh_param.faces)}")
    except Exception as e:
        log(f"[WT] repair skipped/failed: {e}")

    verts_param = np.asarray(mesh_param.vertices, dtype=np.float64)
    faces_param = np.asarray(mesh_param.faces, dtype=np.int32)

    # --- TetGen ONCE (param space volume) ---
    if verts_param.size == 0 or faces_param.size == 0:
        return [], [], cyl_params

    try:
        t = tetgen.TetGen(verts_param, faces_param)
        t.tetrahedralize(
            switches="pq1.2Y",
            verbose=1,
        )

        nodes_param = np.asarray(t.node, dtype=np.float64)
        elems = np.asarray(t.elem, dtype=np.int64)

        if elems.size == 0 or nodes_param.size == 0:
            log("[TET] TetGen produced empty volume")
            return [], [], cyl_params

        # keep linear tets only
        if elems.shape[1] > 4:
            elems = elems[:, :4]

        log(f"[TET] full volume: nodes={nodes_param.shape[0]}, tets={elems.shape[0]}")

    except Exception as e:
        log(f"[TET] TetGen failed (full volume): {e}")
        return [], [], cyl_params

    # --- slice the TET VOLUME along w in param space (no caps!) ---
    w_all = nodes_param[:, 2]
    w_min = float(w_all.min())
    w_max = float(w_all.max())

    tet_slices_param, z_slices = slice_tet_volume_along_w(
        nodes_param=nodes_param,
        elems_0based=elems,
        w_step=cube_size,
        w_min=w_min,
        w_max=w_max,
    )
    log(f"[SLICE-TET] Generated {len(tet_slices_param)} tet slices in param space.")

    # --- map per-slice nodes to WORLD space ---
    tet_slices_world: list[tuple[np.ndarray, np.ndarray]] = []
    for si, (nodes_p, elems_1based) in enumerate(tet_slices_param):
        if nodes_p.size == 0 or elems_1based.size == 0:
            tet_slices_world.append((np.zeros((0, 3), np.float64), np.zeros((0, 4), np.int32)))
            continue

        nodes_world = np.zeros_like(nodes_p)
        for i in range(nodes_p.shape[0]):
            nodes_world[i] = param_to_world_cyl(nodes_p[i], cx_cyl, cz_cyl, R0)

        tet_slices_world.append((nodes_world, elems_1based.astype(np.int32)))
        log(f"[SLICE-TET] Slice {si}: nodes={nodes_world.shape[0]}, tets={elems_1based.shape[0]}")

        # --- optional debug wireframe ---
        try:
            dbg_dir = Path(out_dir) / "TET_DEBUG"
            dbg_dir.mkdir(parents=True, exist_ok=True)
            edges0 = tet_edges_from_elems_1based(elems_1based)
            dbg_path = dbg_dir / f"{Path(input_stl).stem}_tet_slice_{si:03d}.ply"
            write_ply_edges(dbg_path, nodes_world, edges0)
        except Exception as e_dbg:
            log(f"[TET-DBG] Slice {si}: failed debug PLY: {e_dbg}")

    return tet_slices_world, z_slices, cyl_params


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
        def _fmt_ccx_float(val: float) -> str:
            # CCX parses scientific notation reliably; keep it short and ASCII.
            return f"{float(val):.12e}"

        for i, (x, y, z) in enumerate(vertices, start=1):
            if not (math.isfinite(float(x)) and math.isfinite(float(y)) and math.isfinite(float(z))):
                raise ValueError(f"Non-finite node coord at node {i}: {(x,y,z)}")
            f.write(f"{i}, {_fmt_ccx_float(x)}, {_fmt_ccx_float(y)}, {_fmt_ccx_float(z)}\n")

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


from pathlib import Path
import trimesh

def build_bottom_cap_faces_from_loops(
    verts_idx_slice: np.ndarray,
    faces_local: np.ndarray,
    z_band: float = 0.5,
    debug_dir: str | Path | None = None,
    slice_idx: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a radial bottom cap for one MC slice using boundary contour loops.

    Parameters
    ----------
    verts_idx_slice : (N, 3)
        Slice vertices in index/param coords (local indexing [0..N-1]).
    faces_local : (F, 3)
        Slice faces in terms of local vertex indices [0..N-1].
        (All faces of the slice, not pre-filtered by plane.)
    z_band : float
        Max allowed |mean_z(loop) - max_mean_z| in index units for loops
        to be considered "bottom" loops.
    debug_dir : str | Path | None
        If not None, export bottom loops and hole points as PLY point clouds.
    slice_idx : int | None
        Optional slice index used in debug filenames.

    Returns
    -------
    new_verts_idx : (M, 3) float
        Newly created vertices in index coords (will be appended to the slice).
        May be empty if Triangle did not create any Steiner points.
    faces_cap_local : (K, 3) int
        Triangle faces (local vertex indices) forming the bottom cap.
        These indices are in the *extended* local space [0..N+M-1].
    """
    if verts_idx_slice.shape[0] < 3 or faces_local.shape[0] == 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int)

    N0 = verts_idx_slice.shape[0]

    # --- 1) Find boundary edges (edges used exactly once in this slice) ---
    edge_count: dict[tuple[int, int], int] = {}
    for (i0, i1, i2) in faces_local:
        for a, b in ((i0, i1), (i1, i2), (i2, i0)):
            e = (min(a, b), max(a, b))
            edge_count[e] = edge_count.get(e, 0) + 1

    boundary_edges = [e for e, c in edge_count.items() if c == 1]
    if not boundary_edges:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int)

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
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int)

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
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int)

    max_mean_z = max(s[1] for s in loop_stats)
    bottom_loop_ids = [
        li for (li, mean_z, _area) in loop_stats
        if abs(mean_z - max_mean_z) <= z_band
    ]
    if not bottom_loop_ids:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int)

    # store mean_z for each loop (used for z of new points)
    loop_mean_z = {li: mean_z for (li, mean_z, _area) in loop_stats}

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

    # ============ 5) Helpers for triangulation & interior point ==========
    def _polygon_signed_area(poly: np.ndarray) -> float:
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
        Uses ear-clip triangulation:
        - Triangulate polygon.
        - Take largest-area triangle.
        - Return its centroid.
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
            return 0.5 * (poly[0] + poly[1])

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
            if _point_in_poly(best_centroid, poly):
                return best_centroid

        # Fallbacks should almost never be hit in your case:
        xmin, ymin = poly.min(axis=0)
        xmax, ymax = poly.max(axis=0)
        if xmax == xmin and ymax == ymin:
            return poly[0].copy()

        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        center = np.array([cx, cy], dtype=float)

        if _point_in_poly(center, poly):
            return center

        for v in poly:
            cand = v + 0.25 * (center - v)
            if _point_in_poly(cand, poly):
                return cand

        return center

    # ================== 6) TRIANGULATE FOR ALL EVEN DEPTHS =============
    new_verts_list: list[np.ndarray] = []
    faces_cap_local: list[list[int]] = []
    hole_points_debug: list[np.ndarray] = []  # 3D points for holes debug export

    def _alloc_new_vertex(x: float, y: float, z: float) -> int:
        """Create a new vertex in index space and return its local index."""
        idx = N0 + len(new_verts_list)
        new_verts_list.append(np.array([x, y, z], dtype=float))
        return idx

    for outer_li in bottom_loop_ids:
        d_outer = depth[outer_li]
        if d_outer % 2 != 0:
            continue  # only even-depth loops define solid regions

        outer_poly = poly2d_by_loop[outer_li]
        poly_area = abs(_polygon_signed_area(outer_poly))
        if poly_area < 1e-12:
            continue

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

        # 2D coordinates for the PSLG
        pts2d = np.array(
            [verts_idx_slice[v, :2] for v in vert_ids_sorted],
            dtype=float,
        )

        # Map from rounded (x, y) -> existing local vertex index
        existing_xy_to_local: dict[tuple[float, float], int] = {}
        for v_local in vert_ids_sorted:
            x, y = verts_idx_slice[v_local, 0], verts_idx_slice[v_local, 1]
            key = (round(float(x), 6), round(float(y), 6))
            existing_xy_to_local[key] = v_local

        segments: list[tuple[int, int]] = []

        # segments for outer loop
        outer_loop = loops[outer_li]
        L_out = len(outer_loop)
        for i in range(L_out):
            v0 = outer_loop[i]
            v1 = outer_loop[(i + 1) % L_out]
            s0 = vert_ids_sorted.index(v0)
            s1 = vert_ids_sorted.index(v1)
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
                s0 = vert_ids_sorted.index(v0)
                s1 = vert_ids_sorted.index(v1)
                segments.append((s0, s1))

        if not segments:
            continue

        # Hole points: use _find_interior_point
        holes_pts: list[list[float]] = []
        for li in hole_lis:
            poly_hole = poly2d_by_loop[li]
            p_in = _find_interior_point(poly_hole)
            holes_pts.append([float(p_in[0]), float(p_in[1])])

            # store 3D hole point for debug (z = loop_mean_z of this hole loop)
            z_hole = loop_mean_z.get(li, max_mean_z)
            hole_points_debug.append(
                np.array([float(p_in[0]), float(p_in[1]), float(z_hole)], dtype=float)
            )

        data = {
            "vertices": pts2d,
            "segments": np.array(segments, dtype=int),
        }
        if holes_pts:
            data["holes"] = np.array(holes_pts, dtype=float)

        # --- Quality Triangle call (adds Steiner points) -----------------
        min_angle = 25.0  # degrees
        flags = f"pq{min_angle}"

        tri_out = tr.triangulate(data, flags)
        if "triangles" not in tri_out or "vertices" not in tri_out:
            continue

        tri_verts2d = np.asarray(tri_out["vertices"], dtype=float)
        tris_pslg = np.asarray(tri_out["triangles"], dtype=int)
        if tris_pslg.size == 0:
            continue

        # Map Triangle vertex indices -> local vertex indices (old or new)
        tri_to_local: dict[int, int] = {}
        z_plane = loop_mean_z.get(outer_li, max_mean_z)

        for vid_pslg, (x2d, y2d) in enumerate(tri_verts2d):
            key = (round(float(x2d), 6), round(float(y2d), 6))

            if key in existing_xy_to_local:
                tri_to_local[vid_pslg] = existing_xy_to_local[key]
            else:
                local_idx = _alloc_new_vertex(x2d, y2d, z_plane)
                tri_to_local[vid_pslg] = local_idx
                existing_xy_to_local[key] = local_idx

        # Map all Triangle triangles to local indices
        for a, b, c in tris_pslg:
            ga = tri_to_local[int(a)]
            gb = tri_to_local[int(b)]
            gc = tri_to_local[int(c)]
            faces_cap_local.append([ga, gb, gc])

    if not faces_cap_local:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int)

    # ------------------ DEBUG EXPORT: bottom loops & hole points --------
    if debug_dir is not None:
        dbg_dir = Path(debug_dir)
        dbg_dir.mkdir(parents=True, exist_ok=True)

        # bottom loops as connected line segments
        loop_segments = []  # each entry: (2, 3) segment
        for li in bottom_loop_ids:
            loop = loops[li]
            if len(loop) < 2:
                continue

            # connect all points in order, and close the loop
            L = len(loop)
            for i in range(L):
                v0_idx = loop[i]
                v1_idx = loop[(i + 1) % L]  # wrap around to close
                p0 = verts_idx_slice[v0_idx]
                p1 = verts_idx_slice[v1_idx]
                loop_segments.append(np.vstack([p0, p1]))

        if loop_segments:
            segs = np.stack(loop_segments, axis=0)  # (num_segments, 2, 3)
            path = trimesh.load_path(segs)
            if slice_idx is not None:
                fname_loops = f"bottom_loops_slice_{slice_idx:03d}.ply"
            else:
                fname_loops = "bottom_loops.ply"
            path.export((dbg_dir / fname_loops).as_posix())

        # hole interior points (still as points)
        if hole_points_debug:
            pc_holes = trimesh.points.PointCloud(
                np.asarray(hole_points_debug, dtype=float)
            )
            if slice_idx is not None:
                fname_holes = f"bottom_holes_slice_{slice_idx:03d}.ply"
            else:
                fname_holes = "bottom_holes.ply"
            pc_holes.export((dbg_dir / fname_holes).as_posix())

    # --------------------------------------------------------------------
    new_verts_idx = (
        np.vstack(new_verts_list).astype(float) if new_verts_list else np.zeros((0, 3), dtype=float)
    )
    return new_verts_idx, np.asarray(faces_cap_local, dtype=int)




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
    tet_slices, z_slices, cyl_params = generate_global_tet_mesh(
        args.input_stl,
        args.cube_size,
        cyl_radius=args.cyl_radius,
        out_dir=out_dir,
    )

    tet_vertices, tet_elements, tet_slice_to_eids = build_global_tet_from_slices(tet_slices)

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
        cure_shrink_per_unit=0.05,
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

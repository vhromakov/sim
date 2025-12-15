import os
import subprocess
import trimesh
import point_cloud_utils as pcu
import pymeshfix as pfix
import numpy as np
import tetgen

import math
from typing import List, Any
import numpy as np

import math
from typing import List, Any
import numpy as np


def write_calculix_job_tet_single_layer_bottom_fixed(
    path: str,
    tg: Any,  # tetgen.TetGen or pyvista.UnstructuredGrid
    shrinkage_curve: List[float],
    cure_shrink_per_unit: float,
    bottom_thickness: float = 10,   # in model units; pick something like 0.2..2.0
):
    def _fmt(val: float) -> str:
        return f"{float(val):.12e}"

    # ---- extract grid ----
    if hasattr(tg, "points") and hasattr(tg, "cells"):
        grid = tg
    elif hasattr(tg, "grid") and tg.grid is not None:
        grid = tg.grid
    elif hasattr(tg, "tetrahedralize"):
        grid = tg.tetrahedralize()
    else:
        raise TypeError("Expected tetgen.TetGen or pyvista.UnstructuredGrid")

    vertices = np.asarray(grid.points, dtype=float)
    n_nodes = int(vertices.shape[0])

    # ---- tets from pyvista cell buffer: [4,a,b,c,d, 4,a,b,c,d,...] ----
    cells = np.asarray(grid.cells, dtype=np.int64)
    tets = []
    i = 0
    while i < len(cells):
        n = int(cells[i])
        if n != 4:
            raise ValueError("Non-tetrahedral cell encountered")
        a, b, c, d = map(int, cells[i + 1:i + 5])
        tets.append((a + 1, b + 1, c + 1, d + 1))  # -> 1-based for CCX
        i += 1 + n
    n_elems = len(tets)

    # ---- normalize curve ----
    if not shrinkage_curve:
        shrinkage_curve = [1.0]
    total = float(sum(shrinkage_curve))
    if total <= 0.0:
        raise ValueError("shrinkage_curve must have positive sum")
    shrinkage_curve = [float(w) / total for w in shrinkage_curve]

    cure_vals = []
    c = 0.0
    for w in shrinkage_curve:
        c = min(1.0, c + w)
        cure_vals.append(c)

    # ---- bottom nodes (Y-min clamp) ----
    y = vertices[:, 1]
    y_min = float(y.min())
    tol = float(bottom_thickness)
    base_nodes = np.where(y <= (y_min + tol))[0] + 1  # 1-based ids
    base_nodes = base_nodes.astype(int).tolist()

    def _write_ids(fh, ids, per_line=16):
        for k in range(0, len(ids), per_line):
            fh.write(", ".join(str(x) for x in ids[k:k + per_line]) + "\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("**\n")
        f.write("** Single-layer Tet shrink test (C3D4) with bottom-fixed BASE\n")
        f.write("**\n")
        f.write("*HEADING\n")
        f.write("Global shrinkage test on tetrahedral mesh (bottom fixed)\n")

        # ---- nodes ----
        f.write("*NODE\n")
        for nid, (x, yy, z) in enumerate(vertices, start=1):
            if not (math.isfinite(x) and math.isfinite(yy) and math.isfinite(z)):
                raise ValueError(f"Invalid node {nid}")
            f.write(f"{nid}, {_fmt(x)}, {_fmt(yy)}, {_fmt(z)}\n")

        # ---- elements ----
        f.write("*ELEMENT, TYPE=C3D4, ELSET=ALL\n")
        for eid, (n0, n1, n2, n3) in enumerate(tets, start=1):
            f.write(f"{eid}, {n0}, {n1}, {n2}, {n3}\n")

        # ---- sets ----
        f.write("*NSET, NSET=ALLNODES, GENERATE\n")
        f.write(f"1, {n_nodes}, 1\n")

        if not base_nodes:
            raise RuntimeError("BASE node set ended up empty; increase bottom_thickness or check mesh units.")

        f.write("*NSET, NSET=BASE\n")
        _write_ids(f, base_nodes)

        # ---- material ----
        f.write("*MATERIAL, NAME=ABS\n")
        f.write("*DENSITY\n")
        f.write("1.12e-9\n")
        f.write("*ELASTIC\n")
        f.write("2800., 0.35\n")

        # shrink via thermal expansion driven by TEMP(11)
        alpha = -float(cure_shrink_per_unit)
        f.write("*EXPANSION, ZERO=0.\n")
        f.write(f"{alpha:.6E}\n")

        # required by UT-D transient thermal part
        f.write("*CONDUCTIVITY\n")
        f.write("0.20\n")
        f.write("*SPECIFIC HEAT\n")
        f.write("1.30e+9\n")

        f.write("*SOLID SECTION, ELSET=ALL, MATERIAL=ABS\n")

        # ---- initial cure ----
        f.write("*INITIAL CONDITIONS, TYPE=TEMPERATURE\n")
        f.write("ALLNODES, 0.0\n")

        # ---- dummy step ----
        f.write("*STEP\n")
        f.write("*UNCOUPLED TEMPERATURE-DISPLACEMENT\n")
        f.write("1.0, 1.0\n")
        f.write("*BOUNDARY\n")
        f.write("BASE, 1, 3, 0.\n")  # fix X,Y,Z on bottom band
        f.write("*NODE FILE\n")
        f.write("U\n")
        f.write("*END STEP\n")

        # ---- cure steps ----
        prev = 0.0
        for cure in cure_vals:
            f.write("*STEP\n")
            f.write("*UNCOUPLED TEMPERATURE-DISPLACEMENT\n")
            f.write("1.0, 1.0\n")
            f.write("*BOUNDARY\n")
            f.write("BASE, 1, 3, 0.\n")
            if cure != prev:
                f.write(f"ALLNODES, 11, 11, {cure:.6f}\n")
            f.write("*NODE FILE\n")
            f.write("U\n")
            f.write("*END STEP\n")
            prev = cure

    print(f"[CCX] wrote: {path}")
    print(f"[CCX] nodes={n_nodes}, elems={n_elems}, BASE nodes={len(base_nodes)}, y_min={y_min}, tol={tol}")


def run_calculix(job_name: str, ccx_cmd: str = "ccx"):
    print(f"[RUN] Launching CalculiX: {ccx_cmd} {job_name}")

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
        print(f"[RUN] ERROR: CalculiX command not found: {ccx_cmd}")
        return False

    rc = proc.wait()
    print(f"[RUN] CalculiX completed with return code {rc}")
    print(f"[RUN] Full output written to: {log_path}")

    return rc == 0


repaired_mesh = trimesh.load("MODELS/CSC16_U00P_.stl")

vw, fw = pcu.make_mesh_watertight(
    repaired_mesh.vertices.astype(np.float64),
    repaired_mesh.faces.astype(np.int64),
    resolution=50_000
)

tin = pfix.MeshFix(vw, fw)
tin.repair()

repaired_mesh = trimesh.Trimesh(vertices=tin.v, faces=tin.f, process=False)
repaired_mesh.export("CLEAN.stl")

t = tetgen.TetGen(repaired_mesh.vertices, repaired_mesh.faces)
t.tetrahedralize(
    order=1,        # linear tets (C3D4)
    quality=False,  # DO NOT enforce radius-edge ratio
    mindihedral=0,  # disable angle constraints
    steinerleft=0,  # allow NO Steiner points
    verbose=1
)

write_calculix_job_tet_single_layer_bottom_fixed(
    path="OUTPUT/shrink_test.inp",
    tg=t,  # or grid
    shrinkage_curve=[1],
    cure_shrink_per_unit=0.2,
)

run_calculix(
    "OUTPUT/shrink_test",
    "C:/Users/4y5t6/Downloads/PrePoMax v2.4.0/Solver/ccx_dynamic.exe"
)

import re
import numpy as np
import vtk
from vtk.util import numpy_support



# =========================
# 1) Build a VTK tet grid + locator from TetGen/PyVista grid
# =========================
import os
import numpy as np
import vtk


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def write_ply_points(path: str, points_xyz: np.ndarray):
    """
    Write a point-cloud PLY (ASCII) with vertex positions only.
    """
    _ensure_dir(path)
    pts = np.asarray(points_xyz, dtype=float)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for x, y, z in pts:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def write_ply_vector_lines(path: str, p0: np.ndarray, p1: np.ndarray):
    """
    Write a PLY with 2*N vertices (p0 then p1) and N line edges connecting (2*i)->(2*i+1).
    Great for visualizing displacement vectors.
    """
    _ensure_dir(path)
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    assert p0.shape == p1.shape and p0.ndim == 2 and p0.shape[1] == 3

    n = p0.shape[0]
    verts = np.vstack([p0, p1])  # 2N x 3

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {2*n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element edge {n}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("end_header\n")

        for x, y, z in verts:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

        # edges connect i -> i+n (i.e. pre -> post)
        for i in range(n):
            f.write(f"{i} {i+n}\n")


def export_tet_displacement_debug(
    ug: vtk.vtkUnstructuredGrid,
    disp_by_nid: dict,
    out_prefix: str = "DEBUG/tet",
    scale: float = 1.0,
    max_vectors: int = 50000,
    stride: int = 1,
):
    """
    Exports PLY debug artifacts for tet nodal displacement field.

    - Uses ug points as pre positions.
    - Uses disp_by_nid[nid] where nid = pid+1 (CCX 1-based).
    - Writes vector lines (pre->post) for a subset of nodes.

    max_vectors: cap output size
    stride: additionally downsample nodes (1 = all)
    """
    npts = ug.GetNumberOfPoints()
    pre = np.zeros((npts, 3), dtype=float)
    post = np.zeros((npts, 3), dtype=float)
    mag = np.zeros((npts,), dtype=float)

    missing = 0
    for pid in range(npts):
        x, y, z = ug.GetPoint(pid)
        pre[pid] = (x, y, z)
        nid = pid + 1
        d = disp_by_nid.get(nid)
        if d is None:
            missing += 1
            d = np.zeros(3, dtype=float)
        post[pid] = pre[pid] + scale * np.asarray(d, dtype=float)
        mag[pid] = float(np.linalg.norm(d))

    print(f"[DBG] Tet nodes: {npts}, missing disp entries: {missing}")
    print(f"[DBG] Disp magnitude: min={mag.min():.6e} max={mag.max():.6e} mean={mag.mean():.6e}")

    # Choose nodes to export: take largest displacements first (more informative)
    idx = np.argsort(-mag)

    # downsample
    if stride > 1:
        idx = idx[::stride]

    if max_vectors is not None and len(idx) > max_vectors:
        idx = idx[:max_vectors]

    pre_sel = pre[idx]
    post_sel = post[idx]

    write_ply_points(out_prefix + "_nodes_pre.ply", pre_sel)
    write_ply_points(out_prefix + "_nodes_post.ply", post_sel)
    write_ply_vector_lines(out_prefix + "_disp_vectors.ply", pre_sel, post_sel)

    # Extra: print bbox so you can see if “half” is outside / zero
    mn = pre_sel.min(axis=0); mx = pre_sel.max(axis=0)
    print(f"[DBG] Exported vectors: {len(idx)}")
    print(f"[DBG] Export bbox (pre): min={mn}, max={mx}")

def vtk_grid_from_tetgen(tg_or_grid):
    """
    Accepts:
      - tetgen.TetGen object (has .tetrahedralize() or .grid)
      - pyvista.UnstructuredGrid-like (has .points and .cells)

    Returns: vtk.vtkUnstructuredGrid with VTKTETRA cells
    """
    if hasattr(tg_or_grid, "points") and hasattr(tg_or_grid, "cells"):
        grid = tg_or_grid
    elif hasattr(tg_or_grid, "grid") and tg_or_grid.grid is not None:
        grid = tg_or_grid.grid
    elif hasattr(tg_or_grid, "tetrahedralize"):
        grid = tg_or_grid.tetrahedralize()
    else:
        raise TypeError("Expected tetgen.TetGen or pyvista.UnstructuredGrid-like object")

    pts = np.asarray(grid.points, dtype=float)
    cells = np.asarray(grid.cells, dtype=np.int64)

    # Convert PyVista cell buffer: [4, a,b,c,d, 4, a,b,c,d, ...]
    tet_conn = []
    i = 0
    while i < len(cells):
        n = int(cells[i])
        if n != 4:
            raise ValueError(f"Non-tet cell encountered (n={n}). Need pure tetra mesh.")
        a, b, c, d = map(int, cells[i + 1:i + 5])
        tet_conn.append((a, b, c, d))
        i += 1 + n

    # VTK points
    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(numpy_support.numpy_to_vtk(pts, deep=True))

    # VTK cells
    ug = vtk.vtkUnstructuredGrid()
    ug.SetPoints(vtk_pts)

    # Insert tetra cells
    for (a, b, c, d) in tet_conn:
        tet = vtk.vtkTetra()
        tet.GetPointIds().SetId(0, a)
        tet.GetPointIds().SetId(1, b)
        tet.GetPointIds().SetId(2, c)
        tet.GetPointIds().SetId(3, d)
        ug.InsertNextCell(tet.GetCellType(), tet.GetPointIds())

    ug.Modified()      # let VTK know cells/points changed
    ug.BuildLinks()    # optional but useful for some queries
    return ug


def build_cell_locator(ug: vtk.vtkUnstructuredGrid):
    """
    Fast point->containing-cell queries.
    """
    loc = vtk.vtkCellLocator()
    loc.SetDataSet(ug)
    loc.BuildLocator()
    return loc


# =========================
# 2) Read nodal displacements from CalculiX FRD
# =========================

import re
import numpy as np


import re
import numpy as np


import re
import numpy as np

def read_ccx_frd_displacements(frd_path: str):
    """
    Robust CCX FRD DISP reader:
    - finds the LAST '-4  DISP' block
    - parses each '-1' line as: node_id + 3 floats (spacing may be missing)
    """
    float_re = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?")

    in_disp = False
    disp_last = {}

    with open(frd_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            u = line.upper()

            if u.startswith(" -4") and "DISP" in u:
                in_disp = True
                disp_last = {}      # start new block; keep the last one
                continue

            if in_disp and u.strip().startswith("-3"):
                in_disp = False
                continue

            if not in_disp:
                continue

            # data lines begin with -1
            if not line.lstrip().startswith("-1"):
                continue

            # split off the leading "-1" and grab node id + remainder
            # Example: "-1      1562-4.05E-01 2.48E-01-3.22E-01"
            m = re.match(r"^\s*-1\s*(\d+)\s*(.*)$", line)
            if not m:
                continue

            nid = int(m.group(1))
            tail = m.group(2)

            nums = float_re.findall(tail)
            if len(nums) < 3:
                continue

            ux, uy, uz = map(float, nums[:3])
            disp_last[nid] = np.array([ux, uy, uz], dtype=float)

    if not disp_last:
        raise RuntimeError(f"No DISP block parsed from FRD: {frd_path}")

    return disp_last


# =========================
# 3) FEM shape function interpolation (C3D4) using VTK weights
# =========================

def interpolate_displacement_at_point(
    ug: vtk.vtkUnstructuredGrid,
    locator: vtk.vtkCellLocator,
    disp_by_nid: dict,
    p: np.ndarray,
):
    """
    Find containing tet for point p, compute linear-tet weights, return interpolated displacement.

    Returns:
      u (np.ndarray shape (3,)) if inside some tet
      None if not found / outside
    """
    # Find closest cell candidate first
    closest_point = [0.0, 0.0, 0.0]
    cell_id = vtk.reference(0)
    sub_id = vtk.reference(0)
    dist2 = vtk.reference(0.0)

    locator.FindClosestPoint(p.tolist(), closest_point, cell_id, sub_id, dist2)

    cid = int(cell_id)
    if cid < 0:
        return None

    cell = ug.GetCell(cid)  # should be vtkTetra
    if cell.GetNumberOfPoints() != 4:
        return None

    # EvaluatePosition gives us:
    # - inside/outside
    # - param coords (unused)
    # - interpolation weights for the 4 vertices (these are the C3D4 shape functions)
    closest = [0.0, 0.0, 0.0]
    pcoords = [0.0, 0.0, 0.0]
    weights = [0.0] * 4
    dist2_eval = vtk.reference(0.0)

    inside = cell.EvaluatePosition(p.tolist(), closest, sub_id, pcoords, dist2_eval, weights)
    if inside != 1:
        return None

    # Map weights to displacement via node ids.
    # IMPORTANT: your disp dict is 1-based node ids; VTK points are 0-based.
    u = np.zeros(3, dtype=float)
    for local_i in range(4):
        pid0 = int(cell.GetPointId(local_i))        # 0-based
        nid1 = pid0 + 1                             # 1-based (CCX)
        if nid1 not in disp_by_nid:
            # If your FRD doesn't include all nodes, treat missing as zero
            continue
        u += float(weights[local_i]) * disp_by_nid[nid1]

    return u


# =========================
# 4) Read STL, deform vertices, write STL (VTK)
# =========================

def deform_stl_by_tet_field(
    stl_in: str,
    stl_out: str,
    ug: vtk.vtkUnstructuredGrid,
    disp_by_nid: dict,
    scale: float = 1.0,
    outside_mode: str = "keep",  # "keep" or "nearest"
):
    """
    Deform STL vertices by interpolating displacements from tet mesh.

    outside_mode:
      - "keep": if vertex is outside tet mesh -> no displacement
      - "nearest": if outside -> use displacement of closest tet cell (still via closest cell weights, but clamped)
                   (more aggressive; may distort edges)
    """
    locator = build_cell_locator(ug)

    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_in)
    reader.Update()
    poly = reader.GetOutput()

    pts = poly.GetPoints()
    n = pts.GetNumberOfPoints()

    new_pts = vtk.vtkPoints()
    new_pts.SetNumberOfPoints(n)

    missed = 0
    for i in range(n):
        x, y, z = pts.GetPoint(i)
        p = np.array([x, y, z], dtype=float)

        u = interpolate_displacement_at_point(ug, locator, disp_by_nid, p)
        if u is None:
            missed += 1
            if outside_mode == "keep":
                new_pts.SetPoint(i, x, y, z)
                continue
            elif outside_mode == "nearest":
                # fallback: just use closest-point cell weights even if outside
                # (we already used closest cell; re-evaluate weights; if still fails, keep)
                u = np.zeros(3, dtype=float)
                # brute: use closest cell without inside test
                closest_point = [0.0, 0.0, 0.0]
                cell_id = vtk.reference(0)
                sub_id = vtk.reference(0)
                dist2 = vtk.reference(0.0)
                locator.FindClosestPoint(p.tolist(), closest_point, cell_id, sub_id, dist2)
                cid = int(cell_id)
                if cid >= 0:
                    cell = ug.GetCell(cid)
                    # Use the closest point on the cell for stable weights
                    q = np.array(closest_point, dtype=float)
                    closest = [0.0, 0.0, 0.0]
                    pcoords = [0.0, 0.0, 0.0]
                    weights = [0.0] * 4
                    dist2_eval = vtk.reference(0.0)
                    inside2 = cell.EvaluatePosition(q.tolist(), closest, sub_id, pcoords, dist2_eval, weights)
                    if cell.GetNumberOfPoints() == 4:
                        for local_i in range(4):
                            pid0 = int(cell.GetPointId(local_i))
                            nid1 = pid0 + 1
                            if nid1 in disp_by_nid:
                                u += float(weights[local_i]) * disp_by_nid[nid1]
                    new_pts.SetPoint(i, *(p + scale * u))
                    continue

                new_pts.SetPoint(i, x, y, z)
                continue
            else:
                raise ValueError("outside_mode must be 'keep' or 'nearest'")

        new_pts.SetPoint(i, *(p + scale * u))

    out_poly = vtk.vtkPolyData()
    out_poly.ShallowCopy(poly)
    out_poly.SetPoints(new_pts)

    writer = vtk.vtkSTLWriter()
    writer.SetFileName(stl_out)
    writer.SetInputData(out_poly)
    writer.SetFileTypeToBinary()
    writer.Write()

    print(f"[DEFORM] STL verts={n}, missed(outside)={missed} ({missed/max(1,n)*100:.2f}%)")
    print(f"[DEFORM] wrote: {stl_out}")


# =========================
# 5) Convenience wrapper: from TetGen + FRD + STL -> deformed STL
# =========================

def apply_ccx_deformation_to_stl(
    tetgen_or_grid,
    frd_path: str,
    stl_in: str,
    stl_out: str,
    scale: float = 1.0,
    outside_mode: str = "keep",
):
    ug = vtk_grid_from_tetgen(tetgen_or_grid)
    disp = read_ccx_frd_displacements(frd_path)
    print("[CHK] ug points:", ug.GetNumberOfPoints())
    print("[CHK] disp entries:", len(disp))
    print("[CHK] disp id range:", min(disp), max(disp))

    # how many nodes are missing?
    missing = 0
    for pid in range(ug.GetNumberOfPoints()):
        if (pid+1) not in disp:
            missing += 1
    print("[CHK] missing disp for nodes:", missing)
    # DEBUG: export tet node displacement vectors
    export_tet_displacement_debug(
        ug,
        disp,
        out_prefix="DEBUG/tet",
        scale=scale,
        max_vectors=80000,  # adjust if needed
        stride=1,
    )

    deform_stl_by_tet_field(stl_in, stl_out, ug, disp, scale=scale, outside_mode=outside_mode)


apply_ccx_deformation_to_stl(
    tetgen_or_grid=t,                # or grid
    frd_path="OUTPUT/shrink_test.frd",
    stl_in="MODELS/CSC16_U00P_.stl",
    stl_out="original_high_detail_deformed.stl",
    scale=1.0,
    outside_mode="keep",
)

from typing import Any, List, Dict, Set, Optional
from typing import List, Any
from vtk.util import numpy_support
import math
import numpy as np
import os
import point_cloud_utils as pcu
import pymeshfix as pfix
import re
import subprocess
import tetgen
import trimesh
import vtk
import pymeshlab as ml
import triangle as tr

import numpy as np

# scipy
from scipy.spatial import cKDTree

# trimesh core
import trimesh
from trimesh.base import Trimesh

# trimesh helpers used internally
from trimesh.intersections import slice_faces_plane
from trimesh.creation import triangulate_polygon
from trimesh.path import polygons
from trimesh.visual import TextureVisuals

# trimesh math / utils
from trimesh import grouping, geometry, util, transformations as tf


WATERTIGHT_RESOLUTION = 50_000
DECIMATE_NUM_FACES = 50_000
CYLINDER_RADIUS = 199.82
LAYER_HEIGHT = 5
SHRINKAGE = 0.2
SHRINKAGE_CURVE = [5,4,3,2,1]
INPUT_STL = "MODELS/CSC16_U00P_.stl"
OUTPUT_DIR = "OUTPUT"
SIMULATION = f"{OUTPUT_DIR}/simulation"


def pymeshlab_decimate(
    v: np.ndarray,
    f: np.ndarray,
    target_faces: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quadric edge-collapse decimation using PyMeshLab.

    target_faces: desired triangle count (approx).
    Returns (v_out, f_out).
    """
    v = np.asarray(v, dtype=np.float64)
    f = np.asarray(f, dtype=np.int64)

    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(v, f), "watertight")

    # Main decimation
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=int(target_faces),
        preserveboundary=True,
        boundaryweight=1.0,
        preservenormal=True,
        preservetopology=True,
        optimalplacement=True,
        planarquadric=True,
        autoclean=True,
    )

    m2 = ms.current_mesh()
    v2 = m2.vertex_matrix().astype(np.float64)
    f2 = m2.face_matrix().astype(np.int64)
    return v2, f2


def find_bottom_contact_point(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Find the 'bottom' contact point as you used before:
    the vertex with minimum Z in WORLD space.

    Returns: np.ndarray shape (3,) = (x_low, y_low, z_low)
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)
    if V.size == 0:
        raise ValueError("Mesh has no vertices.")
    i0 = int(np.argmin(V[:, 2]))
    return V[i0].copy()


def transform_mesh_to_cylindrical_like_old(
    mesh: trimesh.Trimesh,
    cyl_radius: float,
    bottom_point_world: np.ndarray,
) -> trimesh.Trimesh:
    """
    Same principle as your old snippet, BUT the bottom contact point is provided manually.

    - bottom_point_world = (x_low, y_low, z_low) in WORLD coords
    - cx = x_low
    - cz = z_low +/- R, choose the one closer to bbox z-center
    - axis is +Y (so we unwrap in XZ plane)
    - mapping:
        u = R * wrap(theta - theta0)   (arc length)
        v = y
        w = R - r    (depth inward, clamped >= 0)
    - seam fix: if mesh crosses u=0/2πR, shift high-u half back by 2πR
    """
    if cyl_radius <= 0:
        raise ValueError("cyl_radius must be > 0")

    Vw = np.asarray(mesh.vertices, dtype=np.float64)
    if Vw.size == 0:
        raise ValueError("Mesh has no vertices.")

    bp = np.asarray(bottom_point_world, dtype=np.float64).reshape(3)
    x_low, y_low, z_low = map(float, bp)

    # bbox z center (same as your old code)
    z_min_w = float(np.min(Vw[:, 2]))
    z_max_w = float(np.max(Vw[:, 2]))
    z_center_bbox = 0.5 * (z_min_w + z_max_w)

    cx_cyl = float(x_low)
    R0 = float(cyl_radius)

    cz_plus = z_low + R0
    cz_minus = z_low - R0
    cz_cyl = cz_plus if abs(cz_plus - z_center_bbox) < abs(cz_minus - z_center_bbox) else cz_minus

    # theta0 so that the provided bottom point maps to u=0
    theta0 = math.atan2(z_low - cz_cyl, x_low - cx_cyl)

    # vectorized mapping for all vertices
    x = Vw[:, 0]
    y = Vw[:, 1]
    z = Vw[:, 2]

    dx = x - cx_cyl
    dz = z - cz_cyl

    theta = np.arctan2(dz, dx)
    dtheta = (theta - theta0) % (2.0 * math.pi)

    u = dtheta * R0
    v = y
    r = np.sqrt(dx * dx + dz * dz)
    w = R0 - r
    w = np.maximum(w, 0.0)

    Vp = np.column_stack([u, v, w]).astype(np.float64)

    # ---- seam fix (global) ----
    tw = 2.0 * math.pi * R0
    uu = Vp[:, 0]
    if (uu.max() - uu.min()) > 0.5 * tw:
        uu = np.where(uu > 0.5 * tw, uu - tw, uu)
        Vp[:, 0] = uu

    print(f"[CYL-OLD] bottom={bp}, cx={cx_cyl:.6f}, cz={cz_cyl:.6f}, R0={R0:.6f}, theta0={theta0:.6f}")
    return trimesh.Trimesh(vertices=Vp, faces=mesh.faces.copy(), process=False), (cx_cyl, cz_cyl, R0, theta0)


def transform_mesh_from_cylindrical_like_old(
    cyl_mesh: trimesh.Trimesh,
    cx_cyl: float,
    cz_cyl: float,
    R0: float,
    theta0: float,
) -> trimesh.Trimesh:
    """
    Reverse of the old-style cylindrical param mapping.

    Input mesh vertices are in (u, v, w):
      u = arc length around cylinder (may be seam-shifted negative)
      v = y (axis coordinate)
      w = inward depth from surface (>=0)

    Inverse mapping:
      theta = theta0 + (u / R0)
      r = R0 - w
      x = cx + r*cos(theta)
      z = cz + r*sin(theta)
      y = v

    Returns a new trimesh.Trimesh in WORLD space.
    """
    Vp = np.asarray(cyl_mesh.vertices, dtype=np.float64)
    if Vp.size == 0:
        raise ValueError("Mesh has no vertices.")
    if R0 <= 0:
        raise ValueError("R0 must be > 0")

    u = Vp[:, 0].astype(np.float64)
    v = Vp[:, 1].astype(np.float64)
    w = Vp[:, 2].astype(np.float64)

    # If seam-fix shifted some u negative, normalize back to [0, 2πR0)
    tw = 2.0 * math.pi * float(R0)
    u_norm = np.mod(u, tw)

    theta = float(theta0) + (u_norm / float(R0))
    r = float(R0) - w

    x = float(cx_cyl) + r * np.cos(theta)
    z = float(cz_cyl) + r * np.sin(theta)
    y = v

    Vw = np.column_stack([x, y, z]).astype(np.float64)
    return trimesh.Trimesh(vertices=Vw, faces=cyl_mesh.faces.copy(), process=False)


def compute_cylinder_center_from_bottom_z(pts: np.ndarray, radius: float):
    """
    Assumptions (your clarified convention):
      - model -Z is bottom and touches cylinder surface
      - model +Z points inward toward cylinder axis
      - cylinder axis is global +Y (so center is in XZ)

    We take the point with minimum Z as the contact point on the cylinder.
    Then center is directly +Z from it by radius.
    """
    if radius <= 0:
        raise ValueError("cyl_radius must be > 0")

    i0 = int(np.argmin(pts[:, 2]))  # min Z
    x0 = float(pts[i0, 0])
    z0 = float(pts[i0, 2])

    cx = x0
    cz = z0 + float(radius)

    # reference angle at the contact point (start of printing)
    theta0 = math.atan2(z0 - cz, x0 - cx)  # usually ~ -pi/2
    return cx, cz, theta0


def write_calculix_job_tet_layer_binned(
    path: str,
    tg_or_grid: Any,
    layer_height: float,     # radial thickness
    cyl_radius: float,       # cylinder radius
    shrinkage_curve: List[float],
    cure_shrink_per_unit: float,
    output_stride: int = 1,
):
    """
    C3D4 layered curing job using simple height binning after tet meshing.

    - Creates layers by grouping elements based on tet centroid Y coordinate.
    - Bottom layer is fixed (Ux,Uy,Uz = 0) via SLICE_000_NODES.
    - Applies shrinkage curve by ramping "cure" via TEMP (DOF 11) per layer NSET.
    - Uses MODEL CHANGE to add layers over time, like your previous implementation.
    """

    def log(msg: str) -> None:
        print(msg)

    def _fmt(val: float) -> str:
        return f"{float(val):.12e}"

    if layer_height <= 0:
        raise ValueError("layer_height must be > 0")

    # ---- extract pyvista grid ----
    grid = tg_or_grid.grid

    pts = np.asarray(grid.points, dtype=float)  # (N,3)
    n_nodes = int(pts.shape[0])

    # ---- parse tets from pyvista cell buffer [4,a,b,c,d, 4,a,b,c,d, ...] ----
    cells = np.asarray(grid.cells, dtype=np.int64)
    tets0 = []  # 0-based connectivity
    i = 0
    while i < len(cells):
        n = int(cells[i])
        if n != 4:
            raise ValueError(f"Non-tet cell encountered (n={n}). Need pure tetra mesh.")
        a, b, c, d = map(int, cells[i + 1:i + 5])
        tets0.append((a, b, c, d))
        i += 1 + n

    n_elems = len(tets0)
    if n_elems == 0:
        raise ValueError("No tetrahedra found.")

    # ---- normalize shrinkage curve ----
    if not shrinkage_curve:
        shrinkage_curve = [1.0]
    total_w = float(sum(shrinkage_curve))
    if total_w <= 0.0:
        raise ValueError("shrinkage_curve must have positive sum.")
    shrinkage_curve = [float(w) / total_w for w in shrinkage_curve]
    curve_len = len(shrinkage_curve)

    if output_stride < 1:
        output_stride = 1

    # ---- CYLINDRICAL RADIAL slicing (axis = Y) ----
    if layer_height <= 0:
        raise ValueError("layer_height must be > 0")

    # find cylinder center from bottom-most Z
    i0 = int(np.argmin(pts[:, 2]))   # min Z
    x0 = float(pts[i0, 0])
    z0 = float(pts[i0, 2])

    cx = x0
    cz = z0 + float(cyl_radius)

    slice_to_eids: Dict[int, List[int]] = {}
    slice_node_ids: Dict[int, Set[int]] = {}

    for eid1, (a, b, c, d) in enumerate(tets0, start=1):
        # centroid in XZ plane
        xc = (pts[a, 0] + pts[b, 0] + pts[c, 0] + pts[d, 0]) * 0.25
        zc = (pts[a, 2] + pts[b, 2] + pts[c, 2] + pts[d, 2]) * 0.25

        r = math.sqrt((xc - cx)**2 + (zc - cz)**2)

        # radial penetration from cylinder surface inward
        depth = cyl_radius - r

        # ignore elements outside cylinder (numerical safety)
        if depth < 0:
            depth = 0.0

        sidx = int(math.floor(depth / layer_height))
        if sidx < 0:
            sidx = 0

        slice_to_eids.setdefault(sidx, []).append(eid1)
        ns = slice_node_ids.setdefault(sidx, set())
        ns.update([a + 1, b + 1, c + 1, d + 1])

    print(
        f"[CYL-RADIAL] cx={cx:.6f}, cz={cz:.6f}, "
        f"R={cyl_radius:.6f}, layers={len(slice_to_eids)}"
    )

    existing_slice_idxs = sorted(slice_to_eids.keys())
    n_slices = len(existing_slice_idxs)

    # bottom layer nodes for fixation
    bottom_idx = existing_slice_idxs[0] if existing_slice_idxs else 0
    base_nodes = sorted(slice_node_ids.get(bottom_idx, set()))

    def _write_id_list_lines(fh, ids: List[int], per_line: int = 16):
        for k in range(0, len(ids), per_line):
            chunk = ids[k:k + per_line]
            fh.write(", ".join(str(x) for x in chunk) + "\n")

    # ---- write INP ----
    time_per_layer = 1.0
    time_per_layer_step = 1.0

    with open(path, "w", encoding="utf-8") as f:
        f.write("**\n** Auto-generated incremental-cure shrink job (C3D4 tets, binned layers)\n**\n")
        f.write("*HEADING\n")
        f.write("Tet C3D4 UT-D (MODEL CHANGE + shrinkage curve), layers from CYL-ANGLE binning\n")

        # NODES
        f.write("** Nodes +++++++++++++++++++++++++++++++++++++++++++++++++++\n")
        f.write("*NODE\n")
        for nid, (x, y, z) in enumerate(pts, start=1):
            if not (math.isfinite(float(x)) and math.isfinite(float(y)) and math.isfinite(float(z))):
                raise ValueError(f"Non-finite node coord at node {nid}: {(x,y,z)}")
            f.write(f"{nid}, {_fmt(x)}, {_fmt(y)}, {_fmt(z)}\n")

        # ELEMENTS
        f.write("** Elements ++++++++++++++++++++++++++++++++++++++++++++++++\n")
        f.write("*ELEMENT, TYPE=C3D4, ELSET=ALL\n")
        for eid1, (a, b, c, d) in enumerate(tets0, start=1):
            f.write(f"{eid1}, {a+1}, {b+1}, {c+1}, {d+1}\n")

        # SETS
        f.write("** Node sets +++++++++++++++++++++++++++++++++++++++++++++++\n")
        f.write("*NSET, NSET=ALLNODES, GENERATE\n")
        f.write(f"1, {n_nodes}, 1\n")

        f.write("** Element + node sets (per layer bin) ++++++++++++++++++++++\n")
        for sidx in existing_slice_idxs:
            name = f"SLICE_{sidx:03d}"
            eids = slice_to_eids[sidx]
            nodes = sorted(slice_node_ids[sidx])

            f.write(f"*ELSET, ELSET={name}\n")
            _write_id_list_lines(f, eids)

            f.write(f"*NSET, NSET={name}_NODES\n")
            _write_id_list_lines(f, nodes)

        if base_nodes:
            f.write("** Base node set = bottom binned layer ++++++++++++++++++++++\n")
            f.write("*NSET, NSET=BASE\n")
            _write_id_list_lines(f, base_nodes)
        else:
            f.write("** Warning: BASE set empty (no bottom layer?)\n")

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

        # needed for UT-D transient thermal part
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
        if n_slices == 0:
            f.write("** No slices -> no steps.\n")
        else:
            # per-layer cure state
            cure_state: Dict[int, float] = {idx: 0.0 for idx in existing_slice_idxs}
            applied_count: Dict[int, int] = {idx: 0 for idx in existing_slice_idxs}
            printed: Dict[int, bool] = {idx: False for idx in existing_slice_idxs}

            step_counter = 1

            # Step 1: dummy full model (then we will remove all but first layer at step 2)
            f.write("** --------------------------------------------------------\n")
            f.write("** Step 1: initial dummy step with full model (no curing)\n")
            f.write("** --------------------------------------------------------\n")
            f.write("*STEP\n")
            f.write("*UNCOUPLED TEMPERATURE-DISPLACEMENT, SOLVER=PASTIX\n")
            f.write(f"{time_per_layer_step:.6f}, {time_per_layer:.6f}\n")
            if base_nodes:
                f.write("*BOUNDARY\n")
                f.write("BASE, 1, 3, 0.\n")
            f.write("*END STEP\n")
            step_counter += 1

            total_cure_steps = len(existing_slice_idxs) + curve_len - 1

            for global_k in range(total_cure_steps):
                slice_to_add: Optional[int] = None
                if global_k < len(existing_slice_idxs):
                    slice_to_add = existing_slice_idxs[global_k]
                    printed[slice_to_add] = True

                prev_cure_state = cure_state.copy()

                # advance curve for all printed slices
                for j in existing_slice_idxs:
                    if not printed[j]:
                        continue
                    k_applied = applied_count[j]
                    if k_applied >= curve_len:
                        continue
                    inc = shrinkage_curve[k_applied]
                    if inc != 0.0:
                        cure_state[j] = min(1.0, cure_state[j] + inc)
                    applied_count[j] = k_applied + 1

                f.write("** --------------------------------------------------------\n")
                if slice_to_add is not None:
                    f.write(f"** Step {step_counter}: add layer SLICE_{slice_to_add:03d} and advance shrink curve\n")
                else:
                    f.write(f"** Step {step_counter}: post-cure step (advance shrink curve)\n")
                f.write("** --------------------------------------------------------\n")
                f.write("*STEP\n")
                f.write("*UNCOUPLED TEMPERATURE-DISPLACEMENT, SOLVER=PASTIX\n")
                f.write(f"{time_per_layer_step:.6f}, {time_per_layer:.6f}\n")

                # model change logic
                if slice_to_add is not None:
                    name = f"SLICE_{slice_to_add:03d}"
                    if slice_to_add == existing_slice_idxs[0]:
                        # keep only first slice active
                        remove = [f"SLICE_{other:03d}" for other in existing_slice_idxs if other != slice_to_add]
                        if remove:
                            f.write("*MODEL CHANGE, TYPE=ELEMENT, REMOVE\n")
                            for nm in remove:
                                f.write(f"{nm}\n")
                    else:
                        f.write("*MODEL CHANGE, TYPE=ELEMENT, ADD\n")
                        f.write(f"{name}\n")

                # outputs
                write_outputs = (
                    output_stride <= 1
                    or (global_k + 1) % output_stride == 0
                    or global_k == total_cure_steps - 1
                )
                f.write("*NODE FILE\n")
                if write_outputs:
                    f.write("U\n")

                # boundary + cure application
                f.write("*BOUNDARY\n")

                for j in existing_slice_idxs:
                    if not printed[j]:
                        continue
                    cure_val = cure_state[j]
                    if cure_val == 0.0:
                        continue
                    if cure_val == prev_cure_state.get(j, 0.0):
                        continue
                    f.write(f"SLICE_{j:03d}_NODES, 11, 11, {cure_val:.6f}\n")

                f.write("*END STEP\n")
                step_counter += 1

    log(f"[CCX] Wrote binned-layer UT-D tet job to: {path}")
    log(f"[CCX] Nodes: {n_nodes}, elems: {n_elems}, layers: {n_slices}, layer_height={layer_height}")
    log(f"[CCX] Bottom layer index: {bottom_idx}, BASE nodes: {len(base_nodes)}")


def run_calculix(job_name: str, ccx_cmd: str = "ccx"):
    print(f"[RUN] Launching CalculiX: {ccx_cmd} {job_name}")

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
        print(f"[RUN] ERROR: CalculiX command not found: {ccx_cmd}")
        return False

    rc = proc.wait()
    print(f"[RUN] CalculiX completed with return code {rc}")
    print(f"[RUN] Full output written to: {log_path}")

    return rc == 0


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
    stride: int = 1,
):
    """
    Exports PLY debug artifacts for tet nodal displacement field.

    - Uses ug points as pre positions.
    - Uses disp_by_nid[nid] where nid = pid+1 (CCX 1-based).
    - Writes vector lines (pre->post) for a subset of nodes.

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
    grid = tg_or_grid.grid

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

    print("[CHK] ug points:", vtk_grid.GetNumberOfPoints())
    print("[CHK] disp entries:", len(disp_last))
    print("[CHK] disp id range:", min(disp_last), max(disp_last))
    # how many nodes are missing?
    missing = 0
    for pid in range(vtk_grid.GetNumberOfPoints()):
        if (pid+1) not in disp_last:
            missing += 1
    print("[CHK] missing disp for nodes:", missing)

    return disp_last


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


def slice_mesh_plane(
    mesh,
    plane_normal,
    plane_origin,
    face_index=None,
    cap=False,
    engine=None,
    triangle_args=None,
    **kwargs,
):
    """
    Slice a mesh with a plane returning a new mesh that is the
    portion of the original mesh to the positive normal side
    of the plane.

    Parameters
    ---------
    mesh : Trimesh object
      Source mesh to slice
    plane_normal : (3,) float
      Normal vector of plane to intersect with mesh
    plane_origin :  (3,) float
      Point on plane to intersect with mesh
    cap : bool
      If True, cap the result with a triangulated polygon
    face_index : ((m,) int)
      Indexes of mesh.faces to slice. When no mask is provided, the
      default is to slice all faces.
    cached_dots : (n, 3) float
      If an external function has stored dot
      products pass them here to avoid recomputing
    engine : None or str
      Triangulation engine passed to `triangulate_polygon`
    kwargs : dict
      Passed to the newly created sliced mesh

    Returns
    ----------
    new_mesh : Trimesh object
      Sliced mesh
    """
    # check input for none
    if mesh is None:
        return None

    # avoid circular import
    from scipy.spatial import cKDTree

    # check input plane
    plane_normal = np.asanyarray(plane_normal, dtype=np.float64)
    plane_origin = np.asanyarray(plane_origin, dtype=np.float64)

    # check to make sure origins and normals have acceptable shape
    shape_ok = (
        (plane_origin.shape == (3,) or util.is_shape(plane_origin, (-1, 3)))
        and (plane_normal.shape == (3,) or util.is_shape(plane_normal, (-1, 3)))
        and plane_origin.shape == plane_normal.shape
    )
    if not shape_ok:
        raise ValueError("plane origins and normals must be (n, 3)!")

    # start with copy of original mesh, faces, and vertices
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()

    # We copy the UV coordinates if available
    has_uv = (
        hasattr(mesh.visual, "uv") and np.shape(mesh.visual.uv) == (len(mesh.vertices), 2)
    ) and not cap
    uv = mesh.visual.uv.copy() if has_uv else None

    if "process" not in kwargs:
        kwargs["process"] = False

    # slice away specified planes
    for origin, normal in zip(
        plane_origin.reshape((-1, 3)), plane_normal.reshape((-1, 3))
    ):
        # save the new vertices and faces
        vertices, faces, uv = slice_faces_plane(
            vertices=vertices,
            faces=faces,
            uv=uv,
            plane_normal=normal,
            plane_origin=origin,
            face_index=face_index,
        )
        # check if cap arg specified
        if cap:
            if face_index:
                # This hasn't been implemented yet.
                raise NotImplementedError("face_index and cap can't be used together")

            # start by deduplicating vertices again
            unique, inverse = grouping.unique_rows(vertices)
            vertices = vertices[unique]
            # will collect additional faces
            f = inverse[faces]
            # remove degenerate faces by checking to make sure
            # that each face has three unique indices
            f = f[(f[:, :1] != f[:, 1:]).all(axis=1)]
            # transform to the cap plane
            to_2D = geometry.plane_transform(origin=origin, normal=-normal)
            to_3D = np.linalg.inv(to_2D)

            vertices_2D = tf.transform_points(vertices, to_2D)
            edges = geometry.faces_to_edges(f)
            edges.sort(axis=1)

            on_plane = np.abs(vertices_2D[:, 2]) < 1e-8
            edges = edges[on_plane[edges].all(axis=1)]
            edges = edges[edges[:, 0] != edges[:, 1]]

            unique_edge = grouping.group_rows(edges, require_count=1)
            if len(unique) < 3:
                continue

            # collect new faces
            faces = [f]
            for p in polygons.edges_to_polygons(edges[unique_edge], vertices_2D[:, :2]):
                # triangulate cap and raise an error if any new vertices were inserted
                vn, fn = triangulate_polygon(p, engine=engine, triangle_args=triangle_args)
                # collect the original index for the new vertices
                vn3 = tf.transform_points(util.stack_3D(vn), to_3D)

                # Append new vertices to mesh (ALL vn3 returned, including boundary).
                base = len(vertices)
                vertices = np.vstack([vertices, vn3])

                if uv is not None:
                    # Cap vertices have no meaningful UV; fill zeros (or change as needed)
                    uv = np.vstack([uv, np.zeros((len(vn3), 2), dtype=uv.dtype)])

                nf = fn + base

                nf_ok = (nf[:, 1:] != nf[:, :1]).all(axis=1) & (nf[:, 1] != nf[:, 2])
                faces.append(nf[nf_ok])

            faces = np.vstack(faces)

    visual = (
        TextureVisuals(uv=uv, material=mesh.visual.material.copy()) if has_uv else None
    )

    # return the sliced mesh
    return Trimesh(vertices=vertices, faces=faces, visual=visual, **kwargs)


def slice_mesh_into_z_slabs_by_height(
    mesh: trimesh.Trimesh,
    layer_height: float,
) -> list[trimesh.Trimesh]:
    """
    Slice mesh into slabs along Z using trimesh.intersections.slice_mesh_plane.

    - Slabs are [z0, z0+H], [z0+H, z0+2H], ... where z0 = min Z of mesh bounds.
    - Returns OPEN meshes (not capped). We'll cap later.

    Returns: list of trimesh.Trimesh
    """
    if layer_height <= 0:
        raise ValueError("layer_height must be > 0")

    bounds = np.asarray(mesh.bounds, dtype=float)
    z0 = float(bounds[0, 2])  # min Z
    z1 = float(bounds[1, 2])  # max Z
    if not np.isfinite(z0) or not np.isfinite(z1) or z1 <= z0:
        raise ValueError(f"Bad Z bounds: zmin={z0}, zmax={z1}")

    H = float(layer_height)

    # planes are z=const
    n_pos = np.array([0.0, 0.0, 1.0], dtype=float)   # keep z >= plane
    n_neg = np.array([0.0, 0.0, -1.0], dtype=float)  # keep z <= plane

    n_layers = int(np.ceil((z1 - z0) / H))
    slabs: list[trimesh.Trimesh] = []

    for i in range(n_layers):
        a0 = z0 + i * H
        a1 = min(z1, z0 + (i + 1) * H)

        origin0 = mesh.centroid.copy()
        origin0[2] = float(a0)

        origin1 = mesh.centroid.copy()
        origin1[2] = float(a1)

        # keep z >= a0
        m1 = slice_mesh_plane(
            mesh,
            plane_normal=n_pos,
            plane_origin=origin0,
            cap=True,
            engine="triangle",
            triangle_args="pq15",
        )
        if m1 is None or len(m1.faces) == 0:
            continue

        # keep z <= a1
        m2 = slice_mesh_plane(
            m1,
            plane_normal=n_neg,
            plane_origin=origin1,
            cap=False,
        )
        if m2 is None or len(m2.faces) == 0:
            continue

        slabs.append(m2)

    return slabs


# Input
input_mesh = trimesh.load(INPUT_STL)
contact_point = find_bottom_contact_point(input_mesh)

vw, fw = pcu.make_mesh_watertight(
    input_mesh.vertices.astype(np.float64),
    input_mesh.faces.astype(np.int64),
    resolution=WATERTIGHT_RESOLUTION
)

# Water
water_mesh = trimesh.Trimesh(
    vertices=vw,
    faces=fw,
    process=False
)
water_mesh.export(f"{OUTPUT_DIR}/water_mesh.stl")

# Decimate
vd, fd = pymeshlab_decimate(
    water_mesh.vertices,
    water_mesh.faces,
    target_faces=DECIMATE_NUM_FACES
)

decimated_mesh = trimesh.Trimesh(
    vertices=vd,
    faces=fd,
    process=False
)
decimated_mesh.export(f"{OUTPUT_DIR}/decimated_mesh.stl")

# Cylinder
cylinder_mesh, (cx, cz, R0, theta0) = transform_mesh_to_cylindrical_like_old(
    decimated_mesh,
    CYLINDER_RADIUS,
    contact_point,
)
cylinder_mesh.export(f"{OUTPUT_DIR}/cyl_mesh.stl")

# Repair
repaired_mesh = pfix.MeshFix(
    cylinder_mesh.vertices,
    cylinder_mesh.faces
)
repaired_mesh.repair()

repaired_mesh = trimesh.Trimesh(
    vertices=repaired_mesh.v,
    faces=repaired_mesh.f,
    process=False
)
repaired_mesh.export(f"{OUTPUT_DIR}/repaired_mesh.stl")

# Slice
slabs = slice_mesh_into_z_slabs_by_height(
    repaired_mesh,
    layer_height=LAYER_HEIGHT
)

for i, slab in enumerate(slabs):
    slab.export(f"{OUTPUT_DIR}/slices/slice_{i:03d}.stl")

# world_mesh = transform_mesh_from_cylindrical_like_old(cylinder_mesh, cx, cz, R0, theta0)
# world_mesh.export(f"{OUTPUT_DIR}/world_mesh.stl")

# # Tetrahedralize
# tetgen_mesh = tetgen.TetGen(repaired_mesh.vertices, repaired_mesh.faces)
# tetgen_mesh.tetrahedralize(
#     order=1,        # linear tets (C3D4)
#     quality=True,  # DO NOT enforce radius-edge ratio
#     # mindihedral=0,  # disable angle constraints
#     steinerleft=-1,  # allow NO Steiner points
#     verbose=1
# )

# write_calculix_job_tet_layer_binned(
#     path=f"{SIMULATION}.inp",
#     tg_or_grid=tetgen_mesh,
#     layer_height=LAYER_HEIGHT,
#     cyl_radius=CYLINDER_RADIUS,
#     shrinkage_curve=SHRINKAGE_CURVE,
#     cure_shrink_per_unit=SHRINKAGE,
# )

# run_calculix(
#     SIMULATION,
#     "C:/Users/4y5t6/Downloads/PrePoMax v2.4.0/Solver/ccx_dynamic.exe"
# )

# vtk_grid = vtk_grid_from_tetgen(tetgen_mesh)
# displacements = read_ccx_frd_displacements(f"{SIMULATION}.frd")

# export_tet_displacement_debug(
#     vtk_grid,
#     displacements,
#     out_prefix="DEBUG/tet",
#     scale=1.0,
#     stride=1,
# )

# deform_stl_by_tet_field(
#     INPUT_STL,
#     f"{OUTPUT_DIR}/deformed_stl.stl",
#     vtk_grid,
#     displacements,
#     scale=1.0,
#     outside_mode="keep"
# )

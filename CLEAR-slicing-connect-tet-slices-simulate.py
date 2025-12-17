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

import numpy as np

from trimesh import Trimesh
from trimesh.visual.texture import TextureVisuals

from trimesh import util
from trimesh import geometry
from trimesh import grouping
from trimesh.path import polygons
from trimesh import transformations as tf

from trimesh.creation import triangulate_polygon
from trimesh.intersections import slice_faces_plane


WATERTIGHT_RESOLUTION = 50_000
DECIMATE_NUM_FACES = 50_000
CYLINDER_RADIUS = 199.82
LAYER_HEIGHT = 2 # 1.1
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
    grid_points: Any,
    grid_cells: Any,
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

    pts = np.asarray(grid_points, dtype=float)  # (N,3)
    n_nodes = int(pts.shape[0])

    # ---- parse tets from pyvista cell buffer [4,a,b,c,d, 4,a,b,c,d, ...] ----
    cells = np.asarray(grid_cells, dtype=np.int64)
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


def vtk_grid_from_tetgen(grid_points, grid_cells) -> vtk.vtkUnstructuredGrid:
    """
    Accepts:
      - tetgen.TetGen object (has .tetrahedralize() or .grid)
      - pyvista.UnstructuredGrid-like (has .points and .cells)

    Returns: vtk.vtkUnstructuredGrid with VTKTETRA cells
    """
    pts = np.asarray(grid_points, dtype=float)
    cells = np.asarray(grid_cells, dtype=np.int64)

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


def add_verts_faces_dedup(
    mesh: trimesh.Trimesh,
    new_verts: np.ndarray[np.float64],
    new_faces: np.ndarray[np.int64],
    *,
    tol: float = 1e-8,
    rebuild: bool = True,
) -> trimesh.Trimesh:
    """
    Add (new_verts, new_faces) into an existing mesh, reusing vertices already
    present in mesh (within `tol`) and remapping faces accordingly.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Existing mesh.
    new_verts : (M,3) float64
        Vertices to add.
    new_faces : (K,3) int64
        Faces referencing indices in new_verts.
    tol : float
        Quantization tolerance for vertex matching (world units).
    rebuild : bool
        If True, runs process/re-zero checks on the returned mesh.

    Returns
    -------
    out : trimesh.Trimesh
        New mesh with faces added and vertices reused when possible.
    """
    if new_verts.size == 0 or new_faces.size == 0:
        return mesh.copy()

    v_old = np.asarray(mesh.vertices, dtype=np.float64)
    f_old = np.asarray(mesh.faces, dtype=np.int64)

    new_verts = np.asarray(new_verts, dtype=np.float64)
    new_faces = np.asarray(new_faces, dtype=np.int64)

    if new_verts.ndim != 2 or new_verts.shape[1] != 3:
        raise ValueError("new_verts must be (M,3)")
    if new_faces.ndim != 2 or new_faces.shape[1] != 3:
        raise ValueError("new_faces must be (K,3)")
    if new_faces.min() < 0 or new_faces.max() >= len(new_verts):
        raise ValueError("new_faces indices out of range for new_verts")

    # --- quantize to an integer grid so we can match by exact keys ---
    inv = 1.0 / float(tol)

    def qkey(v: np.ndarray[np.float64]) -> np.ndarray[np.int64]:
        # round to nearest multiple of tol
        return np.round(v * inv).astype(np.int64)

    q_old = qkey(v_old)
    q_new = qkey(new_verts)

    # map quantized coordinate -> index in old mesh
    # note: if old mesh already has duplicates within tol, we keep the first one
    old_map: dict[tuple[int, int, int], int] = {}
    for i, k in enumerate(map(tuple, q_old)):
        old_map.setdefault(k, i)

    # build remap: new vertex idx -> final vertex idx (old reused or appended)
    remap = np.empty(len(new_verts), dtype=np.int64)

    appended = []          # list of actual vertices to append
    appended_keys = []     # quantized keys for the ones we append
    appended_map: dict[tuple[int, int, int], int] = {}  # key -> appended local idx

    base = len(v_old)

    for i, k in enumerate(map(tuple, q_new)):
        if k in old_map:
            remap[i] = old_map[k]
            continue

        # if multiple new verts collapse to same key, reuse within the new batch too
        if k in appended_map:
            remap[i] = base + appended_map[k]
            continue

        appended_map[k] = len(appended)
        remap[i] = base + len(appended)
        appended.append(new_verts[i])
        appended_keys.append(k)

    if len(appended) == 0:
        v_out = v_old
    else:
        v_out = np.vstack([v_old, np.asarray(appended, dtype=np.float64)])

        # also extend old_map so future calls could reuse these (optional, but nice)
        for local_idx, k in enumerate(appended_keys):
            old_map[k] = base + local_idx

    f_new_remapped = remap[new_faces]
    f_out = np.vstack([f_old, f_new_remapped])

    out = trimesh.Trimesh(vertices=v_out, faces=f_out, process=rebuild)
    return out


def weld_vertices_and_remap_faces(verts: np.ndarray, faces: np.ndarray, tol: float = 1e-8):
    """
    Deduplicate vertices within tol and remap faces to the new vertex indices.
    Returns (verts_welded, faces_remapped).
    """
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    inv = 1.0 / float(tol)
    keys = np.round(verts * inv).astype(np.int64)

    # unique keys -> new vertex list
    uniq_keys, uniq_idx, inv_map = np.unique(keys, axis=0, return_index=True, return_inverse=True)

    verts_welded = verts[uniq_idx]
    faces_remapped = inv_map[faces]

    # validate
    if faces_remapped.min() < 0 or faces_remapped.max() >= len(verts_welded):
        raise ValueError("Remap failed: faces out of range after welding")

    return verts_welded, faces_remapped


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

            tree = cKDTree(vertices)
            # collect new faces
            faces = [f]
            tol = 1e-8

            for p in polygons.edges_to_polygons(edges[unique_edge], vertices_2D[:, :2]):

                # triangulate cap polygon (vn can include inserted vertices)
                vn, fn = triangulate_polygon(p, engine=engine, force_vertices=False, triangle_args=triangle_args)

                # vn is 2D -> lift to 3D cap plane, then to world 3D
                vn3 = tf.transform_points(util.stack_3D(vn), to_3D)

                # For each triangulation vertex: reuse existing vertex if close, else append
                distance, vid = tree.query(vn3)  # vid are indices into current `vertices`

                # global indices for all vn vertices
                vmap = vid.copy()

                new_mask = distance > tol
                if np.any(new_mask):
                    # append new vertices
                    start = len(vertices)
                    vertices = np.vstack([vertices, vn3[new_mask]])

                    # assign their new global indices
                    vmap[new_mask] = np.arange(start, start + int(new_mask.sum()), dtype=np.int64)

                    # rebuild tree so future polygons can reuse these appended vertices
                    tree = cKDTree(vertices)

                # remap triangulation faces to global vertex indices
                nf = vmap[fn]

                # remove degenerate faces (still can happen)
                nf_ok = (nf[:, 1:] != nf[:, :1]).all(axis=1) & (nf[:, 1] != nf[:, 2])
                faces.append(nf[nf_ok])

            faces = np.vstack(faces)

    return Trimesh(vertices=vertices, faces=faces, **kwargs)


def _orient_cap(cap: trimesh.Trimesh | None, desired_normal: np.ndarray) -> trimesh.Trimesh | None:
    """
    Ensure cap face winding produces normals roughly aligned with desired_normal.
    """
    if cap is None or cap.faces is None or len(cap.faces) == 0:
        return None

    desired_normal = np.asarray(desired_normal, dtype=float)
    desired_normal /= (np.linalg.norm(desired_normal) + 1e-30)

    # mean normal (robust-ish)
    n = cap.face_normals
    if n is None or len(n) == 0:
        return cap
    mean_n = n.mean(axis=0)
    if np.dot(mean_n, desired_normal) < 0.0:
        # flip winding
        cap = cap.copy()
        cap.faces = cap.faces[:, [0, 2, 1]]
    return cap


def _add_cap(mesh: trimesh.Trimesh, cap: trimesh.Trimesh | None) -> trimesh.Trimesh:
    if cap is None or len(cap.faces) == 0:
        return mesh
    # concatenate keeps things simple (duplicates verts are OK for STL export etc.)
    return trimesh.util.concatenate([mesh, cap])


def slice_mesh_into_z_slabs_by_height(
    mesh: trimesh.Trimesh,
    layer_height: float,
) -> list[trimesh.Trimesh]:
    """
    Slice mesh into Z slabs of height layer_height, returning CLOSED slabs.

    Reuse rule:
      top cap of slab i  == bottom cap of slab (i+1)
    """
    if layer_height <= 0:
        raise ValueError("layer_height must be > 0")

    bounds = np.asarray(mesh.bounds, dtype=float)
    z0 = float(bounds[0, 2])
    z1 = float(bounds[1, 2])
    if not np.isfinite(z0) or not np.isfinite(z1) or z1 <= z0:
        raise ValueError(f"Bad Z bounds: zmin={z0}, zmax={z1}")

    H = float(layer_height)

    n_pos = np.array([0.0, 0.0, 1.0], dtype=float)   # keep z >= plane
    n_neg = np.array([0.0, 0.0, -1.0], dtype=float)  # keep z <= plane

    n_layers = int(np.ceil((z1 - z0) / H))
    slabs: list[trimesh.Trimesh] = []

    prev_slab: trimesh.Trimesh | None = None  # previous slab waiting for its TOP cap

    for i in range(n_layers):
        a0 = z0 + i * H
        a1 = min(z1, z0 + (i + 1) * H)

        origin0 = mesh.centroid.copy()
        origin0[2] = float(a0)

        origin1 = mesh.centroid.copy()
        origin1[2] = float(a1)

        # 1) Keep z >= a0, and generate the cap at z=a0 (this is the BOTTOM cap of current slab)
        if i > 0:
            m1 = slice_mesh_plane(
                mesh,
                plane_normal=n_pos,
                plane_origin=origin0,
                cap=True,
                return_cap=True,
                engine="triangle",
                triangle_args="pYq",
            )
            # m1 = trimesh.intersections.slice_mesh_plane(
            #     mesh,
            #     plane_normal=n_pos,
            #     plane_origin=origin0,
            #     cap=True,
            #     engine="triangle",
            # )
        else:
            m1 = mesh

        if m1 is None or len(m1.faces) == 0:
            continue

        # 2) Keep z <= a1. Only the LAST slab needs an actual top cap computed here.
        is_last = (i == n_layers - 1)

        if (is_last):
            m2 = m1
        else:
            m2 = slice_mesh_plane(
                m1,
                plane_normal=n_neg,
                plane_origin=origin1,
                cap=True,
                return_cap=True,
                engine="triangle",
                triangle_args="pYq",
            )
            # m2 = trimesh.intersections.slice_mesh_plane(
            #     m1,
            #     plane_normal=n_neg,
            #     plane_origin=origin1,
            #     cap=True,
            #     engine="triangle",
            # )

        if m2 is None or len(m2.faces) == 0:
            continue

        slabs.append(m2)
        continue

        # ---- Reuse logic ----
        # The bottom cap of *current* slab at a0 is the TOP cap of the *previous* slab.
        if prev_slab is not None:
            cap_for_prev_top = _orient_cap(bottom_cap, desired_normal=np.array([0.0, 0.0, 1.0]))
            prev_slab.export(f"DEBUG/cap_no_{i}.stl")
            cap_for_prev_top.export(f"DEBUG/cap_{i}.stl")
            prev_closed = _add_cap(prev_slab, cap_for_prev_top)
            prev_closed.merge_vertices()
            prev_closed.remove_degenerate_faces()
            slabs.append(prev_closed)

        # Current slab: add its bottom cap now (normals should point DOWN)
        cur = m2

        # If last slab: also add its top cap now (normals should point UP)
        if is_last:
            prev_slab = None
            m1.merge_vertices()
            m1.remove_degenerate_faces()
            slabs.append(m1)
        else:
            # hold it until next iteration gives us the reused top cap
            prev_slab = cur

    return slabs


import math
import numpy as np

def transform_points_from_cylindrical_like_old(
    points_uv_w: np.ndarray,
    cx_cyl: float,
    cz_cyl: float,
    R0: float,
    theta0: float,
) -> np.ndarray:
    """
    Same inverse mapping as your trimesh version, but for a raw (N,3) point array.

    points_uv_w are (u, v, w):
      u = arc length around cylinder
      v = y
      w = inward depth

    Returns (N,3) world points (x,y,z).
    """
    Vp = np.asarray(points_uv_w, dtype=np.float64)
    if Vp.ndim != 2 or Vp.shape[1] != 3:
        raise ValueError("points_uv_w must be an (N,3) array")
    if Vp.size == 0:
        raise ValueError("No points")
    if R0 <= 0:
        raise ValueError("R0 must be > 0")

    u = Vp[:, 0]
    v = Vp[:, 1]
    w = Vp[:, 2]

    tw = 2.0 * math.pi * float(R0)
    u_norm = np.mod(u, tw)

    theta = float(theta0) + (u_norm / float(R0))
    r = float(R0) - w

    x = float(cx_cyl) + r * np.cos(theta)
    z = float(cz_cyl) + r * np.sin(theta)
    y = v

    return np.column_stack([x, y, z]).astype(np.float64)


def transform_tetgen_grid_from_cylindrical_like_old(
    tetgen_grid,
    cx_cyl: float,
    cz_cyl: float,
    R0: float,
    theta0: float,
):
    """
    Takes tetgen_mesh.grid (a pyvista.UnstructuredGrid), returns a new grid with
    transformed points and identical cell connectivity/types.
    """
    import pyvista as pv

    pts_world = transform_points_from_cylindrical_like_old(
        tetgen_grid.points, cx_cyl, cz_cyl, R0, theta0
    )

    # Keep exact connectivity + cell types
    cells = tetgen_grid.cells.copy()
    celltypes = tetgen_grid.celltypes.copy()

    out = pv.UnstructuredGrid(cells, celltypes, pts_world)

    # Optional: carry over cell/point data if you have any
    out.point_data.update(tetgen_grid.point_data)
    out.cell_data.update(tetgen_grid.cell_data)

    return out


import numpy as np
from typing import Iterable, Tuple, Set


def write_tetgen_wireframe_ply(
    filename: str,
    points: np.ndarray,
    cells: np.ndarray,
):
    """
    Write TetGen tetrahedral mesh as a PLY wireframe:
    - vertices = points
    - edges = unique edges from tetrahedra

    Parameters
    ----------
    filename : str
        Output .ply path
    points : (N,3) ndarray
        World-space vertex coordinates
    cells : (M,) ndarray
        TetGen / PyVista cell array:
        [4, i0, i1, i2, i3, 4, i0, i1, i2, i3, ...]
    """

    points = np.asarray(points, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be (N,3)")
    if cells.ndim != 1:
        raise ValueError("cells must be flat array")

    edges: Set[Tuple[int, int]] = set()

    i = 0
    n = len(cells)
    while i < n:
        if cells[i] != 4:
            raise ValueError(f"Expected tet (4), got {cells[i]} at index {i}")

        a, b, c, d = cells[i + 1 : i + 5]

        # 6 edges per tetrahedron
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((a, c))))
        edges.add(tuple(sorted((a, d))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((b, d))))
        edges.add(tuple(sorted((c, d))))

        i += 5

    edges = np.array(list(edges), dtype=np.int64)

    # ---- Write ASCII PLY ----
    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("end_header\n")

        for x, y, z in points:
            f.write(f"{x} {y} {z}\n")

        for i0, i1 in edges:
            f.write(f"{i0} {i1}\n")


import numpy as np
from typing import Iterable, Tuple, Dict

def merge_tetgen_grids(
    slab_tetgens: Iterable,
    eps: float = 1e-7,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge many TetGen objects (where mesh is in tg.grid.points and tg.grid.cells)
    into one global (points, cells) with NO duplicate vertices on shared interfaces.

    Parameters
    ----------
    slab_tetgens : iterable of tetgen.TetGen objects
        Each must have:
          - tg.grid.points : (Ni,3) float
          - tg.grid.cells  : (Mi,)  int, flat PyVista cell array:
                [4, i0, i1, i2, i3, 4, i0, i1, i2, i3, ...]
    eps : float
        Vertex dedup tolerance. Vertices whose coordinates match within eps (via
        quantization) are treated as identical and merged.

    Returns
    -------
    merged_points : (N,3) float64
    merged_cells  : (M,)  int64
        Flat PyVista cell array of tetrahedra.
    """
    # global storage
    key_to_gid: Dict[tuple, int] = {}
    global_pts = []          # list of (3,) float
    global_cells = []        # list of ints, flat [4, a,b,c,d, 4, ...]
    gid_counter = 0

    def key_of(p: np.ndarray) -> tuple:
        # quantize to a grid of size eps to be robust to float noise
        return (int(np.round(p[0] / eps)),
                int(np.round(p[1] / eps)),
                int(np.round(p[2] / eps)))

    for tg in slab_tetgens:
        pts = np.asarray(tg.grid.points, dtype=np.float64)
        cells = np.asarray(tg.grid.cells, dtype=np.int64)

        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("tg.grid.points must be (N,3)")
        if cells.ndim != 1:
            raise ValueError("tg.grid.cells must be a flat array")

        # local->global vertex map
        l2g = np.empty(len(pts), dtype=np.int64)
        for li, p in enumerate(pts):
            k = key_of(p)
            gi = key_to_gid.get(k)
            if gi is None:
                gi = gid_counter
                gid_counter += 1
                key_to_gid[k] = gi
                global_pts.append(p)
            l2g[li] = gi

        # remap cells (PyVista flat cell array)
        i = 0
        n = len(cells)
        while i < n:
            if cells[i] != 4:
                raise ValueError(f"Expected tet (4), got {cells[i]} at index {i}")
            a, b, c, d = cells[i + 1 : i + 5]
            ga, gb, gc, gd = l2g[a], l2g[b], l2g[c], l2g[d]
            global_cells.extend([4, int(ga), int(gb), int(gc), int(gd)])
            i += 5

    merged_points = np.asarray(global_pts, dtype=np.float64)
    merged_cells = np.asarray(global_cells, dtype=np.int64)
    return merged_points, merged_cells


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

tet_meshes = []

for i, slab in enumerate(slabs):
    slab.export(f"{OUTPUT_DIR}/slices/slice_{i:03d}.stl")

    slab = transform_mesh_from_cylindrical_like_old(
        slab,
        cx, cz, R0, theta0
    )
    slab.export(f"{OUTPUT_DIR}/slices/slice_world_{i:03d}.stl")

    tetgen_mesh = tetgen.TetGen(slab.vertices, slab.faces)
    tetgen_mesh.tetrahedralize(
        quality=False,
        nobisect=False,
        verbose=1
    )


    tetgen_points = tetgen_mesh.grid.points
    tetgen_cells = tetgen_mesh.grid.cells

    # grid_cyl = tetgen_mesh.grid               # pyvista.UnstructuredGrid in (u,v,w)
    # grid_world = transform_tetgen_grid_from_cylindrical_like_old(
    #     grid_cyl, cx, cz, R0, theta0
    # )

    # # If you still want arrays:
    # tetgen_points_world = grid_world.points
    # tetgen_cells = grid_world.cells
    # tetgen_celltypes = grid_world.celltypes

    write_tetgen_wireframe_ply(
        f"{OUTPUT_DIR}/slices/slice_tet_{i:03d}.ply",
        tetgen_points,
        tetgen_cells,
    )

    tet_meshes.append(tetgen_mesh)

mega_points, mega_cells = merge_tetgen_grids(tet_meshes)

write_tetgen_wireframe_ply(
    f"{OUTPUT_DIR}/slices/MEGA.ply",
    mega_points,
    mega_cells,
)

pts_world = transform_points_from_cylindrical_like_old(
    mega_points, cx, cz, R0, theta0
)

write_tetgen_wireframe_ply(
    f"{OUTPUT_DIR}/slices/MEGA_tran.ply",
    pts_world,
    mega_cells,
)

write_calculix_job_tet_layer_binned(
    path=f"{SIMULATION}.inp",
    grid_points=mega_points,
    grid_cells=mega_cells,
    layer_height=LAYER_HEIGHT,
    cyl_radius=CYLINDER_RADIUS,
    shrinkage_curve=SHRINKAGE_CURVE,
    cure_shrink_per_unit=SHRINKAGE,
)

run_calculix(
    SIMULATION,
    "C:/Users/4y5t6/Downloads/PrePoMax v2.4.0/Solver/ccx_dynamic.exe"
)

vtk_grid = vtk_grid_from_tetgen(mega_points, tetgen_cells)
displacements = read_ccx_frd_displacements(f"{SIMULATION}.frd")

export_tet_displacement_debug(
    vtk_grid,
    displacements,
    out_prefix="DEBUG/tet",
    scale=1.0,
    stride=1,
)

deform_stl_by_tet_field(
    INPUT_STL,
    f"{OUTPUT_DIR}/deformed_stl.stl",
    vtk_grid,
    displacements,
    scale=1.0,
    outside_mode="keep"
)

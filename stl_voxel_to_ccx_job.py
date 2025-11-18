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
   - One step per slice ("layer" in Z):
       * Apply heat flux to that layer's ELSET.
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
"""

import argparse
import os
from typing import List, Tuple, Dict

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
#  Voxel mesh -> CalculiX job
# ============================================================

def generate_global_cubic_hex_mesh(
    input_stl: str,
    cube_size: float,
):
    """
    Voxelize input_stl and build a global C3D8R brick mesh.

    Returns:
        vertices: List[(x, y, z)]
        hexes:    List[(v1..v8)]  1-based indices
        slice_to_eids: Dict[slice_index -> List[element_id]]
        z_slices: List[float]  physical z of each slice index (sorted bottom->top)
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

    # map iz -> physical z
    unique_iz = np.unique(indices_sorted[:, 2])
    layer_info = []
    for iz in unique_iz:
        idx_arr = np.array([[0, 0, iz]], dtype=float)
        z_center = float(vox.indices_to_points(idx_arr)[0, 2])
        layer_info.append((iz, z_center))
    layer_info.sort(key=lambda x: x[1])

    iz_to_slice: Dict[int, int] = {}
    z_slices: List[float] = []
    for slice_idx, (iz, z_phys) in enumerate(layer_info):
        iz_to_slice[int(iz)] = slice_idx
        z_slices.append(z_phys)

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
    base_temp: float = 20.0,
    heat_flux: float = 1.0,
):
    """
    Single uncoupled temperature-displacement job, modeled after the
    PrePoMax ABS example:

      - C3D8R bricks
      - ABS material with thermal + mechanical properties
      - Reference temperature 20 C (Expansion, Zero=20)
      - NSET=BASE (bottom nodes) fixed mechanically and held at base_temp
      - One *UNCOUPLED TEMPERATURE-DISPLACEMENT step per slice:
          * In step k:
              - clear previous flux (OP=NEW)
              - apply body flux (BF) only to ELSET=SLICE_k
          * Temperatures and displacements carry over between steps.

    Parameters
    ----------
    base_temp : float
        Clamp temperature for the base (deg C). Default 20 to match Zero=20.
    heat_flux : float
        Body heat generation value used in *DFLUX ... BF, value.
        Units are consistent with your MM_TON_S_C system (W/mm^3).
    """
    n_nodes = len(vertices)
    n_elems = len(hexes)

    # detect bottom nodes as "BASE"
    z_coords = np.array([v[2] for v in vertices], dtype=float)
    z_min = float(z_coords.min())
    tol = (z_coords.max() - z_min + 1e-9) * 1e-3
    base_nodes = [i + 1 for i, z in enumerate(z_coords) if abs(z - z_min) <= tol]

    n_slices = len(z_slices)

    # choose a simple time scheme: 1.0 per slice
    time_per_layer = 1.0

    with open(path, "w", encoding="utf-8") as f:
        f.write("**\n** Auto-generated SLA-style thermo-mechanical job\n**\n")
        f.write("*HEADING\n")
        f.write("Voxel C3D8R uncoupled temperature-displacement (layer-wise BF heating)\n")

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

        if base_nodes:
            f.write("*NSET, NSET=BASE\n")
            _write_id_list_lines(f, base_nodes)
        else:
            f.write("** Warning: no base nodes detected for BASE set.\n")

        f.write("** Element sets (per slice) ++++++++++++++++++++++++++++++++\n")
        for slice_idx in range(n_slices):
            eids = slice_to_eids.get(slice_idx, [])
            if not eids:
                continue
            valid_eids = [eid for eid in eids if 1 <= eid <= n_elems]
            if not valid_eids:
                print(f"[WARN] Slice {slice_idx} has no valid element IDs within 1..{n_elems}")
                continue
            name = f"SLICE_{slice_idx:03d}"
            f.write(f"*ELSET, ELSET={name}\n")
            _write_id_list_lines(f, valid_eids)

        # -------------------- MATERIAL (ABS) --------------------
        f.write("** Materials +++++++++++++++++++++++++++++++++++++++++++++++\n")
        f.write("*MATERIAL, NAME=ABS\n")
        f.write("*DENSITY\n")
        f.write("1.02E-09\n")
        f.write("*ELASTIC\n")
        f.write("2000., 0.394\n")
        f.write("*EXPANSION, ZERO=20.\n")
        f.write("7.4E-05\n")
        f.write("*CONDUCTIVITY\n")
        f.write("0.2256\n")
        f.write("*SPECIFIC HEAT\n")
        f.write("1386000000.\n")
        f.write("*SOLID SECTION, ELSET=ALL, MATERIAL=ABS\n")

        # -------------------- INITIAL TEMPERATURE --------------------
        f.write("** Initial conditions ++++++++++++++++++++++++++++++++++++++\n")
        f.write("*INITIAL CONDITIONS, TYPE=TEMPERATURE\n")
        f.write(f"ALLNODES, {base_temp}\n")

        # -------------------- STEPS --------------------
        f.write("** Steps +++++++++++++++++++++++++++++++++++++++++++++++++++\n")
        if n_slices == 0:
            f.write("** No slices -> no steps.\n")
        else:
            for slice_idx in range(n_slices):
                name = f"SLICE_{slice_idx:03d}"
                z_val = z_slices[slice_idx]
                step_name = f"LAYER_{slice_idx:03d}"
                t_end = (slice_idx + 1) * time_per_layer

                f.write("** --------------------------------------------------------\n")
                f.write(f"** Step {slice_idx+1}: heat slice {name} at z = {z_val}\n")
                f.write("** --------------------------------------------------------\n")
                f.write("*STEP\n")
                f.write(f"*UNCOUPLED TEMPERATURE-DISPLACEMENT\n")
                f.write(f"{t_end:.6f}, {time_per_layer:.6f}\n")

                # Boundary conditions: base fixed mechanically + thermally
                if base_nodes:
                    f.write("** Boundary conditions +++++++++++++++++++++++++++++++++++++\n")
                    f.write("*BOUNDARY, OP=NEW\n")
                    # mechanical clamp
                    f.write("BASE, 1, 6, 0.\n")
                    # temperature clamp (DOF 11)
                    # f.write(f"BASE, 11, 11, {base_temp}\n")

                # Outputs (similar to PrePoMax Node/El file)
                f.write("** Field outputs +++++++++++++++++++++++++++++++++++++++++++\n")
                f.write("*NODE FILE\n")
                f.write("RF, U, NT, RFL\n")
                f.write("*EL FILE\n")
                f.write("S, E, HFL, NOE\n")

                # Loads: surface flux on TOP face (S2) of this slice, previous flux cleared
                # S2 is the top surface for our C3D8R element ordering (1-4 bottom, 5-8 top).
                f.write("** Loads (surface flux on top of current slice) ++++++++++++\n")
                f.write("*DFLUX, OP=NEW\n")
                f.write(f"{name}, S2, {heat_flux:.6E}\n")

                f.write("*END STEP\n")

    print(f"[CCX] Wrote PrePoMax-style UT-D job to: {path}")
    print(f"[CCX] Nodes: {n_nodes}, elements: {n_elems}, slices: {n_slices}")


# ============================================================
#  (Old mechanical job writer – now unused, kept only for reference)
# ============================================================

def write_mechanical_job(*args, **kwargs):
    """
    Deprecated: the pipeline now uses a single *UNCOUPLED TEMPERATURE-DISPLACEMENT job.
    This function is kept only to avoid breaking imports; it is not used in main().
    """
    print("[WARN] write_mechanical_job() is deprecated and not used in this script.")


# ============================================================
#  CalculiX runner
# ============================================================

def run_calculix(job_name: str, ccx_cmd: str = "ccx"):
    """
    Run CalculiX on given job (job_name without .inp).
    """
    print(f"[RUN] Launching CalculiX: {ccx_cmd} {job_name}")
    try:
        result = subprocess.run(
            [ccx_cmd, job_name],
            check=False,
            capture_output=True,
            text=True,
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
    """
    Parse nodal displacements from a CalculiX .frd file.

    We look for the last DISP dataset:

        -4  DISP ...
        -5  D1 ...
        -5  D2 ...
        -5  D3 ...
        -5  ALL ...
        -1  <node> <D1> <D2> <D3>
        ...
        -3

    Lines are parsed using fixed-width columns, because the node ID and
    first value may be glued when the value is negative (so split() is unsafe).
    """
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

            # Start of a DISP result block
            if s.startswith("-4") and "DISP" in s:
                disp = {}
                in_disp = True
                continue

            if not in_disp:
                continue

            # End of this result block
            if s.startswith("-3"):
                in_disp = False
                continue

            # Nodal data line
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
    """
    Build a PyGeM FFD object whose control points coincide (logically)
    with the voxel lattice nodes, and whose control point displacements
    come from the FE nodal displacements.

    This implementation is compatible with the *classic* PyGeM API:

        FFD(n_control_points=[nx, ny, nz])
        ffd.box_origin, ffd.box_length, ffd.rot_angle
        ffd.array_mu_x / array_mu_y / array_mu_z

    The FE displacements (in physical units) are converted to the
    normalized weights used by PyGeM: array_mu_* = disp / box_length.
    """
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

    # First pass: find max logical indices (ix, iy, iz), assuming regular grid
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

    nx = imax + 1  # number of control points in X
    ny = jmax + 1  # in Y
    nz = kmax + 1  # in Z

    print(f"[FFD] Building PyGeM FFD: n_control_points = ({nx}, {ny}, {nz})")

    # --- Create FFD (classic PyGeM API) ---
    ffd = FFD(n_control_points=[nx, ny, nz])

    # Align FFD box with voxel lattice bounding box
    Lx = max(x_max - x_min, 1e-12)
    Ly = max(y_max - y_min, 1e-12)
    Lz = max(z_max - z_min, 1e-12)

    ffd.box_origin[:] = np.array([x_min, y_min, z_min], dtype=float)
    ffd.box_length[:] = np.array([Lx, Ly, Lz], dtype=float)
    ffd.rot_angle[:] = np.array([0.0, 0.0, 0.0], dtype=float)

    # Reset weights if available, otherwise zero arrays manually
    if hasattr(ffd, "reset_weights"):
        ffd.reset_weights()
    else:
        ffd.array_mu_x[:] = 0.0
        ffd.array_mu_y[:] = 0.0
        ffd.array_mu_z[:] = 0.0

    # --- Map FE nodal displacements -> FFD weights ---
    # displacements[nid] is in physical units
    for nid, (ix, iy, iz) in enumerate(logical_idx, start=1):
        if not (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
            continue

        u = displacements.get(nid)
        if u is None:
            continue

        # Convert physical displacement to normalized mu (per box length)
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
    """
    Export lattice as two PLY point clouds:

        - orig_path : original lattice node positions
        - def_path  : deformed lattice node positions

    Both are ASCII PLY files of points (no colors).
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


# ============================================================
#  STL deformation with PyGeM (forward + pre-deformed)
# ============================================================

def ffd_apply_points(ffd, pts: np.ndarray) -> np.ndarray:
    """Wrapper to handle PyGeM API (__call__ vs deform)."""
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
    """
    Full pipeline using PyGeM FFD:

      - Read displacements from thermo-mechanical .frd
      - Optionally export lattice PLYs
      - Build PyGeM FFD lattice from voxel nodes + displacements
      - Load original STL
      - Forward FFD deformation -> deformed STL
      - Pre-deformed STL -> apply FFD with reversed control displacements
    """
    if FFD is None:
        print("[FFD] PyGeM FFD not available; skipping STL deformation.")
        return

    displacements = read_frd_displacements(mech_frd_path)
    if not displacements:
        print("[FFD] No displacements found, skipping STL deformation.")
        return

    print(f"[FFD] Displacements available for {len(displacements)} nodes "
          f"out of {len(vertices)} total.")

    # Optional lattice export
    if lattice_basepath:
        orig_ply = lattice_basepath + "_orig.ply"
        def_ply = lattice_basepath + "_def.ply"
        export_lattice_ply_split(vertices, displacements, orig_ply, def_ply)

    # Build FFD from voxel lattice + FE displacements
    ffd = build_ffd_from_lattice(vertices, cube_size, displacements)

    # Load original STL
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

    # ------------------------------------------------------------------
    # Forward deformation: FFD with +displacements -> deformed STL
    # ------------------------------------------------------------------
    deformed_verts = ffd_apply_points(ffd, orig_verts)
    mesh.vertices = deformed_verts
    mesh.export(output_stl)
    print(f"[FFD] Deformed STL (PyGeM) written to: {output_stl}")

    # ------------------------------------------------------------------
    # Pre-deformation: flip control displacements and apply to original
    # ------------------------------------------------------------------
    print("[FFD] Computing pre-deformed STL by reversing FFD control displacements...")

    # Restore original geometry
    mesh.vertices = orig_verts

    # Flip the FFD weights (classic PyGeM: array_mu_x/y/z)
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

    args = parser.parse_args()

    if not os.path.isfile(args.input_stl):
        print(f"Input STL not found: {args.input_stl}")
        raise SystemExit(1)

    # 1) Mesh
    vertices, hexes, slice_to_eids, z_slices = generate_global_cubic_hex_mesh(
        args.input_stl,
        args.cube_size,
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
        heat_flux=args.heat_flux,
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

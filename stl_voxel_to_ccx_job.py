#!/usr/bin/env python3
"""
STL -> voxel C3D8R/DC3D8 hex mesh -> thermal + mechanical CalculiX jobs
+ optional FFD deformation of the original STL (forward + pre-deformed)
+ optional lattice visualization as PLY point clouds.

Pipeline:

1. Voxelize input STL into cubes of size cube_size.
2. Build global hex mesh:
   - thermal job: DC3D8, per-slice heating
   - mechanical job: C3D8R, reads temps from thermal .frd, computes U
3. If --run-ccx:
   - run thermal job -> <job_name>_therm.frd
   - run mech job    -> <job_name>_mech.frd
   - read nodal displacements from mech .frd
   - treat voxel node grid as FFD lattice
   - forward:  deform input STL -> <job_name>_deformed.stl
   - inverse: pre-deform input STL -> <job_name>_deformed_pre.stl
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
    base_temp: float = 293.0,
    heat_flux: float = 1.0e3,
):
    """
    Pure thermal variant:
      - DC3D8 heat-transfer elements
      - all elements always active
      - ELSET per slice
      - base held at base_temp
      - each step: steady-state HEAT TRANSFER, heat only one slice via DFLUX.
    """
    n_nodes = len(vertices)
    n_elems = len(hexes)

    # detect bottom nodes as "BASE"
    z_coords = np.array([v[2] for v in vertices], dtype=float)
    z_min = float(z_coords.min())
    tol = (z_coords.max() - z_min + 1e-9) * 1e-3
    base_nodes = [i + 1 for i, z in enumerate(z_coords) if abs(z - z_min) <= tol]

    with open(path, "w", encoding="utf-8") as f:
        f.write("*HEADING\n")
        f.write("Voxel DC3D8 build (all layers active, per-slice heating, thermal-only)\n")

        # Nodes
        f.write("*NODE\n")
        for i, (x, y, z) in enumerate(vertices, start=1):
            f.write(f"{i}, {x}, {y}, {z}\n")

        # Elements: thermal brick
        f.write("*ELEMENT, TYPE=DC3D8, ELSET=ALL\n")
        for eid, (n0, n1, n2, n3, n4, n5, n6, n7) in enumerate(hexes, start=1):
            f.write(
                f"{eid}, {n0}, {n1}, {n2}, {n3}, "
                f"{n4}, {n5}, {n6}, {n7}\n"
            )

        # Node sets
        f.write("*NSET, NSET=ALLNODES, GENERATE\n")
        f.write(f"1, {n_nodes}, 1\n")

        if base_nodes:
            f.write("*NSET, NSET=BASE\n")
            _write_id_list_lines(f, base_nodes)
        else:
            f.write("** Warning: no base nodes detected.\n")

        # ELSET per slice (sanitized)
        n_slices = len(z_slices)
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

        # --- Thermal material ---
        f.write("** --- MATERIAL DEFINITION (thermal only) ---\n")
        f.write("*MATERIAL, NAME=MAT1\n")
        f.write("*DENSITY\n")
        f.write("7.8E-9\n")
        f.write("*CONDUCTIVITY\n")
        f.write("45.\n")
        f.write("*SPECIFIC HEAT\n")
        f.write("500.\n")
        f.write("*SOLID SECTION, ELSET=ALL, MATERIAL=MAT1\n")

        # Initial temperature
        f.write("** --- INITIAL CONDITIONS ---\n")
        f.write("*INITIAL CONDITIONS, TYPE=TEMPERATURE\n")
        f.write(f"ALLNODES, {base_temp}\n")

        # --- Steps: all elements active, heat one slice per step ---
        f.write("** --- HEAT-TRANSFER STEPS, PER-SLICE HEATING ---\n")

        if n_slices == 0:
            f.write("** No slices found, no steps generated.\n")
        else:
            for slice_idx in range(n_slices):
                name = f"SLICE_{slice_idx:03d}"
                z_val = z_slices[slice_idx]

                f.write("** ----------------------------------------\n")
                f.write(f"** Heat-transfer step heating {name} at z = {z_val}\n")
                f.write("*STEP\n")
                f.write("*HEAT TRANSFER, STEADY STATE\n")
                f.write("1., 1.\n")

                if base_nodes:
                    f.write("*BOUNDARY\n")
                    # Fix base temperature at base_temp (DOF 11)
                    f.write(f"BASE, 11, 11, {base_temp}\n")

                # Output: nodal temperatures every step
                f.write("*NODE FILE, FREQUENCY=1\n")
                f.write("NT\n")

                # Heat only this slice
                f.write("*DFLUX\n")
                f.write(f"{name}, S2, {heat_flux:.6E}\n")

                f.write("*END STEP\n")

    print(f"[CCX] Wrote CalculiX job to: {path}")
    print(f"[CCX] Nodes: {n_nodes}, elements: {n_elems}, slices: {len(z_slices)}")


def write_mechanical_job(
    path: str,
    vertices: List[Tuple[float, float, float]],
    hexes: List[Tuple[int, int, int, int, int, int, int, int]],
    base_temp: float,
    thermal_job_name: str,
    n_slices: int,
):
    """
    Mechanical deformation job:

      - C3D8R thermo-elastic elements
      - same mesh as thermal job
      - base fixed mechanically
      - reads final temperature field from thermal job .frd
      - single STATIC step computes thermal expansion deformation
    """
    n_nodes = len(vertices)
    n_elems = len(hexes)

    # detect bottom nodes as "BASE" (same logic as thermal job)
    z_coords = np.array([v[2] for v in vertices], dtype=float)
    z_min = float(z_coords.min())
    tol = (z_coords.max() - z_min + 1e-9) * 1e-3
    base_nodes = [i + 1 for i, z in enumerate(z_coords) if abs(z - z_min) <= tol]

    with open(path, "w", encoding="utf-8") as f:
        f.write("*HEADING\n")
        f.write("Thermo-elastic deformation from thermal FRD (auto-generated)\n")

        # Nodes (same coordinates)
        f.write("*NODE\n")
        for i, (x, y, z) in enumerate(vertices, start=1):
            f.write(f"{i}, {x}, {y}, {z}\n")

        # Elements: structural bricks now
        f.write("*ELEMENT, TYPE=C3D8R, ELSET=ALL\n")
        for eid, (n0, n1, n2, n3, n4, n5, n6, n7) in enumerate(hexes, start=1):
            f.write(
                f"{eid}, {n0}, {n1}, {n2}, {n3}, "
                f"{n4}, {n5}, {n6}, {n7}\n"
            )

        # Node sets
        f.write("*NSET, NSET=ALLNODES, GENERATE\n")
        f.write(f"1, {n_nodes}, 1\n")

        if base_nodes:
            f.write("*NSET, NSET=BASE\n")
            _write_id_list_lines(f, base_nodes)
        else:
            f.write("** Warning: no base nodes detected.\n")

        # Material: linear thermo-elastic
        f.write("** --- MATERIAL: thermo-elastic for deformation ---\n")
        f.write("*MATERIAL, NAME=MAT1\n")
        f.write("*ELASTIC\n")
        f.write("210000., 0.30\n")
        f.write("*EXPANSION\n")
        f.write("1.2E-5\n")
        f.write("*DENSITY\n")
        f.write("7.8E-9\n")
        f.write("*SOLID SECTION, ELSET=ALL, MATERIAL=MAT1\n")

        # Initial temperature (must match thermal job reference)
        f.write("** --- INITIAL TEMPERATURE (reference) ---\n")
        f.write("*INITIAL CONDITIONS, TYPE=TEMPERATURE\n")
        f.write(f"ALLNODES, {base_temp}\n")

        # Mechanical step
        f.write("** --- STATIC STEP: apply temperature field from thermal job ---\n")
        f.write("*STEP\n")
        f.write("*STATIC\n")
        f.write("1., 1.\n")

        # Fix base mechanically
        if base_nodes:
            f.write("*BOUNDARY\n")
            f.write("BASE, 1, 3, 0.\n")

        # Read temperatures from thermal job .frd, last step = n_slices
        f.write(
            f"*TEMPERATURE, FILE={thermal_job_name}.frd, "
            f"BSTEP={n_slices}, ESTEP={n_slices}\n"
        )
        f.write("ALLNODES\n")

        # Output: displacements + stresses
        f.write("*NODE FILE, FREQUENCY=1\n")
        f.write("U\n")
        f.write("*EL FILE, FREQUENCY=1\n")
        f.write("S, E\n")

        f.write("*END STEP\n")

    print(f"[CCX] Wrote mechanical job to: {path}")


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
#  FRD parsing + FFD helpers
# ============================================================

def read_frd_displacements(frd_path: str) -> Dict[int, np.ndarray]:
    """
    Read nodal displacements from a CalculiX .frd file.

    Parses the *DISP dataset in fixed-width format:

        columns:
          1-2   : code  (should be '-1')
          4-13  : node id (I10)
          14-25 : Ux (E12.5)
          26-37 : Uy (E12.5)
          38-49 : Uz (E12.5)

    Returns:
        disp: dict { node_id: np.array([ux, uy, uz]) }
    """
    if not os.path.isfile(frd_path):
        print(f"[FRD] File not found: {frd_path}")
        return {}

    disp: Dict[int, np.ndarray] = {}
    state = "search"

    print(f"[FRD] Parsing displacements from: {frd_path}")

    with open(frd_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.lstrip()
            if not stripped:
                continue

            if state == "search":
                # Look for the DISP result header
                if stripped.startswith("-4") and "DISP" in stripped:
                    state = "in_header"
                continue

            elif state == "in_header":
                # Skip the -5 D1/D2/D3/ALL lines, go to data at first -1
                if stripped.startswith("-1"):
                    state = "in_data"
                elif stripped.startswith("-3"):
                    # no data after all, go back to search
                    state = "search"
                continue

            elif state == "in_data":
                # End of this dataset
                if stripped.startswith("-3"):
                    state = "search"
                    continue

                # Only process lines with code -1 in columns 1-3
                code = line[1:3]
                if code != "-1":
                    continue

                try:
                    nid = int(line[3:13])
                    ux = float(line[13:25])
                    uy = float(line[25:37])
                    uz = float(line[37:49])
                except ValueError:
                    # Badly formatted line, skip
                    continue

                disp[nid] = np.array([ux, uy, uz], dtype=float)

    print(f"[FRD] Parsed displacement data for {len(disp)} nodes.")
    return disp


def build_voxel_lattice_from_vertices(
    vertices: List[Tuple[float, float, float]],
    cube_size: float,
):
    """
    Reconstruct a regular voxel lattice from the voxel node coordinates.

    Returns:
        origin: (x_min, y_min, z_min)  of lattice
        node_map: Dict[(I,J,K) -> node_id]
        nx_cells, ny_cells, nz_cells: number of voxel cells along each axis
    """
    coords = np.array(vertices, dtype=float)
    x_min = float(coords[:, 0].min())
    y_min = float(coords[:, 1].min())
    z_min = float(coords[:, 2].min())

    node_map: Dict[Tuple[int, int, int], int] = {}
    imax = jmax = kmax = 0

    inv_h = 1.0 / float(cube_size)

    for nid, (x, y, z) in enumerate(vertices, start=1):
        ix = int(round((x - x_min) * inv_h))
        iy = int(round((y - y_min) * inv_h))
        iz = int(round((z - z_min) * inv_h))

        node_map[(ix, iy, iz)] = nid
        if ix > imax:
            imax = ix
        if iy > jmax:
            jmax = iy
        if iz > kmax:
            kmax = iz

    nx_cells = max(imax, 1)
    ny_cells = max(jmax, 1)
    nz_cells = max(kmax, 1)

    print(
        "[FFD] Voxel lattice: nodes (0..{}, 0..{}, 0..{}), cells: {} x {} x {}".format(
            imax, jmax, kmax, nx_cells, ny_cells, nz_cells
        )
    )

    origin = (x_min, y_min, z_min)
    return origin, node_map, nx_cells, ny_cells, nz_cells


def deform_point_trilinear(
    p: np.ndarray,
    origin: Tuple[float, float, float],
    cube_size: float,
    node_map: Dict[Tuple[int, int, int], int],
    nx_cells: int,
    ny_cells: int,
    nz_cells: int,
    displacements: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, bool]:
    """
    Given a point p in world coordinates, compute its displacement
    by trilinear interpolation from voxel lattice node displacements.

    Returns:
        (u, has_data):
            u        - interpolated displacement
            has_data - True if at least one lattice corner contributed
    """
    x_min, y_min, z_min = origin
    h = float(cube_size)

    rx = (p[0] - x_min) / h
    ry = (p[1] - y_min) / h
    rz = (p[2] - z_min) / h

    eps = 1e-6
    rx = max(0.0, min(rx, nx_cells - eps))
    ry = max(0.0, min(ry, ny_cells - eps))
    rz = max(0.0, min(rz, nz_cells - eps))

    ix = int(np.floor(rx))
    iy = int(np.floor(ry))
    iz = int(np.floor(rz))

    xi = rx - ix
    eta = ry - iy
    zeta = rz - iz

    corners = [
        (ix,     iy,     iz),
        (ix + 1, iy,     iz),
        (ix,     iy + 1, iz),
        (ix + 1, iy + 1, iz),
        (ix,     iy,     iz + 1),
        (ix + 1, iy,     iz + 1),
        (ix,     iy + 1, iz + 1),
        (ix + 1, iy + 1, iz + 1),
    ]

    w000 = (1 - xi) * (1 - eta) * (1 - zeta)
    w100 = xi       * (1 - eta) * (1 - zeta)
    w010 = (1 - xi) * eta       * (1 - zeta)
    w110 = xi       * eta       * (1 - zeta)
    w001 = (1 - xi) * (1 - eta) * zeta
    w101 = xi       * (1 - eta) * zeta
    w011 = (1 - xi) * eta       * zeta
    w111 = xi       * eta       * zeta
    weights = [w000, w100, w010, w110, w001, w101, w011, w111]

    u = np.zeros(3, dtype=float)
    used_weight_sum = 0.0

    for (ijk, w) in zip(corners, weights):
        nid = node_map.get(ijk)
        if nid is None:
            continue
        u_n = displacements.get(nid)
        if u_n is None:
            continue
        u += w * u_n
        used_weight_sum += w

    if used_weight_sum > 0.0:
        u /= used_weight_sum
        return u, True
    else:
        return u, False


def make_nearest_displacement_func(
    vertices: List[Tuple[float, float, float]],
    displacements: Dict[int, np.ndarray],
):
    """
    Prepare a simple nearest-node displacement lookup.

    Returns:
        nearest_disp(p: np.ndarray) -> np.ndarray
    """
    if not displacements:
        def zero_disp(_p: np.ndarray) -> np.ndarray:
            return np.zeros(3, dtype=float)
        return zero_disp

    node_ids = np.array(sorted(displacements.keys()), dtype=int)
    node_coords = np.array([vertices[nid - 1] for nid in node_ids], dtype=float)
    node_u = np.array([displacements[nid] for nid in node_ids], dtype=float)

    def nearest_disp(p: np.ndarray) -> np.ndarray:
        diff = node_coords - p
        dist2 = np.einsum("ij,ij->i", diff, diff)
        idx = int(np.argmin(dist2))
        return node_u[idx]

    return nearest_disp


def inverse_deform_point_fixedpoint(
    p_target: np.ndarray,
    origin: Tuple[float, float, float],
    cube_size: float,
    node_map: Dict[Tuple[int, int, int], int],
    nx_cells: int,
    ny_cells: int,
    nz_cells: int,
    displacements: Dict[int, np.ndarray],
    nearest_disp,
    max_iter: int = 15,
    tol: float = 1e-4,
) -> np.ndarray:
    """
    Given a target point p_target (desired *deformed* position),
    find q such that q + u(q) ≈ p_target using fixed-point iteration:

        q_{k+1} = q_k - (q_k + u(q_k) - p_target)

    u(q) is obtained via trilinear interpolation with nearest-node fallback.
    """
    q = p_target.copy()

    for _ in range(max_iter):
        u_q, has_data = deform_point_trilinear(
            q,
            origin,
            cube_size,
            node_map,
            nx_cells,
            ny_cells,
            nz_cells,
            displacements,
        )
        if not has_data:
            u_q = nearest_disp(q)

        f_q = q + u_q
        r = f_q - p_target
        norm_r = np.linalg.norm(r)
        if norm_r < tol:
            break
        q = q - r

    return q


# ============================================================
#  Lattice visualization (two separate PLYs)
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

    Both are ASCII PLY files of points (no colors, easy to view in ParaView).
    """
    if not vertices:
        print("[PLY] No vertices, skipping lattice export.")
        return

    if not orig_path and not def_path:
        return

    verts = np.array(vertices, dtype=float)
    n_nodes = verts.shape[0]

    # Build full displacement array (zero where missing)
    disp_arr = np.zeros_like(verts)
    for nid, u in displacements.items():
        if 1 <= nid <= n_nodes:
            disp_arr[nid - 1] = u

    original_points = verts
    deformed_points = verts + disp_arr

    # Subsample if too many points
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
#  STL deformation (forward + pre-deformed)
# ============================================================

def deform_input_stl_with_frd(
    input_stl: str,
    mech_frd_path: str,
    vertices: List[Tuple[float, float, float]],
    cube_size: float,
    output_stl: str,
    lattice_basepath: str = None,
):
    """
    Full pipeline:

      - Read displacements from mechanical .frd
      - Optionally export lattice as two PLYs:
            lattice_basepath + "_orig.ply"
            lattice_basepath + "_def.ply"
      - Build structured voxel lattice from 'vertices'
      - Build nearest-node fallback displacement
      - Load original STL (can be arbitrary shape)
      - For each STL vertex:
          * forward deformation -> deformed STL
          * inverse map -> pre-deformed STL
      - Save both STLs
    """
    displacements = read_frd_displacements(mech_frd_path)
    if not displacements:
        print("[FFD] No displacements found, skipping STL deformation.")
        return

    print(f"[FFD] Displacements available for {len(displacements)} nodes "
          f"out of {len(vertices)} total.")

    # Optional lattice export (original + deformed nodes as separate PLYs)
    if lattice_basepath:
        orig_ply = lattice_basepath + "_orig.ply"
        def_ply = lattice_basepath + "_def.ply"
        export_lattice_ply_split(vertices, displacements, orig_ply, def_ply)

    origin, node_map, nx_cells, ny_cells, nz_cells = build_voxel_lattice_from_vertices(
        vertices,
        cube_size,
    )

    nearest_disp = make_nearest_displacement_func(vertices, displacements)

    mesh = trimesh.load(input_stl)
    if not isinstance(mesh, trimesh.Trimesh):
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
        else:
            print("[FFD] Could not load input STL mesh, aborting deformation.")
            return

    orig_verts = mesh.vertices.copy()
    n_verts = orig_verts.shape[0]
    print(f"[FFD] Deforming input STL with {n_verts} vertices ...")

    # Forward deformation: p' = p + u(p)
    deformed_verts = orig_verts.copy()
    for i in range(n_verts):
        p = orig_verts[i]
        u, has_data = deform_point_trilinear(
            p,
            origin,
            cube_size,
            node_map,
            nx_cells,
            ny_cells,
            nz_cells,
            displacements,
        )
        if not has_data:
            u = nearest_disp(p)
        deformed_verts[i] = p + u

    mesh.vertices = deformed_verts
    mesh.export(output_stl)
    print(f"[FFD] Deformed STL written to: {output_stl}")

    # Inverse deformation: find q so that q + u(q) ≈ p_original
    predeformed_verts = orig_verts.copy()
    print("[FFD] Computing pre-deformed STL (inverse FFD) ...")

    for i in range(n_verts):
        p_target = orig_verts[i]
        q = inverse_deform_point_fixedpoint(
            p_target,
            origin,
            cube_size,
            node_map,
            nx_cells,
            ny_cells,
            nz_cells,
            displacements,
            nearest_disp=nearest_disp,
        )
        predeformed_verts[i] = q

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
            "Voxelize an STL and generate thermal + mechanical CalculiX jobs "
            "with C3D8R/DC3D8 hexahedra and layer-by-layer steady-state "
            "heat-transfer steps, then deform the input STL using the "
            "mechanical displacements as an FFD lattice (forward + pre-deformed) "
            "and optionally export the lattice as separate PLY point clouds."
        )
    )
    parser.add_argument("input_stl", help="Path to input STL file")
    parser.add_argument(
        "job_name",
        help=(
            "Base job name (thermal/mech .inp will be '<job_name>_therm.inp' and "
            "'<job_name>_mech.inp'; CalculiX will produce corresponding .frd)."
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
        help="If set, run CalculiX on the generated jobs.",
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

    # 2) Thermal job
    thermal_job = args.job_name + "_therm"
    thermal_inp = thermal_job + ".inp"

    write_calculix_job(
        thermal_inp,
        vertices,
        hexes,
        slice_to_eids,
        z_slices,
        base_temp=args.base_temp,
        heat_flux=args.heat_flux,
    )

    # 3) Mechanical job
    mech_job = args.job_name + "_mech"
    mech_inp = mech_job + ".inp"

    write_mechanical_job(
        mech_inp,
        vertices,
        hexes,
        base_temp=args.base_temp,
        thermal_job_name=thermal_job,
        n_slices=n_slices,
    )

    # 4) Optional runs + FFD deformation + lattice export
    if args.run_ccx:
        ok_therm = run_calculix(thermal_job, ccx_cmd=args.ccx_cmd)
        if not ok_therm:
            print("[RUN] Thermal job failed, skipping mechanical job and FFD.")
            return
        ok_mech = run_calculix(mech_job, ccx_cmd=args.ccx_cmd)
        if not ok_mech:
            print("[RUN] Mechanical job failed, skipping FFD.")
            return

        mech_frd = mech_job + ".frd"
        if os.path.isfile(mech_frd):
            deformed_stl = args.job_name + "_deformed.stl"
            lattice_basepath = args.job_name + "_lattice" if args.export_lattice else None
            deform_input_stl_with_frd(
                args.input_stl,
                mech_frd,
                vertices,
                args.cube_size,
                deformed_stl,
                lattice_basepath=lattice_basepath,
            )
        else:
            print(f"[FFD] Mechanical FRD '{mech_frd}' not found, skipping STL deformation.")


if __name__ == "__main__":
    main()

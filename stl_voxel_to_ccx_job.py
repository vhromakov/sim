#!/usr/bin/env python3
"""
STL -> voxel C3D8R hex mesh -> global CalculiX .inp job (layered build skeleton)
+ optional run of CalculiX.

Simple layered thermal simulation:
  - Each voxel Z-layer becomes one "printed" layer.
  - For each layer we:
      * Activate that layer's elements (MODEL CHANGE, ADD).
      * Keep previously activated layers "solid".
      * Apply a constant heat flux on the top face of that layer.
      * Hold the base nodes at fixed temperature.
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

    This avoids mechanical DOFs and is robust even for finer voxel sizes
    and disconnected components.
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
#  CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Voxelize an STL and generate a single global CalculiX .inp job "
            "with C3D8R hexahedra and layer-by-layer steady-state "
            "heat-transfer steps (MODEL CHANGE + DFLUX)."
        )
    )
    parser.add_argument("input_stl", help="Path to input STL file")
    parser.add_argument(
        "job_name",
        help=(
            "Job name (output .inp will be '<job_name>.inp', "
            "CalculiX will produce '<job_name>.frd', etc.)"
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

    # 2) Thermal job name
    thermal_job = args.job_name + "_therm"
    thermal_inp = thermal_job + ".inp"

    write_calculix_job(  # <- your thermal-only DC3D8 writer
        thermal_inp,
        vertices,
        hexes,
        slice_to_eids,
        z_slices,
        base_temp=args.base_temp,
        heat_flux=args.heat_flux,
    )

    # 3) Mechanical job name
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

    # 4) Optional runs
    if args.run_ccx:
        ok_therm = run_calculix(thermal_job, ccx_cmd=args.ccx_cmd)
        if not ok_therm:
            print("[RUN] Thermal job failed, skipping mechanical job.")
            return
        run_calculix(mech_job, ccx_cmd=args.ccx_cmd)


if __name__ == "__main__":
    main()

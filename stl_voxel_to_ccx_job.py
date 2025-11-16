#!/usr/bin/env python3
"""
STL -> voxel C3D8R hex mesh -> global CalculiX .inp job (layered build skeleton)

Pipeline:
  1. Voxelize an STL into cubes with pitch = cube_size.
  2. Build a single global C3D8R brick mesh (brick per voxel).
  3. Group elements by Z "slice" and create ELSET=SLICE_xxx.
  4. Detect bottom nodes as BASE NSET.
  5. Write a CalculiX .inp file with:
       * NODE, ELEMENT, ELSETs, NSETs
       * Material stub
       * Initial temperature
       * Multi-step skeleton using *MODEL CHANGE, TYPE=ELEMENT
  6. (Optional) Run CalculiX via --run-ccx.
"""

import argparse
import os
from typing import List, Tuple, Dict

import numpy as np
import trimesh
import subprocess


# ------------------------------------------------------------
# Mesh generation: global voxel C3D8R mesh
# ------------------------------------------------------------

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

    # Load mesh
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

    indices = vox.sparse_indices  # shape (N, 3) with (ix, iy, iz)
    if indices.size == 0:
        print("[VOXEL] No voxels found – check cube size or input mesh.")
        return [], [], {}, []

    total_voxels = indices.shape[0]
    print(f"[VOXEL] Total filled voxels (cubes): {total_voxels}")

    # Sort voxels: by iz (bottom->top), then iy, then ix
    order = np.lexsort((indices[:, 0], indices[:, 1], indices[:, 2]))
    indices_sorted = indices[order]

    # Build mapping from iz -> physical z (center)
    unique_iz = np.unique(indices_sorted[:, 2])
    layer_info = []
    for iz in unique_iz:
        idx_arr = np.array([[0, 0, iz]], dtype=float)
        z_center = float(vox.indices_to_points(idx_arr)[0, 2])
        layer_info.append((iz, z_center))
    # sort by physical z
    layer_info.sort(key=lambda x: x[1])

    # Map each iz to a slice index [0..S-1]
    iz_to_slice: Dict[int, int] = {}
    z_slices: List[float] = []
    for slice_idx, (iz, z_phys) in enumerate(layer_info):
        iz_to_slice[iz] = slice_idx
        z_slices.append(z_phys)

    # Node + element building
    vertex_index_map: Dict[Tuple[int, int, int], int] = {}
    vertices: List[Tuple[float, float, float]] = []
    hexes: List[Tuple[int, int, int, int, int, int, int, int]] = []
    slice_to_eids: Dict[int, List[int]] = {i: [] for i in range(len(z_slices))}

    def get_vertex_index(
        key: Tuple[int, int, int],
        coord: Tuple[float, float, float],
    ) -> int:
        """
        Return 1-based vertex index for given logical key (ix_node, iy_node, iz_node).
        Create new vertex if needed.
        """
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

        # Bottom face (z0)
        v0 = get_vertex_index((ix,   iy,   iz),   (x0, y0, z0))  # (x-,y-,z-)
        v1 = get_vertex_index((ix+1, iy,   iz),   (x1, y0, z0))  # (x+,y-,z-)
        v2 = get_vertex_index((ix+1, iy+1, iz),   (x1, y1, z0))  # (x+,y+,z-)
        v3 = get_vertex_index((ix,   iy+1, iz),   (x0, y1, z0))  # (x-,y+,z-)

        # Top face (z1)
        v4 = get_vertex_index((ix,   iy,   iz+1), (x0, y0, z1))  # (x-,y-,z+)
        v5 = get_vertex_index((ix+1, iy,   iz+1), (x1, y0, z1))  # (x+,y-,z+)
        v6 = get_vertex_index((ix+1, iy+1, iz+1), (x1, y1, z1))  # (x+,y+,z+)
        v7 = get_vertex_index((ix,   iy+1, iz+1), (x0, y1, z1))  # (x-,y+,z+)

        hexes.append((v0, v1, v2, v3, v4, v5, v6, v7))

        eid = len(hexes)  # 1-based element id
        slice_idx = iz_to_slice[int(iz)]
        slice_to_eids[slice_idx].append(eid)

    print(
        f"[VOXEL] Built mesh: {len(vertices)} nodes, "
        f"{len(hexes)} hex elements, {len(z_slices)} slices."
    )
    return vertices, hexes, slice_to_eids, z_slices


# ------------------------------------------------------------
# CalculiX .inp writing
# ------------------------------------------------------------

def _write_id_list_lines(f, ids: List[int], per_line: int = 16):
    """
    Write a list of integer IDs split across multiple lines.
    """
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
):
    """
    Write a single global CalculiX .inp job:

      - *NODE, *ELEMENT, TYPE=C3D8R
      - *ELSET for each slice: SLICE_000, SLICE_001, ...
      - *NSET for base nodes: BASE
      - *MATERIAL stub (edit as needed)
      - *INITIAL CONDITIONS for temperature
      - Multi-step skeleton using *MODEL CHANGE, TYPE=ELEMENT
        (steady-state heat-transfer steps).
    """
    n_nodes = len(vertices)
    n_elems = len(hexes)

    # Find base (lowest z) nodes
    z_coords = np.array([v[2] for v in vertices], dtype=float)
    z_min = float(z_coords.min())
    # tolerance: small fraction of overall height
    tol = (z_coords.max() - z_min + 1e-9) * 1e-3
    base_nodes = [i + 1 for i, z in enumerate(z_coords) if abs(z - z_min) <= tol]

    with open(path, "w", encoding="utf-8") as f:
        f.write("*HEADING\n")
        f.write("Layered voxel C3D8R build (auto-generated)\n")

        # ---- Nodes ----
        f.write("*NODE\n")
        for i, (x, y, z) in enumerate(vertices, start=1):
            f.write(f"{i}, {x}, {y}, {z}\n")

        # ---- Elements ----
        f.write("*ELEMENT, TYPE=C3D8R, ELSET=ALL\n")
        for eid, (n0, n1, n2, n3, n4, n5, n6, n7) in enumerate(hexes, start=1):
            f.write(
                f"{eid}, {n0}, {n1}, {n2}, {n3}, "
                f"{n4}, {n5}, {n6}, {n7}\n"
            )

        # ---- Node sets ----
        f.write("*NSET, NSET=ALLNODES, GENERATE\n")
        f.write(f"1, {n_nodes}, 1\n")

        if base_nodes:
            f.write("*NSET, NSET=BASE\n")
            _write_id_list_lines(f, base_nodes)
        else:
            f.write("** Warning: no base nodes detected.\n")

        # ---- Element sets per slice ----
        n_slices = len(z_slices)
        for slice_idx in range(n_slices):
            eids = slice_to_eids.get(slice_idx, [])
            if not eids:
                continue
            name = f"SLICE_{slice_idx:03d}"
            f.write(f"*ELSET, ELSET={name}\n")
            _write_id_list_lines(f, eids)

        # ---- Material stub ----
        f.write("** --- MATERIAL DEFINITION (edit as needed) ---\n")
        f.write("*MATERIAL, NAME=MAT1\n")
        f.write("*DENSITY\n")
        f.write("7800.\n")
        f.write("*CONDUCTIVITY\n")
        f.write("45.\n")
        f.write("*SPECIFIC HEAT\n")
        f.write("500.\n")
        f.write("*SOLID SECTION, ELSET=ALL, MATERIAL=MAT1\n")

        # ---- Initial temperature ----
        f.write("** --- INITIAL CONDITIONS ---\n")
        f.write("*INITIAL CONDITIONS, TYPE=TEMPERATURE\n")
        f.write(f"ALLNODES, {base_temp}\n")

        # ---- Steady-state heat-transfer steps per slice ----
        f.write("** --- LAYER-BY-LAYER HEAT-TRANSFER STEPS ---\n")

        if n_slices == 0:
            f.write("** No slices found, no steps generated.\n")
        else:
            for slice_idx in range(n_slices):
                name = f"SLICE_{slice_idx:03d}"
                z_val = z_slices[slice_idx]

                f.write("** ----------------------------------------\n")
                f.write(f"** Heat-transfer step for {name} at z = {z_val}\n")
                f.write("*STEP\n")
                f.write("*HEAT TRANSFER, STEADY STATE\n")
                # Optionally an explicit time line; for steady-state,
                # CalculiX will ignore the time but we can still give one:
                f.write("1., 1.\n")

                # Fix base temperature
                if base_nodes:
                    f.write("*BOUNDARY\n")
                    f.write(f"BASE, 11, 11, {base_temp}\n")

                # Model change: first step removes all, then add first slice;
                # subsequent steps just add new slice on top.
                if slice_idx == 0:
                    f.write("*MODEL CHANGE, TYPE=ELEMENT, REMOVE\n")
                    f.write("ALL\n")

                f.write("*MODEL CHANGE, TYPE=ELEMENT, ADD\n")
                f.write(f"{name}\n")

                # Flux placeholder on this slice
                f.write("** Example: uniform heat flux on this slice (edit or remove):\n")
                f.write("** *DFLUX\n")
                f.write(f"** {name}, S1, 1.E6\n")

                f.write("*END STEP\n")

        # Done
    print(f"[CCX] Wrote CalculiX job to: {path}")
    print(f"[CCX] Nodes: {n_nodes}, elements: {n_elems}, slices: {len(z_slices)}")


# ------------------------------------------------------------
# Optional: run CalculiX
# ------------------------------------------------------------

def run_calculix(job_name: str, ccx_cmd: str = "ccx"):
    """
    Run CalculiX on given job (job_name without .inp).

    ccx_cmd: executable name or full path, e.g. "ccx", "ccx_static", "C:\\path\\ccx.exe"
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
        return

    print(f"[RUN] CalculiX return code: {result.returncode}")
    if result.stdout:
        print("----- CalculiX STDOUT -----")
        print(result.stdout)
    if result.stderr:
        print("----- CalculiX STDERR -----")
        print(result.stderr)
    print("[RUN] Done.")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Voxelize an STL and generate a single global CalculiX .inp job "
            "with C3D8R hexahedra and layer-by-layer heat-transfer steps."
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
        "--run-ccx",
        action="store_true",
        help="If set, run CalculiX on the generated job.",
    )
    parser.add_argument(
        "--ccx-cmd",
        default="ccx",
        help="CalculiX executable (default 'ccx'). Example: 'ccx', 'ccx_static', 'C:\\\\ccx\\\\ccx.exe'",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input_stl):
        print(f"Input STL not found: {args.input_stl}")
        raise SystemExit(1)

    # 1) Generate global voxel mesh
    vertices, hexes, slice_to_eids, z_slices = generate_global_cubic_hex_mesh(
        args.input_stl,
        args.cube_size,
    )
    if not vertices or not hexes:
        print("No mesh generated, aborting.")
        raise SystemExit(1)

    # 2) Write CalculiX job
    inp_path = args.job_name + ".inp"
    write_calculix_job(
        inp_path,
        vertices,
        hexes,
        slice_to_eids,
        z_slices,
        base_temp=args.base_temp,
    )

    # 3) Optionally run CalculiX
    if args.run_ccx:
        run_calculix(args.job_name, ccx_cmd=args.ccx_cmd)


if __name__ == "__main__":
    main()

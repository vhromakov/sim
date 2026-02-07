#!/usr/bin/env python3
"""
VTK volumetric mesh (.vtk / .vtu) -> CalculiX .inp
with the SAME layer-by-layer MODEL CHANGE + shrinkage-curve curing logic
as your write_calculix_job().

Key difference:
- In VTK we don't have slices/layers, so we SPLIT elements into layers by
  element centroid along a chosen axis, using a user-provided layer height.

Dependencies:
  pip install numpy trimesh meshio

Notes:
- Only 8-node hex elements are written (C3D8R).
- Supports cell types: "hexahedron" and "voxel" (voxel is reordered to hex order).
- BASE node set is derived robustly from the first layer:
    For each element in the first layer, we pick the quad face with the minimum
    average coordinate along base_face_axis (default: z), and add its nodes to BASE.
  This matches your "outer radial face" idea in unrolled space where offset=Z and wall is Z≈0.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    import meshio
except ImportError as e:
    raise SystemExit(
        "meshio is required. Install with:\n  pip install meshio\n"
        f"Original error: {e}"
    )


def log(msg: str) -> None:
    print(msg, flush=True)


def _axis_index(axis: str) -> int:
    a = axis.lower()
    if a == "x":
        return 0
    if a == "y":
        return 1
    if a == "z":
        return 2
    raise ValueError("axis must be one of: x, y, z")


def _write_id_list_lines(f, ids: Sequence[int], per_line: int = 16) -> None:
    ids = list(ids)
    for i in range(0, len(ids), per_line):
        chunk = ids[i : i + per_line]
        f.write(", ".join(str(x) for x in chunk) + "\n")


# Faces in "Abaqus/CalculiX-friendly" hex ordering (0..7):
# bottom: 0-1-2-3, top: 4-5-6-7, sides...
HEX_FACES = [
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
]


def _voxel_to_hex(conn: np.ndarray) -> np.ndarray:
    """
    VTK_VOXEL node order differs from VTK_HEXAHEDRON.

    Voxel (common): 0,1,2,3,4,5,6,7 with 2/3 and 6/7 swapped vs hex.
    Convert voxel -> hexahedron order:
      hex = [0, 1, 3, 2, 4, 5, 7, 6]
    """
    if conn.shape[-1] != 8:
        raise ValueError("voxel connectivity must have 8 nodes")
    return conn[:, [0, 1, 3, 2, 4, 5, 7, 6]]


@dataclass
class HexMesh:
    points: np.ndarray          # (N,3) float64
    hexes_0based: np.ndarray    # (M,8) int64, in C3D8R-friendly order


def read_vtk_hex_mesh(path: str) -> HexMesh:
    m = meshio.read(path)
    pts = np.asarray(m.points, dtype=np.float64)
    if pts.shape[1] != 3:
        raise ValueError(f"Expected 3D points, got shape {pts.shape}")

    hex_blocks = []
    for cell_block in m.cells:
        ctype = cell_block.type.lower()
        data = np.asarray(cell_block.data, dtype=np.int64)
        if ctype == "hexahedron":
            if data.shape[1] != 8:
                continue
            hex_blocks.append(data)
        elif ctype == "voxel":
            if data.shape[1] != 8:
                continue
            hex_blocks.append(_voxel_to_hex(data))

    if not hex_blocks:
        raise ValueError(
            "No 8-node hexahedra found. Supported cell types: hexahedron, voxel."
        )

    hexes = np.vstack(hex_blocks).astype(np.int64)
    return HexMesh(points=pts, hexes_0based=hexes)


def build_layers_by_height(
    pts: np.ndarray,
    hexes_0based: np.ndarray,
    *,
    layer_axis: str,
    layer_height: float,
    origin: Optional[float] = None,
) -> tuple[Dict[int, List[int]], List[float]]:
    """
    Returns:
      slice_to_eids: dict[layer_idx] -> list of element IDs (1-based)
      layer_positions: list where layer_positions[i] is the layer "z_slices" value for i

    Layering is based on element centroid coordinate along layer_axis.
    """
    if layer_height <= 0:
        raise ValueError("layer_height must be > 0")

    ax = _axis_index(layer_axis)

    # centroid of each element along axis
    elem_pts = pts[hexes_0based]                 # (M,8,3)
    cent = elem_pts.mean(axis=1)                 # (M,3)
    c = cent[:, ax]                              # (M,)

    cmin = float(c.min()) if origin is None else float(origin)
    # numeric stability: small epsilon
    eps = 1e-9 * max(1.0, abs(layer_height))
    layer_idx = np.floor((c - cmin) / layer_height + eps).astype(np.int64)

    slice_to_eids: Dict[int, List[int]] = {}
    for local_ei, li in enumerate(layer_idx):
        eid = local_ei + 1  # 1-based element ID in .inp
        slice_to_eids.setdefault(int(li), []).append(eid)

    # Create a dense layer_positions list from min..max existing indices
    if not slice_to_eids:
        return {}, []

    min_li = min(slice_to_eids.keys())
    max_li = max(slice_to_eids.keys())
    n_layers = max_li - min_li + 1

    # Remap layers to start at 0 (so your SLICE_000 is the first)
    remapped: Dict[int, List[int]] = {}
    for li, eids in slice_to_eids.items():
        remapped[li - min_li] = eids

    # layer "z_slices": use the nominal layer start coordinate (like a layer plane)
    layer_positions = [cmin + (i + 0) * layer_height for i in range(n_layers)]
    return remapped, layer_positions


def compute_base_nodes_from_first_layer(
    pts: np.ndarray,
    hexes_1based: List[Tuple[int, int, int, int, int, int, int, int]],
    first_layer_eids: List[int],
    *,
    base_face_axis: str = "z",
) -> List[int]:
    """
    Robust replacement for "nodes 4..7" assumption:
    - For each element in the first layer, choose the quad face that has MINIMUM
      average coordinate along base_face_axis (default Z).
    - Add those face nodes to BASE.
    """
    ax = _axis_index(base_face_axis)
    base: Set[int] = set()

    V = pts  # 0-based points for coordinate lookup

    for eid in first_layer_eids:
        conn_1based = hexes_1based[eid - 1]
        conn0 = [n - 1 for n in conn_1based]  # to 0-based for indexing pts

        best_face = None
        best_val = None
        for face in HEX_FACES:
            face_nodes0 = [conn0[i] for i in face]
            avg = float(V[face_nodes0, ax].mean())
            if best_val is None or avg < best_val:
                best_val = avg
                best_face = face

        assert best_face is not None
        for i in best_face:
            base.add(conn_1based[i])

    return sorted(base)


def write_calculix_job_from_layers(
    path: str,
    vertices: List[Tuple[float, float, float]],
    hexes: List[Tuple[int, int, int, int, int, int, int, int]],
    slice_to_eids: Dict[int, List[int]],
    z_slices: List[float],
    shrinkage_curve: List[float],
    cure_shrink_per_unit: float,
    output_stride: int = 1,
    *,
    base_nodes: Optional[List[int]] = None,
) -> None:
    """
    Same logic as your function, but:
    - BASE can be passed in (recommended, computed robustly from VTK).
    """
    n_nodes = len(vertices)
    n_elems = len(hexes)
    n_slices = len(z_slices)

    time_per_layer = 1.0
    time_per_layer_step = 1.0

    total_weight = float(sum(shrinkage_curve)) if shrinkage_curve else 1.0
    if total_weight == 0:
        total_weight = 1.0
    shrinkage_curve = [float(w) / total_weight for w in shrinkage_curve]

    if output_stride < 1:
        output_stride = 1

    with open(path, "w", encoding="utf-8") as f:
        f.write("**\n** Auto-generated incremental-cure shrink job\n**\n")
        f.write("*HEADING\n")
        f.write(
            "VTK C3D8R uncoupled temperature-displacement "
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

        f.write("** Element + node sets (per slice) +++++++++++++++++++++++++\n")
        slice_names: List[str] = []
        slice_node_ids: Dict[int, List[int]] = {}

        for slice_idx in range(n_slices):
            eids = slice_to_eids.get(slice_idx, [])
            if not eids:
                continue
            valid_eids = [eid for eid in eids if 1 <= eid <= n_elems]
            if not valid_eids:
                log(f"[WARN] Slice {slice_idx} has no valid element IDs within 1..{n_elems}")
                continue

            name = f"SLICE_{slice_idx:03d}"
            slice_names.append(name)
            f.write(f"*ELSET, ELSET={name}\n")
            _write_id_list_lines(f, valid_eids)

            nodes_in_slice: Set[int] = set()
            for eid in valid_eids:
                nodes_in_slice.update(hexes[eid - 1])

            node_list = sorted(nodes_in_slice)
            slice_node_ids[slice_idx] = node_list

            f.write(f"*NSET, NSET={name}_NODES\n")
            _write_id_list_lines(f, node_list)

        # -------------------- BASE --------------------
        if base_nodes is None:
            f.write("** Warning: BASE node set not provided.\n")
            base_nodes = []
        else:
            f.write("** Base node set +++++++++++++++++++++++++++++++++++++++++++\n")
            f.write("*NSET, NSET=BASE\n")
            _write_id_list_lines(f, base_nodes)

        # -------------------- MATERIAL --------------------
        f.write("** Materials +++++++++++++++++++++++++++++++++++++++++++++++\n")
        f.write("*MATERIAL, NAME=ABS\n")
        f.write("*DENSITY\n")
        f.write("1.12e-9\n")
        f.write("*ELASTIC\n")
        f.write("2800., 0.35\n")

        alpha = -float(cure_shrink_per_unit)  # higher T -> shrink
        f.write("*EXPANSION, ZERO=0.\n")
        f.write(f"{alpha:.6E}\n")

        f.write("*CONDUCTIVITY\n")
        f.write("0.20\n")
        f.write("*SPECIFIC HEAT\n")
        f.write("1.30e+9\n")
        f.write("*SOLID SECTION, ELSET=ALL, MATERIAL=ABS\n")

        # -------------------- INITIAL CONDITIONS --------------------
        f.write("** Initial conditions (cure variable) ++++++++++++++++++++++\n")
        f.write("*INITIAL CONDITIONS, TYPE=TEMPERATURE\n")
        f.write("ALLNODES, 0.0\n")

        # -------------------- STEPS --------------------
        f.write("** Steps +++++++++++++++++++++++++++++++++++++++++++++++++++\n")
        if n_slices == 0 or not slice_names:
            f.write("** No slices -> no steps.\n")
        else:
            existing_slice_idxs = sorted(slice_node_ids.keys())
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
                    z_val = z_slices[slice_to_add] if slice_to_add < len(z_slices) else float("nan")
                    f.write(
                        f"** Step {step_counter}: add slice {name} at layer_pos = {z_val} "
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
                        remove = [f"SLICE_{other:03d}" for other in existing_slice_idxs if other != slice_to_add]
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
                    f.write("** Field outputs disabled for this step\n")
                    f.write("*NODE FILE\n")  # wipe selection

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

    log(f"[CCX] Wrote incremental-cure UT-D job to: {path}")
    log(
        f"[CCX] Nodes: {n_nodes}, elements: {n_elems}, "
        f"slices: {n_slices}, shrinkage_curve={shrinkage_curve}, "
        f"cure_shrink_per_unit={cure_shrink_per_unit}"
    )


def parse_curve(s: str) -> List[float]:
    # Accept: "5,4,3,2,1" or "1 0.5 0.25"
    parts = [p.strip() for p in s.replace(",", " ").split()]
    if not parts:
        return [1.0]
    return [float(x) for x in parts]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_vtk", help="input .vtk/.vtu volumetric mesh with hexes")
    ap.add_argument("output_inp", help="output CalculiX .inp")

    ap.add_argument("--layer-height", type=float, required=True, help="layer height for splitting elements")
    ap.add_argument("--layer-axis", choices=["x", "y", "z"], default="y",
                    help="axis along which layers are defined (default: y)")

    ap.add_argument("--layer-origin", type=float, default=None,
                    help="optional origin coordinate along layer-axis. Default = min element centroid.")
    ap.add_argument("--base-face-axis", choices=["x", "y", "z"], default="z",
                    help="axis used to pick BASE faces in first layer (min avg along this axis). Default: z")

    ap.add_argument("--shrinkage-curve", default="5,4,3,2,1",
                    help="comma/space-separated weights per layer-cure substep (default: 5,4,3,2,1)")
    ap.add_argument("--cure-shrink-per-unit", type=float, required=True,
                    help="shrink strain per unit cure (mapped via *EXPANSION)")
    ap.add_argument("--output-stride", type=int, default=1,
                    help="write node outputs every N curing steps (default: 1)")

    args = ap.parse_args()

    mesh = read_vtk_hex_mesh(args.input_vtk)

    slice_to_eids, layer_positions = build_layers_by_height(
        mesh.points,
        mesh.hexes_0based,
        layer_axis=args.layer_axis,
        layer_height=args.layer_height,
        origin=args.layer_origin,
    )

    # Convert points + hexes to the exact format your writer expects
    vertices = [tuple(map(float, p)) for p in mesh.points.tolist()]
    hexes_1based = [(int(a)+1, int(b)+1, int(c)+1, int(d)+1, int(e)+1, int(f)+1, int(g)+1, int(h)+1)
                    for (a,b,c,d,e,f,g,h) in mesh.hexes_0based.tolist()]

    # BASE nodes from first layer (layer 0 after remap)
    first_layer_eids = slice_to_eids.get(0, [])
    base_nodes = compute_base_nodes_from_first_layer(
        mesh.points,
        hexes_1based,
        first_layer_eids,
        base_face_axis=args.base_face_axis,
    )

    curve = parse_curve(args.shrinkage_curve)

    write_calculix_job_from_layers(
        path=args.output_inp,
        vertices=vertices,
        hexes=hexes_1based,
        slice_to_eids=slice_to_eids,
        z_slices=layer_positions,
        shrinkage_curve=curve,
        cure_shrink_per_unit=args.cure_shrink_per_unit,
        output_stride=args.output_stride,
        base_nodes=base_nodes,
    )


if __name__ == "__main__":
    main()

# python vtk_to_ccx_inp.py asmoOctree_0.vtk job.inp --layer-height 1 --layer-axis z --base-face-axis z --cure-shrink-per-unit 0.05 --shrinkage-curve 5,4,3,2,1 --output-stride 1
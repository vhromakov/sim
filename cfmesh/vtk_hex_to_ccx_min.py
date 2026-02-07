#!/usr/bin/env python3
"""
Minimal VTK (UnstructuredGrid) -> CalculiX .inp (C3D8) writer.

- Reads .vtu (XML) or legacy .vtk that contains hex cells
- Writes nodes + C3D8 elements
- Applies a simple constraint:
    * If a "base plane" is detected at min-Y (enough nodes), fixes that plane (1..3)
    * Otherwise uses 3-node constraints to remove rigid body motion
- Applies a tiny nodal load at the max-Y node so the run produces non-zero results

Usage:
  python vtk_hex_to_ccx_min.py mesh.vtu job.inp
  python vtk_hex_to_ccx_min.py mesh.vtk job.inp --reorder
"""

import argparse
import sys
from typing import List, Tuple

import numpy as np

try:
    import vtk  # pip install vtk
except ImportError as e:
    raise SystemExit("ERROR: VTK python package not found. Install with: pip install vtk") from e


VOXEL_TO_HEX = [0, 1, 3, 2, 4, 5, 7, 6]  # VTK_VOXEL point order -> VTK_HEXAHEDRON order


def read_unstructured_grid(path: str) -> "vtk.vtkUnstructuredGrid":
    p = path.lower()
    if p.endswith(".vtu"):
        r = vtk.vtkXMLUnstructuredGridReader()
        r.SetFileName(path)
        r.Update()
        ug = r.GetOutput()
    else:
        # Legacy .vtk (could store various dataset types)
        r = vtk.vtkGenericDataObjectReader()
        r.SetFileName(path)
        r.Update()
        ug = vtk.vtkUnstructuredGrid.SafeDownCast(r.GetOutput())

    if ug is None:
        raise RuntimeError("Input is not a vtkUnstructuredGrid (or could not be read).")
    return ug


def reorder_hex_by_geometry(xyz8: np.ndarray) -> Tuple[List[int], bool]:
    """
    Best-effort reorder of an 8-node hex to Abaqus/CalculiX C3D8 order using local PCA axes.

    Returns:
        (order_indices, ok)
        where order_indices are indices 0..7 into xyz8 for nodes 1..8
    """
    c = xyz8.mean(axis=0)
    X = xyz8 - c
    # PCA axes
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    ex, ey, ez = vt[0], vt[1], vt[2]

    # enforce right-handed basis
    if np.dot(np.cross(ex, ey), ez) < 0:
        ez = -ez

    P = np.stack([X @ ex, X @ ey, X @ ez], axis=1)

    # classify corners by sign
    signs = []
    for i in range(8):
        sx = 1 if P[i, 0] >= 0 else -1
        sy = 1 if P[i, 1] >= 0 else -1
        sz = 1 if P[i, 2] >= 0 else -1
        signs.append((sx, sy, sz))

    # Abaqus C3D8 corner sign targets:
    targets = [
        (-1, -1, -1),  # 1
        (+1, -1, -1),  # 2
        (+1, +1, -1),  # 3
        (-1, +1, -1),  # 4
        (-1, -1, +1),  # 5
        (+1, -1, +1),  # 6
        (+1, +1, +1),  # 7
        (-1, +1, +1),  # 8
    ]

    mapping = {}
    for idx, s in enumerate(signs):
        mapping.setdefault(s, []).append(idx)

    order = []
    for t in targets:
        cand = mapping.get(t, [])
        if len(cand) != 1:
            return list(range(8)), False  # ambiguous / degenerate
        order.append(cand[0])

    return order, True


def write_id_list(f, ids: List[int], per_line: int = 16):
    for i in range(0, len(ids), per_line):
        chunk = ids[i : i + per_line]
        f.write(" " + ", ".join(str(x) for x in chunk) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_vtk", help="Input .vtu or legacy .vtk (UnstructuredGrid with hex cells)")
    ap.add_argument("output_inp", help="Output CalculiX .inp file")
    ap.add_argument("--reorder", action="store_true", help="Reorder each hex to C3D8 order via geometry (best-effort)")
    ap.add_argument("--force", type=float, default=1.0, help="Nodal force magnitude applied at max-Y node (default 1.0)")
    args = ap.parse_args()

    ug = read_unstructured_grid(args.input_vtk)

    pts_vtk = ug.GetPoints()
    if pts_vtk is None or pts_vtk.GetNumberOfPoints() == 0:
        raise SystemExit("ERROR: no points in the VTK grid.")

    n_points = pts_vtk.GetNumberOfPoints()
    pts = np.array([pts_vtk.GetPoint(i) for i in range(n_points)], dtype=float)

    # Collect hexes
    hexes: List[Tuple[int, int, int, int, int, int, int, int]] = []
    skipped = 0

    for ci in range(ug.GetNumberOfCells()):
        ct = ug.GetCellType(ci)
        if ct not in (vtk.VTK_HEXAHEDRON, vtk.VTK_VOXEL):
            skipped += 1
            continue

        cell = ug.GetCell(ci)
        if cell.GetNumberOfPoints() != 8:
            skipped += 1
            continue

        ids = [cell.GetPointId(k) for k in range(8)]
        if ct == vtk.VTK_VOXEL:
            ids = [ids[k] for k in VOXEL_TO_HEX]

        if args.reorder:
            xyz8 = pts[np.array(ids, dtype=int)]
            order, ok = reorder_hex_by_geometry(xyz8)
            if ok:
                ids = [ids[k] for k in order]

        # 1-based for CalculiX
        hexes.append(tuple(i + 1 for i in ids))

    if not hexes:
        raise SystemExit("ERROR: no hex/voxel cells found to export.")
    if skipped:
        print(f"[INFO] Skipped non-hex cells: {skipped}", file=sys.stderr)
    print(f"[INFO] Exporting nodes={n_points}, hexes={len(hexes)}", file=sys.stderr)

    # Find a "base plane" at min-Y
    y = pts[:, 1]
    y_min, y_max = float(y.min()), float(y.max())
    span = max(1e-12, y_max - y_min)
    tol = 1e-6 * span
    base_nodes = np.where(y <= y_min + tol)[0] + 1  # 1-based

    # Load node at max-Y
    load_node = int(np.argmax(y) + 1)

    # If base plane is tiny, fall back to 3-node constraints
    use_plane_fix = len(base_nodes) >= 10

    if not use_plane_fix:
        # pick 3 well-separated nodes to remove rigid motion
        na = int(np.argmin(pts[:, 0] + pts[:, 1] + pts[:, 2]) + 1)
        a = pts[na - 1]
        nb = int(np.argmax(np.linalg.norm(pts - a, axis=1)) + 1)
        b = pts[nb - 1]
        ab = b - a
        abn = np.linalg.norm(ab)
        if abn < 1e-12:
            nc = int(np.argmax(pts[:, 2]) + 1)
        else:
            d = np.linalg.norm(np.cross(pts - a, ab), axis=1) / abn
            nc = int(np.argmax(d) + 1)

    # Write INP
    with open(args.output_inp, "w", encoding="utf-8", newline="\n") as f:
        f.write("*HEADING\n")
        f.write("VTK hex mesh -> minimal CalculiX job\n")
        f.write("*PREPRINT, ECHO=NO, MODEL=NO, HISTORY=NO, CONTACT=NO\n")

        f.write("*NODE\n")
        for i in range(n_points):
            x, yy, z = pts[i]
            f.write(f"{i+1}, {x:.9g}, {yy:.9g}, {z:.9g}\n")

        f.write("*ELEMENT, TYPE=C3D8, ELSET=EALL\n")
        for eid, conn in enumerate(hexes, start=1):
            # if eid in [3110, 177960, 188776, 85252, 160012, 150323]: # vh
            if eid in [23705, 65824, 71981, 56389, 79267]: # vh
            # if eid in [27793, 112596, 50248, 29538, 73994, 54702,
            #            57145, 120703, 78850, 79081, 79999, 81013]: # vh
                continue # vh

            f.write(f"{eid}, {', '.join(map(str, conn))}\n")

        f.write("*MATERIAL, NAME=MAT1\n")
        f.write("*ELASTIC\n")
        f.write("2.0e9, 0.30\n")
        f.write("*SOLID SECTION, ELSET=EALL, MATERIAL=MAT1\n")

        f.write("*NSET, NSET=LOAD\n")
        f.write(f" {load_node}\n")

        if use_plane_fix:
            f.write("*NSET, NSET=FIX\n")
            write_id_list(f, base_nodes.tolist())
            f.write("*BOUNDARY\n")
            f.write("FIX, 1, 3, 0.\n")
        else:
            f.write("*BOUNDARY\n")
            f.write(f"{na}, 1, 3, 0.\n")
            f.write(f"{nb}, 2, 3, 0.\n")
            f.write(f"{nc}, 3, 3, 0.\n")

        f.write("*STEP\n")
        f.write("*STATIC\n")
        f.write("*CLOAD\n")
        # load in +Y direction by default (flip sign if you want the opposite)
        f.write(f"LOAD, 2, {args.force:.9g}\n")

        f.write("*NODE FILE\n")
        f.write("U\n")
        f.write("*EL FILE\n")
        f.write("S, E\n")
        f.write("*END STEP\n")

    print(f"[OK] Wrote: {args.output_inp}", file=sys.stderr)


if __name__ == "__main__":
    main()

"""
python vtk_hex_to_ccx_min.py "C:\cfMesh-v1.1.1\MSYS\home\4y5t6\tutorials\cartesianMesh\asmoOctree\VTK\asmoOctree_0.vtk" job.inp
"""
#!/usr/bin/env python3
"""
vtk2ccx_simple.py

Reads a VTK mesh and writes a very simple single-step CalculiX/Abaqus .inp job:
- Supports:
  * Legacy VTK (.vtk) Unstructured Grid, ASCII (fallback parser)
  * XML VTK (.vtu) if `meshio` is installed
- Converts:
  * VTK_HEXAHEDRON (12) -> C3D8
  * VTK_TETRA (10)      -> C3D4
- Creates:
  * *NODE, *ELEMENT
  * bottom node set FIXED (min axis band)
  * *STATIC step with GRAV

Usage:
  python vtk2ccx_simple.py input.vtk job.inp
  python vtk2ccx_simple.py input.vtu job.inp

Notes:
- Element node ordering:
  VTK hex ordering matches C3D8 for standard VTK hex (0..7) in most cases.
  Tetra ordering generally matches C3D4 too.
"""

from __future__ import annotations
import argparse
import math
import os
from typing import Iterator, List, Tuple, Dict, Optional


VTK_TETRA = 10
VTK_HEX = 12


def token_stream_from_text_file(path: str) -> Iterator[str]:
    """Streaming whitespace token generator (memory-friendly for large ASCII VTK)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # keep things simple: split by whitespace
            for tok in line.split():
                yield tok


def read_legacy_vtk_unstructured_ascii(path: str) -> Tuple[List[Tuple[float, float, float]],
                                                          List[List[int]],
                                                          List[int]]:
    """
    Minimal legacy VTK ASCII parser for UNSTRUCTURED_GRID:
      POINTS n <type>
      CELLS ncells totalSize
      CELL_TYPES ncells
    Returns (points, cells_connectivity, cell_types)
    """
    ts = token_stream_from_text_file(path)

    # Scan tokens until POINTS
    for tok in ts:
        if tok.upper() == "POINTS":
            break
    else:
        raise ValueError("Could not find POINTS section. Is this a legacy VTK ASCII file?")

    n_points = int(next(ts))
    _vtk_dtype = next(ts)  # float/double/etc; ignore
    pts: List[Tuple[float, float, float]] = []
    for _ in range(n_points):
        x = float(next(ts)); y = float(next(ts)); z = float(next(ts))
        pts.append((x, y, z))

    # Scan until CELLS
    for tok in ts:
        if tok.upper() == "CELLS":
            break
    else:
        raise ValueError("Could not find CELLS section.")

    n_cells = int(next(ts))
    _total_size = int(next(ts))  # ignore
    cells: List[List[int]] = []
    for _ in range(n_cells):
        k = int(next(ts))
        conn = [int(next(ts)) for _ in range(k)]
        cells.append(conn)

    # Scan until CELL_TYPES
    for tok in ts:
        if tok.upper() == "CELL_TYPES":
            break
    else:
        raise ValueError("Could not find CELL_TYPES section.")

    n_types = int(next(ts))
    if n_types != n_cells:
        # Some files can be weird, but usually they match.
        raise ValueError(f"CELL_TYPES count ({n_types}) != CELLS count ({n_cells})")

    cell_types = [int(next(ts)) for _ in range(n_types)]
    return pts, cells, cell_types


def try_read_with_meshio(path: str):
    """Try reading via meshio (handles .vtu and many other VTK variants)."""
    try:
        import meshio  # type: ignore
    except Exception:
        return None

    m = meshio.read(path)

    points = [(float(p[0]), float(p[1]), float(p[2])) for p in m.points]
    # meshio stores cells as a list of CellBlock(type, data)
    # We'll collect tetra + hexa
    hexes: List[Tuple[int, int, int, int, int, int, int, int]] = []
    tets: List[Tuple[int, int, int, int]] = []

    for block in m.cells:
        ctype = block.type.lower()
        data = block.data
        if ctype in ("hexahedron", "hex"):
            for row in data:
                hexes.append(tuple(int(i) for i in row[:8]))  # type: ignore
        elif ctype in ("tetra", "tetrahedron"):
            for row in data:
                tets.append(tuple(int(i) for i in row[:4]))  # type: ignore

    return points, hexes, tets


def write_id_list(f, ids: List[int], per_line: int = 16):
    """Write IDs in Abaqus list style (comma separated)."""
    for i in range(0, len(ids), per_line):
        chunk = ids[i:i + per_line]
        f.write(", ".join(str(x) for x in chunk))
        f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_vtk", help="Input VTK (.vtk legacy ASCII or .vtu if meshio installed)")
    ap.add_argument("output_inp", help="Output CalculiX/Abaqus .inp")
    ap.add_argument("--E", type=float, default=2000.0, help="Young's modulus")
    ap.add_argument("--nu", type=float, default=0.35, help="Poisson ratio")
    ap.add_argument("--rho", type=float, default=1.0e-9, help="Density (consistent units!)")
    ap.add_argument("--g", type=float, default=9.81, help="Gravity magnitude")
    ap.add_argument("--gdir", type=str, default="0,0,-1", help="Gravity direction vector, e.g. 0,0,-1")
    ap.add_argument("--fix-axis", choices=["x", "y", "z"], default="z", help="Axis to define bottom fixation band")
    ap.add_argument("--fix-frac", type=float, default=0.001,
                    help="Fix band thickness as fraction of bbox size along fix-axis (default 0.1%)")
    args = ap.parse_args()

    inpath = args.input_vtk
    outpath = args.output_inp

    # First try meshio if available (especially for .vtu)
    meshio_res = try_read_with_meshio(inpath)
    if meshio_res is not None:
        points, hexes0, tets0 = meshio_res
        # meshio indices are 0-based; keep as 0-based for now
        hexes = [tuple(h) for h in hexes0]
        tets = [tuple(t) for t in tets0]
    else:
        # Fallback: legacy VTK ASCII parser
        # This will work for many foamToVTK outputs saved as legacy .vtk ASCII.
        points, cells, cell_types = read_legacy_vtk_unstructured_ascii(inpath)
        hexes = []
        tets = []
        for conn, ct in zip(cells, cell_types):
            if ct == VTK_HEX and len(conn) == 8:
                hexes.append(tuple(conn))  # type: ignore
            elif ct == VTK_TETRA and len(conn) == 4:
                tets.append(tuple(conn))  # type: ignore

    if not hexes and not tets:
        raise SystemExit("No supported HEX(12) or TET(10) cells found. (Or meshio not installed for this format.)")

    # Compute bottom node set
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]

    if args.fix_axis == "x":
        arr = xs
    elif args.fix_axis == "y":
        arr = ys
    else:
        arr = zs

    amin = min(arr)
    amax = max(arr)
    span = max(amax - amin, 0.0)
    band = args.fix_frac * span
    # if span is ~0, still catch something
    if band <= 0.0:
        band = 1e-9

    fixed_nodes_1based: List[int] = []
    for i, val in enumerate(arr):
        if val <= amin + band + 1e-15:
            fixed_nodes_1based.append(i + 1)

    # Parse gravity direction
    gdx, gdy, gdz = (float(s) for s in args.gdir.split(","))
    norm = math.sqrt(gdx*gdx + gdy*gdy + gdz*gdz)
    if norm == 0:
        raise SystemExit("Invalid --gdir (zero vector).")
    gdx /= norm; gdy /= norm; gdz /= norm

    # Write .inp
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("*HEADING\n")
        f.write(f"** Simple VTK -> CalculiX job\n")
        f.write(f"** Input: {os.path.basename(inpath)}\n")
        f.write("**\n")
        f.write("*PREPRINT, ECHO=NO, MODEL=NO, HISTORY=NO, CONTACT=NO\n")

        # Nodes (1-based)
        f.write("*NODE\n")
        for i, (x, y, z) in enumerate(points, start=1):
            f.write(f"{i}, {x:.9g}, {y:.9g}, {z:.9g}\n")

        # Elements
        eid = 1
        elsets: List[str] = []

        if hexes:
            elsets.append("EHEX")
            f.write("*ELEMENT, TYPE=C3D8, ELSET=EHEX\n")
            # VTK hex ordering usually matches C3D8 (1..4 bottom, 5..8 top)
            for h in hexes:
                n = [int(v) + 1 for v in h]  # to 1-based
                f.write(f"{eid}, {n[0]}, {n[1]}, {n[2]}, {n[3]}, {n[4]}, {n[5]}, {n[6]}, {n[7]}\n")
                eid += 1

        if tets:
            elsets.append("ETET")
            f.write("*ELEMENT, TYPE=C3D4, ELSET=ETET\n")
            for t in tets:
                n = [int(v) + 1 for v in t]
                f.write(f"{eid}, {n[0]}, {n[1]}, {n[2]}, {n[3]}\n")
                eid += 1

        # Create a combined set for loads/section
        f.write("*ELSET, ELSET=EALL\n")
        if len(elsets) == 1:
            f.write(f"{elsets[0]}\n")
        else:
            # Abaqus allows referencing sets by name in *ELSET with GENERATE too,
            # but simplest: union by listing set names in multiple lines
            for s in elsets:
                f.write(f"{s}\n")

        # Material + section
        f.write("*MATERIAL, NAME=MAT1\n")
        f.write("*ELASTIC\n")
        f.write(f"{args.E}, {args.nu}\n")
        f.write("*DENSITY\n")
        f.write(f"{args.rho}\n")
        f.write("*SOLID SECTION, ELSET=EALL, MATERIAL=MAT1\n")
        f.write("1.\n")

        # Fixed nodes
        f.write("*NSET, NSET=FIXED\n")
        write_id_list(f, fixed_nodes_1based, per_line=16)

        # Step (single static)
        f.write("*STEP\n")
        f.write("*STATIC\n")
        f.write("1., 1., 1e-05, 1.\n")

        f.write("*BOUNDARY\n")
        # FIXED, dof1..dof3, value0
        f.write("FIXED, 1, 3, 0.\n")

        f.write("*DLOAD\n")
        # EALL, GRAV, g, dirx, diry, dirz
        f.write(f"EALL, GRAV, {args.g}, {gdx}, {gdy}, {gdz}\n")

        f.write("*NODE FILE\n")
        f.write("U\n")
        f.write("*EL FILE\n")
        f.write("S, E\n")
        f.write("*END STEP\n")

    print(f"Wrote: {outpath}")
    print(f"Nodes: {len(points)} | HEX: {len(hexes)} | TET: {len(tets)} | FIXED nodes: {len(fixed_nodes_1based)}")


if __name__ == "__main__":
    main()

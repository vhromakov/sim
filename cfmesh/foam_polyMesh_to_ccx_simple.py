#!/usr/bin/env python3
"""
OpenFOAM constant/polyMesh (points/faces/owner/neighbour/boundary, possibly .gz)
-> CalculiX .inp (single-step uniform shrink)

- Reads ASCII OpenFOAM mesh files (optionally gzipped).
- Extracts ONLY "true" hex cells:
    * cell has exactly 6 faces
    * each face has 4 vertices
    * cell has exactly 8 unique vertices
- Orders C3D8 nodes by centroid-octant classification (works well for cartesian meshes).
- Writes a minimal .inp:
    *NODE, *ELEMENT C3D8R
    NSET=ALLNODES, NSET=BASE
    One step with *UNCOUPLED TEMPERATURE-DISPLACEMENT:
      BASE fixed, ALLNODES DOF 11 set to 1.0, shrink via *EXPANSION alpha=-cure_shrink_per_unit

Notes:
- If your polyMesh is written in binary, this script will fail (ASCII expected).
- If cfMesh produced polyhedral transition cells near boundaries, they are skipped.
- BASE can be selected either from a boundary patch (recommended) or by min coordinate plane.

Deps:
  pip install numpy
"""

from __future__ import annotations

import argparse
import gzip
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def log(msg: str) -> None:
    print(msg, flush=True)


# Canonical C3D8 order in terms of centroid octants:
# bit = (x>cx)*1 + (y>cy)*2 + (z>cz)*4
# C3D8 nodes: 0(-,-,-)=0, 1(+,-,-)=1, 2(+,+,-)=3, 3(-,+,-)=2, 4(-,-,+)=4, 5(+,-,+)=5, 6(+,+,+)=7, 7(-,+,+)=6
_CANON_BITS = np.array([0, 1, 3, 2, 4, 5, 7, 6], dtype=np.int64)


def _axis_index(axis: str) -> int:
    a = axis.lower()
    if a == "x":
        return 0
    if a == "y":
        return 1
    if a == "z":
        return 2
    raise ValueError("axis must be one of x/y/z")


def read_text_maybe_gz(path: str) -> str:
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            return f.read()
    with open(path, "rt", encoding="utf-8", errors="ignore") as f:
        return f.read()


def strip_foam_comments(text: str) -> str:
    # remove /* ... */ blocks and // line comments
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*?$", "", text, flags=re.M)
    return text


def find_outer_list(text: str) -> Tuple[int, int, int]:
    """
    Find:
      count N
      index of outer '(' (start)
      index of matching ')' (end)
    Returns: (N, start_paren_index, end_paren_index)
    """
    t = strip_foam_comments(text)

    # Try pattern: "N(" or "N ("
    m = re.search(r"(?m)^\s*(\d+)\s*\(\s*$", t)
    if m:
        N = int(m.group(1))
        start = t.find("(", m.end() - 1)
        end = match_paren(t, start)
        return N, start, end

    # Pattern: "N" alone on a line, then later '('
    for m2 in re.finditer(r"(?m)^\s*(\d+)\s*$", t):
        N = int(m2.group(1))
        j = m2.end()
        # find next non-whitespace
        while j < len(t) and t[j].isspace():
            j += 1
        if j < len(t) and t[j] == "(":
            start = j
            end = match_paren(t, start)
            return N, start, end

    raise ValueError("Could not locate OpenFOAM list header 'N (...)' in file.")


def match_paren(text: str, start: int) -> int:
    """Return index of the ')' that matches the '(' at start."""
    if start < 0 or start >= len(text) or text[start] != "(":
        raise ValueError("match_paren: start is not '('")
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("Unbalanced parentheses in OpenFOAM file.")


def parse_points(path: str) -> np.ndarray:
    text = read_text_maybe_gz(path)
    N, start, end = find_outer_list(text)
    block = strip_foam_comments(text)[start + 1 : end]
    # Replace parentheses with spaces and parse floats
    trans = block.translate(str.maketrans({"(": " ", ")": " "}))
    arr = np.fromstring(trans, sep=" ", dtype=np.float64)
    if arr.size != 3 * N:
        raise ValueError(f"points: expected {3*N} floats, got {arr.size} (is this ASCII?)")
    pts = arr.reshape((-1, 3))
    if not np.isfinite(pts).all():
        bad = np.where(~np.isfinite(pts).all(axis=1))[0][:10]
        raise ValueError(f"points contain NaN/Inf. Example indices: {bad.tolist()}")
    return pts


@dataclass
class FacesRagged:
    tok: np.ndarray         # all ints: [k v0 v1 ... vk-1 k v0 ...]
    start: np.ndarray       # start index of vertices inside tok for each face
    length: np.ndarray      # k per face


def parse_faces(path: str) -> FacesRagged:
    text = read_text_maybe_gz(path)
    N, start, end = find_outer_list(text)
    block = strip_foam_comments(text)[start + 1 : end]
    # Convert parentheses to spaces; parse all ints (includes k and vertex indices)
    trans = block.translate(str.maketrans({"(": " ", ")": " "}))
    tok = np.fromstring(trans, sep=" ", dtype=np.int64)
    if tok.size == 0:
        raise ValueError("faces: no integers parsed (is this ASCII?)")

    starts = np.empty(N, dtype=np.int64)
    lens = np.empty(N, dtype=np.int32)

    i = 0
    for f in range(N):
        if i >= tok.size:
            raise ValueError(f"faces: ran out of tokens at face {f}/{N}")
        k = int(tok[i])
        i += 1
        lens[f] = k
        starts[f] = i
        i += k

    return FacesRagged(tok=tok, start=starts, length=lens)


def parse_label_list(path: str) -> np.ndarray:
    text = read_text_maybe_gz(path)
    N, start, end = find_outer_list(text)
    block = strip_foam_comments(text)[start + 1 : end]
    arr = np.fromstring(block, sep=" ", dtype=np.int64)
    if arr.size != N:
        raise ValueError(f"{os.path.basename(path)}: expected {N} ints, got {arr.size} (is this ASCII?)")
    return arr


def parse_boundary(path: str) -> Dict[str, Tuple[int, int]]:
    """
    Returns dict: patch_name -> (startFace, nFaces)
    """
    text = read_text_maybe_gz(path)
    N, start, end = find_outer_list(text)
    block = strip_foam_comments(text)[start + 1 : end]

    # Each entry: name { ... startFace X; nFaces Y; ... }
    patches: Dict[str, Tuple[int, int]] = {}
    # Non-greedy match per patch block
    for m in re.finditer(r"(?ms)^\s*([A-Za-z0-9_:\-\.]+)\s*\{(.*?)\}", block):
        name = m.group(1)
        body = m.group(2)
        sf = re.search(r"\bstartFace\s+(\d+)\s*;", body)
        nf = re.search(r"\bnFaces\s+(\d+)\s*;", body)
        if sf and nf:
            patches[name] = (int(sf.group(1)), int(nf.group(1)))

    # boundary file says N patches, but some parsers may miss weird names; that's ok
    return patches


def face_vertices(faces: FacesRagged, face_id: int) -> np.ndarray:
    s = int(faces.start[face_id])
    k = int(faces.length[face_id])
    return faces.tok[s : s + k]


def build_hex_elements(
    points: np.ndarray,
    faces: FacesRagged,
    owner: np.ndarray,
    neighbour: np.ndarray,
) -> List[np.ndarray]:
    """
    Returns list of hex element connectivity (0-based point indices) in canonical C3D8 order.
    Skips non-hex cells.
    """
    nFaces = owner.size
    nInternal = neighbour.size
    nCells = int(max(owner.max(initial=0), neighbour.max(initial=0)) + 1)

    cell_faces: List[List[int]] = [[] for _ in range(nCells)]

    # Assign faces to owner
    for fi in range(nFaces):
        cell_faces[int(owner[fi])].append(fi)
    # Assign internal faces to neighbour cell (only for internal faces)
    for fi in range(nInternal):
        cell_faces[int(neighbour[fi])].append(fi)

    hexes: List[np.ndarray] = []
    skipped_nonhex = 0
    skipped_order = 0

    for c in range(nCells):
        fids = cell_faces[c]
        if len(fids) != 6:
            continue

        # Check all faces are quads
        if not all(int(faces.length[fi]) == 4 for fi in fids):
            skipped_nonhex += 1
            continue

        # Collect unique vertices
        verts = np.concatenate([face_vertices(faces, fi) for fi in fids]).astype(np.int64)
        u = np.unique(verts)
        if u.size != 8:
            skipped_nonhex += 1
            continue

        # Order into C3D8 by octants around centroid
        X = points[u]  # (8,3)
        C = X.mean(axis=0)

        bx = (X[:, 0] > C[0]).astype(np.int64)
        by = (X[:, 1] > C[1]).astype(np.int64)
        bz = (X[:, 2] > C[2]).astype(np.int64)
        bits = bx + 2 * by + 4 * bz  # (8,)

        # Need exactly one vertex per required bit in _CANON_BITS
        ordered = np.empty(8, dtype=np.int64)
        ok = True
        for k, b in enumerate(_CANON_BITS.tolist()):
            idx = np.where(bits == b)[0]
            if idx.size != 1:
                ok = False
                break
            ordered[k] = u[idx[0]]

        if not ok:
            skipped_order += 1
            continue

        hexes.append(ordered)

    log(f"[MESH] Cells: {nCells}, Faces: {nFaces}, InternalFaces: {nInternal}")
    log(f"[MESH] Hexes extracted: {len(hexes)}")
    if skipped_nonhex:
        log(f"[MESH] Skipped non-hex/poly cells: {skipped_nonhex}")
    if skipped_order:
        log(f"[MESH] Skipped hex-like cells that couldn't be ordered: {skipped_order}")

    return hexes


def pick_base_nodes_from_patch(
    patch_name: str,
    patches: Dict[str, Tuple[int, int]],
    faces: FacesRagged,
    used_node_old_ids: np.ndarray,
    node_map_old_to_new: np.ndarray,
) -> List[int]:
    if patch_name not in patches:
        raise ValueError(f"Patch '{patch_name}' not found in boundary. Available: {list(patches.keys())[:20]} ...")

    startFace, nFaces = patches[patch_name]
    fids = range(startFace, startFace + nFaces)

    nodes_old = set()
    for fi in fids:
        fv = face_vertices(faces, fi)
        for v in fv.tolist():
            nodes_old.add(int(v))

    # Keep only nodes that exist in the extracted hex mesh
    base_new = []
    for old in nodes_old:
        new = int(node_map_old_to_new[old])
        if new >= 0:
            base_new.append(new + 1)  # to 1-based
    return sorted(set(base_new))


def pick_base_nodes_by_plane(
    points_new: np.ndarray,
    axis: str,
    pick: str,
    tol_rel: float,
) -> List[int]:
    ax = _axis_index(axis)
    coord = points_new[:, ax]
    extreme = float(coord.min() if pick == "min" else coord.max())
    span = float(coord.max() - coord.min())
    tol = max(1e-12, tol_rel * max(1.0, span))
    sel = np.where(np.abs(coord - extreme) <= tol)[0]
    return (sel + 1).astype(int).tolist()  # 1-based


def write_ccx_inp_single_step(
    out_path: str,
    points_new: np.ndarray,
    hexes_new: np.ndarray,  # (M,8) 0-based in new node indexing
    base_nodes_1based: List[int],
    cure_shrink_per_unit: float,
    write_outputs: bool = True,
) -> None:
    n_nodes = int(points_new.shape[0])
    n_elems = int(hexes_new.shape[0])

    alpha = -float(cure_shrink_per_unit)

    def fmt(x: float) -> str:
        return f"{float(x):.12e}"

    with open(out_path, "w", encoding="ascii", errors="strict") as f:
        f.write("**\n")
        f.write("** Auto-generated from OpenFOAM polyMesh (hex-only)\n")
        f.write("** Single-step uniform shrink via *EXPANSION and TEMP DOF 11\n")
        f.write("**\n")
        f.write("*HEADING\n")
        f.write("polyMesh C3D8R single-step shrink sanity check\n")

        f.write("*NODE\n")
        for i in range(n_nodes):
            x, y, z = points_new[i]
            f.write(f"{i+1}, {fmt(x)}, {fmt(y)}, {fmt(z)}\n")

        f.write("*ELEMENT, TYPE=C3D8R, ELSET=ALL\n")
        for eid in range(n_elems):
            conn = (hexes_new[eid] + 1).astype(int)  # to 1-based
            f.write(f"{eid+1}, {conn[0]}, {conn[1]}, {conn[2]}, {conn[3]}, {conn[4]}, {conn[5]}, {conn[6]}, {conn[7]}\n")

        f.write("*NSET, NSET=ALLNODES, GENERATE\n")
        f.write(f"1, {n_nodes}, 1\n")

        if base_nodes_1based:
            f.write("*NSET, NSET=BASE\n")
            # write in chunks
            for i in range(0, len(base_nodes_1based), 16):
                chunk = base_nodes_1based[i:i+16]
                f.write(", ".join(str(x) for x in chunk) + "\n")
        else:
            f.write("** WARNING: BASE is empty\n")

        # Material
        f.write("*MATERIAL, NAME=ABS\n")
        f.write("*DENSITY\n")
        f.write("1.12e-9\n")
        f.write("*ELASTIC\n")
        f.write("2800., 0.35\n")
        f.write("*EXPANSION, ZERO=0.\n")
        f.write(f"{alpha:.6e}\n")
        f.write("*CONDUCTIVITY\n")
        f.write("0.20\n")
        f.write("*SPECIFIC HEAT\n")
        f.write("1.30e+9\n")
        f.write("*SOLID SECTION, ELSET=ALL, MATERIAL=ABS\n")

        # Initial temperature/cure
        f.write("*INITIAL CONDITIONS, TYPE=TEMPERATURE\n")
        f.write("ALLNODES, 0.0\n")

        # One step
        f.write("*STEP\n")
        f.write("*UNCOUPLED TEMPERATURE-DISPLACEMENT, SOLVER=PASTIX\n")
        f.write("1.0, 1.0\n")

        f.write("*BOUNDARY\n")
        if base_nodes_1based:
            f.write("BASE, 1, 6, 0.\n")
        # Apply cure=1.0 everywhere (TEMP DOF 11)
        f.write("ALLNODES, 11, 11, 1.0\n")

        if write_outputs:
            f.write("*NODE FILE\n")
            f.write("U\n")

        f.write("*END STEP\n")

    log(f"[CCX] Wrote: {out_path}")
    log(f"[CCX] Nodes={n_nodes}, Elems={n_elems}, BASE={len(base_nodes_1based)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("polyMesh_dir", help="Path to constant/polyMesh")
    ap.add_argument("output_inp", help="Output CalculiX .inp")

    ap.add_argument("--cure-shrink-per-unit", type=float, required=True, help="e.g. 0.05 for 5%% shrink")
    ap.add_argument("--base-axis", choices=["x", "y", "z"], default="z", help="Used if no --base-patch")
    ap.add_argument("--base-pick", choices=["min", "max"], default="min", help="Used if no --base-patch")
    ap.add_argument("--base-tol-rel", type=float, default=1e-6, help="Plane tolerance (relative) for BASE")
    ap.add_argument("--base-patch", default=None, help="Boundary patch name to use as BASE (recommended)")
    ap.add_argument("--no-outputs", action="store_true", help="Disable *NODE FILE outputs")

    args = ap.parse_args()

    d = args.polyMesh_dir

    # Accept either "points.gz" or "points"
    def pick_file(name: str) -> str:
        p1 = os.path.join(d, name + ".gz")
        p2 = os.path.join(d, name)
        if os.path.exists(p1):
            return p1
        if os.path.exists(p2):
            return p2
        raise FileNotFoundError(f"Missing {name}[.gz] in {d}")

    points_path = pick_file("points")
    faces_path = pick_file("faces")
    owner_path = pick_file("owner")
    neigh_path = pick_file("neighbour")
    boundary_path = pick_file("boundary")

    log("[IO] Reading points...")
    pts = parse_points(points_path)

    log("[IO] Reading faces...")
    faces = parse_faces(faces_path)

    log("[IO] Reading owner/neighbour...")
    owner = parse_label_list(owner_path)
    neigh = parse_label_list(neigh_path)

    log("[IO] Reading boundary...")
    patches = parse_boundary(boundary_path)
    if args.base_patch:
        log(f"[IO] Base patch requested: {args.base_patch}")

    log("[MESH] Building hex elements...")
    hexes_old = build_hex_elements(pts, faces, owner, neigh)
    if not hexes_old:
        raise SystemExit("No hex elements extracted. (Mesh may be polyhedral or parsing failed.)")

    hexes_old_arr = np.vstack(hexes_old).astype(np.int64)  # (M,8)

    # Reindex nodes to only used ones
    used_old = np.unique(hexes_old_arr.reshape(-1))
    node_map = np.full(pts.shape[0], -1, dtype=np.int64)
    node_map[used_old] = np.arange(used_old.size, dtype=np.int64)

    pts_new = pts[used_old]
    hexes_new = node_map[hexes_old_arr]  # (M,8) now 0-based in new indexing

    # BASE nodes
    base_nodes_1based: List[int]
    if args.base_patch:
        base_nodes_1based = pick_base_nodes_from_patch(
            args.base_patch, patches, faces, used_old, node_map
        )
        log(f"[BASE] From patch '{args.base_patch}': {len(base_nodes_1based)} nodes")
    else:
        base_nodes_1based = pick_base_nodes_by_plane(
            pts_new, axis=args.base_axis, pick=args.base_pick, tol_rel=args.base_tol_rel
        )
        log(f"[BASE] From {args.base_pick}({args.base_axis}) plane: {len(base_nodes_1based)} nodes (tol_rel={args.base_tol_rel})")

    write_ccx_inp_single_step(
        args.output_inp,
        pts_new,
        hexes_new,
        base_nodes_1based=base_nodes_1based,
        cure_shrink_per_unit=args.cure_shrink_per_unit,
        write_outputs=not args.no_outputs,
    )


if __name__ == "__main__":
    main()

"""
python foam_polyMesh_to_ccx_simple.py "C:\cfMesh-v1.1.1\MSYS\home\4y5t6\tutorials\cartesianMesh\asmoOctree\constant\polyMesh" job.inp --cure-shrink-per-unit 0.05 --base-patch fixed
"""
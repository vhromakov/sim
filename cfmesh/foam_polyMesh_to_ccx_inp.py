#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import os
import re
from typing import List, Tuple, Dict, Optional

import numpy as np


# ---------------------------
# Low-level OpenFOAM readers
# ---------------------------

def _open_text_maybe_gz(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "rt", encoding="utf-8", errors="ignore")


def _read_foam_format(path: str) -> str:
    """Detect 'format ascii;' vs 'format binary;' from FoamFile header."""
    with _open_text_maybe_gz(path) as f:
        header = []
        for _ in range(200):  # header is small
            line = f.readline()
            if not line:
                break
            header.append(line)
            if "}" in line:
                break
        txt = "".join(header)
    m = re.search(r"\bformat\s+(\w+)\s*;", txt)
    return m.group(1).lower() if m else "unknown"


def read_points(path: str) -> List[Tuple[float, float, float]]:
    fmt = _read_foam_format(path)
    if fmt == "binary":
        raise RuntimeError(f"{path} is binary; convert mesh to ASCII first.")

    pts: List[Tuple[float, float, float]] = []
    with _open_text_maybe_gz(path) as f:
        # Skip header until we find the count line (an integer)
        n = None
        for line in f:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            if re.fullmatch(r"\d+", line_stripped):
                n = int(line_stripped)
                break
        if n is None:
            raise RuntimeError(f"Failed to find point count in {path}")

        # Next non-empty line should be '('
        for line in f:
            if line.strip():
                if line.strip().startswith("("):
                    break

        # Read points until ')'
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith(")"):
                break
            # Common forms: "(x y z)" or "x y z"
            nums = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", s)
            if len(nums) < 3:
                continue
            pts.append((float(nums[0]), float(nums[1]), float(nums[2])))

    if len(pts) != n:
        # Some meshes split vectors differently, but usually this is exact.
        print(f"[WARN] points: expected {n}, parsed {len(pts)}")
    return pts


def read_int_list(path: str) -> List[int]:
    fmt = _read_foam_format(path)
    if fmt == "binary":
        raise RuntimeError(f"{path} is binary; convert mesh to ASCII first.")

    out: List[int] = []
    with _open_text_maybe_gz(path) as f:
        n = None
        for line in f:
            s = line.strip()
            if not s:
                continue
            if re.fullmatch(r"\d+", s):
                n = int(s)
                break
        if n is None:
            raise RuntimeError(f"Failed to find list count in {path}")

        # Consume up to '('
        for line in f:
            if line.strip():
                if line.strip().startswith("("):
                    break

        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith(")"):
                break
            # May have multiple ints per line in some cases
            for tok in s.replace("(", " ").replace(")", " ").split():
                if re.fullmatch(r"[-+]?\d+", tok):
                    out.append(int(tok))

    if len(out) != n:
        print(f"[WARN] {os.path.basename(path)}: expected {n}, parsed {len(out)}")
    return out


def read_faces(path: str) -> List[List[int]]:
    fmt = _read_foam_format(path)
    if fmt == "binary":
        raise RuntimeError(f"{path} is binary; convert mesh to ASCII first.")

    faces: List[List[int]] = []
    with _open_text_maybe_gz(path) as f:
        n = None
        for line in f:
            s = line.strip()
            if not s:
                continue
            if re.fullmatch(r"\d+", s):
                n = int(s)
                break
        if n is None:
            raise RuntimeError(f"Failed to find face count in {path}")

        # Consume up to '('
        for line in f:
            if line.strip():
                if line.strip().startswith("("):
                    break

        # Faces are usually one per line: 4(0 1 2 3)
        # We'll also handle cases where a face spills across lines by accumulating until ')'.
        buf = ""
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith(")"):
                break

            buf += " " + s
            # If we have a complete "k(...)" we can parse one (or more) faces from buf.
            # We'll repeatedly extract leading face definitions.
            while True:
                m = re.search(r"\s*(\d+)\s*\(\s*([^\)]*?)\s*\)", buf)
                if not m:
                    # Need more text
                    break
                k = int(m.group(1))
                inside = m.group(2).strip()
                idxs = [int(x) for x in inside.split() if re.fullmatch(r"[-+]?\d+", x)]
                if k != len(idxs):
                    # Some formatting weirdness; still keep what we got if plausible
                    pass
                faces.append(idxs)
                # Remove parsed portion
                buf = buf[m.end():]

    if len(faces) != n:
        print(f"[WARN] faces: expected {n}, parsed {len(faces)}")
    return faces


# ---------------------------
# Cell -> Hex detection + ordering
# ---------------------------

def order_hex_nodes_pca(points_xyz: np.ndarray) -> Optional[np.ndarray]:
    """
    points_xyz: (8,3) unique vertex positions of a hex cell.
    Returns indices (8,) into points_xyz giving C3D8 order, or None on failure.
    """
    c = points_xyz.mean(axis=0)
    X = points_xyz - c
    C = X.T @ X
    w, V = np.linalg.eigh(C)          # columns of V are eigenvectors
    V = V[:, np.argsort(w)[::-1]]     # descending eigenvalues

    # Make axes right-handed
    if np.linalg.det(V) < 0:
        V[:, 2] *= -1.0

    local = X @ V  # (8,3): u,v,w coords
    # Primary attempt: sign-coded corners
    eps = 1e-12 * max(1.0, np.max(np.linalg.norm(local, axis=1)))
    bu = (local[:, 0] > eps).astype(int)
    bv = (local[:, 1] > eps).astype(int)
    bw = (local[:, 2] > eps).astype(int)
    code = bu + 2 * bv + 4 * bw

    # Expect exactly one vertex per code 0..7
    if len(set(code.tolist())) == 8:
        inv = {int(code[i]): i for i in range(8)}
        want = [0, 1, 3, 2, 4, 5, 7, 6]  # C3D8 corner pattern
        return np.array([inv[k] for k in want], dtype=int)

    # Fallback: split by w into bottom/top, then sort each by (v,u)
    idx = np.argsort(local[:, 2])  # low w first
    bot = idx[:4]
    top = idx[4:]

    def sort_face(face_idx: np.ndarray) -> np.ndarray:
        uv = local[face_idx][:, :2]
        # sort by v then u
        o = np.lexsort((uv[:, 0], uv[:, 1]))
        face_sorted = face_idx[o]
        # Now map to (-u,-v), (+u,-v), (+u,+v), (-u,+v)
        uv2 = local[face_sorted][:, :2]
        # group by v sign
        vneg = face_sorted[uv2[:, 1] <= 0]
        vpos = face_sorted[uv2[:, 1] > 0]
        if len(vneg) != 2 or len(vpos) != 2:
            return face_sorted  # last resort
        vneg = vneg[np.argsort(local[vneg][:, 0])]  # u ascending: (-u, +u)
        vpos = vpos[np.argsort(local[vpos][:, 0])]  # u ascending
        return np.array([vneg[0], vneg[1], vpos[1], vpos[0]], dtype=int)

    bot4 = sort_face(bot)
    top4 = sort_face(top)
    if len(set(np.concatenate([bot4, top4]).tolist())) == 8:
        return np.concatenate([bot4, top4])
    return None


# ---------------------------
# CalculiX writer
# ---------------------------

def write_inp(
    out_path: str,
    points: List[Tuple[float, float, float]],
    hex_elems: List[Tuple[int, int, int, int, int, int, int, int]],
):
    pts = np.asarray(points, dtype=np.float64)
    zmin = float(np.min(pts[:, 2]))
    zrange = float(np.max(pts[:, 2]) - zmin)
    eps = max(1e-9, 1e-9 * max(1.0, zrange))
    fixed_nodes = np.where(pts[:, 2] <= zmin + eps)[0] + 1  # 1-based

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("*HEADING\n")
        f.write("OpenFOAM polyMesh -> CalculiX (hex only)\n")
        f.write("**\n")

        f.write("*NODE\n")
        for i, (x, y, z) in enumerate(points, start=1):
            f.write(f"{i}, {x:.16g}, {y:.16g}, {z:.16g}\n")

        f.write("**\n")
        f.write("*ELEMENT, TYPE=C3D8, ELSET=EALL\n")
        for eid, conn in enumerate(hex_elems, start=1):
            f.write(f"{eid}, {conn[0]}, {conn[1]}, {conn[2]}, {conn[3]}, {conn[4]}, {conn[5]}, {conn[6]}, {conn[7]}\n")

        f.write("**\n")
        f.write("*MATERIAL, NAME=MAT1\n")
        f.write("*ELASTIC\n")
        f.write("210000, 0.3\n")
        f.write("*SOLID SECTION, ELSET=EALL, MATERIAL=MAT1\n")
        f.write("1.0\n")

        f.write("**\n")
        f.write("*NSET, NSET=FIXED\n")
        # write node ids with line breaks
        line = []
        for nid in fixed_nodes.tolist():
            line.append(str(nid))
            if len(line) >= 16:
                f.write(", ".join(line) + "\n")
                line = []
        if line:
            f.write(", ".join(line) + "\n")

        f.write("**\n")
        f.write("*BOUNDARY\n")
        f.write("FIXED, 1, 3, 0\n")

        f.write("**\n")
        f.write("*STEP\n")
        f.write("*STATIC\n")
        f.write("1., 1.\n")
        f.write("*END STEP\n")


# ---------------------------
# Main conversion logic
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="OpenFOAM constant/polyMesh -> CalculiX .inp (hex-only)")
    ap.add_argument("--polyMesh", required=True, help="Path to constant/polyMesh folder")
    ap.add_argument("--out", default="job.inp", help="Output .inp filename")
    args = ap.parse_args()

    poly = args.polyMesh
    points_path = os.path.join(poly, "points.gz") if os.path.exists(os.path.join(poly, "points.gz")) else os.path.join(poly, "points")
    faces_path = os.path.join(poly, "faces.gz") if os.path.exists(os.path.join(poly, "faces.gz")) else os.path.join(poly, "faces")
    owner_path = os.path.join(poly, "owner.gz") if os.path.exists(os.path.join(poly, "owner.gz")) else os.path.join(poly, "owner")
    neigh_path = os.path.join(poly, "neighbour.gz") if os.path.exists(os.path.join(poly, "neighbour.gz")) else os.path.join(poly, "neighbour")

    points = read_points(points_path)
    faces = read_faces(faces_path)
    owner = read_int_list(owner_path)

    if os.path.exists(neigh_path):
        neigh = read_int_list(neigh_path)
    else:
        neigh = []

    n_faces = len(faces)
    if len(owner) != n_faces:
        raise RuntimeError(f"owner length {len(owner)} != faces length {n_faces}")

    n_internal = len(neigh)  # by OpenFOAM convention: neighbour only for internal faces
    n_cells = 1 + max([max(owner)] + ([max(neigh)] if neigh else [0]))

    cell_faces: List[List[int]] = [[] for _ in range(n_cells)]
    for fi in range(n_faces):
        c0 = owner[fi]
        cell_faces[c0].append(fi)
        if fi < n_internal:
            c1 = neigh[fi]
            cell_faces[c1].append(fi)

    pts_np = np.asarray(points, dtype=np.float64)

    hex_elems: List[Tuple[int, int, int, int, int, int, int, int]] = []
    skipped_nonhex = 0
    skipped_bad_order = 0

    for ci, flist in enumerate(cell_faces):
        # Hex filter
        if len(flist) != 6:
            skipped_nonhex += 1
            continue

        fverts = [faces[fi] for fi in flist]
        if any(len(v) != 4 for v in fverts):
            skipped_nonhex += 1
            continue

        uniq = sorted(set(v for fv in fverts for v in fv))
        if len(uniq) != 8:
            skipped_nonhex += 1
            continue

        cell_xyz = pts_np[np.array(uniq, dtype=int)]
        order = order_hex_nodes_pca(cell_xyz)
        if order is None:
            skipped_bad_order += 1
            continue

        # Convert ordering indices into global node ids (1-based for CalculiX)
        conn = tuple(int(uniq[i] + 1) for i in order.tolist())
        hex_elems.append(conn)

    print(f"[INFO] points: {len(points)}")
    print(f"[INFO] faces:  {len(faces)} (internal faces: {n_internal})")
    print(f"[INFO] cells:  {n_cells}")
    print(f"[INFO] hex elements written: {len(hex_elems)}")
    print(f"[INFO] skipped non-hex/poly cells: {skipped_nonhex}")
    print(f"[INFO] skipped (couldn't order corners): {skipped_bad_order}")

    if not hex_elems:
        print("[WARN] No hex elements detected. Your mesh may be polyhedral/tet/prism-dominant.")
        print("       In that case you must tetrahedralize or use a different conversion approach.")
        # still write nodes so you can inspect
    write_inp(args.out, points, hex_elems)
    print(f"[OK] Wrote {args.out}")


if __name__ == "__main__":
    main()

"""
python foam_polyMesh_to_ccx_inp.py --polyMesh "C:\cfMesh-v1.1.1\MSYS\home\4y5t6\tutorials\cartesianMesh\asmoOctree\constant\polyMesh" --out job.inp
"""
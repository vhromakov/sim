#!/usr/bin/env python3
from __future__ import annotations
import re, sys, math
from typing import Dict, Tuple, List, Optional
import numpy as np

XI   = np.array([-1, +1, +1, -1, -1, +1, +1, -1], dtype=float)
ETA  = np.array([-1, -1, +1, +1, -1, -1, +1, +1], dtype=float)
ZETA = np.array([-1, -1, -1, -1, +1, +1, +1, +1], dtype=float)

def detJ_c3d8_center(X: np.ndarray) -> float:
    dxdxi   = (X[:, 0] * XI).sum() / 8.0
    dxdeta  = (X[:, 0] * ETA).sum() / 8.0
    dxdzeta = (X[:, 0] * ZETA).sum() / 8.0
    dydxi   = (X[:, 1] * XI).sum() / 8.0
    dydeta  = (X[:, 1] * ETA).sum() / 8.0
    dydzeta = (X[:, 1] * ZETA).sum() / 8.0
    dzdxi   = (X[:, 2] * XI).sum() / 8.0
    dzdeta  = (X[:, 2] * ETA).sum() / 8.0
    dzdzeta = (X[:, 2] * ZETA).sum() / 8.0
    J = np.array([[dxdxi, dxdeta, dxdzeta],
                  [dydxi, dydeta, dydzeta],
                  [dzdxi, dzdeta, dzdzeta]], dtype=float)
    return float(np.linalg.det(J))

def parse_csv_ints(line: str) -> List[int]:
    return [int(x.strip()) for x in line.split(",") if x.strip()]

def ccw_order_xy(nids: List[int], nodes: Dict[int, Tuple[float,float,float]]) -> List[int]:
    pts = np.array([nodes[n] for n in nids], dtype=float)
    c = pts[:, :2].mean(axis=0)
    ang = np.arctan2(pts[:,1]-c[1], pts[:,0]-c[0])
    order = np.argsort(ang)
    return [nids[i] for i in order]

def reorder_hex_global_z(conn: List[int], nodes: Dict[int, Tuple[float,float,float]]) -> Optional[List[int]]:
    pts = np.array([nodes[n] for n in conn], dtype=float)  # (8,3)
    z = pts[:,2]
    idx = np.argsort(z)  # ascending
    bottom_idx = idx[:4]
    top_idx    = idx[4:]

    bottom = [conn[i] for i in bottom_idx]
    top    = [conn[i] for i in top_idx]

    # if z separation is tiny, this method may be unreliable
    zb = z[bottom_idx].mean()
    zt = z[top_idx].mean()
    if abs(zt - zb) < 1e-12 * max(1.0, abs(zb), abs(zt)):
        return None

    bottom_ccw = ccw_order_xy(bottom, nodes)

    # match each bottom vertex to closest top vertex in XY
    top_pts = np.array([nodes[n][:2] for n in top], dtype=float)
    used = set()
    top_matched: List[int] = []
    for b in bottom_ccw:
        bxy = np.array(nodes[b][:2], dtype=float)
        d2 = np.sum((top_pts - bxy)**2, axis=1)
        # pick nearest unused
        for j in np.argsort(d2):
            if j not in used:
                used.add(j)
                top_matched.append(top[j])
                break

    if len(top_matched) != 4:
        return None

    # C3D8 canonical: 1-4 bottom CCW, 5-8 top corresponding to 1-4
    cand = bottom_ccw + top_matched

    X = np.array([nodes[n] for n in cand], dtype=float)
    det = detJ_c3d8_center(X)
    if det > 0:
        return cand

    # flip orientation: reverse bottom order (swap 2<->4 effectively)
    bottom_ccw_rev = [bottom_ccw[0], bottom_ccw[3], bottom_ccw[2], bottom_ccw[1]]
    top_matched2: List[int] = []
    used = set()
    top_pts = np.array([nodes[n][:2] for n in top], dtype=float)
    for b in bottom_ccw_rev:
        bxy = np.array(nodes[b][:2], dtype=float)
        d2 = np.sum((top_pts - bxy)**2, axis=1)
        for j in np.argsort(d2):
            if j not in used:
                used.add(j)
                top_matched2.append(top[j])
                break
    cand2 = bottom_ccw_rev + top_matched2
    X2 = np.array([nodes[n] for n in cand2], dtype=float)
    if detJ_c3d8_center(X2) > 0:
        return cand2

    return None

def main(inp: str, out: str):
    lines = open(inp, "r", encoding="utf-8", errors="ignore").read().splitlines()

    # parse nodes
    nodes: Dict[int, Tuple[float,float,float]] = {}
    i = 0
    while i < len(lines):
        if lines[i].strip().upper().startswith("*NODE"):
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith("*"):
                s = lines[i].strip()
                if not s or s.startswith("**"):
                    i += 1
                    continue
                parts = [p.strip() for p in lines[i].split(",")]
                nid = int(parts[0])
                nodes[nid] = (float(parts[1]), float(parts[2]), float(parts[3]))
                i += 1
            break
        i += 1
    if not nodes:
        raise SystemExit("No *NODE block found.")

    type_re = re.compile(r"TYPE\s*=\s*([A-Z0-9]+)", re.IGNORECASE)

    fixed = 0
    still_bad = 0
    skipped = 0

    out_lines: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if s.upper().startswith("*ELEMENT"):
            m = type_re.search(line)
            etype = m.group(1).upper() if m else None
            out_lines.append(line)
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith("*"):
                raw = lines[i]
                if raw.lstrip().startswith("**") or not raw.strip():
                    out_lines.append(raw)
                    i += 1
                    continue
                parts = parse_csv_ints(raw)
                eid = parts[0]
                conn = parts[1:]

                if etype == "C3D8" and len(conn) == 8:
                    new_conn = reorder_hex_global_z(conn, nodes)
                    if new_conn is None:
                        skipped += 1
                        out_lines.append(raw)
                    else:
                        X = np.array([nodes[n] for n in new_conn], dtype=float)
                        if detJ_c3d8_center(X) > 0:
                            # count if original was bad
                            Xo = np.array([nodes[n] for n in conn], dtype=float)
                            if detJ_c3d8_center(Xo) <= 0:
                                fixed += 1
                            out_lines.append(f"{eid}, " + ", ".join(map(str, new_conn)))
                        else:
                            still_bad += 1
                            out_lines.append(raw)
                else:
                    out_lines.append(raw)

                i += 1
            continue

        out_lines.append(line)
        i += 1

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")

    print(f"Wrote: {out}")
    print(f"Hex fixed: {fixed} | Hex still bad: {still_bad} | Hex skipped (z-ambiguous): {skipped}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python fix_ccx_hex_global.py input.inp output.inp")
    main(sys.argv[1], sys.argv[2])

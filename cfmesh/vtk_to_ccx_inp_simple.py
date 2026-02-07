#!/usr/bin/env python3
"""
VTK (.vtk/.vtu) hex mesh -> CalculiX .inp (single-step uniform shrink)

Primary goal: make CCX run (sanity check).

Fixes:
- Node reordering per hex via octant classification (works for voxel/brick-ish hexes)
- Optional dropping of hexes that cannot be canonicalized by octants
- Jacobian det at center (C3D8R integration point) computed in canonical order
- Flip xi parity for det<0
- Optional dropping of hexes with det<=eps (default eps=0 to match CCX "nonpositive")

Also fixes BASE selection (avoid BASE=1):
- selects nodes near extreme plane using tol_abs, with auto-widen fallback

Deps:
  pip install numpy meshio

Example:
  python vtk_to_ccx_inp_simple.py asmoOctree_0.vtk job.inp \
    --cure-shrink-per-unit 0.05 --drop-bad
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Sequence

import numpy as np
import meshio


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
    raise ValueError("axis must be x/y/z")


def _write_id_list_lines(f, ids: Sequence[int], per_line: int = 16) -> None:
    ids = list(ids)
    for i in range(0, len(ids), per_line):
        chunk = ids[i : i + per_line]
        f.write(", ".join(str(x) for x in chunk) + "\n")


def _voxel_to_hex(conn: np.ndarray) -> np.ndarray:
    # VTK_VOXEL -> VTK_HEXAHEDRON common mapping
    return conn[:, [0, 1, 3, 2, 4, 5, 7, 6]]


def read_vtk_hex_mesh(path: str) -> tuple[np.ndarray, np.ndarray]:
    m = meshio.read(path)
    pts = np.asarray(m.points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points, got {pts.shape}")
    if not np.isfinite(pts).all():
        bad = np.where(~np.isfinite(pts).all(axis=1))[0]
        raise ValueError(f"Found NaN/Inf points. Example indices: {bad[:10].tolist()}")

    blocks = []
    for cb in m.cells:
        ctype = cb.type.lower()
        data = np.asarray(cb.data, dtype=np.int64)
        if ctype == "hexahedron" and data.shape[1] == 8:
            blocks.append(data)
        elif ctype == "voxel" and data.shape[1] == 8:
            blocks.append(_voxel_to_hex(data))

    if not blocks:
        raise ValueError("No 8-node hex cells found (need hexahedron or voxel).")

    hexes0 = np.vstack(blocks).astype(np.int64)

    n = pts.shape[0]
    if hexes0.min() < 0 or hexes0.max() >= n:
        raise ValueError(f"Connectivity out of range: min={hexes0.min()} max={hexes0.max()} n_nodes={n}")

    return pts, hexes0


# Canonical C3D8 octant bits relative to centroid:
# bits = (x>cx)*1 + (y>cy)*2 + (z>cz)*4
# canonical positions 0..7 correspond to these bitmasks:
# 0(-,-,-)=0, 1(+,-,-)=1, 2(+,+,-)=3, 3(-,+,-)=2, 4(-,-,+)=4, 5(+,-,+)=5, 6(+,+,+)=7, 7(-,+,+)=6
_CANON_BITS = np.array([0, 1, 3, 2, 4, 5, 7, 6], dtype=np.int64)

# parity flip (mirror in xi): swap 0<->1, 2<->3, 4<->5, 6<->7
_FLIP_XI = np.array([1, 0, 3, 2, 5, 4, 7, 6], dtype=np.int64)

# dN/d(xi,eta,zeta) at center for canonical order (already /8)
_XI  = np.array([-1,  1,  1, -1, -1,  1,  1, -1], dtype=np.float64) / 8.0
_ETA = np.array([-1, -1,  1,  1, -1, -1,  1,  1], dtype=np.float64) / 8.0
_ZET = np.array([-1, -1, -1, -1,  1,  1,  1,  1], dtype=np.float64) / 8.0


def canonicalize_by_octants(
    pts: np.ndarray,
    hexes0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      hexes_canon: (M,8) reordered to canonical C3D8 order (where possible)
      ok: (M,) True if element had exactly one node per canonical octant
    """
    X = pts[hexes0]            # (M,8,3)
    C = X.mean(axis=1)         # (M,3)

    bx = (X[:, :, 0] > C[:, None, 0]).astype(np.int64)
    by = (X[:, :, 1] > C[:, None, 1]).astype(np.int64)
    bz = (X[:, :, 2] > C[:, None, 2]).astype(np.int64)
    bits = bx + 2 * by + 4 * bz

    M = hexes0.shape[0]
    idx = np.zeros((M, 8), dtype=np.int64)
    ok = np.ones((M,), dtype=bool)

    for k, b in enumerate(_CANON_BITS.tolist()):
        mask = (bits == b)
        cnt = mask.sum(axis=1)
        ok &= (cnt == 1)
        idx[:, k] = mask.argmax(axis=1)  # only valid where cnt==1

    rows = np.arange(M)[:, None]
    hexes_canon = hexes0[rows, idx]
    return hexes_canon, ok


def detJ_center(pts: np.ndarray, hexes0_canon: np.ndarray) -> np.ndarray:
    X = pts[hexes0_canon]  # (M,8,3)

    dx_dxi  = (X[:, :, 0] * _XI ).sum(axis=1)
    dy_dxi  = (X[:, :, 1] * _XI ).sum(axis=1)
    dz_dxi  = (X[:, :, 2] * _XI ).sum(axis=1)

    dx_deta = (X[:, :, 0] * _ETA).sum(axis=1)
    dy_deta = (X[:, :, 1] * _ETA).sum(axis=1)
    dz_deta = (X[:, :, 2] * _ETA).sum(axis=1)

    dx_dzet = (X[:, :, 0] * _ZET).sum(axis=1)
    dy_dzet = (X[:, :, 1] * _ZET).sum(axis=1)
    dz_dzet = (X[:, :, 2] * _ZET).sum(axis=1)

    return (
        dx_dxi * (dy_deta * dz_dzet - dz_deta * dy_dzet)
        - dy_dxi * (dx_deta * dz_dzet - dz_deta * dx_dzet)
        + dz_dxi * (dx_deta * dy_dzet - dy_deta * dx_dzet)
    )


def write_report(path: str, header: str, ids_1based: Sequence[int]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(header.rstrip() + "\n")
        for eid in ids_1based:
            f.write(f"{eid}\n")


def repair_mesh_for_ccx(
    pts: np.ndarray,
    hexes0: np.ndarray,
    *,
    eps: float,
    drop_bad: bool,
    oct_bad_report: Optional[str],
    jac_bad_report: Optional[str],
) -> np.ndarray:
    n_in = hexes0.shape[0]

    hexes_canon, ok_oct = canonicalize_by_octants(pts, hexes0)
    oct_bad_ids = (np.where(~ok_oct)[0] + 1).tolist()
    if oct_bad_report is not None:
        write_report(
            oct_bad_report,
            f"# Elements that fail octant-uniqueness (cannot be canonicalized by centroid octants). count={len(oct_bad_ids)}",
            oct_bad_ids,
        )

    if len(oct_bad_ids) > 0:
        log(f"[OCT] bad={len(oct_bad_ids)} / {n_in}")
        if drop_bad:
            hexes_canon = hexes_canon[ok_oct]
            log(f"[OCT] Dropped {len(oct_bad_ids)} octant-invalid elements.")

    det0 = detJ_center(pts, hexes_canon)
    neg = det0 < 0.0  # strict: negative orientation
    n_flip = int(neg.sum())
    if n_flip:
        hexes_canon[neg] = hexes_canon[neg][:, _FLIP_XI]

    det1 = detJ_center(pts, hexes_canon)

    # IMPORTANT: default eps=0.0 matches CCX "nonpositive"
    bad = det1 <= eps
    jac_bad_ids = (np.where(bad)[0] + 1).tolist()

    if jac_bad_report is not None:
        write_report(
            jac_bad_report,
            f"# Elements with detJ(center) <= eps after canonicalize+flip. eps={eps} count={len(jac_bad_ids)}",
            jac_bad_ids,
        )

    log(f"[JAC] after_oct={hexes_canon.shape[0]} flipped={n_flip} bad_after={len(jac_bad_ids)} eps={eps:.3e}")
    log(f"[JAC] min_det_before={float(det0.min()):.3e} min_det_after={float(det1.min()):.3e}")

    if len(jac_bad_ids) > 0:
        if drop_bad:
            hexes_canon = hexes_canon[~bad]
            log(f"[JAC] Dropped {len(jac_bad_ids)} Jacobian-invalid elements.")
        else:
            log("[JAC] WARNING: bad elements remain; CCX will fail unless --drop-bad or mesh is fixed upstream.")

    return hexes_canon


def compute_base_nodes(
    pts: np.ndarray,
    *,
    base_axis: str,
    pick: str,
    tol_abs: Optional[float],
    min_target: int = 100,
) -> List[int]:
    """
    Select BASE nodes as nodes close to extreme plane.
    If too few nodes are selected, auto-increase tolerance until we reach min_target
    or until tolerance becomes "large enough".
    """
    ax = _axis_index(base_axis)
    coord = pts[:, ax]
    extreme = float(coord.min() if pick == "min" else coord.max())

    # scale
    span = float(coord.max() - coord.min())
    diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))

    # start tolerance
    if tol_abs is None:
        # start with something reasonable; your mesh seems to have very fine coords
        tol = max(1e-9, 1e-6 * max(1.0, span))
    else:
        tol = float(tol_abs)

    # widen if needed
    for _ in range(12):
        sel = np.where(np.abs(coord - extreme) <= tol)[0]
        if sel.size >= min_target:
            return sorted((sel + 1).tolist())  # 1-based
        # widen
        tol *= 10.0
        # stop widening if it becomes too big (avoid selecting half the model)
        if diag > 0 and tol > 1e-2 * diag:
            break

    # final selection (might still be small)
    sel = np.where(np.abs(coord - extreme) <= tol)[0]
    return sorted((sel + 1).tolist())


def write_inp_single_step(
    out_path: str,
    pts: np.ndarray,
    hexes0: np.ndarray,
    *,
    cure_shrink_per_unit: float,
    base_nodes_1based: List[int],
    write_outputs: bool,
) -> None:
    n_nodes = int(pts.shape[0])
    n_elems = int(hexes0.shape[0])

    # Fortran-friendly float format
    def fmt(x: float) -> str:
        return f"{float(x):.12e}"

    alpha = -float(cure_shrink_per_unit)

    with open(out_path, "w", encoding="ascii", errors="strict") as f:
        f.write("**\n")
        f.write("** Auto-generated CalculiX input (single-step)\n")
        f.write("** Uniform shrink via *EXPANSION and TEMP DOF 11\n")
        f.write("**\n")
        f.write("*HEADING\n")
        f.write("VTK C3D8R single-step shrink sanity check\n")

        f.write("*NODE\n")
        for i in range(n_nodes):
            x, y, z = pts[i]
            f.write(f"{i+1}, {fmt(x)}, {fmt(y)}, {fmt(z)}\n")

        f.write("*ELEMENT, TYPE=C3D8R, ELSET=ALL\n")
        for eid in range(n_elems):
            a, b, c, d, e, f2, g, h = (int(v) + 1 for v in hexes0[eid])
            f.write(f"{eid+1}, {a}, {b}, {c}, {d}, {e}, {f2}, {g}, {h}\n")

        f.write("*NSET, NSET=ALLNODES, GENERATE\n")
        f.write(f"1, {n_nodes}, 1\n")

        if base_nodes_1based:
            f.write("*NSET, NSET=BASE\n")
            _write_id_list_lines(f, base_nodes_1based)
        else:
            f.write("** WARNING: BASE set is empty.\n")

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

        f.write("*INITIAL CONDITIONS, TYPE=TEMPERATURE\n")
        f.write("ALLNODES, 0.0\n")

        f.write("*STEP\n")
        f.write("*UNCOUPLED TEMPERATURE-DISPLACEMENT, SOLVER=PASTIX\n")
        f.write("1.0, 1.0\n")

        f.write("*BOUNDARY\n")
        if base_nodes_1based:
            f.write("BASE, 1, 6, 0.\n")
        f.write("ALLNODES, 11, 11, 1.0\n")

        if write_outputs:
            f.write("*NODE FILE\n")
            f.write("U\n")

        f.write("*END STEP\n")

    log(f"[OK] wrote {out_path}")
    log(f"[OK] nodes={n_nodes}, elems={n_elems}, BASE={len(base_nodes_1based)}, alpha={alpha:.6e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_vtk")
    ap.add_argument("output_inp")
    ap.add_argument("--cure-shrink-per-unit", type=float, required=True)

    ap.add_argument("--drop-bad", action="store_true",
                    help="Drop octant-invalid and Jacobian-invalid elements (recommended for sanity run)")
    ap.add_argument("--eps", type=float, default=0.0,
                    help="detJ threshold for 'nonpositive'. Default 0.0 to match CCX.")
    ap.add_argument("--oct-report", default="octant_bad_elements.txt")
    ap.add_argument("--jac-report", default="jacobian_bad_elements.txt")

    ap.add_argument("--base-axis", choices=["x", "y", "z"], default="z")
    ap.add_argument("--base-pick", choices=["min", "max"], default="min")
    ap.add_argument("--base-tol-abs", type=float, default=None,
                    help="Absolute tolerance for picking BASE nodes near extreme plane (auto-widens if too few).")
    ap.add_argument("--no-outputs", action="store_true")

    args = ap.parse_args()

    pts, hexes0 = read_vtk_hex_mesh(args.input_vtk)

    hexes0_fixed = repair_mesh_for_ccx(
        pts,
        hexes0,
        eps=float(args.eps),
        drop_bad=args.drop_bad,
        oct_bad_report=args.oct_report,
        jac_bad_report=args.jac_report,
    )

    base_nodes = compute_base_nodes(
        pts,
        base_axis=args.base_axis,
        pick=args.base_pick,
        tol_abs=args.base_tol_abs,
        min_target=200,  # try to avoid BASE=1
    )

    write_inp_single_step(
        args.output_inp,
        pts,
        hexes0_fixed,
        cure_shrink_per_unit=args.cure_shrink_per_unit,
        base_nodes_1based=base_nodes,
        write_outputs=not args.no_outputs,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
vtk_to_internal_edges_ply.py

Read a VTK dataset (legacy .vtk, XML .vtu, .vtm, etc.) and export a PLY
that contains:
- vertex list (points)
- edge list (line segments) representing *all* mesh edges (including internal)

Requires: pyvista (which pulls in VTK)
    pip install pyvista

Usage:
    python vtk_to_internal_edges_ply.py input.vtk output.ply
    python vtk_to_internal_edges_ply.py input.vtm output.ply --max-edges 500000
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, List, Tuple, Set

def parse_polydata_lines_to_edges(lines: Iterable[int]) -> List[Tuple[int, int]]:
    """
    VTK "lines" array format:
      [n, p0, p1, ..., pn-1,  n, q0, q1, ...,]
    For each polyline, emit edges between consecutive points.
    """
    lines = list(lines)
    edges: List[Tuple[int, int]] = []
    i = 0
    L = len(lines)
    while i < L:
        n = lines[i]
        i += 1
        if n <= 1:
            i += max(n, 0)
            continue
        pts = lines[i : i + n]
        i += n
        for a, b in zip(pts, pts[1:]):
            if a != b:
                edges.append((int(a), int(b)))
    return edges

def write_ply_with_edges(
    out_path: Path,
    points: List[Tuple[float, float, float]],
    edges: List[Tuple[int, int]],
) -> None:
    """
    ASCII PLY with 'vertex' and 'edge' elements.
    MeshLab supports this well for wireframe visualization.
    """
    with out_path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("end_header\n")

        for (x, y, z) in points:
            f.write(f"{x:.9g} {y:.9g} {z:.9g}\n")

        for (a, b) in edges:
            f.write(f"{a} {b}\n")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=str, help="Input VTK file (.vtk/.vtu/.vtm/...)")
    ap.add_argument("output", type=str, help="Output PLY file (.ply)")
    ap.add_argument("--max-edges", type=int, default=0,
                    help="If >0, randomly keep at most this many edges (helps huge meshes).")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for --max-edges sampling.")
    ap.add_argument("--dedup", action="store_true",
                    help="Deduplicate edges (can be slower, but reduces size).")
    args = ap.parse_args()

    try:
        import pyvista as pv
    except ImportError as e:
        raise SystemExit(
            "pyvista is not installed.\n"
            "Install it with: pip install pyvista\n"
        ) from e

    in_path = Path(args.input)
    out_path = Path(args.output)

    ds = pv.read(str(in_path))

    # foamToVTK often writes a multi-block dataset (.vtm). Combine if needed.
    if isinstance(ds, pv.MultiBlock):
        # Combine all blocks that actually contain cells/points
        blocks = [b for b in ds if b is not None and getattr(b, "n_points", 0) > 0]
        if not blocks:
            raise SystemExit("MultiBlock has no non-empty blocks to combine.")
        ds = blocks[0]
        for b in blocks[1:]:
            ds = ds.merge(b, merge_points=True)

    if ds.n_points == 0:
        raise SystemExit("Dataset has 0 points.")

    # Extract ALL edges (includes internal edges)
    edges_pd = ds.extract_all_edges()

    if edges_pd.n_points == 0:
        raise SystemExit("Edge extraction produced 0 points (unexpected).")

    # Points for PLY come from the extracted-edges polydata (consistent indexing for lines)
    pts = edges_pd.points
    points_xyz: List[Tuple[float, float, float]] = [(float(p[0]), float(p[1]), float(p[2])) for p in pts]

    # Convert VTK line cells -> pair edges
    raw_edges = parse_polydata_lines_to_edges(edges_pd.lines)

    if args.dedup:
        # Deduplicate undirected edges
        seen: Set[Tuple[int, int]] = set()
        deduped: List[Tuple[int, int]] = []
        for a, b in raw_edges:
            aa, bb = (a, b) if a < b else (b, a)
            if (aa, bb) not in seen:
                seen.add((aa, bb))
                deduped.append((aa, bb))
        raw_edges = deduped

    if args.max_edges and len(raw_edges) > args.max_edges:
        random.seed(args.seed)
        raw_edges = random.sample(raw_edges, args.max_edges)

    write_ply_with_edges(out_path, points_xyz, raw_edges)

    print(f"Wrote: {out_path}")
    print(f"Vertices: {len(points_xyz)}")
    print(f"Edges:    {len(raw_edges)}")
    print("\nTip: Open in MeshLab. Use Render -> Show Edges / Wireframe as needed.")

if __name__ == "__main__":
    main()

"""
python vtk_to_internal_edges_ply.py "C:\cfMesh-v1.1.1\MSYS\home\4y5t6\tutorials\cartesianMesh\asmoOctree\VTK\asmoOctree_0.vtk" vtk-edges.ply --dedup
"""
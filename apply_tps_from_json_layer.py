#!/usr/bin/env python3
import argparse
import json
import vtk
import numpy as np


def read_stl(path: str) -> vtk.vtkPolyData:
    r = vtk.vtkSTLReader()
    r.SetFileName(path)
    r.Update()
    out = vtk.vtkPolyData()
    out.ShallowCopy(r.GetOutput())
    return out


def write_stl(path: str, poly: vtk.vtkPolyData) -> None:
    w = vtk.vtkSTLWriter()
    w.SetFileTypeToBinary()
    w.SetFileName(path)
    w.SetInputData(poly)
    w.Write()


def build_tps(p0: np.ndarray, p1: np.ndarray) -> vtk.vtkThinPlateSplineTransform:
    src = vtk.vtkPoints()
    dst = vtk.vtkPoints()
    n = int(p0.shape[0])

    src.SetNumberOfPoints(n)
    dst.SetNumberOfPoints(n)

    for i in range(n):
        src.SetPoint(i, float(p0[i, 0]), float(p0[i, 1]), float(p0[i, 2]))
        dst.SetPoint(i, float(p1[i, 0]), float(p1[i, 1]), float(p1[i, 2]))

    tps = vtk.vtkThinPlateSplineTransform()
    tps.SetSourceLandmarks(src)
    tps.SetTargetLandmarks(dst)
    tps.SetBasisToR()
    tps.Update()
    return tps


def apply_transform_in_bbox(poly: vtk.vtkPolyData,
                            tps: vtk.vtkAbstractTransform,
                            bbox_min: np.ndarray,
                            bbox_max: np.ndarray) -> tuple[vtk.vtkPolyData, int]:
    pts = poly.GetPoints()
    npts = pts.GetNumberOfPoints()

    xmin, ymin, zmin = map(float, bbox_min)
    xmax, ymax, zmax = map(float, bbox_max)

    new_pts = vtk.vtkPoints()
    new_pts.SetNumberOfPoints(npts)

    changed = 0
    for i in range(npts):
        x, y, z = pts.GetPoint(i)
        if (xmin <= x <= xmax) and (ymin <= y <= ymax) and (zmin <= z <= zmax):
            x2, y2, z2 = tps.TransformPoint((x, y, z))
            new_pts.SetPoint(i, float(x2), float(y2), float(z2))
            changed += 1
        else:
            new_pts.SetPoint(i, float(x), float(y), float(z))

    out = vtk.vtkPolyData()
    out.ShallowCopy(poly)
    out.SetPoints(new_pts)
    out.Modified()
    return out, changed


def load_pairs_from_json(json_path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    layers = data.get("layers", [])
    if not layers:
        raise RuntimeError("JSON has no layers")

    p0_all = []
    p1_all = []
    for layer in layers:
        for pr in layer.get("points", []):
            p0_all.append(pr["p0"])
            p1_all.append(pr["p1"])

    if not p0_all:
        raise RuntimeError("JSON contains no point pairs")

    return np.asarray(p0_all, dtype=float), np.asarray(p1_all, dtype=float)


def dedup_by_p0(p0: np.ndarray, p1: np.ndarray, ndp: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove duplicate pairs where p0 are equal (within rounding).
    Keeps the first occurrence.
    """
    seen = set()
    keep_idx = []
    for i in range(p0.shape[0]):
        k = (round(float(p0[i, 0]), ndp),
             round(float(p0[i, 1]), ndp),
             round(float(p0[i, 2]), ndp))
        if k in seen:
            continue
        seen.add(k)
        keep_idx.append(i)

    keep_idx = np.asarray(keep_idx, dtype=int)
    return p0[keep_idx], p1[keep_idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--input-stl", required=True)
    ap.add_argument("--output-stl", required=True)
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--bbox-pad", type=float, default=0.0)
    ap.add_argument("--dedup-ndp", type=int, default=6)
    ap.add_argument("--write-intermediate", action="store_true")
    args = ap.parse_args()

    poly = read_stl(args.input_stl)

    p0, p1 = load_pairs_from_json(args.json)
    print(f"[CHUNK] loaded pairs: {len(p0)}")

    p0, p1 = dedup_by_p0(p0, p1, ndp=args.dedup_ndp)
    print(f"[CHUNK] after dedup: {len(p0)}")

    # sort bottom->top by Z of p0
    order = np.argsort(p0[:, 2])
    p0 = p0[order]
    p1 = p1[order]

    n = p0.shape[0]
    chunk = max(1, int(args.chunk))
    pad = float(args.bbox_pad)

    for k in range(0, n, chunk):
        p0c = p0[k:k + chunk]
        p1c = p1[k:k + chunk]
        if p0c.shape[0] < 4:
            continue  # TPS needs at least a few points

        bbox_min = p0c.min(axis=0) - pad
        bbox_max = p0c.max(axis=0) + pad

        print(f"[CHUNK] {k//chunk:04d}: points={len(p0c)} "
              f"bbox_min={bbox_min} bbox_max={bbox_max}")

        tps = build_tps(p0c, p1c)
        poly, changed = apply_transform_in_bbox(poly, tps, bbox_min, bbox_max)
        print(f"[CHUNK] {k//chunk:04d}: moved {changed} STL vertices")

        if args.write_intermediate:
            out_i = args.output_stl
            if out_i.lower().endswith(".stl"):
                out_i = out_i[:-4] + f"_chunk_{k//chunk:04d}.stl"
            else:
                out_i = out_i + f"_chunk_{k//chunk:04d}.stl"
            write_stl(out_i, poly)

    write_stl(args.output_stl, poly)
    print(f"[CHUNK] Done. Wrote: {args.output_stl}")


if __name__ == "__main__":
    main()

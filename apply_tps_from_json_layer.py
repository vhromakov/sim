#!/usr/bin/env python3
import argparse
import json
import vtk

import numpy as np

def build_tps(p0, p1):
    src = vtk.vtkPoints()
    dst = vtk.vtkPoints()
    n = len(p0)

    src.SetNumberOfPoints(n)
    dst.SetNumberOfPoints(n)

    i = 0

    for j in range(len(p0)):
        src.SetPoint(i, float(p0[j][0]), float(p0[j][1]), float(p0[j][2]))
        dst.SetPoint(i, float(p1[j][0]), float(p1[j][1]), float(p1[j][2]))
        i += 1

    tps = vtk.vtkThinPlateSplineTransform()
    tps.SetSourceLandmarks(src)
    tps.SetTargetLandmarks(dst)
    tps.SetBasisToR()  # 3D TPS
    tps.Update()
    return tps

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


def apply_transform(poly: vtk.vtkPolyData, tps: vtk.vtkAbstractTransform) -> vtk.vtkPolyData:
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputData(poly)
    tf.SetTransform(tps)
    tf.Update()
    out = vtk.vtkPolyData()
    out.ShallowCopy(tf.GetOutput())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="layer_surface_displacements.json")
    ap.add_argument("--input-stl", required=True)
    ap.add_argument("--output-stl", required=True)
    ap.add_argument(
        "--min-points",
        type=int,
        default=10,
        help="Skip layers with fewer control points than this",
    )
    ap.add_argument(
        "--write-intermediate",
        action="store_true",
        help="Write STL after each layer: <output-stl>_slice_XXX.stl",
    )
    args = ap.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    layers = data.get("layers", [])
    if not layers:
        raise RuntimeError("JSON has no layers")

    # Sort by slice_idx so it’s deterministic
    layers = sorted(layers, key=lambda L: int(L.get("slice_idx", 0)))

    poly = read_stl(args.input_stl)

    for index, layer in enumerate(layers):
        s = int(layer["slice_idx"])
        pairs = layer["points"]
        p0 = np.asarray([pr["p0"] for pr in pairs], dtype=float)
        p1 = np.asarray([pr["p1"] for pr in pairs], dtype=float)

        tps = build_tps(p0, p1)
        poly = apply_transform(poly, tps)  # your existing global apply

        if args.write_intermediate:
            # insert before extension
            if args.output_stl.lower().endswith(".stl"):
                out_i = args.output_stl[:-4] + f"_slice_{s:03d}.stl"
            else:
                out_i = args.output_stl + f"_slice_{s:03d}.stl"
            write_stl(out_i, poly)
            print(f"[TPS] Slice {s:03d}: wrote {out_i}")

    write_stl(args.output_stl, poly)
    print(f"[TPS] Done. Wrote final STL: {args.output_stl}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import vtk


def load_ctrl_points_from_ffd_json(path: str) -> np.ndarray:
    """
    Load your FFD JSON and return control point positions as an array
    of shape (N, 3) in world coordinates.

    JSON format:
      {
        "nx": int,
        "ny": int,
        "nz": int,
        "box_origin": [ox, oy, oz],
        "box_length": [lx, ly, lz],
        "mu_x": [[[...]], ...],   # (nx, ny, nz)
        "mu_y": [[[...]], ...],
        "mu_z": [[[...]], ...]
      }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nx = int(data["nx"])
    ny = int(data["ny"])
    nz = int(data["nz"])

    ox, oy, oz = map(float, data["box_origin"])
    lx, ly, lz = map(float, data["box_length"])

    mu_x = np.array(data["mu_x"], dtype=float)
    mu_y = np.array(data["mu_y"], dtype=float)
    mu_z = np.array(data["mu_z"], dtype=float)

    expected = (nx, ny, nz)
    if mu_x.shape != expected or mu_y.shape != expected or mu_z.shape != expected:
        raise ValueError(
            f"{path}: mu_x/mu_y/mu_z shapes {mu_x.shape},{mu_y.shape},{mu_z.shape} "
            f"do not match (nx,ny,nz)={expected}"
        )

    ctrl = np.zeros((nx, ny, nz, 3), dtype=float)

    # parametric coords in [0,1]
    alphas = np.linspace(0.0, 1.0, nx) if nx > 1 else np.zeros(1)
    betas  = np.linspace(0.0, 1.0, ny) if ny > 1 else np.zeros(1)
    gammas = np.linspace(0.0, 1.0, nz) if nz > 1 else np.zeros(1)

    for i in range(nx):
        base_x = ox + alphas[i] * lx
        for j in range(ny):
            base_y = oy + betas[j] * ly
            for k in range(nz):
                base_z = oz + gammas[k] * lz

                dx = mu_x[i, j, k]
                dy = mu_y[i, j, k]
                dz = mu_z[i, j, k]

                ctrl[i, j, k, 0] = base_x + dx
                ctrl[i, j, k, 1] = base_y + dy
                ctrl[i, j, k, 2] = base_z + dz

    # Flatten to (N, 3) – the relative ordering must match
    # between the two JSONs, which it will if both were
    # exported by the same code.
    return ctrl.reshape(-1, 3)


def deform_stl_with_tps(
    input_stl: str,
    ffd_json_1: str,
    ffd_json_2: str,
    output_stl: str,
):
    """
    Use VTK Thin Plate Spline transform to deform an STL:

      - source landmarks: control points from ffd_json_1
      - target landmarks: control points from ffd_json_2
      - deformation applied to all STL vertices
    """

    # --- Load control point sets ---
    pts1 = load_ctrl_points_from_ffd_json(ffd_json_1)
    pts2 = load_ctrl_points_from_ffd_json(ffd_json_2)

    if pts1.shape != pts2.shape:
        raise ValueError(
            f"FFD lattices must have same number of points, got {pts1.shape} vs {pts2.shape}"
        )

    n = pts1.shape[0]

    # --- Build VTK landmark sets ---
    src_pts = vtk.vtkPoints()
    dst_pts = vtk.vtkPoints()
    src_pts.SetNumberOfPoints(n)
    dst_pts.SetNumberOfPoints(n)

    for i in range(n):
        x1, y1, z1 = pts1[i]
        x2, y2, z2 = pts2[i]
        src_pts.SetPoint(i, x1, y1, z1)
        dst_pts.SetPoint(i, x2, y2, z2)

    # --- Thin Plate Spline transform (3D) ---
    tps = vtk.vtkThinPlateSplineTransform()
    tps.SetSourceLandmarks(src_pts)
    tps.SetTargetLandmarks(dst_pts)
    tps.SetBasisToR()  # 3D TPS (r basis)

    # --- Read STL ---
    reader = vtk.vtkSTLReader()
    reader.SetFileName(input_stl)
    reader.Update()

    # --- Apply transform to STL ---
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputConnection(reader.GetOutputPort())
    tf.SetTransform(tps)
    tf.Update()

    deformed = tf.GetOutput()

    # --- Write deformed STL ---
    out_dir = os.path.dirname(os.path.abspath(output_stl))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    writer = vtk.vtkSTLWriter()
    writer.SetFileTypeToBinary()
    writer.SetFileName(output_stl)
    writer.SetInputData(deformed)
    writer.Write()

    print(f"[VTK] Deformed STL written to {output_stl}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Deform an STL using VTK Thin Plate Spline based on "
            "two FFD JSON lattices (source → target)."
        )
    )
    parser.add_argument("--input-stl", required=True, help="Input STL file")
    parser.add_argument(
        "--ffd-json-1",
        required=True,
        help="FFD JSON whose control points hug the original STL (source lattice)",
    )
    parser.add_argument(
        "--ffd-json-2",
        required=True,
        help="FFD JSON with target control point positions (target lattice)",
    )
    parser.add_argument(
        "--output-stl",
        required=True,
        help="Output deformed STL file",
    )

    args = parser.parse_args()

    deform_stl_with_tps(
        input_stl=args.input_stl,
        ffd_json_1=args.ffd_json_1,
        ffd_json_2=args.ffd_json_2,
        output_stl=args.output_stl,
    )


if __name__ == "__main__":
    main()

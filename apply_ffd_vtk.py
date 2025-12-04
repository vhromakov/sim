#!/usr/bin/env python3
import json
import argparse
import numpy as np
import vtk
from vtk.util import numpy_support as nps


def flatten_mu(mu_list, nx, ny, nz, name="mu"):
    flat = []
    for i in range(len(mu_list)):
        plane = mu_list[i]
        for j in range(len(plane)):
            row = plane[j]
            flat.extend(row)

    flat = np.asarray(flat, dtype=np.float32)
    expected = nx * ny * nz
    if flat.size != expected:
        raise ValueError(
            f"{name}: expected {expected} values (nx*ny*nz = {nx}*{ny}*{nz}), "
            f"got {flat.size}"
        )
    return flat.reshape((nx, ny, nz))


# ----------------------------
# Load JSON
# ----------------------------
def load_ffd_json(path):
    with open(path, "r") as f:
        data = json.load(f)

    nx = int(data["nx"])
    ny = int(data["ny"])
    nz = int(data["nz"])

    box_origin = np.array(data["box_origin"], dtype=np.float32)
    box_length = np.array(data["box_length"], dtype=np.float32)

    mu_x = flatten_mu(data["mu_x"], nx, ny, nz, name="mu_x")
    mu_y = flatten_mu(data["mu_y"], nx, ny, nz, name="mu_y")
    mu_z = flatten_mu(data["mu_z"], nx, ny, nz, name="mu_z")

    # (nx, ny, nz, 3)
    mu = np.stack([mu_x, mu_y, mu_z], axis=-1)

    return nx, ny, nz, box_origin, box_length, mu


# ----------------------------
# Build VTK GridTransform
# ----------------------------
def build_vtk_grid(nx, ny, nz, box_origin, box_length, mu):
    """
    mu shape: (nx, ny, nz, 3)
    Build vtkImageData with 3-component *Scalars* as displacement field.
    """

    image = vtk.vtkImageData()
    image.SetDimensions(nx, ny, nz)
    image.SetSpacing(
        float(box_length[0]) / float(nx - 1),
        float(box_length[1]) / float(ny - 1),
        float(box_length[2]) / float(nz - 1),
    )
    image.SetOrigin(
        float(box_origin[0]),
        float(box_origin[1]),
        float(box_origin[2])
    )

    # vtkGridTransform expects: 3-component scalars (x,y,z displacement)
    mu_flat = mu.reshape(-1, 3)  # (nx*ny*nz, 3)

    scalars = nps.numpy_to_vtk(mu_flat, deep=True, array_type=vtk.VTK_FLOAT)
    scalars.SetNumberOfComponents(3)
    scalars.SetName("Displacement")

    image.GetPointData().SetScalars(scalars)

    grid_transform = vtk.vtkGridTransform()
    grid_transform.SetDisplacementGridData(image)
    grid_transform.SetInterpolationModeToLinear()
    # Optional: scale / shift if needed
    # grid_transform.SetDisplacementScale(1.0)
    # grid_transform.SetDisplacementShift(0.0)

    return grid_transform


# ----------------------------
# Apply transformation to STL
# ----------------------------
def apply_ffd_to_stl(input_stl, json_path, output_stl):

    print("[INFO] Loading FFD JSON...")
    nx, ny, nz, box_origin, box_length, mu = load_ffd_json(json_path)
    print(f"[INFO] Grid size: nx={nx}, ny={ny}, nz={nz}")

    print("[INFO] Building VTK GridTransform...")
    grid_transform = build_vtk_grid(nx, ny, nz, box_origin, box_length, mu)

    print("[INFO] Loading STL mesh...")
    reader = vtk.vtkSTLReader()
    reader.SetFileName(input_stl)
    reader.Update()
    polydata = reader.GetOutput()

    if polydata is None or polydata.GetNumberOfPoints() == 0:
        raise RuntimeError("Failed to load STL or STL has no points.")

    print("[INFO] Applying FFD deformation (VTK C++ fast path)...")
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetTransform(grid_transform)
    transform_filter.SetInputData(polydata)
    transform_filter.Update()

    print("[INFO] Writing output STL...")
    writer = vtk.vtkSTLWriter()
    writer.SetFileTypeToBinary()
    writer.SetFileName(output_stl)
    writer.SetInputData(transform_filter.GetOutput())
    writer.Write()

    print(f"[DONE] Deformed STL saved to {output_stl}")


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_stl")
    parser.add_argument("ffd_json")
    parser.add_argument("output_stl")

    args = parser.parse_args()
    apply_ffd_to_stl(args.input_stl, args.ffd_json, args.output_stl)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import vtk

import numpy as np
import trimesh


import numpy as np
import trimesh


import numpy as np
import trimesh


def get_random_and_shifted_vertices_from_stl(
    stl_path: str,
    n_vertices: int,
    shift_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load an STL and return:
      - random_vertices: (M, 3) array of up to n_vertices randomly selected vertices
      - shifted_vertices: same points, but only those in the "left half" of the model
        (x < bbox_center_x) are shifted toward the STL bounding-box center by a
        given factor. Others remain unchanged.

    Parameters
    ----------
    stl_path : str
        Path to STL file.
    n_vertices : int
        Number of vertices to randomly select. If n_vertices >= total vertex count,
        return all vertices.
    shift_factor : float
        How far to move selected points toward the bbox center:
          0.0 -> no movement
          1.0 -> move directly to center
          values >1.0 -> overshoot past center in that direction.

    Returns
    -------
    random_vertices : np.ndarray
        Array of shape (M, 3), original random vertices.
    shifted_vertices : np.ndarray
        Array of shape (M, 3), same vertices but with only those in the left half
        shifted toward the center.
    """

    if n_vertices <= 0:
        empty = np.empty((0, 3), dtype=float)
        return empty, empty

    mesh = trimesh.load(stl_path)

    # If it's a Scene, merge everything into a single mesh
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    vertices = np.asarray(mesh.vertices, dtype=float)
    num_vertices = vertices.shape[0]

    if num_vertices == 0:
        empty = np.empty((0, 3), dtype=float)
        return empty, empty

    # Bounding-box center of the STL
    bbox_min, bbox_max = mesh.bounds  # each is (3,)
    center = 0.5 * (bbox_min + bbox_max)
    center_x = center[0]

    # If requested more than available, use all vertices
    if n_vertices >= num_vertices:
        random_vertices = vertices
    else:
        idx = np.random.choice(num_vertices, size=n_vertices, replace=False)
        random_vertices = vertices[idx]

    # Start with shifted_vertices identical to random_vertices
    shifted_vertices = random_vertices.copy()

    # Mask: only shift points in the "left half" (x < center_x)
    mask = random_vertices[:, 0] < center_x
    if np.any(mask):
        direction_to_center = center - random_vertices[mask]
        shifted_vertices[mask] = random_vertices[mask] + shift_factor * direction_to_center

    return random_vertices, shifted_vertices



def deform_stl_with_tps(
    input_stl: str,
    output_stl: str,
):
    """
    Use VTK Thin Plate Spline transform to deform an STL:

      - source landmarks: control points from ffd_json_1
      - target landmarks: control points from ffd_json_2
      - deformation applied to all STL vertices
    """

    # --- Load control point sets ---
    pts1, pts2 = get_random_and_shifted_vertices_from_stl(input_stl, 10000, 0.3)

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
        "--output-stl",
        required=True,
        help="Output deformed STL file",
    )

    args = parser.parse_args()

    deform_stl_with_tps(
        input_stl=args.input_stl,
        output_stl=args.output_stl,
    )


if __name__ == "__main__":
    main()

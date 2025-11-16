#!/usr/bin/env python3
import argparse
import os
from typing import List, Tuple, Dict

import numpy as np
import trimesh


# ------------------------------------------------------------
# Utility: write MEDIT .mesh file with Tetrahedra only
# ------------------------------------------------------------

def write_medit_tet_mesh(
    path: str,
    vertices: List[Tuple[float, float, float]],
    tets: List[Tuple[int, int, int, int]],
) -> None:
    """
    Write a 3D MEDIT .mesh file with:
      - Vertices
      - Tetrahedra

    All items are given region id 1.
    """
    with open(path, "w") as f:
        f.write("MeshVersionFormatted 2\n\n")
        f.write("Dimension 3\n\n")

        # Vertices
        f.write("Vertices\n")
        f.write(f"{len(vertices)}\n")
        for (x, y, z) in vertices:
            # last number = reference/region id
            f.write(f"{x} {y} {z} 1\n")
        f.write("\n")

        # Tetrahedra
        f.write("Tetrahedra\n")
        f.write(f"{len(tets)}\n")
        for (n1, n2, n3, n4) in tets:
            # 4 vertex indices + region id
            f.write(f"{n1} {n2} {n3} {n4} 1\n")
        f.write("\nEnd\n")


# ------------------------------------------------------------
# Orientation fix for tets
# ------------------------------------------------------------

def _fix_tet_orientation(
    vertices: List[Tuple[float, float, float]],
    tets: List[Tuple[int, int, int, int]],
    eps: float = 1e-12,
) -> List[Tuple[int, int, int, int]]:
    """
    Ensure all tetrahedra have positive signed volume.

    For each tet (a,b,c,d):
      - compute signed volume using vertex coordinates
      - if volume < 0, swap two vertices to flip orientation
      - if |volume| < eps, drop the tet (degenerate)

    Returns a new list of tets.
    """
    verts = np.asarray(vertices, dtype=float)
    fixed: List[Tuple[int, int, int, int]] = []

    for (a, b, c, d) in tets:
        pa = verts[a - 1]
        pb = verts[b - 1]
        pc = verts[c - 1]
        pd = verts[d - 1]

        # signed volume ~ det([b-a, c-a, d-a]) / 6
        v1 = pb - pa
        v2 = pc - pa
        v3 = pd - pa
        det = np.dot(np.cross(v1, v2), v3)

        if abs(det) < eps:
            # degenerate or nearly so – skip
            continue

        if det < 0.0:
            # flip orientation by swapping two vertices (b,c)
            fixed.append((a, c, b, d))
        else:
            fixed.append((a, b, c, d))

    return fixed


# ------------------------------------------------------------
# Main voxel-based meshing logic (cube -> 6 correct tets)
# ------------------------------------------------------------

def generate_cubic_tet_mesh_for_layer(
    layer_indices: np.ndarray,
    vox: "trimesh.voxel.VoxelGrid",
    cube_size: float,
) -> Tuple[List[Tuple[float, float, float]],
           List[Tuple[int, int, int, int]]]:
    """
    Given voxel indices for one Z-layer (iz fixed), build:

    - shared vertex list (unique)
    - list of tetrahedral elements, 6 per voxel cube

    Voxel indices: array of shape (N, 3) with columns [ix, iy, iz].
    """
    centers = vox.indices_to_points(layer_indices.astype(float))
    half = cube_size / 2.0

    # Map logical grid-vertex key -> 1-based vertex index
    vertex_index_map: Dict[Tuple[int, int, int], int] = {}
    vertices: List[Tuple[float, float, float]] = []
    tets: List[Tuple[int, int, int, int]] = []

    def get_vertex_index(key: Tuple[int, int, int],
                         coord: Tuple[float, float, float]) -> int:
        """
        Return 1-based vertex index for given logical key.
        Create new vertex if needed.
        """
        if key in vertex_index_map:
            return vertex_index_map[key]
        idx = len(vertices) + 1
        vertex_index_map[key] = idx
        vertices.append(coord)
        return idx

    for (ix, iy, iz), center in zip(layer_indices, centers):
        cx, cy, cz = center
        x0, x1 = cx - half, cx + half
        y0, y1 = cy - half, cy + half
        z0, z1 = cz - half, cz + half

        # Corner vertices (shared across cubes in the slice)
        v0 = get_vertex_index((ix,   iy,   0), (x0, y0, z0))
        v1 = get_vertex_index((ix+1, iy,   0), (x1, y0, z0))
        v2 = get_vertex_index((ix+1, iy+1, 0), (x1, y1, z0))
        v3 = get_vertex_index((ix,   iy+1, 0), (x0, y1, z0))

        v4 = get_vertex_index((ix,   iy,   1), (x0, y0, z1))
        v5 = get_vertex_index((ix+1, iy,   1), (x1, y0, z1))
        v6 = get_vertex_index((ix+1, iy+1, 1), (x1, y1, z1))
        v7 = get_vertex_index((ix,   iy+1, 1), (x0, y1, z1))

        # Geometric 6-tet decomposition of cube (v0..v7):
        #   T1: (v0, v1, v3, v4)
        #   T2: (v1, v2, v3, v6)
        #   T3: (v1, v3, v4, v6)
        #   T4: (v1, v4, v5, v6)
        #   T5: (v3, v4, v6, v7)
        #   T6: (v2, v3, v6, v7)
        tets.append((v0, v1, v3, v4))
        tets.append((v1, v2, v3, v6))
        tets.append((v1, v3, v4, v6))
        tets.append((v1, v4, v5, v6))
        tets.append((v3, v4, v6, v7))
        tets.append((v2, v3, v6, v7))

    # Fix orientation of all tets so Jacobian > 0
    tets = _fix_tet_orientation(vertices, tets)

    return vertices, tets


def stl_to_cubic_slice_meshes_tet(
    input_stl: str,
    output_prefix: str,
    cube_size: float,
) -> List[str]:
    """
    Voxelize an STL and export one MEDIT .mesh file per Z-layer (slice).

    Each output .mesh:
      - represents a structured voxel grid,
      - uses 6 tetrahedra per cube (conforming, fills the cube),
      - has no separate shell part (no Triangles section).
    """
    if cube_size <= 0:
        raise ValueError("cube_size must be positive")

    # Load the mesh
    mesh = trimesh.load(input_stl)
    if not isinstance(mesh, trimesh.Trimesh):
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
        else:
            raise ValueError("Failed to load STL as a valid 3D mesh")

    print(f"Loaded mesh from {input_stl}")
    print(f"Watertight: {mesh.is_watertight}, bounding box size: {mesh.extents}")

    print(f"Voxelizing with cube size = {cube_size} ...")
    vox = mesh.voxelized(pitch=cube_size)
    vox.fill()

    indices = vox.sparse_indices
    if indices.size == 0:
        print("No voxels found – check cube size or input mesh.")
        return []

    total_voxels = indices.shape[0]
    print(f"Total filled voxels (cubes): {total_voxels}")

    # Unique Z indices (layers) and sort bottom → top by physical Z
    unique_z = np.unique(indices[:, 2])
    layer_info = []
    for iz in unique_z:
        idx_arr = np.array([[0, 0, iz]], dtype=float)
        z_center = float(vox.indices_to_points(idx_arr)[0, 2])
        layer_info.append((iz, z_center))
    layer_info.sort(key=lambda x: x[1])

    output_mesh_files: List[str] = []
    slice_counter = 0

    for iz, z_center in layer_info:
        layer_mask = (indices[:, 2] == iz)
        layer_indices = indices[layer_mask]
        if layer_indices.shape[0] == 0:
            continue

        print(
            f"Building voxel tet mesh for slice {slice_counter} "
            f"(iz={iz}, z≈{z_center:.3f}, cubes={layer_indices.shape[0]})"
        )

        vertices, tets = generate_cubic_tet_mesh_for_layer(layer_indices, vox, cube_size)
        if not tets:
            print(f"  Slice {slice_counter} has no tet elements, skipping.")
            continue

        mesh_path = f"{output_prefix}_z{slice_counter:03d}.mesh"
        write_medit_tet_mesh(mesh_path, vertices, tets)
        output_mesh_files.append(mesh_path)

        print(
            f"  Wrote {len(vertices)} vertices and {len(tets)} tetrahedra "
            f"to {mesh_path}"
        )

        slice_counter += 1

    print(f"Done. Exported {slice_counter} non-empty slice meshes.")
    return output_mesh_files


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Voxelize an STL and generate MEDIT .mesh files with tetrahedra "
            "for each Z slice (6 tets per voxel cube, fixed orientation)."
        )
    )
    parser.add_argument("input_stl", help="Path to input STL file")
    parser.add_argument(
        "output_prefix",
        help=(
            "Prefix for output .mesh files, e.g. 'out/part' -> "
            "'out/part_z000.mesh', 'out/part_z001.mesh', ..."
        ),
    )
    parser.add_argument(
        "--cube-size",
        "-s",
        type=float,
        required=True,
        help="Edge length of each voxel cube (same units as STL, e.g. mm)",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input_stl):
        print(f"Input STL not found: {args.input_stl}")
        raise SystemExit(1)

    stl_to_cubic_slice_meshes_tet(args.input_stl, args.output_prefix, args.cube_size)


if __name__ == "__main__":
    main()

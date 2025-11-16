#!/usr/bin/env python3
import argparse
import os
from typing import List, Tuple, Dict

import numpy as np
import trimesh


# ------------------------------------------------------------
# Utility: write Abaqus/CalculiX .inp with C3D8R elements
# ------------------------------------------------------------

def write_abaqus_c3d8r(
    path: str,
    vertices: List[Tuple[float, float, float]],
    hexes: List[Tuple[int, int, int, int, int, int, int, int]],
    heading: str = "Voxel slice C3D8R mesh",
) -> None:
    """
    Write an Abaqus/CalculiX .inp file with:
      - *NODE
      - *ELEMENT, TYPE=C3D8R  (8-node reduced-integration brick)

    Node and element IDs are 1-based.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("*HEADING\n")
        f.write(f"{heading}\n")

        # Nodes
        f.write("*NODE\n")
        for i, (x, y, z) in enumerate(vertices, start=1):
            f.write(f"{i}, {x}, {y}, {z}\n")

        # Elements (reduced-integration bricks)
        f.write("*ELEMENT, TYPE=C3D8R, ELSET=EALL\n")
        for eid, (n0, n1, n2, n3, n4, n5, n6, n7) in enumerate(hexes, start=1):
            f.write(
                f"{eid}, {n0}, {n1}, {n2}, {n3}, "
                f"{n4}, {n5}, {n6}, {n7}\n"
            )


# ------------------------------------------------------------
# Main voxel-based meshing logic (cube -> 1 hex)
# ------------------------------------------------------------

def generate_cubic_hex_mesh_for_layer(
    layer_indices: np.ndarray,
    vox: "trimesh.voxel.VoxelGrid",
    cube_size: float,
) -> Tuple[List[Tuple[float, float, float]],
           List[Tuple[int, int, int, int, int, int, int, int]]]:
    """
    Given voxel indices for one Z-layer (iz fixed), build:

    - shared vertex list (unique)
    - list of hexahedral elements, 1 per voxel cube

    Node numbering uses the standard C3D8/C3D8R order:
      1:(x-,y-,z-), 2:(x+,y-,z-), 3:(x+,y+,z-), 4:(x-,y+,z-),
      5:(x-,y-,z+), 6:(x+,y-,z+), 7:(x+,y+,z+), 8:(x-,y+,z+)
    """
    centers = vox.indices_to_points(layer_indices.astype(float))
    half = cube_size / 2.0

    # Map logical grid-vertex key -> 1-based vertex index
    vertex_index_map: Dict[Tuple[int, int, int], int] = {}
    vertices: List[Tuple[float, float, float]] = []
    hexes: List[Tuple[int, int, int, int, int, int, int, int]] = []

    def get_vertex_index(
        key: Tuple[int, int, int],
        coord: Tuple[float, float, float],
    ) -> int:
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

        # Bottom face (z0)
        v0 = get_vertex_index((ix,   iy,   0), (x0, y0, z0))  # (x-,y-,z-)
        v1 = get_vertex_index((ix+1, iy,   0), (x1, y0, z0))  # (x+,y-,z-)
        v2 = get_vertex_index((ix+1, iy+1, 0), (x1, y1, z0))  # (x+,y+,z-)
        v3 = get_vertex_index((ix,   iy+1, 0), (x0, y1, z0))  # (x-,y+,z-)

        # Top face (z1)
        v4 = get_vertex_index((ix,   iy,   1), (x0, y0, z1))  # (x-,y-,z+)
        v5 = get_vertex_index((ix+1, iy,   1), (x1, y0, z1))  # (x+,y-,z+)
        v6 = get_vertex_index((ix+1, iy+1, 1), (x1, y1, z1))  # (x+,y+,z+)
        v7 = get_vertex_index((ix,   iy+1, 1), (x0, y1, z1))  # (x-,y+,z+)

        hexes.append((v0, v1, v2, v3, v4, v5, v6, v7))

    return vertices, hexes


def stl_to_cubic_slice_meshes_c3d8r(
    input_stl: str,
    output_prefix: str,
    cube_size: float,
) -> List[str]:
    """
    Voxelize an STL and export one Abaqus/CalculiX .inp file per Z-layer (slice).

    Each output .inp:
      - represents a structured voxel grid,
      - uses 1 C3D8R (8-node reduced-integration brick) per voxel cube.
    """
    if cube_size <= 0:
        raise ValueError("cube_size must be positive")

    # Load mesh
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

    # Unique Z indices (layers)
    unique_z = np.unique(indices[:, 2])

    # Compute physical Z centers to sort bottom -> top
    layer_info: List[Tuple[int, float]] = []
    for iz in unique_z:
        idx_arr = np.array([[0, 0, iz]], dtype=float)
        z_center = float(vox.indices_to_points(idx_arr)[0, 2])
        layer_info.append((iz, z_center))
    layer_info.sort(key=lambda x: x[1])

    output_inp_files: List[str] = []
    slice_counter = 0

    for iz, z_phys in layer_info:
        layer_mask = (indices[:, 2] == iz)
        layer_indices = indices[layer_mask]
        if layer_indices.shape[0] == 0:
            continue

        print(
            f"Building voxel C3D8R mesh for slice {slice_counter} "
            f"(iz={iz}, z={z_phys:.3f}, cubes={layer_indices.shape[0]})"
        )

        vertices, hexes = generate_cubic_hex_mesh_for_layer(
            layer_indices, vox, cube_size
        )
        if not hexes:
            print(f"  Slice {slice_counter} has no hex elements, skipping.")
            continue

        inp_path = f"{output_prefix}_z{slice_counter:03d}.inp"
        heading = (
            f"C3D8R voxel slice {slice_counter} "
            f"(iz={iz}, z={z_phys:.3f})"
        )
        write_abaqus_c3d8r(inp_path, vertices, hexes, heading=heading)
        output_inp_files.append(inp_path)

        print(
            f"  Wrote {len(vertices)} nodes and {len(hexes)} C3D8R elements "
            f"to {inp_path}"
        )

        slice_counter += 1

    print(f"Done. Exported {slice_counter} non-empty slice meshes (.inp).")
    return output_inp_files


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Voxelize an STL and generate Abaqus/CalculiX .inp files with "
            "C3D8R hexahedra for each Z slice "
            "(1 brick per voxel cube)."
        )
    )
    parser.add_argument("input_stl", help="Path to input STL file")
    parser.add_argument(
        "output_prefix",
        help=(
            "Prefix for output .inp files, e.g. 'out/part' -> "
            "'out/part_z000.inp', 'out/part_z001.inp', ..."
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

    stl_to_cubic_slice_meshes_c3d8r(
        args.input_stl,
        args.output_prefix,
        args.cube_size,
    )


if __name__ == "__main__":
    main()

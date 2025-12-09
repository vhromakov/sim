#!/usr/bin/env python3
import gmsh
import sys
import os


def mesh_stl_to_volume(stl_path: str, mesh_size: float | None = None) -> str:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)

    try:
        gmsh.model.add("stl_volume")

        # Import STL as a surface mesh
        gmsh.merge(stl_path)

        # Classify and create geometry from STL mesh
        angle = 40  # angle between surface normals (deg) to consider an edge a crease
        force_parametrizable_patches = True
        include_boundary = True
        curve_angle = 180

        # --- KEY CHANGE: positional args, no keywords ---
        gmsh.model.mesh.classifySurfaces(
            angle,
            include_boundary,
            force_parametrizable_patches,
            curve_angle,
        )
        # ------------------------------------------------

        gmsh.model.mesh.createGeometry()

        # Get all surfaces and make one volume
        surfs = gmsh.model.getEntities(2)
        sloop = [s[1] for s in surfs]
        gmsh.model.geo.addSurfaceLoop(sloop, 1)
        gmsh.model.geo.addVolume([1], 1)

        # Synchronize before meshing
        gmsh.model.geo.synchronize()

        # Optional: set global mesh size
        if mesh_size is not None:
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

        # Generate 3D mesh (tetra)
        gmsh.model.mesh.generate(3)

        # Decide output path
        base, _ = os.path.splitext(stl_path)
        out_path = base + "_vol.msh"

        gmsh.write(out_path)
        print(f"Written volumetric mesh to: {out_path}")
        return out_path

    finally:
        gmsh.finalize()


def main():
    if len(sys.argv) < 2:
        print("Usage: python mesh_from_stl.py input.stl [mesh_size]")
        sys.exit(1)

    stl_path = sys.argv[1]
    if not os.path.isfile(stl_path):
        print(f"Error: file not found: {stl_path}")
        sys.exit(1)

    mesh_size = None
    if len(sys.argv) >= 3:
        try:
            mesh_size = float(sys.argv[2])
        except ValueError:
            print(f"Warning: invalid mesh_size '{sys.argv[2]}', ignoring.")

    mesh_stl_to_volume(stl_path, mesh_size)


if __name__ == "__main__":
    main()

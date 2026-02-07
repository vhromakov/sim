#!/usr/bin/env python3
import gmsh
import math
import os

# === Params ===
input_stl = "MODELS/torus.stl"   # set your STL path here
output_msh = "split_halves.msh"

gmsh.initialize()
gmsh.model.add("split_stl_example")

# ------------------------------------------------------------------
# 1. Load STL as a surface mesh
# ------------------------------------------------------------------
if not os.path.exists(input_stl):
    raise FileNotFoundError(f"STL file not found: {input_stl}")

gmsh.merge(input_stl)

# ------------------------------------------------------------------
# 2. Classify surfaces and create geometry from the STL mesh (geo kernel)
# ------------------------------------------------------------------
angle = 40 * math.pi / 180.0

gmsh.model.mesh.classifySurfaces(
    angle,
    boundary=True,
    forReparametrization=True,
    curveAngle=180,
)

gmsh.model.mesh.createGeometry()
gmsh.model.geo.synchronize()

surfaces = gmsh.model.getEntities(2)
if not surfaces:
    gmsh.finalize()
    raise RuntimeError("No 2D entities found after creating geometry from STL.")

surf_loop = gmsh.model.geo.addSurfaceLoop([s[1] for s in surfaces])
stl_volume_geo = gmsh.model.geo.addVolume([surf_loop])
gmsh.model.geo.synchronize()

# ------------------------------------------------------------------
# 3. Copy the GEO volume into OCC so we can use OCC booleans
# ------------------------------------------------------------------
copied = gmsh.model.occ.copy([(3, stl_volume_geo)])
gmsh.model.occ.synchronize()

stl_volume_occ = copied[0][1]

# ------------------------------------------------------------------
# 4. Bounding box of the OCC volume and creation of two OCC boxes
#    We split by X: plane x = xmid
# ------------------------------------------------------------------
xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(3, stl_volume_occ)
xmid = 0.5 * (xmin + xmax)

dx_left = xmid - xmin
dx_right = xmax - xmid
dy = ymax - ymin
dz = zmax - zmin

if dx_left <= 0 or dx_right <= 0 or dy <= 0 or dz <= 0:
    gmsh.finalize()
    raise RuntimeError("Degenerate bounding box, cannot split.")

left_box = gmsh.model.occ.addBox(
    xmin,
    ymin,
    zmin,
    dx_left,
    dy,
    dz,
)

right_box = gmsh.model.occ.addBox(
    xmid,
    ymin,
    zmin,
    dx_right,
    dy,
    dz,
)

gmsh.model.occ.synchronize()

# ------------------------------------------------------------------
# 5. Boolean fragment: cut the STL volume with the two boxes (OCC)
# ------------------------------------------------------------------
gmsh.model.occ.fragment(
    [(3, stl_volume_occ)],
    [(3, left_box), (3, right_box)],
)
gmsh.model.occ.synchronize()

# ------------------------------------------------------------------
# 6. Physical groups for the two resulting volumes
# ------------------------------------------------------------------
volumes = gmsh.model.occ.getEntities(3)

if len(volumes) < 2:
    print(f"Warning: got {len(volumes)} OCC volumes after fragment (expected >= 2).")

# Sort by center-of-mass X so we know which one is "left" and "right"
def volume_center_x(dim_tag):
    dim, tag = dim_tag
    bxmin, bymin, bzmin, bxmax, bymax, bzmax = gmsh.model.getBoundingBox(dim, tag)
    return 0.5 * (bxmin + bxmax)

volumes_sorted = sorted(volumes, key=volume_center_x)

left_tag = volumes_sorted[0][1]
right_tag = volumes_sorted[1][1]

gmsh.model.addPhysicalGroup(3, [left_tag], 1)
gmsh.model.setPhysicalName(3, 1, "LeftHalf")

gmsh.model.addPhysicalGroup(3, [right_tag], 2)
gmsh.model.setPhysicalName(3, 2, "RightHalf")

# ------------------------------------------------------------------
# 7. Generate volumetric mesh and write output
# ------------------------------------------------------------------
gmsh.model.mesh.generate(3)
gmsh.write(output_msh)

print(f"Wrote volumetric mesh with split halves to: {output_msh}")

gmsh.finalize()

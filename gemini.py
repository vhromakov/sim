import gmsh

gmsh.initialize()
gmsh.model.add("split_mesh_example")

# Define coordinates for a cube that will be split along the y-z plane at x=0
# We need points for both halves

# Left half points (x <= 0)
p1 = gmsh.model.occ.addPoint(-1, 0, 0, 0.1)
p2 = gmsh.model.occ.addPoint(0, 0, 0, 0.1) # Common boundary point
p3 = gmsh.model.occ.addPoint(0, 1, 0, 0.1) # Common boundary point
p4 = gmsh.model.occ.addPoint(-1, 1, 0, 0.1)
p5 = gmsh.model.occ.addPoint(-1, 0, 1, 0.1)
p6 = gmsh.model.occ.addPoint(0, 0, 1, 0.1) # Common boundary point
p7 = gmsh.model.occ.addPoint(0, 1, 1, 0.1) # Common boundary point
p8 = gmsh.model.occ.addPoint(-1, 1, 1, 0.1)

# ... lines and surfaces for the left volume ...
# (This is tedious to do manually, generally you use higher level functions
# like gmsh.model.occ.addBox and boolean operations)

# A more robust method for complex shapes involves using the boolean
# operations with a cutting tool (a plane or another box).

# Create a full solid body first
full_box = gmsh.model.occ.addBox(-1, 0, 0, 2, 1, 1) # A box from x=-1 to x=1

# Create two cutting boxes that define the two halves
cut_box_left = gmsh.model.occ.addBox(-1, 0, 0, 1, 1, 1) # x=-1 to x=0
cut_box_right = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1) # x=0 to x=1

gmsh.model.occ.synchronize()

# Use boolean fragmentation to split the original volume into two distinct volumes
# We keep the original box and fragment it with the two half-boxes
# The result will be two volumes sharing a conformal mesh boundary (the y-z plane at x=0)
gmsh.model.occ.fragment([(3, full_box)], [(3, cut_box_left), (3, cut_box_right)])

gmsh.model.occ.synchronize()

# Now, assign physical groups to the two resulting volumes to distinguish them
# The fragment operation might renumber entities, we can find the new volumes:
volumes = gmsh.model.occ.getEntities(dim=3)

if len(volumes) == 2:
    gmsh.model.addPhysicalGroup(3, [volumes[0][1]], 1, "LeftHalf")
    gmsh.model.addPhysicalGroup(3, [volumes[1][1]], 2, "RightHalf")
else:
    print("Error: Fragmentation did not result in exactly two volumes.")

# Generate the mesh
gmsh.model.mesh.generate(3)

# Write the file (this single .msh file will contain elements grouped by their physical volumes)
gmsh.write("split_halves.msh")

gmsh.finalize()

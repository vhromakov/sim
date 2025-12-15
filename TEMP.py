import numpy as np
import trimesh
from trimesh import repair
import point_cloud_utils as pcu
import pymesh


# ---------- Helpers ----------

def compute_edge_stats(mesh: trimesh.Trimesh):
    """Return (num_boundary_edges, num_nonmanifold_edges) for a triangle mesh."""
    faces = mesh.faces

    e01 = faces[:, [0, 1]]
    e12 = faces[:, [1, 2]]
    e20 = faces[:, [2, 0]]

    edges = np.vstack((e01, e12, e20))
    edges = np.sort(edges, axis=1)

    edges_unique, counts = np.unique(edges, axis=0, return_counts=True)

    num_boundary_edges = int(np.sum(counts == 1))
    num_nonmanifold_edges = int(np.sum(counts > 2))
    return num_boundary_edges, num_nonmanifold_edges


def heal_component_trimesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    'Current workflow' healing applied to a single connected component:
    - remove duplicate/degenerate faces
    - remove unreferenced verts, merge verts
    - fix normals
    - fill holes once
    (We avoid crazy iterative nonmanifold surgery here, and let PyMesh+boolean
     handle self-intersections.)
    """
    print("  [HEAL] faces(before):", len(mesh.faces), "verts:", len(mesh.vertices))

    # Deduplicate faces
    mesh.update_faces(mesh.unique_faces())
    # Remove degenerate (zero-area) faces
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    repair.fix_normals(mesh)

    # Fill obvious boundary loops
    repair.fill_holes(mesh)
    mesh.remove_unreferenced_vertices()

    nb, nn = compute_edge_stats(mesh)
    print("  [HEAL] faces(after):", len(mesh.faces), "verts:", len(mesh.vertices))
    print("  [HEAL] boundary_edges:", nb, "nonmanifold_edges:", nn)
    print("  [HEAL] trimesh.is_watertight:", mesh.is_watertight)

    return mesh


def trimesh_to_pymesh(mesh: trimesh.Trimesh):
    """Convert a Trimesh to a PyMesh mesh."""
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    return pymesh.form_mesh(v, f)


def pymesh_cleanup(mesh):
    """Basic PyMesh cleanup after boolean."""
    mesh, _ = pymesh.remove_duplicated_vertices(mesh, tol=1e-12)
    mesh, _ = pymesh.remove_duplicated_faces(mesh)
    mesh, _ = pymesh.remove_degenerated_triangles(mesh, 100)
    mesh, _ = pymesh.remove_isolated_vertices(mesh)
    mesh = pymesh.resolve_self_intersection(mesh)
    mesh, _ = pymesh.remove_duplicated_vertices(mesh, tol=1e-12)
    mesh, _ = pymesh.remove_duplicated_faces(mesh)
    mesh, _ = pymesh.remove_isolated_vertices(mesh)
    return mesh


def pymesh_boolean_union(meshes):
    """Boolean union of a list of PyMesh meshes."""
    assert len(meshes) >= 1
    acc = meshes[0]
    for i, m in enumerate(meshes[1:], start=1):
        print(f"[BOOL] Union {i}/{len(meshes)-1} ...")
        acc = pymesh.boolean(acc, m, operation="union", engine="igl")
        acc = pymesh_cleanup(acc)
        print("       -> result verts:", acc.num_vertices, "faces:", acc.num_faces)
    return acc



# ---------- MAIN ----------

input_path = "MODELS/CSC16_U00P_.stl"

mesh = trimesh.load(input_path, process=False)
if isinstance(mesh, trimesh.Scene):
    mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])

print("[INFO] Loaded:")
print("  faces:", len(mesh.faces))
print("  verts:", len(mesh.vertices))
print("  watertight:", mesh.is_watertight)

# Split into connected components BEFORE heavy surgery
pieces = mesh.split()  # older trimesh versions: just split()
print("[INFO] Split into", len(pieces), "components")

healed_trimesh_components = []
pymesh_components = []

for idx, comp in enumerate(pieces):
    print(f"\n[COMP {idx}] start:")
    print("  faces:", len(comp.faces), "verts:", len(comp.vertices))
    comp_healed = heal_component_trimesh(comp)
    healed_trimesh_components.append(comp_healed)

    pm = trimesh_to_pymesh(comp_healed)
    print(
        f"  [COMP {idx}] PyMesh form: verts={pm.num_vertices}, faces={pm.num_faces}"
    )
    pymesh_components.append(pm)

if len(pymesh_components) == 0:
    raise RuntimeError("No components found after split.")

# Boolean union of all healed components
print("\n[BOOL] Starting boolean union of all components...")
union_pm = pymesh_boolean_union(pymesh_components)
print("[BOOL] Final union:")
print("  verts:", union_pm.num_vertices, "faces:", union_pm.num_faces)

# Convert union back to numpy arrays / Trimesh / PCU
union_v = union_pm.vertices
union_f = union_pm.faces.astype(np.int64)

union_trimesh = trimesh.Trimesh(union_v, union_f, process=False)
nb, nn = compute_edge_stats(union_trimesh)
print("\n[FINAL UNION STATS]")
print("  faces:", len(union_trimesh.faces))
print("  verts:", len(union_trimesh.vertices))
print("  watertight (trimesh):", union_trimesh.is_watertight)
print("  boundary edges:", nb)
print("  nonmanifold edges:", nn)

# Export for your slicing/caps pipeline
union_trimesh.export("WATER_union_components.ply")
pcu.save_mesh_vf("WATER_union_components_pcuvf.ply", union_v, union_f)

import open3d as o3d

# Load the SMPL mesh
smpl_mesh = o3d.io.read_triangle_mesh("smplx_uv.obj")
# smpl_mesh.compute_vertex_normals() # will give specular effects

# Load texture
smpl_mesh.textures = [o3d.io.read_image("smplx_uv_altas_colored.png")]

# Visualize
o3d.visualization.draw_geometries([smpl_mesh])

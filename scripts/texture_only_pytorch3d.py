import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
    HardPhongShader,
    TexturesVertex,
    TexturesUV,
)
import cv2
import numpy as np


from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_view_transform,
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
    HardPhongShader,
    PointLights,
)
from pytorch3d.renderer.mesh.shader import ShaderBase

from ipdb import set_trace as st

# === Custom Shader for Texture-only (no lighting)
class TextureOnlyShader(ShaderBase):
    def forward(self, fragments, meshes, **kwargs):
        
        texture = meshes.textures.sample_textures(fragments)
        # print(type(meshes.textures))
        
        return texture

def render_configurable(
    mesh,
    image_size=512,
    device="cuda",
    shader_mode="original",  # options: "original", "albedo_only", "texture_only"
    background_color=(1.0, 1.0, 1.0),  # add this
):
    # === 1. Camera: fixed view
    R, T = look_at_view_transform(dist=2.5, elev=0, azim=0)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # === 2. Rasterizer
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

    # === 3. Shader based on mode
    if shader_mode == "original":
        # Default HardPhongShader with implicit lights
        shader = HardPhongShader(device=device, cameras=cameras)

    elif shader_mode == "albedo_only":
        # Pure ambient light = approximate albedo
        lights = PointLights(
            device=device,
            ambient_color=((1.0, 1.0, 1.0)),
            diffuse_color=((0.0, 0.0, 0.0)),
            specular_color=((0.0, 0.0, 0.0)),
            location=[[0.0, 0.0, 3.0]],
        )
        shader = HardPhongShader(device=device, cameras=cameras, lights=lights)

    elif shader_mode == "texture_only":
        # Bypass lighting, directly use UV texture color
        shader = TextureOnlyShader(device=device, cameras=cameras)

    else:
        raise ValueError(f"Unknown shader_mode: {shader_mode}")

    # === 4. Renderer
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

    # === 5. Render
    with torch.no_grad():
        images = renderer(mesh)

    print("images:", images.shape)
    if shader_mode == "texture_only":
        # For texture-only shader, we need to remove the alpha channel
        return images[0, ..., 0, :3].cpu().numpy()
    elif shader_mode == "original":
        return images[0,..., :3].cpu().numpy()


def render_safe_vertex_albedo(obj_path, image_size=256, device="cuda"):
    # 1. Load the mesh
    verts, faces, aux = load_obj(obj_path, load_textures=False)
    verts = verts.to(device)
    faces_idx = faces.verts_idx.to(device)
    
    _, faces, aux = load_obj("smplx_uv.obj", load_textures=False)
    verts_uvs = aux.verts_uvs.numpy()
    faces_uvs = faces.textures_idx.numpy()

    texture_path = "smplx_uv_altas_colored_i_no_flip.png"
    texture_image = cv2.imread(texture_path)[:, :, ::-1]  # BGR to RGB

    pure_color_texture = False
    if pure_color_texture:
        mask = (texture_image > 0).any(axis=-1, keepdims=False)
        # print("mask", mask.shape, mask.min(), mask.max(), mask.mean(axis=(0,1)))
        texture_image[mask] = 125
    
    
    print("texture_image (unique values)", np.unique(texture_image))
    # print("texture_image", texture_image.shape, texture_image.min(), texture_image.max(), texture_image.mean(axis=(0,1)))
    
    verts_uvs = torch.from_numpy(verts_uvs).float().unsqueeze(0).to(device) # (1, Vt, 2)
    faces_uvs = torch.from_numpy(faces_uvs).long().unsqueeze(0).to(device)  # (1, F, 3)

    # Load texture image
    texture_image = texture_image.astype(np.float32) / 255.0
    texture_tensor = torch.from_numpy(texture_image).unsqueeze(0).to(device)  # (1, 3, H, W)
    # print("texture_tensor:", texture_tensor.shape)
    
    textures = TexturesUV(
        maps=texture_tensor,
        faces_uvs=faces_uvs,
        verts_uvs=verts_uvs,
        sampling_mode="nearest",  # "bilinear" or "nearest"
    )
    mesh = Meshes(verts=[verts], faces=[faces_idx], textures=textures)
    
        
    shader_mode="texture_only"
    # shader_mode="albedo_only"
    # shader_mode="original"
    
    image = render_configurable(mesh, image_size=512, device="cuda", 
                                shader_mode=shader_mode,
                                )
    return image

obj_path = "smplx_uv.obj"
obj_path = "/home/liu-compute/Downloads/CatDensepose/smplx_params/0000/mesh_smplx.obj"

image = render_safe_vertex_albedo(obj_path, image_size=256, device="cuda")
print("image:", image.shape, image[...,0].min(), image[...,0].max())

image = (image * 255).astype(np.uint8)
print("image 255 unique values", np.unique(image))
# Save the image
output_path = "texture_only_p3d_posed_smplx.png"
cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
print(f"Saved: {output_path}")

# import matplotlib.pyplot as plt
# plt.imshow(image)
# plt.axis("off")
# plt.title("Vertex Color Albedo (Red)")
# plt.show()

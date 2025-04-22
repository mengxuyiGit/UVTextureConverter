import torch
import numpy as np
import cv2
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
    ShaderBase,
    TexturesUV,
)
import matplotlib.pyplot as plt

class NoLightShader(ShaderBase):
    def forward(self, fragments, meshes, **kwargs):
        # No lighting, directly sample texture via UV
        return meshes.textures.sample_textures(fragments)

def render_smpl_with_texture(
    verts,            # [V, 3] SMPL vertices
    faces,            # [F, 3] triangle indices
    verts_uvs,        # [Vt, 2] UV coordinates for texture vertices
    faces_uvs,        # [F, 3] UV triangle indices
    texture_image,    # [H, W, 3] RGB UV texture map, in numpy format
    image_size=512,   # Render resolution
    device='cuda',
):
    # Convert inputs to tensors
    verts = torch.from_numpy(verts).float().unsqueeze(0).to(device)
    faces = torch.from_numpy(faces).long().unsqueeze(0).to(device)
    verts_uvs = torch.from_numpy(verts_uvs).float().unsqueeze(0).to(device)
    faces_uvs = torch.from_numpy(faces_uvs).long().unsqueeze(0).to(device)

    # Prepare texture
    texture_image = texture_image.astype(np.float32) / 255.0
    texture_tensor = torch.from_numpy(texture_image).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, H, W)

    # Create texture object
    textures = TexturesUV(
        maps=texture_tensor,
        faces_uvs=faces_uvs,
        verts_uvs=verts_uvs
    )

    # Mesh
    mesh = Meshes(verts=verts, faces=faces, textures=textures)

    # Camera at identity
    cameras = FoVPerspectiveCameras(device=device)

    # Rasterizer and shader
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    shader = NoLightShader(device=device, cameras=cameras)
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

    # Render
    images = renderer(mesh)
    return images[0, ..., :3].cpu().numpy()  # Remove batch and alpha

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
import einops


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

import numpy as np

def generate_orbit_cameras(num_views=8, pos=[-1, 0, 1], dist=3.2, radius=0.55):
    """
    Args:
        num_views (int): Number of views per elevation layer.
        pos (list): List of elevation shifts [-1, 0, 1] typically.
        dist (float): Distance from camera to origin center.
        radius (float): Radius to control focal length.
        
    Returns:
        Rs: (N, 3, 3) torch-compatible rotation matrices
        Ts: (N, 3) torch-compatible translation vectors
    """
    all_R = []
    all_T = []

    for p in pos:
        for k in range(num_views):
            # Horizontal orbit (azimuthal)
            y0 = k * 2 * np.pi / num_views
            sy = np.sin(y0)
            cy = np.cos(y0)
            R_C_y = np.array([
                [ cy, 0.0,  sy],
                [0.0, 1.0, 0.0],
                [-sy, 0.0, cy]
            ], dtype=np.float32)

            # Vertical tilt (here fixed at 0 elevation)
            x0 = 0.0
            sx = np.sin(x0)
            cx = np.cos(x0)
            R_C_x = np.array([
                [1.0, 0.0, 0.0],
                [0.0, cx, -sx],
                [0.0, sx,  cx]
            ], dtype=np.float32)

            # Camera location
            loc = np.array([0, p, dist], dtype=np.float32)
            loc = (R_C_x @ R_C_y).T @ loc

            # Target always (0, p, 0)
            # Not directly used here, but blender uses it for look-at

            # Camera extrinsics (world-to-camera)
            R = R_C_x @ R_C_y
            T = -R @ loc

            # Blender to OpenCV convention flip
            Flip = np.diag([-1, 1, -1]).astype(np.float32)
            R = Flip @ R
            T = Flip @ T

            all_R.append(R)
            all_T.append(T)

    all_R = np.stack(all_R, axis=0)  # (N, 3, 3)
    all_T = np.stack(all_T, axis=0)  # (N, 3)

    return all_R, all_T


import numpy as np
import torch
from pytorch3d.renderer import PerspectiveCameras

def generate_orbit_cameras_with_intrinsics(
    num_views=8,
    pos=[-1, 0, 1],
    dist=3.2,
    radius=0.55,
    img_size=(2048, 1500),  # (width, height)
):
    """
    Args:
        num_views (int): Number of views per elevation layer.
        pos (list): List of elevation shifts [-1, 0, 1] typically.
        dist (float): Distance from camera to origin center.
        radius (float): Radius to control focal length.
        img_size (tuple): (width, height) of rendered image.
        
    Returns:
        cameras: PerspectiveCameras ready to use
    """
    all_R = []
    all_T = []
    all_focal = []
    all_principal = []

    img_w, img_h = img_size

    for p in pos:
        for k in range(num_views):
            # Horizontal orbit (azimuthal)
            y0 = k * np.pi / num_views
            sy = np.sin(y0)
            cy = np.cos(y0)
            R_C_y = np.array([
                [ cy, 0.0,  sy],
                [0.0, 1.0, 0.0],
                [-sy, 0.0, cy]
            ], dtype=np.float32)

            # Vertical tilt (fixed at 0 elevation)
            x0 = 0.0
            sx = np.sin(x0)
            cx = np.cos(x0)
            R_C_x = np.array([
                [1.0, 0.0, 0.0],
                [0.0, cx, -sx],
                [0.0, sx,  cx]
            ], dtype=np.float32)

            # Camera location
            loc = np.array([0, p, dist], dtype=np.float32)
            loc = (R_C_x @ R_C_y).T @ loc

            # Camera extrinsics (world-to-camera)
            R = R_C_x @ R_C_y
            T = -R @ loc

            # Flip to OpenGL/Blender convention
            
            
            Flip = np.diag([-1, 1, -1]).astype(np.float32)
            R = Flip @ R
            T = Flip @ T

            all_R.append(R)
            all_T.append(T)

            # Compute focal length and principal point
            lens = 12 * np.linalg.norm(loc) / radius  # Same as Blender
            fx = fy = lens / 24 * img_h
            cx = img_w / 2
            cy = img_h / 2

            # Normalize to NDC space
            fx_norm = fx / (img_w / 2)
            fy_norm = fy / (img_h / 2)
            px_norm = (cx - img_w/2) / (img_w/2)  # 0
            py_norm = (cy - img_h/2) / (img_h/2)  # 0

            all_focal.append([fx_norm, fy_norm])
            all_principal.append([px_norm, py_norm])

    # Stack all
    R = torch.from_numpy(np.stack(all_R, axis=0)).float()  # (N, 3, 3)
    T = torch.from_numpy(np.stack(all_T, axis=0)).float()  # (N, 3)
    focal = torch.tensor(all_focal).float()                # (N, 2)
    principal = torch.tensor(all_principal).float()        # (N, 2)

    return R, T, focal, principal


import json

def load_cameras_from_json(json_path, device="cuda"):
    """
    Args:
        json_path: path to your calibration JSON file (containing R, T, K, imgSize)
        device: torch device for the output
        
    Returns:
        cameras: a pytorch3d.renderer.PerspectiveCameras object
        R: (N, 3, 3) torch Tensor of rotation matrices
        T: (N, 3) torch Tensor of translation vectors
        focal_length: (N, 2) torch Tensor of fx, fy
        principal_point: (N, 2) torch Tensor of cx, cy
        image_size: (N, 2) torch Tensor of (H, W)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    Rs = []
    Ts = []
    focals = []
    principals = []
    image_sizes = []

    for key in sorted(data.keys()):
        cam_info = data[key]

        K = cam_info["K"]
        R = cam_info["R"]
        T = cam_info["T"]
        img_size = cam_info["imgSize"]  # [H, W]

        
        Flip = np.diag([1, -1, 1]).astype(np.float32) # blender -> pytorch3d, and need to flip fx in K
        R = Flip @ R
        T = Flip @ T

        # # ## v1: offsetted rendering
        # fx = -K[0][0]# blender -> pytorch3d
        # fy = K[1][1]
        # cx = K[0][2] 
        # cy = K[1][2]


        # v2: centered rendering by manual correction, not real fix
        fx = -K[0][0] # blender -> pytorch3d
        fy = K[1][1] 
        # cx = K[0][2] * 1.375
        # cy = K[1][2] * 0.725
        cx = 1024
        cy = 750

        # ## v2.a (for smpl_uv.obj): centered rendering by manual correction, not real fix
        # fx = -K[0][0] # blender -> pytorch3d
        # fy = K[1][1]
        # cx = K[0][2] * 1.375
        # cy = K[1][2] * 0.45
        
        # # ## v3, fix with sensor size
        # print("fx, fy", K[0][0], K[1][1])
        # # f_mm = 12  # Blender focal length in mm
        # # sensor_width = 36.0  # Blender sensor width in mm
        # # img_width_px = 2048
        # # img_height_px = 1500

        # # fx_px = f_mm / sensor_width * img_width_px
        # # fy_px = f_mm / sensor_width * img_height_px  # because Blender sensor fit vertical
        
        # fx = - 12 / 36.0 * 1500
        # fy = 12 / 36.0 * 2048
        # cx = cx * 0
        # cy = cy * 0
        # print("[corrected] fx, fy", fx, fy)
        
        print("cx, cy", cx, cy)

        Rs.append(torch.tensor(R, dtype=torch.float32))
        Ts.append(torch.tensor(T, dtype=torch.float32))
        focals.append(torch.tensor([fx, fy], dtype=torch.float32))
        principals.append(torch.tensor([cx, cy], dtype=torch.float32))
        image_sizes.append(torch.tensor(img_size, dtype=torch.float32))  # (H, W)

    R = torch.stack(Rs, dim=0).to(device)  # (N, 3, 3)
    T = torch.stack(Ts, dim=0).to(device)  # (N, 3)
    focal_length = torch.stack(focals, dim=0).to(device)  # (N, 2)
    principal_point = torch.stack(principals, dim=0).to(device)  # (N, 2)
    image_size = torch.stack(image_sizes, dim=0).to(device)  # (N, 2)

    
    # After loading focal_length, principal_point, image_size
    # Normalize the principal_point from pixel to NDC
    # print("principal_point", principal_point.shape, principal_point.min(), principal_point.max(), principal_point)
    # principal_point_norm = principal_point.clone()
    # principal_point_norm[:, 0] = (principal_point[:, 0] / image_size[:, 1] - 0.5) * 2  # normalize w.r.t width
    # principal_point_norm[:, 1] = (principal_point[:, 1] / image_size[:, 0] - 0.5) * 2  # normalize w.r.t height
    # principal_point = principal_point_norm


    return R, T, focal_length, principal_point, image_size


def render_configurable(
    mesh,
    image_size=512,
    device="cuda",
    shader_mode="original",  # options: "original", "albedo_only", "texture_only"
    background_color=(1.0, 1.0, 1.0),  # add this
    num_cameras=8,
    dist=2.5,
    camera_sample_mode=None,  # options: "pytorch3d", "custom", "custom_with_intrinsics", "load_json"
):
    # === 1. Camera: multi view, fixed elev=0
    # camera_sample_mode = "custom"
    if camera_sample_mode == "pytorch3d":
        azim = np.linspace(0, 360, num_cameras, endpoint=False)  # (0, 45, 90, ..., 315)
        azim = torch.from_numpy(azim).float()
        elev = torch.zeros_like(azim)  # (8,) all zeros
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    elif camera_sample_mode == "custom":
        R_np, T_np = generate_orbit_cameras(num_views=num_cameras, pos=[0], dist=dist)
        # print("R v.s. R_np \n", R, "\n", R_np)
        # print("T v.s. T_np \n", T, "\n", T_np) 
        R = torch.from_numpy(R_np).to(device)
        T = torch.from_numpy(T_np).to(device)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    elif camera_sample_mode == "custom_with_intrinsics":
        R, T, focal, principal = generate_orbit_cameras_with_intrinsics(
            num_views=num_cameras,
            pos=[0],
            dist=dist,
            img_size=image_size,
        )

        cameras = PerspectiveCameras(
            device=device,
            R=R.to(device),
            T=T.to(device),
            focal_length=focal.to(device),
            principal_point=principal.to(device),
            in_ndc=True,
        )
    elif camera_sample_mode == "load_json":
        json_path = "test_data/camera_thuman2.json"
        device = "cuda"

        R, T, focal, principal, img_size = load_cameras_from_json(json_path, device=device)

        R = R[:num_cameras]
        T = T[:num_cameras]
        focal = focal[:num_cameras]
        principal = principal[:num_cameras]
        img_size = img_size[:num_cameras]
        
        # Make a PerspectiveCameras object
        cameras = PerspectiveCameras(
            focal_length=focal,
            principal_point=principal,
            R=R,
            T=T,
            device=device,
            in_ndc=False,  # Important: because your K is in pixel units
            image_size=img_size,
        )

        print(cameras)
        print(R.shape, T.shape, focal.shape, principal.shape, img_size)


    # === 2. Rasterizer
    # print("image size to render:", image_size)
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

    # print("images:", images.shape)
    if shader_mode == "texture_only":
        # For texture-only shader, we need to remove the alpha channel
        
        return images[..., 0, :3].cpu().numpy()
    elif shader_mode == "original":
        return images[0,..., :3].cpu().numpy()


def render_safe_vertex_albedo(obj_path, texture_path, num_views, image_size=256, device="cuda", dist=2.5, camera_sample_mode = None):
    # 1. Load the mesh
    verts, faces, aux = load_obj(obj_path, load_textures=False)
    verts = verts.to(device)
    faces_idx = faces.verts_idx.to(device)
    print("verts", verts.shape, "faces", faces_idx.shape)
    
    # st()
    normalize_mesh = False
    if normalize_mesh:
        print("normalizing mesh")
        verts -= verts.mean(axis=0, keepdims=True)  # Center the mesh
        # verts -= torch.tensor([[0, -0.4, 0]], device=verts.device)  # Center the mesh
    
    _, faces, aux = load_obj("smplx_uv.obj", load_textures=False)
    verts_uvs = aux.verts_uvs.numpy()
    faces_uvs = faces.textures_idx.numpy()

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
    


    
    verts_list = [verts]*num_views
    faces_idx_list = [faces_idx]*num_views
    verts_uvs = verts_uvs.repeat(num_views, 1, 1)  # (24, Vt, 2)
    faces_uvs = faces_uvs.repeat(num_views, 1, 1)  # (24, F, 3)
    texture_tensor = texture_tensor.repeat(num_views, 1, 1, 1)  # (24, 3, H, W)
    
    
    
    print("texture_tensor:", texture_tensor.shape)
    
    textures = TexturesUV(
        maps=texture_tensor,
        faces_uvs=faces_uvs,
        verts_uvs=verts_uvs,
        sampling_mode="nearest",  # "bilinear" or "nearest"
    )
    mesh = Meshes(verts=verts_list, faces=faces_idx_list, textures=textures)
    
        
    shader_mode="texture_only"
    # shader_mode="albedo_only"
    # shader_mode="original"
    
    image = render_configurable(mesh, image_size=image_size, device="cuda", 
                                shader_mode=shader_mode,
                                num_cameras=num_views, dist=dist,
                                camera_sample_mode = camera_sample_mode, # "pytorch3d", "custom", "custom_with_intrinsics", "load_json"
                                )
    return image

# obj_path = "smplx_uv.obj"
# obj_path = "/home/liu-compute/Downloads/CatDensepose/smplx_params/0000/mesh_smplx.obj"
# obj_path = "test_data/0000_QIAO_google/output_scaled.obj"
obj_path = "/home/liu-compute/Repo/UVTextureConverter/test_data/0480_smplx_pkl/smplx_param_myvis_scaled_back.obj"
# obj_path = "/home/liu-compute/Repo/UVTextureConverter/test_data/0480/smplx_mesh/mesh_smplx.obj"

# smpl_texture_path = "smplx_uv_altas_colored_i_no_flip.png"
# smpl_texture_path = "smplx_uv_altas_colored_u_no_flip.png"
# smpl_texture_path = "smplx_uv_altas_colored_v_no_flip.png"
smpl_texture_path = "smplx_uv_altas_colored_iuv_no_flip.png"

num_views = 16
images_nviews = render_safe_vertex_albedo(obj_path, smpl_texture_path, num_views, image_size=(2048, 1500), device="cuda", 
                                        dist=2, camera_sample_mode = "load_json")
print("images_nviews:", images_nviews.shape, images_nviews.min(), images_nviews.max())
# save each view independently
out_dir = "texture_only_p3d_posed_views"
import os
os.makedirs(out_dir, exist_ok=True)
for i, img in enumerate(images_nviews):
    img = (img * 255).astype(np.uint8)
    print("img", img.shape, img.min(), img.max())
    # Save the image
    output_path = f"{out_dir}/smplx_0480_{i:04d}.png"
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Saved: {output_path}")


image =  einops.rearrange(images_nviews, "b h w c -> h (b w) c")
print("image:", image.shape, image[...,0].min(), image[...,0].max())

image = (image * 255).astype(np.uint8)
print("image 255 unique values", np.unique(image))

# Save the image
output_path = "texture_only_p3d_posed_smplx.png"
cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
print(f"Saved: {output_path}")


## check alignment by overlay with the RGB renderings
import glob
from PIL import Image
# image_folder = "/home/liu-compute/Repo/UVTextureConverter/test_data/0000"
image_folder = "/home/liu-compute/Repo/UVTextureConverter/test_data/0480"
images_paths = sorted(glob.glob(f"{image_folder}/*.png"))

check_overlay = True
if check_overlay:
    im1 = [np.array(Image.open(_img_path)) for _img_path in images_paths]
    im1 = Image.fromarray(np.concatenate(im1, axis=1))
    print("im1", im1.size, im1.mode)
    im2 = Image.open(output_path)
    print("im2", im2.size, im2.mode)
    im_overlay = Image.blend(im1, im2, alpha=0.5)
    im_overlay.save("texture_overlay.png")
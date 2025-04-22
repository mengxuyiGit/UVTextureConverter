import open3d as o3d
import numpy as np
import cv2
import open3d.visualization.rendering as rendering

def render_smplx(mesh_path, texture_path, camera_views, output_dir="renders", lighting=False, normals=False, 
                save_as_grid=False, grid_size=(2, 2), save_as_npy=False, save_as_video=False):
    """
    Renders the SMPL-X mesh with texture from given camera views.
    
    Args:
        mesh_path (str): Path to the SMPLX mesh (.obj).
        texture_path (str): Path to the texture image (.png or .jpg).
        camera_views (list): List of camera extrinsic matrices (4x4 numpy arrays).
        output_dir (str): Directory to save rendered images.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Load the SMPL mesh
    smpl_mesh = o3d.io.read_triangle_mesh(mesh_path)
    if normals:
        smpl_mesh.compute_vertex_normals()  # Compute normals if needed
        print("Computed vertex normals for the mesh.")
    # smpl_mesh.textures = [o3d.io.read_image(texture_path)]
    
    # mesh_center = smpl_mesh.get_center()
    mesh_center = np.array([0, -0.2, 0])
    
    print(f"Mesh center: {mesh_center}")
    move = "cam"
    if move == "mesh":
        smpl_mesh.translate(-mesh_center)
        look_at_center = np.array([0, 0, 0])  # Look at the origin after centering the mesh
    elif move == "cam":
        # Move the camera to the mesh center
        look_at_center = mesh_center
    # smpl_mesh.translate((0, 0.2, 0))  # Shift it down to see if anything changes

    
    # # Visualize
    # o3d.visualization.draw_geometries([smpl_mesh])


    material = rendering.MaterialRecord()
    if lighting:
        print("Lighting is enabled.")
        material.shader = "defaultLit"
    else:
        print("Lighting is disabled.")
        material.shader = "defaultUnlit"
        
    material.albedo_img = o3d.io.read_image(texture_path)
    # material.base_color = [1.0, 1.0, 1.0, 1.0] 
    material.sRGB_color = False  # Use linear color space instead of sRGB


    scene = rendering.OffscreenRenderer(1024, 1024)
    scene.scene.add_geometry("mesh", smpl_mesh, material)
    scene.scene.set_background(np.array([0.0, 0.0, 0.0, 1.0]))  # Set background to black

    # Define intrinsics (assume a simple pinhole camera model)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=1024, height=1024, fx=1000, fy=1000, cx=512, cy=512
    )
    
    images = []

    for i, extrinsic in enumerate(camera_views):
        # Set camera parameters
        # scene.scene.camera.set_projection(intrinsic.intrinsic_matrix, 0.1, 10.0)
        # Fix the projection setup
        scene.scene.camera.set_projection(
            intrinsic.intrinsic_matrix,  # 3x3 intrinsic matrix
            0.1,  # Near plane
            10.0, # Far plane
            intrinsic.width,  # Image width
            intrinsic.height  # Image height
        )


        scene.scene.camera.look_at(
            # center=[0, -0.2, 0],  # Target (center of the mesh)
            center=look_at_center,  # Target (center of the mesh)
            eye=extrinsic[:3, 3],  # Camera position
            up=extrinsic[:3, 1]  # Camera up direction
        )

        gamma_correction = True
        # Render the image
        if gamma_correction:
            image = scene.render_to_image()
            image_np = np.asarray(image)
        else:
            ## inverse gamma correction
            image = np.asarray(scene.render_to_image(), dtype=np.float32) / 255.0
            image = image ** (1.0 / 2.2)  # Apply inverse gamma correction
            image_np = (image * 255).astype(np.uint8)
        
        # Save or store the image
        if save_as_npy or save_as_grid:
            images.append(image_np)
        else:
            output_path = os.path.join(output_dir, f"render_{i}.png")
            cv2.imwrite(output_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            print(f"Saved: {output_path}")


    # Save as grid if needed
    if save_as_grid:
        grid_output_path = os.path.join(output_dir, "render_grid.png")
        save_images_as_grid(images, grid_size, grid_output_path)

    # Save as .npy if needed
    if save_as_npy:
        npy_output_path = os.path.join(output_dir, "renders.npy")
        np.save(npy_output_path, np.array(images))
        print(f"Saved images as .npy at {npy_output_path}")
        
    if save_as_video:
        video_output_path = os.path.join(output_dir, "render_video.mp4")
        save_images_as_video(images, video_output_path, fps=30)


def save_images_as_grid(images, grid_size, output_path):
    """Helper function to save images as a grid."""
    height, width, _ = images[0].shape
    grid_image = np.zeros((grid_size[0] * height, grid_size[1] * width, 3), dtype=np.uint8)
    
    for i, image in enumerate(images):
        row = i // grid_size[1]
        col = i % grid_size[1]
        grid_image[row * height: (row + 1) * height, col * width: (col + 1) * width] = image

    cv2.imwrite(output_path, cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR))
    print(f"Saved grid image at {output_path}")


def save_images_as_video(images, output_path, fps=30):
    """
    Saves a sequence of images as a video.

    Parameters:
        images (list of np.array): List of images (H, W, 3).
        output_path (str): Path to save the output video (should end with .mp4).
        fps (int): Frames per second for the video.
    """
    if not images:
        print("No images to save as video.")
        return

    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img in images:
        video_writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # Convert to BGR for OpenCV

    video_writer.release()
    print(f"Saved video at {output_path}")


import numpy as np

def look_at_matrix(camera_position, target, up=np.array([0, 1, 0])):
    """
    Creates a 4x4 transformation matrix to make the camera look at a target.

    Parameters:
        camera_position (ndarray): (3,) Camera world position.
        target (ndarray): (3,) Target position (usually the origin).
        up (ndarray): (3,) World up direction.

    Returns:
        ndarray: (4,4) Camera transformation matrix.
    """
    forward = target - camera_position
    forward /= np.linalg.norm(forward)  # Normalize forward vector

    right = np.cross(up, forward)
    right /= np.linalg.norm(right)  # Normalize right vector

    up = np.cross(forward, right)  # Recalculate true up vector

    # Construct the transformation matrix
    cam_matrix = np.eye(4)
    cam_matrix[:3, 0] = right
    cam_matrix[:3, 1] = up
    cam_matrix[:3, 2] = forward
    cam_matrix[:3, 3] = camera_position

    return cam_matrix

def generate_camera_trajectory(radius=2.0, height=3.0, steps=4):
    """
    Generates a 360-degree camera trajectory around an object, ensuring 
    the camera always looks at the object center.

    Parameters:
        radius (float): Distance of the camera from the object center.
        height (float): Height of the camera above the object.
        steps (int): Number of steps (e.g., 4 for 90-degree increments).

    Returns:
        List of 4x4 camera transformation matrices.
    """
    angles = np.linspace(0, 2 * np.pi, steps, endpoint=False)  # 0°, 90°, 180°, 270°
    target = np.array([0, -4.0, 0])  # Mesh center at the origin
    camera_poses = []

    for angle in angles:
        # Compute the camera position in world coordinates
        cam_pos = np.array([radius * np.cos(angle), height, radius * np.sin(angle)])
        
        # Compute the look-at matrix
        cam_matrix = look_at_matrix(cam_pos, target)

        camera_poses.append(cam_matrix)

    return camera_poses


if __name__ == "__main__":
    # Example camera extrinsics (3 views)
    # camera_views = [
    #     np.array([[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 3], [0, 0, 0, 1]]),  # Front
    #     np.array([[1, 0, 0, -2], [0, 1, 0, 0], [0, 0, 1, 3], [0, 0, 0, 1]]),  # Back
    #     np.array([[0, -1, 0, 0], [1, 0, 0, -2], [0, 0, 1, 3], [0, 0, 0, 1]])   # Side
    # ]
    
    
    # Generate and print camera trajectories
    camera_views = generate_camera_trajectory(radius=3.0, height=0.0, steps=8)
    for idx, view in enumerate(camera_views):
        print(f"Camera {idx}:\n{view}\n")


    # render_smplx("smplx_uv.obj", 
    #             # "/home/liu-compute/Repo/UVTextureConverter/smplx_uv_altas_colored.png", # uv
    #             "smplx_uv_altas_colored_i_no_flip.png", # i
    #             camera_views,
    #             output_dir="renders_I",
    #             lighting=False,
    #             save_as_grid=True,
    #             grid_size=(4, 8),
    #             save_as_video=True,
    # )

    
    render_smplx("smplx_uv.obj", 
                # "/home/liu-compute/Repo/UVTextureConverter/smplx_uv_altas_colored.png", # uv
                "smplx_uv_altas_colored_i_no_flip.png", # i
                camera_views,
                output_dir="renders_debug2",
                # lighting=True, normals=True,
                # save_as_grid=True,
                # grid_size=(4, 8),
                # save_as_video=True,
    )


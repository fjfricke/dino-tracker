import torch

from mesh_video_generator import MeshVideoGenerator
from pytorch3d.renderer import PerspectiveCameras

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

import plotly.graph_objects as go

def screen_to_ndc_depth(depth_map, image_size):
    """
    Converts screen-space (H, W) coordinates to NDC coordinates while keeping depth in NDC.

    Args:
        depth_map (torch.Tensor): (B, H, W, 1) depth values in NDC.
        image_size (tuple): (H, W) Image height and width.

    Returns:
        torch.Tensor: (B, H, W, 3) points in NDC space.
    """
    B, H, W, _ = depth_map.shape
    device = depth_map.device

    # Generate pixel grid
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij"
    )

    # Normalize to NDC (-1 to 1 range)
    x_ndc = 1 - (xx / (W - 1)) * 2  # Flip and scale  x to [-1, 1]
    y_ndc = 1 - (yy / (H - 1)) * 2  # Flip and scale to [-1, 1]

    # Expand for batch size
    x_ndc = x_ndc.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
    y_ndc = y_ndc.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)

    # Depth remains unchanged
    z_ndc = depth_map.squeeze(-1)  # Remove last dim (B, H, W)

    # Stack to (B, H, W, 3)
    ndc_points = torch.stack([x_ndc, y_ndc, z_ndc], dim=-1)

    return ndc_points

def plot_3d_points_interactive(points, title="3D Point Cloud"):
    """
    Plots a list of 3D points using Plotly for an interactive visualization.

    Args:
        points (numpy.ndarray or torch.Tensor): Shape (N, 3), where each row is (x, y, z).
        title (str): Title of the plot.
    """
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0], 
        y=points[:, 1], 
        z=points[:, 2], 
        mode='markers',
        marker=dict(size=5, color=points[:, 2], colorscale='Viridis', opacity=0.8)
    )])

    # Set the axis ranges to [-1, 1]
    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=dict(range=[-1, 1]),
            yaxis=dict(range=[-1, 1]),
            zaxis=dict(range=[-1, 1]),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                up=dict(x=0, y=1, z=0)
            )
        )
    )

    # fig.update_layout(title=title, margin=dict(l=0, r=0, b=0, t=40))
    fig.show()

def reshape_depth_map(depth_maps):
    B, H, W, D = depth_maps.shape

    # Create a grid of x, y coordinates
    y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    x_coords = x_coords.to(depth_maps.device)
    y_coords = y_coords.to(depth_maps.device)

    # Flatten the coordinates and depth maps
    x_coords_flat = x_coords.flatten().unsqueeze(0).expand(B, -1)
    y_coords_flat = y_coords.flatten().unsqueeze(0).expand(B, -1)
    depth_flat = depth_maps.view(B, -1, D).squeeze(-1)

    # Concatenate x, y, and depth
    coords_depth = torch.stack((x_coords_flat, y_coords_flat, depth_flat), dim=-1)

    return coords_depth

def compute_pixel_flow(world_points, cameras, image_size, mask, max_world_dist=0.1):
    """
    Computes optical flow in pixel coordinates by matching world points across frames 
    and transforming them back to screen space.

    Args:
        world_points (torch.Tensor): 
            Tensor of shape (F, H, W, 3), containing valid 3D world points per frame.
        cameras (list of PerspectiveCameras): 
            A list of PyTorch3D cameras per frame.
        image_size (tuple): 
            (H, W) Image height and width.
        max_world_dist (float): 
            Maximum allowable distance in 3D world coordinates for valid matches.

    Returns:
        flow (torch.Tensor): 
            Tensor of shape (F-1, H, W, 2), representing (dx, dy) flow per pixel.
        valid_mask (torch.Tensor): 
            Tensor of shape (F-1, H, W), indicating valid matches (1=valid, 0=invalid).
    """
    N, H, W, _ = world_points.shape  # Extract frame count and spatial dimensions
    flow_list = []
    mask_list = []

    for t in range(N - 1):  # Iterate over consecutive frames
        world_t = world_points[t].cpu().numpy().reshape(-1, 3)  # (H*W, 3)
        world_tp1 = world_points[t + 1].cpu().numpy().reshape(-1, 3)  # (H*W, 3)

        # Flatten the valid mask for frame t
        mask_t = mask[t].reshape(-1).cpu().numpy().astype(bool)  # (H*W,)
        mask_tp1 = mask[t + 1].reshape(-1).cpu().numpy().astype(bool)  # (H*W,)
        # Step 1: Use KD-Tree for nearest neighbor search (only on valid points)
        world_t_valid = world_t[mask_t]  # (V, 3) only valid points
        world_tp1_valid = world_tp1[mask_tp1]  # (V, 3) only valid points

        # Step 1: Find nearest match in world coordinates using KD-Tree
        tree = cKDTree(world_tp1_valid)  # Build KDTree for fast lookup
        dists, indices = tree.query(world_t_valid)  # Match points in frame t to t+1

        # Step 2: Apply world-space distance threshold
        # valid_matches = dists < max_world_dist  # True if within threshold
        # valid_indices = indices[valid_matches]  # Indices of valid matches

        # Select valid world points
        # world_t_valid = world_t[valid_indices]  # (V, 3)
        world_tp1_valid = world_tp1_valid[indices]  # (V, 3)

        # Convert to tensors
        device = cameras[t].device
        world_t_tensor = torch.tensor(world_t_valid, dtype=torch.float32, device=device)  # (V, 3)
        world_tp1_tensor = torch.tensor(world_tp1_valid, dtype=torch.float32, device=device)  # (V, 3)

        # Step 3: Transform world coordinates to pixel-space
        screen_t = cameras[t].transform_points_screen(world_t_tensor)  # (V, 3)
        screen_tp1 = cameras[t + 1].transform_points_screen(world_tp1_tensor)  # (V, 3)

        # Step 4: Compute optical flow (dx, dy) in pixels
        flow = screen_tp1 - screen_t  # (V, 2)
        flow = flow[:, :2]

        # Create output tensors (default to zero flow)
        flow_full = torch.zeros((H * W, 2), dtype=torch.float32, device=device)
        updated_mask = torch.zeros((H * W,), dtype=torch.float32, device=device)

        # Assign computed flow only to valid matches
        valid_idx_flat = mask_t.nonzero()[0]  # Indices in (H*W) for valid points
        flow_full[valid_idx_flat] = flow  # Assign flow only to matched valid points
        updated_mask[valid_idx_flat] = 1  # Mark valid matches

        # Reshape back to (H, W)
        flow_list.append(flow_full.view(H, W, 2))  # (H, W, 2)
        mask_list.append(updated_mask.view(H, W))  # (H, W)

    # Stack across frames to get final shape (F-1, H, W, 2)
    return torch.stack(flow_list), torch.stack(mask_list)


def compute_optical_flow_with_mask(cameras, depth_maps, threshold=0.1):
    """
    Compute optical flow between consecutive frames with a validity mask.
    
    Args:
        cameras (PerspectiveCameras): PyTorch3D cameras.
        depth_maps (torch.Tensor): Depth maps of shape (N, H, W, 1).
        threshold (float): Depth difference threshold for occlusion detection
                           (not used here, but kept in signature).
    
    Returns:
        optical_flow (torch.Tensor): Optical flow of shape (N-1, H, W, 2).
        validity_mask (torch.Tensor): Binary mask (1 = valid, 0 = invalid) of shape (N-1, H, W).
    """
    N, H, W, _ = depth_maps.shape
    device = depth_maps.device

    depth_in_ndc = screen_to_ndc_depth(depth_maps, (H, W))

    # depth_squeezed = reshape_depth_map(depth_in_ndc)
    mask = depth_in_ndc[:, :, :, 2] != -1

    cameras_cpu = PerspectiveCameras(
        # focal_length=cameras.focal_length.cpu(),
        # principal_point=cameras.principal_point.cpu(),
        R=cameras.R.cpu(),
        T=cameras.T.cpu(),
        image_size=torch.tensor((H, W)).repeat(N, 1)
    )

    # depth_in_ndc = cameras_cpu.transform_points_screen(depth_squeezed)

    depth_in_ndc_flat = depth_in_ndc.view(N, H * W, 3)
    depth_in_wc_flat = cameras_cpu.unproject_points(depth_in_ndc_flat, image_size=(H, W), world_coordinates=True, from_ndc=True)
    depth_in_wc = depth_in_wc_flat.view(N, H, W, 3)

    # plot_3d_points_interactive(depth_in_wc[0])

    # mask_expanded = mask.unsqueeze(-1).expand_as(depth_in_wc)
    # filtered_tensor = depth_in_wc_flat[mask_expanded].view(-1, 3)

    # depth_in_ws_squeezed = depth_in_wc_flat.view(N * H * W, 3)

    # plot_3d_points_interactive(filtered_tensor[:1000000, :])

    # depth_in_ws_masked = depth_in_wc[mask]

    flow, valid_mask = compute_pixel_flow(depth_in_wc[[100,107], :, :, :], cameras_cpu[[100,107]], (H, W), mask[[100,107], :, :])

    # visualize_warped_images(depth_maps[12], depth_maps[13], flow[0])

    # plot_flow(flow[100], valid_mask[100])

    visualize_optical_flow_quiver(flow[0], valid_mask[0])

    return flow, valid_mask

def visualize_optical_flow_quiver(flow, mask=None, step=10):
    """
    Plots optical flow as arrows on a grid.
    
    Args:
        flow (torch.Tensor): Optical flow of shape (H, W, 2).
        step (int): Sampling step for arrows (to reduce density).
        mask (torch.Tensor, optional): Validity mask (H, W).
    """
    H, W = flow.shape[:2]
    
    # Convert flow to numpy
    flow_np = flow.cpu().numpy()
    X, Y = np.meshgrid(np.arange(0, W, step), np.arange(0, H, step))

    U = flow_np[::step, ::step, 0]  # X-displacement
    V = flow_np[::step, ::step, 1]  # Y-displacement

    plt.figure(figsize=(10, 10))
    plt.imshow(np.zeros((H, W)), cmap='gray')  # Background
    plt.quiver(X, Y, U, V, angles="xy", scale_units="xy", scale=1, color="red")
    
    if mask is not None:
        mask_np = mask.cpu().numpy()
        mask_resampled = mask_np[::step, ::step]
        plt.quiver(X[mask_resampled == 0], Y[mask_resampled == 0], U[mask_resampled == 0], V[mask_resampled == 0], 
                   angles="xy", scale_units="xy", scale=1, color="gray", alpha=0.3)

    plt.axis("off")
    plt.title("Optical Flow Quiver Plot")
    plt.show()

def warp_image(image, flow):
    """
    Warps an image using the optical flow field.
    
    Args:
        image (torch.Tensor): Input image of shape (H, W, 3).
        flow (torch.Tensor): Optical flow of shape (H, W, 2).
    
    Returns:
        warped (numpy.ndarray): Warped image.
    """
    H, W = flow.shape[:2]

    # Create pixel grid
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))

    # Get flow displacement
    flow_np = flow.cpu().numpy()
    x_new = np.clip(xx + flow_np[..., 0], 0, W-1).astype(np.float32)
    y_new = np.clip(yy + flow_np[..., 1], 0, H-1).astype(np.float32)

    # Convert image to numpy
    image_np = image.cpu().numpy()

    # Warp image
    warped = cv2.remap(image_np, x_new, y_new, interpolation=cv2.INTER_LINEAR)

    # Create mask for zero flow
    zero_flow_mask = (np.abs(flow_np[..., 0]) < 1e-1) & (np.abs(flow_np[..., 1]) < 1e-1)
    warped[zero_flow_mask] = 0  # Set pixels with zero flow to black
    
    return warped

def visualize_warped_images(image1, image2, flow):
    """
    Visualizes the warped previous image and compares it with the next frame.
    
    Args:
        image1 (torch.Tensor): Previous frame (H, W, 3).
        image2 (torch.Tensor): Next frame (H, W, 3).
        flow (torch.Tensor): Optical flow of shape (H, W, 2).
    """
    warped_image = warp_image(image1, flow)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image1.cpu().numpy())
    plt.title("Previous Frame")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(image2.cpu().numpy())
    plt.title("Next Frame")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(warped_image)
    plt.title("Warped Image (Using Flow)")
    plt.axis("off")

    plt.show()

def visualize_optical_flow_hsv(flow, mask=None):
    """
    Visualizes optical flow using an HSV colormap.
    
    Args:
        flow (torch.Tensor): Optical flow tensor of shape (H, W, 2).
        mask (torch.Tensor, optional): Validity mask (H, W), 1 for valid, 0 for invalid.
    
    Returns:
        flow_img (numpy.ndarray): Optical flow visualization (H, W, 3).
    """
    H, W = flow.shape[:2]
    
    # Convert flow to numpy
    flow_np = flow.cpu().numpy()

    # Compute magnitude and angle
    mag, ang = cv2.cartToPolar(flow_np[..., 0], flow_np[..., 1])

    # Normalize magnitude to fit in [0, 1]
    mag_norm = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)

    # Create HSV image
    hsv = np.zeros((H, W, 3), dtype=np.float32)
    hsv[..., 0] = ang * 180 / np.pi / 2  # Hue (direction)
    hsv[..., 1] = 1.0  # Saturation (full color)
    hsv[..., 2] = mag_norm  # Value (magnitude)

    # Convert to RGB
    flow_img = cv2.cvtColor((hsv * 255).astype(np.uint8), cv2.COLOR_HSV2RGB)

    # Apply mask if provided
    if mask is not None:
        mask_np = mask.cpu().numpy()
        flow_img[mask_np < 0.5] = 0  # Black out invalid pixels

    return flow_img

# Example usage
def plot_flow(flow, mask=None):
    flow_img = visualize_optical_flow_hsv(flow, mask)
    plt.figure(figsize=(10, 10))
    plt.imshow(flow_img)
    plt.axis("off")
    plt.title("Optical Flow Visualization")
    plt.show()

def load_renderings(path):
    with open(path, "rb") as f:
        return torch.load(f, map_location=torch.device("cpu"))
    
def save_video(renderings, path):
    video_gen = MeshVideoGenerator(device="cpu")
    video_gen.save_video(renderings["normal_batched_renderings"], path, fps=30, display_frames=True)
    

if __name__ == "__main__":
    files = load_renderings("./datasets/pickled_renderings/render_data_cow.pt")
    flow, mask = compute_optical_flow_with_mask(files["camera"], files["depth"])
    plot_flow(flow[25], mask[25])
    # save_video(files, "./datasets/pickled_renderings/output.mp4")
    print(files)
import argparse
import os
import cv2
import numpy as np
import torch

# from mesh_video_generator import MeshVideoGenerator
from pytorch3d.renderer import PerspectiveCameras

from scipy.spatial import cKDTree

from build_trajectories_from_flow import build_and_pad_trajectories
from create_trajectories_and_resize import create_trajectories_for_all_frames, resize_flows
from plot_flow import visualize_optical_flow_quiver, visualize_optical_flow_video


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

def compute_pixel_flow(world_points, cameras, image_size, mask, max_world_dist=0.01):
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
        valid_matches = dists < max_world_dist  # True if within threshold

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
        updated_mask = torch.zeros((H * W,), dtype=torch.bool, device=device)

        # Assign computed flow only to valid matches
        valid_idx_flat = mask_t.nonzero()[0][valid_matches]  # Indices in (H*W) for valid points
        flow_full[valid_idx_flat] = flow[valid_matches]  # Assign flow only to matched valid points
        updated_mask[valid_idx_flat] = 1  # Mark valid matches

        # Reshape back to (H, W)
        flow_list.append(flow_full.view(H, W, 2))  # (H, W, 2)
        mask_list.append(updated_mask.view(H, W))  # (H, W)

    device = cameras[0].device
    last_mask = mask[-1].to(device).bool()

    flow_stack = torch.stack(flow_list, dim=0)  # => (N-1, H, W, 2)
    mask_stack = torch.stack(mask_list, dim=0)  # => (N-1, H, W)
    
    # Append the final frame's mask
    mask_stack = torch.cat([mask_stack, last_mask.unsqueeze(0)], dim=0)
    # Stack across frames to get final shape (F-1, H, W, 2)
    return flow_stack, mask_stack


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

    depth_in_ndc_flat = depth_in_ndc.view(N, H * W, 3)
    depth_in_wc_flat = cameras_cpu.unproject_points(depth_in_ndc_flat, image_size=(H, W), world_coordinates=True, from_ndc=True)
    depth_in_wc = depth_in_wc_flat.view(N, H, W, 3)

    flow, valid_mask = compute_pixel_flow(depth_in_wc, cameras_cpu, (H, W), mask)

    # visualize_optical_flow_quiver(flow[0], valid_mask[0])

    return flow, valid_mask

def load_renderings(path):
    with open(path, "rb") as f:
        return torch.load(f, map_location=torch.device("cpu"))
    
def save_with_torch(trajectories, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        torch.save(trajectories, f)
    
# def save_video(renderings, path, filename):
#     video_gen = MeshVideoGenerator(device="cpu")
#     video_gen.save_video(renderings, path, filename, fps=30, display_frames=False)

def save_mask(masks, path, resize=False, h=476, w=854):
    os.makedirs(path, exist_ok=True)
    for i, mask in enumerate(masks):
        if resize:
            mask = cv2.resize(mask.cpu().numpy(), (w, h))
        else:
            mask = mask.cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(path, f"{i:05d}.png"), mask)

def save_video(renderings, path, resize=False, h=476, w=854):
    os.makedirs(path, exist_ok=True)
    for i, rendering in enumerate(renderings):
        if resize:
            rendering = cv2.resize(rendering.cpu().numpy(), (w, h))
        else:
            rendering = rendering.cpu().numpy()
        # Ensure rendering is in the correct format without alpha channel
        if rendering.shape[-1] == 4:  # Check if there's an alpha channel
            rendering = rendering[..., :3]  # Remove the alpha channel
        # Scale the rendering values from [0, 1] to [0, 255]
        rendering = (rendering * 255).astype(np.uint8)  # Convert to uint8
        rendering = cv2.cvtColor(rendering, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(path, f"{i:05d}.png"), rendering)

def main(data_path, input_path, dino_h, dino_w):
    files = load_renderings(input_path)

    save_video(files["renderings"], os.path.join(data_path, "video"), resize=True)

    flows, masks = compute_optical_flow_with_mask(files["camera"], files["depth"])

    visualize_optical_flow_video(flows, masks, output_path=os.path.join(data_path, "flow_video.mp4"))
    flows_resized, masks_resized = resize_flows(flows, masks, h=dino_h, w=dino_w)
    trajectories = build_and_pad_trajectories(flows_resized, masks_resized)
    save_with_torch(trajectories, os.path.join(data_path, "of_trajectories/fg_trajectories.pt"))
    save_mask(masks_resized, os.path.join(data_path, "masks"), resize=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="dataset/rendered_mesh_output", type=str)
    parser.add_argument("--input-path", default="dataset/rendered_mesh_output/rendered_mesh_output_cow.pt", type=str)
    parser.add_argument("--dino-h", default=476, type=int)
    parser.add_argument("--dino-w", default=854, type=int)

    args = parser.parse_args()

    # Call the refactored main function
    main(args.data_path, args.input_path, args.dino_h, args.dino_w)

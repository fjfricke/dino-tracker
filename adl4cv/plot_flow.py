import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

# import plotly.graph_objects as go

# def plot_3d_points_interactive(points, title="3D Point Cloud"):
#     """
#     Plots a list of 3D points using Plotly for an interactive visualization.

#     Args:
#         points (numpy.ndarray or torch.Tensor): Shape (N, 3), where each row is (x, y, z).
#         title (str): Title of the plot.
#     """
#     if isinstance(points, torch.Tensor):
#         points = points.cpu().numpy()

#     fig = go.Figure(data=[go.Scatter3d(
#         x=points[:, 0], 
#         y=points[:, 1], 
#         z=points[:, 2], 
#         mode='markers',
#         marker=dict(size=5, color=points[:, 2], colorscale='Viridis', opacity=0.8)
#     )])

#     # Set the axis ranges to [-1, 1]
#     fig.update_layout(
#         title=title,
#         margin=dict(l=0, r=0, b=0, t=40),
#         scene=dict(
#             xaxis=dict(range=[-1, 1]),
#             yaxis=dict(range=[-1, 1]),
#             zaxis=dict(range=[-1, 1]),
#             aspectmode='cube',
#             camera=dict(
#                 eye=dict(x=1.5, y=1.5, z=1.5),
#                 up=dict(x=0, y=1, z=0)
#             )
#         )
#     )

#     # fig.update_layout(title=title, margin=dict(l=0, r=0, b=0, t=40))
#     fig.show()

def visualize_optical_flow_quiver(flow, mask=None, step=10, save_path=None):
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
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()
    pass

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

def visualize_optical_flow_video(flow, mask=None, step=10, output_path="optical_flow_video.mp4"):
    _, H, W, _ = flow.shape
    N = flow.shape[0]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (W, H))  

    for i in range(N - 1):  
        flow_np = flow[i].cpu().numpy()
        mask_np = mask[i].cpu().numpy() if mask is not None else None

        X, Y = np.meshgrid(np.arange(0, W, step), np.arange(0, H, step))
        U = flow_np[::step, ::step, 0]  
        V = flow_np[::step, ::step, 1]  

        X = X[:U.shape[0], :U.shape[1]]
        Y = Y[:V.shape[0], :V.shape[1]]

        fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
        fig.set_dpi(100)
        ax.set_facecolor("white")  # White background
        ax.imshow(np.ones((H, W)), cmap='gray', alpha=0)  
        ax.quiver(X, Y, U, V, angles="xy", scale_units="xy", scale=1./3, color="red")

        if mask_np is not None:
            mask_resampled = mask_np[::step, ::step]
            ax.quiver(X[mask_resampled == 0], Y[mask_resampled == 0], 
                      U[mask_resampled == 0], V[mask_resampled == 0], 
                      angles="xy", scale_units="xy", scale=1, color="gray", alpha=0.3)

        ax.axis("off")
        ax.set_title(f"Optical Flow - Frame {i+1}/{N-1}")

        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(height, width, 3)  # Ensure correct shape

        frame = cv2.resize(frame, (W, H))  
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  

        plt.clf()
        plt.close(fig)

    out.release()
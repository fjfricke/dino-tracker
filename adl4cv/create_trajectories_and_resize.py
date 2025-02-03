import torch
import torch.nn.functional as F

################################################################################
# 1) Flow/mask resizing with flow-vector scaling
################################################################################

def resize_flows(flows, masks, h, w):
    """
    Resize flows to new height and width, and scale the flow vectors accordingly.

    Inputs:
        flows: (N, H_old, W_old, 2) float tensor
               The last dimension is the flow vector (dx, dy).
        masks: (N+1, H_old, W_old) int/bool tensor
               Binary masks or similar, one mask per frame + 1.
        h, w : int
               The desired new height and width.

    Outputs:
        flows_resized: (N, h, w, 2)
        masks_resized: (N+1, h, w)
    """
    N, H_old, W_old, _ = flows.shape
    scale_x = w / W_old
    scale_y = h / H_old

    # -- flows => (N, 2, H_old, W_old) for interpolation
    flows_2ch = flows.permute(0, 3, 1, 2)  # shape: (N, 2, H_old, W_old)
    flows_2ch = F.interpolate(
        flows_2ch, size=(h, w), mode='bilinear', align_corners=False
    )
    # scale x/y
    flows_2ch[:, 0, :, :] *= scale_x  # dx
    flows_2ch[:, 1, :, :] *= scale_y  # dy
    # back => (N, h, w, 2)
    flows_resized = flows_2ch.permute(0, 2, 3, 1)

    # -- masks => (N+1, 1, H_old, W_old) for interpolation
    masks_4d = masks.unsqueeze(1).float()
    masks_4d = F.interpolate(
        masks_4d, size=(h, w), mode='nearest'
    )
    masks_resized = masks_4d.squeeze(1)  # => (N+1, h, w)

    return flows_resized, masks_resized

################################################################################
# 2) Trajectory creation for a single frame
################################################################################

def bilinear_sample_flow(flow_map, x, y):
    """
    flow_map: (h, w, 2)
    x, y: scalar float coords
    Return dx, dy by bilinear interpolation from flow_map at (x,y).
    """
    h, w, _ = flow_map.shape
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1

    # clamp
    x0c = torch.clamp(x0, 0, w - 1)
    x1c = torch.clamp(x1, 0, w - 1)
    y0c = torch.clamp(y0, 0, h - 1)
    y1c = torch.clamp(y1, 0, h - 1)

    # weights
    wx = x - x0.float()
    wy = y - y0.float()
    wx1 = 1.0 - wx
    wy1 = 1.0 - wy

    f00 = flow_map[y0c, x0c, :]
    f01 = flow_map[y0c, x1c, :]
    f10 = flow_map[y1c, x0c, :]
    f11 = flow_map[y1c, x1c, :]

    flow_val = (
        f00 * (wx1 * wy1) +
        f01 * (wx  * wy1) +
        f10 * (wx1 * wy ) +
        f11 * (wx  * wy )
    )
    return flow_val[0], flow_val[1]  # (dx, dy)


def check_mask_fractional(mask_slice, x, y):
    """
    mask_slice: (h, w)
    x, y: scalar float coords
    Return True if mask is valid (1) around that position, else False.
      - If integer (x,y), check that single pixel.
      - If fractional, check the 4 corners => if any corner is 0 => False.
    """
    h, w = mask_slice.shape
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1

    # clamp corners
    x0c = torch.clamp(x0, 0, w - 1)
    x1c = torch.clamp(x1, 0, w - 1)
    y0c = torch.clamp(y0, 0, h - 1)
    y1c = torch.clamp(y1, 0, h - 1)

    frac_x = (x % 1.0) != 0
    frac_y = (y % 1.0) != 0
    is_fractional = frac_x or frac_y

    if not is_fractional:
        # integer pixel => check one position
        return (mask_slice[y0c, x0c] == 1).item()
    else:
        # fractional => check 4 corners
        corners = [
            mask_slice[y0c, x0c],
            mask_slice[y0c, x1c],
            mask_slice[y1c, x0c],
            mask_slice[y1c, x1c],
        ]
        for c in corners:
            if c.item() == 0:
                return False
        return True


def create_trajectory_for_frame(flows, masks, frame_idx):
    """
    For flows starting at frame_idx, create a trajectory by following the flow
    vectors until the mask is 0 or we go out of bounds. 

    flows: (N, h, w, 2)
    masks: (N+1, h, w)
    frame_idx: which frame to start the trajectory from

    Returns:
        trajectories: (T, N+1, 2)
          T = number of valid start-pixels in masks[frame_idx].
          We store each pixel's path from frame_idx.. up to where it remains valid.
          Values for frames before frame_idx or after termination => NaN.
    """
    device = flows.device
    N, h, w, _ = flows.shape

    start_mask = masks[frame_idx]  # shape (h, w)
    valid_starts = torch.nonzero(start_mask)  # shape (T, 2) => [[y,x], ...]
    T = valid_starts.shape[0]

    trajectories = torch.full((T, N+1, 2), float('nan'), device=device)

    for t_idx in range(T):
        y0, x0 = valid_starts[t_idx]
        # store the initial (x,y) at 'frame_idx'
        trajectories[t_idx, frame_idx, 0] = x0.float()
        trajectories[t_idx, frame_idx, 1] = y0.float()

        curr_x, curr_y = x0.float(), y0.float()

        for i in range(frame_idx, N):
            next_idx = i + 1
            if next_idx > N:
                break

            # check mask in next frame
            next_mask_slice = masks[next_idx]
            if not check_mask_fractional(next_mask_slice, curr_x, curr_y):
                break

            # sample flow from flows[i]
            flow_map = flows[i]  # shape (h, w, 2)
            dx, dy = bilinear_sample_flow(flow_map, curr_x, curr_y)
            new_x, new_y = curr_x + dx, curr_y + dy

            # out-of-bounds check
            if (new_x < 0 or new_x > (w - 1) or 
                new_y < 0 or new_y > (h - 1)):
                break

            trajectories[t_idx, next_idx, 0] = new_x
            trajectories[t_idx, next_idx, 1] = new_y

            curr_x, curr_y = new_x, new_y
    print(f"Index {frame_idx} done.")
    return trajectories

################################################################################
# 3) Generate trajectories for ALL frames and concatenate
################################################################################

def create_trajectories_for_all_frames(flows, masks):
    """
    Runs create_trajectory_for_frame_fn for each frame_idx in [0..N]
    and concatenates results along dim=0.

    flows: (N, h, w, 2)
    masks: (N+1, h, w)
    """
    N = flows.shape[0]
    all_trajectories = []

    for frame_idx in range(N+1):
        trajs_i = create_trajectory_for_frame(flows, masks, frame_idx)
        if trajs_i.shape[0] > 0:
            all_trajectories.append(trajs_i)

    if len(all_trajectories) == 0:
        return torch.empty((0, N+1, 2), dtype=flows.dtype, device=flows.device)

    return torch.cat(all_trajectories, dim=0)  # => (sum_of_Ti, N+1, 2)


################################################################################
# 4) Main / Example usage
################################################################################

if __name__ == "__main__":
    # Decide device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: load or create your real flows/masks on CPU:
    # flows_np = np.load("flows.npy")   # shape (N, H_old, W_old, 2)
    # masks_np = np.load("masks.npy")   # shape (N+1, H_old, W_old)
    #
    # For demonstration, let's just create a small synthetic example:
    import numpy as np

    N, H_old, W_old = 3, 4, 5
    flows_np = np.zeros((N, H_old, W_old, 2), dtype=np.float32)
    flows_np[..., 0] = 1.0  # move right
    # all mask =1 except one hole
    masks_np = np.ones((N+1, H_old, W_old), dtype=np.uint8)
    masks_np[2,2,2] = 0

    # Move to torch CPU
    flows_cpu = torch.from_numpy(flows_np)
    masks_cpu = torch.from_numpy(masks_np)

    # Move to GPU for processing
    flows_gpu = flows_cpu.to(device)
    masks_gpu = masks_cpu.to(device)

    # Suppose we want final shape h=476, w=854
    video_resh = 476
    video_resw = 854

    # 1) Resize
    flows_resized, masks_resized = resize_flows(
        flows_gpu, masks_gpu, h=video_resh, w=video_resw
    )
    # flows_resized => (N, 476, 854, 2)
    # masks_resized => (N+1, 476, 854)

    # 2) Create ALL trajectories
    all_trajs_gpu = create_trajectories_for_all_frames(
        flows_resized, 
        masks_resized
    )
    # shape => (T_total, N+1, 2), on GPU

    # 3) Move final result back to CPU
    all_trajs_cpu = all_trajs_gpu.cpu()
    print("All trajectories shape on CPU =", all_trajs_cpu.shape)
    # Optional: do something with all_trajs_cpu (save, etc.)
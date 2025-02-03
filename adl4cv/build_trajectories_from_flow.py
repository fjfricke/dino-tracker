import torch
from tqdm import tqdm  # Add this import at the top of your file

def filter_short_trajectories(trajectories, min_valid_frames=2):
    """
    Removes trajectories that have fewer than 'min_valid_frames' valid points.
    We interpret "valid" as non-NaN in (x,y).

    Args:
        trajectories: (N, T, 2) float tensor
                      Each row is a trajectory, each column is a frame, last dim is (x,y).
        min_valid_frames: int, minimum number of valid points required

    Returns:
        filtered: (M, T, 2) float tensor, with M <= N
    """
    # We say a coordinate is valid if it isn't NaN in either x or y
    valid_mask = ~trajectories.isnan().any(dim=-1)  # shape (N, T) bool

    # Count valid frames per trajectory
    valid_counts = valid_mask.sum(dim=1)  # shape (N,)

    # Keep only those that have at least 'min_valid_frames' valid points
    keep = (valid_counts >= min_valid_frames)
    filtered = trajectories[keep]  # shape (M, T, 2)

    return filtered


def build_and_pad_trajectories(flow, valid_mask):
    """
    Re-initializes trajectories from each frame k in [0..T-2]. 
    Aligns them to a global time dimension [0..T-1] so each track
    can be used with your LongRangeSampler (N, T, 2) format.

    Args:
      flow: (T-1, H, W, 2) float tensor
            flow[i, y, x, :] = (dx, dy) from frame i -> i+1
      valid_mask: (T-1, H, W) bool tensor
            valid_mask[i, y, x] = True if flow is valid at (x,y) in frame i

    Returns:
      all_trajectories: (N, T, 2) float tensor of sub-tracks
        - N = total # of partial sub-trajectories across frames
        - T = total # of frames
        - each row has valid coords for frames [start_k..end], and NaN outside
    """
    device = flow.device
    T_minus_1, H, W, _ = flow.shape
    T = T_minus_1 + 1

    # Prepare a meshgrid for seeding
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    start_coords = torch.stack([x_coords, y_coords], dim=-1).float()  # (H, W, 2)

    sub_trajectories = []  # we will store a padded (H*W, T, 2) for each k

    for k in tqdm(range(T - 1), desc="Building trajectories"):
        # We'll track from frame k up to frame T-1
        sub_flow = flow[k:]        # shape: ((T-1)-k, H, W, 2)
        sub_valid = valid_mask[k:] # shape: ((T-1)-k, H, W)
        sub_T = sub_flow.shape[0] + 1  # => T - k

        # partial_4d is shape (H, W, sub_T, 2), local time dimension [0..sub_T-1]
        partial_4d = torch.full((H, W, sub_T, 2), float('nan'), device=device)
        # seed every pixel in local time 0
        partial_4d[..., 0, :] = start_coords

        # track forward sub_T - 1 steps
        for i in range(sub_T - 1):
            curr_coords = partial_4d[..., i, :]  # (H, W, 2)
            not_nan = ~torch.isnan(curr_coords).any(dim=-1)  # (H, W) bool

            if not_nan.sum() == 0:
                break

            # Round to integer pixel coords
            x_int = curr_coords[..., 0].round().long()
            y_int = curr_coords[..., 1].round().long()

            inside = (
                (x_int >= 0) & (x_int < W) &
                (y_int >= 0) & (y_int < H)
            ) & not_nan

            # Make sure sub_valid is bool:
            sub_valid = sub_valid > 0.5  # or sub_valid = sub_valid.bool()

            # Then create a bool mask_ok:
            mask_ok = torch.zeros(inside.shape, dtype=torch.bool, device=inside.device)

            # Now you can safely assign bool → bool:
            mask_ok[inside] = sub_valid[i, y_int[inside], x_int[inside]]

            dxdy = torch.zeros_like(curr_coords)
            dxdy[mask_ok] = sub_flow[i, y_int[mask_ok], x_int[mask_ok], :]

            next_coords = curr_coords + dxdy
            next_coords[~mask_ok] = float('nan')
            partial_4d[..., i + 1, :] = next_coords

        # Now partial_4d is (H, W, sub_T, 2).
        # We want to place partial_4d into a "global timeline" array
        # => shape (H, W, T, 2), filling frames [k..k+sub_T-1].
        partial_global_4d = torch.full((H, W, T, 2), float('nan'), device=device)

        # Fill the slice [k..k+sub_T-1] in time dimension with partial_4d
        partial_global_4d[..., k : k+sub_T, :] = partial_4d

        # Flatten: (H, W, T, 2) -> (H*W, T, 2)
        partial_global_3d = partial_global_4d.view(-1, T, 2)
        sub_trajectories.append(partial_global_3d)

    # Concatenate along the first dimension => shape (sum(H*W for k), T, 2)
    all_trajectories = torch.cat(sub_trajectories, dim=0)  # (N, T, 2)

    all_trajectories_out = filter_short_trajectories(all_trajectories, min_valid_frames=2)

    return all_trajectories_out
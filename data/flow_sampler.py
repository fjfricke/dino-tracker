"""
flow_sampler.py

A drop-in replacement for DinoTrackerSampler that:
  1) Takes in raw flow + mask of shape (T-1, H, W, 2) and (T-1, H, W).
  2) On each forward pass, samples 'batch_size' random seeds:
       (frame_idx in [0..T-2], x in [0..W-1], y in [0..H-1])
  3) Tracks each seed forward up to 'max_steps' frames or until invalid/out-of-bounds.
  4) Extracts two valid points (t1, x1, y1) and (t2, x2, y2) from each track.
  5) Normalizes (x,y,t) using the provided RangeNormalizer (to e.g. [-1,1]).
  6) Returns the dictionary:

    sample = {
      "frames_set_t": frames_set_t,                 # shape: T  (i.e. [0..T-1])
      "source_frame_indices": source_frame_indices,  # shape: B
      "target_frame_indices": target_frame_indices,  # shape: B
      "t1_points_normalized": t1_points_normalized,  # shape: B x 3
      "t2_points_normalized": t2_points_normalized,  # shape: B x 3
      "t1_points": t1_points,                       # shape: B x 3
      "target_times": t2_points[:, 2],              # shape: B
    }
"""

import torch
import torch.nn as nn

class FlowSampler(nn.Module):
    """
    Replaces DinoTrackerSampler but uses raw flow images (T-1, H, W, 2) & mask (T-1, H, W)
    to generate random correspondences.

    Example usage in your DINOTracker code:
        from flow_sampler import FlowSampler

        ...
        flows = ...        # shape: (T-1, H, W, 2)
        valid_mask = ...   # shape: (T-1, H, W)
        flow_sampler = FlowSampler(
            flows, valid_mask,
            range_normalizer=range_normalizer,   # your RangeNormalizer
            dst_range=(-1, 1),
            batch_size=64,
            max_steps=5,
            device=device
        )
        sample = flow_sampler()
        # Then feed 'sample' into the rest of your pipeline
    """
    def __init__(
        self,
        flows: torch.Tensor,       # shape (T-1, H, W, 2)
        valid_mask: torch.Tensor,  # shape (T-1, H, W), 0/1 or bool
        range_normalizer,          # your RangeNormalizer(...) instance
        dst_range=(-1,1),
        batch_size=16,
        max_steps=5,
        device="cuda"
    ):
        super().__init__()
        self.flows = flows.to(device)
        self.valid_mask = valid_mask.to(device)
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.range_normalizer = range_normalizer
        self.dst_range = dst_range

        self.device = device

        # flows.shape => (T-1, H, W, 2)
        # We'll define T, H, W for convenience
        self.T_minus_1, self.H, self.W, _ = self.flows.shape
        self.T = self.T_minus_1 + 1

    @torch.no_grad()
    def forward(self):
        """
        Returns a dict:

            sample = {
              "frames_set_t": frames_set_t,  # shape: T  => [0..T-1]
              "source_frame_indices": source_frame_indices, # shape: B
              "target_frame_indices": target_frame_indices, # shape: B
              "t1_points_normalized": t1_points_normalized, # shape: B x 3
              "t2_points_normalized": t2_points_normalized, # shape: B x 3
              "t1_points": t1_points,                       # shape: B x 3
              "target_times": t2_points[:, 2],             # shape: B
            }
        """
        # 1) Random seeds => (frame_idx, x, y)
        frame_idxs = torch.randint(
            low=0, high=self.T-1, size=(self.batch_size,), device=self.device
        )
        xs = torch.randint(
            low=0, high=self.W, size=(self.batch_size,), device=self.device
        )
        ys = torch.randint(
            low=0, high=self.H, size=(self.batch_size,), device=self.device
        )

        # We'll track coords in shape (B, max_steps+1, 3) => columns: (x, y, t)
        # Fill with NaN to indicate "no valid coordinate."
        tracks = torch.full(
            (self.batch_size, self.max_steps+1, 3),
            float('nan'),
            device=self.device
        )

        # Initialize time=0 in each track
        tracks[:, 0, 0] = xs.float()          # x
        tracks[:, 0, 1] = ys.float()          # y
        tracks[:, 0, 2] = frame_idxs.float()  # t

        # 2) Follow flow up to `max_steps`
        for step in range(self.max_steps):
            curr_x = tracks[:, step, 0]
            curr_y = tracks[:, step, 1]
            curr_t = tracks[:, step, 2].long()  # frames are int

            # valid if not NaN, t in [0..T-2]
            valid = (
                ~torch.isnan(curr_x) &
                ~torch.isnan(curr_y) &
                (curr_t >= 0) & (curr_t < self.T - 1)
            )
            if valid.sum() == 0:
                break

            # Round to integer pixel coords (purely integer sampling)
            x_int = curr_x.round().long().clamp(0, self.W - 1)
            y_int = curr_y.round().long().clamp(0, self.H - 1)

            # Check mask => must be True (or == 1)
            sub_mask = self.valid_mask[curr_t[valid], y_int[valid], x_int[valid]]
            good_mask = (sub_mask == 1)
            # Convert local good_mask -> global
            good_mask_global = torch.zeros_like(valid)
            good_mask_global[valid] = good_mask
            valid = valid & good_mask_global

            if valid.sum() == 0:
                break

            # read flow => shape (N_valid, 2)
            sub_flow = self.flows[curr_t[valid], y_int[valid], x_int[valid]]
            dx = sub_flow[:, 0]
            dy = sub_flow[:, 1]

            # next coords
            next_x = torch.full_like(curr_x, float('nan'))
            next_y = torch.full_like(curr_y, float('nan'))
            next_t = torch.full_like(curr_t, float('nan'), dtype=torch.float)

            next_x[valid] = curr_x[valid] + dx
            next_y[valid] = curr_y[valid] + dy
            next_t[valid] = curr_t[valid].float() + 1.0

            # out-of-bounds => invalid
            in_bounds = (
                (next_x >= 0) & (next_x < self.W) &
                (next_y >= 0) & (next_y < self.H)
            )
            valid = valid & in_bounds

            next_x[~valid] = float('nan')
            next_y[~valid] = float('nan')
            next_t[~valid] = float('nan')

            if step + 1 < (self.max_steps + 1):
                tracks[:, step+1, 0] = next_x
                tracks[:, step+1, 1] = next_y
                tracks[:, step+1, 2] = next_t

        # 3) Now pick the FIRST TWO valid frames from each track => t1_points, t2_points
        t1_points = torch.full((self.batch_size, 3), float('nan'), device=self.device)
        t2_points = torch.full((self.batch_size, 3), float('nan'), device=self.device)

        for b in range(self.batch_size):
            track_b = tracks[b]  # shape (max_steps+1, 3)
            is_nan = torch.isnan(track_b).any(dim=-1)
            valid_idxs = (~is_nan).nonzero(as_tuple=True)[0]  # e.g. [0,1,2...]

            if len(valid_idxs) >= 2:
                idx1 = valid_idxs[0]
                idx2 = valid_idxs[1]
                t1_points[b] = track_b[idx1]  # (x,y,t)
                t2_points[b] = track_b[idx2]  # (x,y,t)

        # 4) Build sample dictionary

        # 4a) frames_set_t => we typically keep [0..T-1]
        frames_set_t = torch.arange(self.T, device=self.device).int()  # shape: T

        # 4b) For each t in t1_points[:,2] => find index in frames_set_t
        #     For each t in t2_points[:,2] => find index in frames_set_t
        #     shape => B each
        def map_frames_to_indices(times, frames):
            # times => shape (B,) float
            out = []
            for val in times:
                if torch.isnan(val):
                    out.append(torch.tensor([-1], device=self.device))
                else:
                    idx = (frames == val.int()).nonzero(as_tuple=True)[0]
                    if len(idx) == 0:
                        out.append(torch.tensor([-1], device=self.device))
                    else:
                        out.append(idx[0])
            return torch.stack(out, dim=0)

        source_frame_indices = map_frames_to_indices(t1_points[:,2], frames_set_t)
        target_frame_indices = map_frames_to_indices(t2_points[:,2], frames_set_t)

        # 4c) Normalize (x,y,t) with your RangeNormalizer
        # Our RangeNormalizer expects shape (B, d) => typically (B,3) with [x,y,t],
        #   so we do: range_normalizer(t1_points, dst=(-1,1), dims=[0,1,2]) => but
        #   your RangeNormalizer might be expecting shape [N, something].
        # We'll call it directly:
        t1_points_normalized = self.range_normalizer(t1_points.clone(), dst=self.dst_range, dims=[0,1,2])
        t2_points_normalized = self.range_normalizer(t2_points.clone(), dst=self.dst_range, dims=[0,1,2])

        # 4d) If you want to store the normalized time *inside* t1_points, as in DinoTrackerSampler:
        t1_points_mod = t1_points.clone()
        t1_points_mod[:, 2] = t1_points_normalized[:, 2]  # just the time

        sample = {
            "frames_set_t": frames_set_t,               # shape: T
            "source_frame_indices": source_frame_indices, # shape: B
            "target_frame_indices": target_frame_indices, # shape: B
            "t1_points_normalized": t1_points_normalized, # shape: B x 3
            "t2_points_normalized": t2_points_normalized, # shape: B x 3
            "t1_points": t1_points_mod,                  # shape: B x 3
            "target_times": t2_points[:, 2],             # shape: B
        }

        return sample
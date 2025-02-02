import logging
import os
import torch
import yaml
from pathlib import Path
from tqdm import tqdm
from einops import rearrange
from PIL import Image
from models.utils import filter_bb_foreground_pairs, get_last_ckpt_iter, get_feature_cos_sims, get_vit_feature_coords_from_mask
from models.tracker import Tracker
from optimization.schedulers import get_cnn_refiner_scheduler
from data.data_utils import load_video
from data.dataset import DinoTrackerSampler, RangeNormalizer
from preprocessing.split_trajectories_to_fg_bg import load_masks
from utils import add_config_paths


device = "cuda:0" if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")



class DINOTracker():
    def __init__(self, args):
        
        self.load_config(args.config)
        self.set_paths(args.data_path)

        self.orig_video_res_h, self.orig_video_res_w, video_rest = self.get_original_video_res(self.video_path)
        self.range_normalizer = RangeNormalizer(shapes=(self.config["video_resw"], self.config["video_resh"], video_rest), device=device).to(device) # nn.Module
        self.of_loss_fn = torch.nn.HuberLoss(delta=1/32, reduction='none')
        
    def load_fg_masks(self):
        if self.config["use_fg_masks"]:
            self.fg_masks = torch.from_numpy(load_masks(self.fg_masks_path, h_resize=self.config["video_resh"])).to(device)        
        else:
            self.fg_masks = torch.ones(self.config["video_resh"], self.config["video_resw"]).to(device)
    
    def set_paths(self, data_path):
        config_paths = add_config_paths(data_path, {})
        self.video_path = config_paths["video_folder"]
        # self.fg_masks_path = config_paths["masks_path"]
        self.dino_embed_path = config_paths["dino_embed_video_path"]
        # self.fg_trajectories_path = config_paths["fg_trajectories_file"]
        # self.bg_trajectories_path = config_paths["bg_trajectories_file"]
        self.trajectories_path = config_paths["trajectories_file"]
        # self.dino_bb_path = os.path.join(config_paths["dino_bb_dir"], "dino_best_buddies_filtered.pt")
        self.ckpt_folder = config_paths["ckpt_folder"]
        # self.trajectories_dir = config_paths['trajectories_dir']
        # self.occlusions_dir = config_paths['occlusions_dir']
        # self.grid_trajectories_dir = config_paths['grid_trajectories_dir']
        # self.grid_occlusions_dir = config_paths['grid_occlusions_dir']
        os.makedirs(self.ckpt_folder, exist_ok=True)
        
    def load_config(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f.read())


    def get_original_video_res(self, video_path):
        video_frames_list = sorted(list(Path(video_path).glob("*.jpg")) + list(Path(video_path).glob("*.png")))
        video_rest = len(video_frames_list)
        # read first frame using PIL.Image to get resolution
        frame = Image.open(video_frames_list[0])
        video_res_hw = frame.size[::-1]
        return video_res_hw + (video_rest,)

    def load_trajectories(self):
        assert os.path.exists(self.trajectories_path), "trajectory files don't exist"
        
        trj_device = torch.device('cpu') if self.config['keep_traj_in_cpu'] else device
        train_fg_trajectories = torch.load(self.trajectories_path, map_location=trj_device)
        train_bg_trajectories = torch.empty_like(train_fg_trajectories)
        return train_fg_trajectories, train_bg_trajectories
    
    def get_sampler(self):        
        train_fg_trajectories, train_bg_trajectories = self.load_trajectories()
        train_sampler = DinoTrackerSampler(fg_trajectories=train_fg_trajectories,
                                           bg_trajectories=train_bg_trajectories,
                                           fg_traj_ratio=self.config["fg_traj_ratio"],
                                           batch_size=self.config["train_batch_size"],
                                           range_normalizer=self.range_normalizer,
                                           dst_range=(-1, 1),
                                           num_frames=self.config["batch_n_frames"],
                                           keep_in_cpu=self.config['keep_traj_in_cpu'])
        return train_sampler
    
    def get_model(self):
        video = load_video(video_folder=self.video_path, resize=(self.config["video_resh"], self.config["video_resw"])).to(device)
        tracker_args = {
            "video":video,
            "device":device,
            "dino_embed_path": self.dino_embed_path,
            "dino_patch_size": self.config["dino_patch_size"],
            "stride":self.config["stride"],
            "ckpt_path": self.ckpt_folder,
            
            "cyc_n_frames": self.config["cyc_n_frames"],
            "cyc_batch_size_per_frame": self.config["cyc_batch_size_per_frame"],
            "cyc_fg_points_ratio": self.config["cyc_fg_points_ratio"],
            "cyc_thresh": self.config["cyc_thresh"]
        }
        
        model = Tracker(**tracker_args).to(device)
            
        self.init_iter = get_last_ckpt_iter(self.ckpt_folder)
        if self.init_iter > 0:
            model.load_weights(self.init_iter)
        
        return model
    
    def train_setup(self):
        model = self.get_model()
        params = [{"params": model.delta_dino.parameters(), "lr": self.config["lr_delta_dino"]},
                  {"params": model.tracker_head.parameters(), "lr": self.config["lr_cnn_refiner"]}]
        optimizer = torch.optim.Adam(params)                    
        scheduler = get_cnn_refiner_scheduler(optimizer, gamma=self.config['scheduler_gamma'], apply_every=self.config['apply_scheduler_every'])
        
        if self.init_iter > 0:
            self.init_scheduler(scheduler, self.init_iter)
        print("------- INIT ITER", self.init_iter)        
        
        return model, optimizer, scheduler
    
    def init_scheduler(self, scheduler, iter):
        for i in range(iter):
            scheduler.step()
    
    def get_inputs_and_labels(self, sampler):
        sample = sampler()
        labels = sample["t2_points_normalized"][:, :-1]
        inputs = (sample["t1_points"], sample["source_frame_indices"], sample["target_frame_indices"], sample["frames_set_t"])
        return inputs, labels
    
    def set_model_train(self, model):
        model.train()

    def get_cycle_consistency_loss(self, model, inputs):
        cycle_consistency_preds = model.get_cycle_consistent_preds(inputs[-1], self.fg_masks)
        consistent_track_weight = self.config["cyc_gamma"] ** cycle_consistency_preds["cycle_consistency_dists"]
        source_target_tracking_loss = consistent_track_weight[:, None] * self.of_loss_fn(cycle_consistency_preds["source_target_coords"], cycle_consistency_preds["target_coords"][:, :2])
        target_source_tracking_loss = consistent_track_weight[:, None] * self.of_loss_fn(cycle_consistency_preds["target_source_coords"], cycle_consistency_preds["source_coords"][:, :2])
        consistent_track_loss = (source_target_tracking_loss.mean() + target_source_tracking_loss.mean()) / 2
        
        return consistent_track_loss
    
    def init_losses(self):
        self.running_loss_total = 0.
        self.running_loss_of = 0.
        self.running_loss_cyc = 0.

    def update_losses(self, loss_total, loss_of, loss_cyc):
        self.running_loss_total += loss_total
        self.running_loss_of += loss_of
        self.running_loss_cyc += loss_cyc

    def log_losses(self, i, log_interval=100):
        loss_of = self.running_loss_of / log_interval
        loss_cyc = self.running_loss_cyc / log_interval if i >= self.config.get("apply_cyc_after", 0) else None
        loss_total = self.running_loss_total / log_interval
        
        loss_str = f"loss_of: {loss_of:.4f}"
        if loss_cyc is not None:
            loss_str += f", loss_cyc: {loss_cyc:.4f}"
        loss_str += f", loss_total: {loss_total:.4f}"
        
        logging.info(loss_str)
        self.init_losses()

    def train(self):
        # self.load_fg_masks()
        # Get values from config
        total_iterations = self.config["total_iterations"]
        checkpoint_interval = self.config["checkpoint_interval"]
        sampler_batch_iterations = self.config.get("sampler_batch_iterations", 100_000) # only relevant if in config, 100k is never reached

        train_sampler = self.get_sampler()
        model, optimizer, scheduler = self.train_setup()
        self.set_model_train(model)
        self.init_losses()
        
        for i in tqdm(range(self.init_iter, total_iterations)):
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            inputs, labels = self.get_inputs_and_labels(train_sampler)
            
            coords = model(inputs)
            tracking_loss = self.of_loss_fn(coords, labels).mean()
            loss = tracking_loss
            
            if i >= self.config.get("apply_cyc_after", 0):
                consistent_track_loss = self.get_cycle_consistency_loss(model, inputs)
                loss += self.config["lambda_cyc"] * consistent_track_loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # logging losses
            self.update_losses(loss.item(), tracking_loss.item(), consistent_track_loss.item() if i >= self.config.get("apply_cyc_after", 0) else 0)
            if i % 100 == 0:
                self.log_losses(i, log_interval=100)
            
            # saving checkpoint
            if (i == total_iterations - 1 or i % checkpoint_interval == 0):
                model.save_weights(i)

            # loading next batch of trajectories
            if (i % sampler_batch_iterations == 0 and i > 0):
                print("Loading next batch", flush=True)
                train_sampler.load_next_batch()

        model.save_weights(total_iterations)

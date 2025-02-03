import torch

if __name__ == "__main__":

    # Load the trajectories from the specified file
    trajectories = torch.load('dataset/rendered_mesh_output/of_trajectories/fg_trajectories.pt')
    new_trajectories = torch.load('dataset/dino-tracker-output/of_trajectories/fg_trajectories.pt')

    print(trajectories)

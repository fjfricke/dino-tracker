import torch

if __name__ == "__main__":

    # Load the trajectories from the specified file
    trajectories = torch.load('/workspace/dino-tracker/dataset/rendered_mesh_output/of_trajectories/trajectories.pt')
    new_trajectories = torch.load('/workspace/dino-tracker/dataset/dino-tracker-output/of_trajectories/trajectories.pt')

    print(trajectories)

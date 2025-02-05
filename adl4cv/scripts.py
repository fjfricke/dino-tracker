import os
from pathlib import Path
from adl4cv.load_files import main as load_files_main  # Import the refactored main function
from adl4cv.train import main as train_main
from tqdm import tqdm
from adl4cv.save_dino_embed_video import save_dino_embed_video  # Import the save_dino_embed_video function
from adl4cv.get_refined_embeddings import get_refined_embeddings  # Import the get_refined_embeddings function

INPUT_DIRS = [Path(os.getcwd()) / "dataset/shrec20_tex", Path(os.getcwd()) / "dataset/shrec20_untex"]
DINO_H = 476
DINO_W = 854

def get_filenames_and_data_paths():
    # Directories to process
    filenames_and_input_dirs = []
    for input_dir in INPUT_DIRS:
        for file_name in os.listdir(input_dir):
            if file_name.endswith(".pt"):
                filenames_and_input_dirs.append((file_name, input_dir / file_name))
    return filenames_and_input_dirs
    

def load_files():
    # Get filenames and their paths using the helper function
    filenames_and_input_dirs = get_filenames_and_data_paths()

    # Iterate over each collected file
    for file_name, file_path in tqdm(filenames_and_input_dirs, desc="Processing Files"):
        # Extract the filename without the extension
        file_base_name = os.path.splitext(file_name)[0]

        # Set the data path dynamically
        data_path = os.path.join("dataset", file_base_name)

        print(f"Processing file: {file_path}")
        print(f"Output data path: {data_path}")

        # Call the main function of load_files.py directly
        load_files_main(
            data_path=data_path,
            input_path=file_path,
            dino_h=DINO_H,
            dino_w=DINO_W
        )

def save_dino_embeddings():
    config = Path(__file__).parent.parent / "config" / "preprocessing.yaml"

    for file_name, _ in tqdm(get_filenames_and_data_paths(), desc="Saving DINO Embeddings"):
        # Extract the filename without the extension
        file_base_name = os.path.splitext(file_name)[0]

        # Set the data path dynamically
        data_path = os.path.join("dataset", file_base_name)

        # Prepare arguments for save_dino_embed_video
        class Args:
            def __init__(self, config, data_path, for_mask=False):
                self.config = config
                self.data_path = data_path
                self.for_mask = for_mask

        args = Args(config=str(config), data_path=data_path, for_mask=False)

        # Call the save_dino_embed_video function
        save_dino_embed_video(args)

def train():
    config = Path(__file__).parent.parent / "config" / "train.yaml"
    seed = 2

    for file_name, _ in tqdm(get_filenames_and_data_paths(), desc="Training Models"):
        # Extract the filename without the extension
        file_base_name = os.path.splitext(file_name)[0]

        # Set the data path dynamically
        data_path = os.path.join("dataset", file_base_name)

        train_main(config, data_path, seed)

def generate_refined_embeddings():
    config = Path(__file__).parent.parent / "config" / "train.yaml"
    seed = 2

    for file_name, _ in tqdm(get_filenames_and_data_paths(), desc="Generating Refined Embeddings"):
        # Extract the filename without the extension
        file_base_name = os.path.splitext(file_name)[0]

        # Set the data path dynamically
        data_path = os.path.join("dataset", file_base_name)

        # Call the get_refined_embeddings function
        get_refined_embeddings(
            config=str(config),
            data_path=data_path,
            seed=seed
        )
    


if __name__ == "__main__":
    load_files()  # step 1: calculates flow and mask videos for each object
    save_dino_embeddings()  # step 2: saves DINO embeddings for each object
    train()  # step 3: trains the model
    generate_refined_embeddings()  # step 4: generates refined embeddings
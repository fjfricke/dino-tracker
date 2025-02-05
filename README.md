# Adjusted DINO-Tracker for 3D shape correspondence based on ```https://github.com/AssafSinger94/dino-tracker.git```


## Usage

1. [Setup](#setup)
2. [Preprocessing](#preprocessing)
3. [Training](#training)
4. [Inference](#inference)


## Setup

Switch to the project directory:

```cd dino-tracker```

To setup the environment, run:

```
conda create -n dino-tracker python=3.9
conda activate dino-tracker
pip install -r requirements.txt
```

Add current path to ```PYTHONPATH```:

```export PYTHONPATH=`pwd`:$PYTHONPATH```

## Add files to process

Put the ```.pt``` files containing the renderings, cameras and depth images for each frame coming from the Diff3f pipeline into the ```./dataset/``` folder as such:

```
├──datasets/
    ├──shrec20_untex/
        ├──bear_rendered.pt
        ├──bison_rendered.pt
        ├──...
    ├──shrec20_tex/
        ├──bear_tex_rendered.pt
        ├──bison_tex_rendered.pt
        ├──...
```
## Training and inference

Run ```./adl4cv/scripts.py```. This script performs the following:
- loads all the ```.pt``` files
- saves the original DINOv2 embeddings
- trains the DINO-Tracker on each video
- saves the refined embeddings (DINOv2 embeddings + DeltaDINO embeddings)

It outputs the following structure:

```
├──datasets/
    ├──bear<_text>_rendered/
        ├──dino_embeddings/
            ├──dino_embed_video.pt (the original DINOv2 embeddings)
            ├──refined_embeddings.pt (the refined embeddings)
        ├──masks/
            ├──00000.png (foreground mask of first frame)
            ├──00001.png
            ├──...
        ├──video/
            ├──00000.png (rendering of first frame)
            ├──00001.png
            ├──...
        ├──models/
            ├──delta_dino_5000.pt (the DeltaDINO model after iteration 5000)
            ├──... (also saves the tracker head that is unused)
        ├──of_trajectories/
            ├──fg_trajectories.pt (simulated optical flow trajectory for all frames)
        ├──flow_video.mp4 (simulated optical flow video)
    ├──...
```

The files in the ```./adl4cv/``` folder are the ones that were created for the 3D shape correspondence task.

```./models/tracker.py```, ```./config/train.yml``` and ```./dinotracker.py``` were adjusted for the 3D shape correspondence task.

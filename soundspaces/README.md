# SoundSpaces Dataset

## Overview
The SoundSpaces dataset includes audio renderings (room impulse responses) for two datasets, metadata of each scene, episode datasets and mono sound files. 

## Linking the Required Files
0. Create a folder named "data" under $ROOT directory
1. use command 'ln -s /data2/datasets/tmp/supaul2/soundspace/to_share/data/* $ROOT/data/' to symlink all the following required datasets.
2. Done. Check if it matches with the data folder structure give below.
3. Replace `./data/datasets/semantic_audionav_dialog_approx` with [semantic_audionav_dialog_approx](https://drive.google.com/drive/folders/1N4i-vj_ZsH9g8NDII6iUErie46YZZSWi?usp=sharing)


## Download
0. Create a folder named "data" under $ROOT directory
1. Download [Matterport3D](https://niessner.github.io/Matterport). Or clone it from `/data2/datasets/tmp/supaul2/mp3d_habitat/v1/tasks/mp3d/`. 
Keep it inside `$ROOT/data/scene_datasets` folder (follow the data folder structure). 
2. Run the commands below in the **data** directory to download partial binaural RIRs (867G), metadata (1M), datasets (77M) and sound files (13M). Note that this partial binaural RIRs only contain renderings for nodes accessible by the agent on the navigation graph. 
the dowloaded files are available in `/data2/datasets/tmp/supaul2/soundspace/`
```
wget http://dl.fbaipublicfiles.com/SoundSpaces/binaural_rirs.tar && tar xvf binaural_rirs.tar
wget http://dl.fbaipublicfiles.com/SoundSpaces/metadata.tar.xz && tar xvf metadata.tar.xz
wget http://dl.fbaipublicfiles.com/SoundSpaces/sounds.tar.xz && tar xvf sounds.tar.xz
wget http://dl.fbaipublicfiles.com/SoundSpaces/datasets.tar.xz && tar xvf datasets.tar.xz
wget http://dl.fbaipublicfiles.com/SoundSpaces/pretrained_weights.tar.xz && tar xvf pretrained_weights.tar.xz
```
3. Run the command below in the root directory to cache observations for two datasets
```
python scripts/cache_observations.py
```
4. copy/symlink node2view.json from `/data2/datasets/tmp/supaul2/soundspace/`
5. copy/symlink view2node.json from `/data2/datasets/tmp/supaul2/soundspace/`
6. (Optional) Download the full ambisonic (3.6T for Matterport) and binaural (682G for Matterport and 81G for Replica) RIRs data by running the following script in the root directory. Remember to first back up the downloaded bianural RIR data.
```
python scripts/download_data.py --dataset mp3d --rir-type binaural_rirs
python scripts/download_data.py --dataset replica --rir-type binaural_rirs
```



## Data Folder Structure
```
    .
    ├── ...
    ├── metadata                                  # stores metadata of environments
    │   └── [dataset]
    │       └── [scene]
    │           ├── point.txt                     # coordinates of all points in mesh coordinates
    │           ├── graph.pkl                     # points are pruned to a connectivity graph
    ├── binaural_rirs                             # binaural RIRs of 2 channels
    │   └── [dataset]
    │       └── [scene]
    │           └── [angle]                       # azimuth angle of agent's heading in mesh coordinates
    │               └── [receiver]-[source].wav
    ├── datasets                                  # stores datasets of episodes of different splits
    │   └── [dataset]
    │       └── [version]
    │           └── [split]
    │               ├── [split].json.gz
    │               └── content
    │                   └── [scene].json.gz
    ├── sounds                                    # stores all 102 copyright-free sounds
    │   └── 1s_all
    │       └── [sound].wav
    ├── scene_datasets                            # scene_datasets
    │   └── [dataset]
    │       └── [scene]
    │           └── [scene].house (habitat/mesh_sementic.glb)
    ├── scene_observations                        # pre-rendered scene observations
    │   └── [dataset]
    │       └── [scene].pkl                       # dictionary is in the format of {(receiver, rotation): sim_obs}
    ├── pretrained_weights                        # weights provided by soundspace (not giving proper result)
    │   └── [dataset]
    │       └── savi
    │           └── [].pth
    ├── node2view.json
    └── view2node.json
```

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# SPDX-License-Identifier: CC-BY-4.0

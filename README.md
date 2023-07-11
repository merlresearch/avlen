<!--
Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# AVLEN: Audio-Visual-Language Embodied Navigation in 3D Environments
![](./images/task_avlen.png)

## Overview
This repository contains training and testing codes used in the NeurIPS 2022 paper 'AVLEN: Audio-Visual-Language Embodied Navigation in 3D Environments' by Sudipta Paul, Amit K. Roy-Chowdhury, and Anoop Cherian.

Our code is built on [SoundSpaces 1.0](https://github.com/facebookresearch/sound-spaces).

## Installation

### AVLEN
1. `git clone https://github.com/merlresearch/avlen.git`

2. `export ROOT=<path to avlen>`

3. Create a virtual env with python=3.7, this will be used throughout:\
   `conda create -n avlen_env python=3.7 cmake=3.14.0`

4. The directories are assumed to be organized as follows:
```
├── project
        ├── avlen                                 # ROOT directory
        |     |── habitat-lab-dialog              # modified v0.1.7 of habitat-lab
        |     |── ...                             # other files and folders
        |
        └── habitat-sim                           # v0.1.7 of habitat-sim
```
5. Install [habitat-lab-dialog](https://github.com/spaul007/habitat-lab-dialog.git) (modified version of habitat-lab v0.1.7).

```
cd $ROOT/habitat-lab-dialog
pip install -r requirements.txt
python setup.py develop --all # install habitat and habitat_baselines
```

6. Install [habitat-sim v0.1.7](https://github.com/facebookresearch/habitat-sim/tree/v0.1.7) (with `--headless` and `--with-cuda`) \
   `git clone --branch v0.1.7 https://github.com/facebookresearch/habitat-sim.git`\
   Check further instructions from [here](https://github.com/facebookresearch/habitat-sim/blob/v0.1.7/BUILD_FROM_SOURCE.md)

7. Set `PYTHONPATH=$PYTHONPATH:$ROOT/habitat-lab-dialog`

8. Install `avlen` repo into pip by running the following command:
```
cd $ROOT
pip install -e .
```
9. Follow instructions on the [dataset](https://github.com/facebookresearch/sound-spaces/tree/main/soundspaces) page to download the rendered audio data and datasets and put them under `$ROOT/data/` folder.

10. Add connectivity files by from [here](https://drive.google.com/drive/folders/11GjL3RZnbRPGUv05wzP9sxSYjJVRCEJ6?usp=sharing) to ` $ROOT/connectivity/`

11. `export PYTHONPATH=$PYTHONPATH:<path to habitat-lab-dialog>`

12. Download the repurposed dataset for training language-based policy $\pi_l$ from [here](https://drive.google.com/drive/folders/1zGLDG3vxeETO13dBQde2H5qmxgKiAlJH?usp=sharing) and place it to `$ROOT/data/datasets/semantic_audionav_dialog_approx/`

13. Install CLIP from [here](https://github.com/openai/CLIP)

14. Download `node2view.json` and `view2node.json` from [here](https://drive.google.com/drive/folders/1TjnFdupuC7dEVnmz9-6gWPqx1gCGXApB?usp=sharing) and place it in `$ROOT/data` folder.

### Speaker (Location: $ROOT/ss_baselines/savi/dialog/speaker)

1. Compile the Matterport3D Simulator:
```
cd $ROOT/ss_baselines/savi/dialog/speaker
mkdir build && cd build
cmake -DPYTHON_EXECUTABLE:FILEPATH=/path/to/your/bin/python ..
make
cd ../
```
This will install v0.1. Use `-DPYTHON_EXECUTABLE` if you want to build with specific virtual env, otherwise just use `cmake ..` \
Check [Matterport3D Simulator v0.1](https://github.com/peteanderson80/Matterport3DSimulator/tree/v0.1) for further dependency installation.

2. Download the precomputed ResNet Image Features from [here](https://www.dropbox.com/s/o57kxh2mn5rkx4o/ResNet-152-imagenet.zip?dl=1) and place it to `$ROOT/ss_baselines/savi/dialog/speaker/img_features/`
```
mkdir -p $ROOT/ss_baselines/savi/dialog/speaker/img_features/
cd $ROOT/ss_baselines/savi/dialog/speaker/img_features/
wget https://url/ResNet-152-imagenet.zip -O ResNet-152-imagenet.zip
unzip ResNet-152-imagenet.zip
```

3. Download FGR2R dataset from [here](https://github.com/YicongHong/Fine-Grained-R2R) and place it to `$ROOT/ss_baselines/savi/dialog/speaker/tasks/R2R/data/`

4. Download pretrained weights of the speaker model from [here](http://url/speaker_model_weights.zip). Unzip it and place the two files in `$ROOT/ss_baselines/savi/dialog/speaker/tasks/R2R/speaker/snapshots/`

### May also require to Install
- CUDA compatible pytorch version (>=1.7.1)
- torchtext compatible with pytorch version

## Pretrained Weights for $\pi_g$ and $\pi_l$

Download the weights from [here](http://url/pretrained_weights.zip), unzip it and place the two folders: (i) semantic_audionav and (ii) semantic_audionav_distractor at `$ROOT/data/pretrained_weights/`


## Instruction on Training AVLEN:

To use multiple GPUs, submit slurm.sh from the `$ROOT` dir.  `slurm.sh` file is in `$ROOT/ss_baselines/savi/`. To submit slurm:
- Edit the python command based on the type of training
- Change master port to run multiple instance at the same time

### Heard/Unheard Sound

There are two stages of training:
- 1st stage: does not uses history information
- 2nd stage: uses the history information

#### 1st stage:
```
python ss_baselines/savi/run.py --exp-config ss_baselines/savi/config/semantic_audionav/savi_interactive_1st_stage.yaml --model-dir data/models/AVLEN RL.SOFT_QUERY_REWARD True ALLOW_STOP True RL.QUERY_REWARD -1.2 RL.CONSECUTIVE_REWARD -0.5 REPLAY_STORE True
```

#### 2nd stage:
```
python ss_baselines/savi/run.py --exp-config ss_baselines/savi/config/semantic_audionav/savi_interactive_2nd_stage.yaml --model-dir data/models/AVLEN RL.SOFT_QUERY_REWARD True ALLOW_STOP True RL.QUERY_REWARD -1.2 RL.CONSECUTIVE_REWARD -0.5 RESUME_CHECKPOINT True
```

For the first stage training use `savi_interactive_1st_stage.yaml` and for the second stage training use `savi_interactive_2nd_stage.yaml`.

### Distractor Sound:

#### 1st stage:
```
python ss_baselines/savi/run.py --exp-config ss_baselines/savi/config/semantic_audionav_distractor/savi_interactive_1st_stage.yaml --model-dir data/models/AVLEN_dis RL.SOFT_QUERY_REWARD True ALLOW_STOP True RL.QUERY_REWARD -1.2 RL.CONSECUTIVE_REWARD -0.5
```
#### 2nd stage:
```
python ss_baselines/savi/run.py --exp-config ss_baselines/savi/config/semantic_audionav_distractor/savi_interactive_2nd_stage.yaml --model-dir data/models/AVLEN_dis RL.SOFT_QUERY_REWARD True ALLOW_STOP True RL.QUERY_REWARD -1.2 RL.CONSECUTIVE_REWARD -1.0 RESUME_CHECKPOINT True
```

## Instruction on Training $\pi_l$

Trained using single gpu
```
python ss_baselines/savi/run.py --exp-config ss_baselines/savi/config/semantic_audionav/savi_pretraining_dialog_training.yaml --model-dir data/models/AVLEN_VLN
```

## Pretrained Weights for $\pi_q$

Download the weights from [here](http://url/ckpt.119.pth) and place it as `$ROOT/data/models/AVLEN/data/ckpt.119.pth`(from general) or `$ROOT/data/models/AVLEN_dis/data/ckpt.119.pth` (from distractor). (considering `$ROOT/data/models/AVLEN/` and `$ROOT/data/models/AVLEN_dis/` are the model directories for general and distractor case respectively)


## Instruction on Testing AVLEN

To evaluate a single checkpoint, indicate the checkpoint path for `EVAL_CKPT_PATH_DIR`

### Unheard and Heard

```
python ss_baselines/savi/run.py --run-type eval --exp-config ss_baselines/savi/config/semantic_audionav/savi_pretraining_interactive.yaml EVAL_CKPT_PATH_DIR <path to checkpoint> EVAL.SPLIT test USE_SYNC_VECENV True RL.DDPPO.pretrained False
```
Above mentioned command will use 'unheard' sound. If you want to use 'heard' sound, update L214 of `$ROOT/soundspaces/tasks/semantic_audionav_task.py`

### Distractor
```
python ss_baselines/savi/run.py --run-type eval --exp-config ss_baselines/savi/config/semantic_audionav_distractor/savi_pretraining_interactive.yaml EVAL_CKPT_PATH_DIR <path to checkpoint> EVAL.SPLIT test_distractor USE_SYNC_VECENV True RL.DDPPO.pretrained False
```

## Instruction on Testing $\pi_l$

```
python ss_baselines/savi/run.py --run-type eval --exp-config ss_baselines/savi/config/semantic_audionav/savi_pretraining_dialog_training.yaml EVAL_CKPT_PATH_DIR <path to $\pi_l$ model> val USE_SYNC_VECENV True RL.DDPPO.pretrained False
```

## Contact
Anoop Cherian, cherian@merl.com or Sudipta Paul, spaul007@ucr.edu.

## Citation
```
@article{paul2022avlen,
  title={AVLEN: Audio-Visual-Language Embodied Navigation in 3D Environments},
  author={Paul, Sudipta and Roy-Chowdhury, Amit K and Cherian, Anoop},
  journal={arXiv preprint arXiv:2210.07940},
  year={2022}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.

## License

Released under `AGPL-3.0-or-later` license, as found in the [LICENSE.md](LICENSE.md) file.

All files, except as noted below:

```
Copyright (c) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
```

`SoundSpaces` was adapted from https://github.com/facebookresearch/sound-spaces (`CC-BY-4.0` license as found in [LICENSES/CC-BY-4.0.txt](LICENSES/CC-BY-4.0.txt)).

`Habitat Lab Dialog` was adapted from https://github.com/facebookresearch/habitat-lab/tree/v0.1.7 (`MIT` License as found in [LICENSES/MIT.txt](LICENSES/MIT.txt)).

`ss_baselines/savi/dialog/speaker` was adapted from https://github.com/ronghanghu/speaker_follower/blob/master/ (`BSD-2-Clause` license as found in [LICENSES/BSD-2-Clause.txt](LICENSES/BSD-2-Clause.txt)).

`ss_baselines/savi/dialog/ques_gen` was adapted from https://github.com/ranjaykrishna/iq/ (`MIT` license as found in [LICENSES/MIT.txt](LICENSES/MIT.txt)).

`ss_baselines/savi/dialog/speaker/pybind11` is from https://github.com/pybind/pybind11/ (`BSD-3-Clause` license as found in [LICENSES/BSD-3-Clause.txt](LICENSES/BSD-3-Clause.txt)).

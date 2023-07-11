#!/usr/bin/env python3

# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: CC-BY-4.0

import os
import time
import logging
from collections import deque, defaultdict
from typing import Dict, List, Any
import json
import random
import glob
import sys
import copy
import clip
import itertools as it
import matplotlib.pyplot as plt
from copy import deepcopy
from math import floor
import math
import scipy.stats
from torchvision.utils import save_image
from PIL import Image


import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from numpy.linalg import norm

from habitat import Config, logger
from ss_baselines.common.utils import observations_to_image
from ss_baselines.common.base_trainer import BaseRLTrainer
from ss_baselines.common.baseline_registry import baseline_registry
from ss_baselines.common.env_utils import construct_envs
from ss_baselines.common.environments import get_env_class
from ss_baselines.common.rollout_storage import RolloutStorage
from ss_baselines.common.tensorboard_utils import TensorboardWriter
from ss_baselines.common.utils import (
    batch_obs,
    generate_video,
    linear_decay,
    plot_top_down_map,
    resize_observation,
    NpEncoder
)
from ss_baselines.savi.ppo.policy import AudioNavBaselinePolicy, AudioNavSMTPolicy
from ss_baselines.savi.ppo.ppo import PPO
from ss_baselines.savi.ppo.slurm_utils import (
    EXIT,
    REQUEUE,
    load_interrupted_state,
    requeue_job,
    save_interrupted_state,
)
from ss_baselines.savi.models.rollout_storage import RolloutStorage, ExternalMemory
from ss_baselines.savi.models.belief_predictor import BeliefPredictor
from habitat.tasks.nav.nav import IntegratedPointGoalGPSAndCompassSensor
from soundspaces.tasks.nav import LocationBelief, CategoryBelief, SpectrogramSensor


# ----------------------------------------------------------
from habitat.tasks.utils import cartesian_to_polar

from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)

import pynvml
from pynvml.smi import nvidia_smi
pynvml.nvmlInit()


SPEAKER = True #(also change in ppo_trainer.py)


class DataParallelPassthrough(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


@baseline_registry.register_trainer(name="savi")
class PPOTrainer(BaseRLTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None

        self._static_smt_encoder = False
        self._encoder = None

    def _setup_actor_critic_agent(self, ppo_cfg: Config, observation_space=None) -> None:
        r"""Sets up actor critic and agent for PPO.
        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        if observation_space is None:
            observation_space = self.envs.observation_spaces[0]

        if not ppo_cfg.use_external_memory:
            self.actor_critic = AudioNavBaselinePolicy(
                observation_space=observation_space,
                action_space=self.envs.action_spaces[0],
                hidden_size=ppo_cfg.hidden_size,
                goal_sensor_uuid=self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
                extra_rgb=self.config.EXTRA_RGB
            )
        else:
            smt_cfg = ppo_cfg.SCENE_MEMORY_TRANSFORMER
            self.actor_critic = AudioNavSMTPolicy(
                observation_space=observation_space,
                action_space=self.envs.action_spaces[0],
                hidden_size=smt_cfg.hidden_size,
                nhead=smt_cfg.nhead,
                num_encoder_layers=smt_cfg.num_encoder_layers,
                num_decoder_layers=smt_cfg.num_decoder_layers,
                dropout=smt_cfg.dropout,
                activation=smt_cfg.activation,
                use_pretrained=smt_cfg.use_pretrained,
                pretrained_path=smt_cfg.pretrained_path,
                use_belief_as_goal=ppo_cfg.use_belief_predictor,
                use_label_belief=smt_cfg.use_label_belief,
                use_location_belief=smt_cfg.use_location_belief
            )

            if ppo_cfg.use_belief_predictor:
                belief_cfg = ppo_cfg.BELIEF_PREDICTOR
                smt = self.actor_critic.net.smt_state_encoder
                self.belief_predictor = BeliefPredictor(belief_cfg, self.device, smt._input_size, smt._pose_indices,
                                                        smt.hidden_state_size, self.envs.num_envs,
                                                        ).to(device=self.device)
                for param in self.belief_predictor.parameters():
                    param.requires_grad = False

        self.agent = PPO(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
        )

        if self.config.RESUME:
            ckpt_dict = self.load_checkpoint('data/models/smt_with_pose/ckpt.400.pth', map_location="cpu")
            self.agent.actor_critic.net.visual_encoder.load_state_dict(self.search_dict(ckpt_dict, 'visual_encoder'))
            self.agent.actor_critic.net.goal_encoder.load_state_dict(self.search_dict(ckpt_dict, 'goal_encoder'))
            self.agent.actor_critic.net.action_encoder.load_state_dict(self.search_dict(ckpt_dict, 'action_encoder'))

        if ppo_cfg.use_external_memory and smt_cfg.freeze_encoders:
            self._static_smt_encoder = True
            self.actor_critic.net.freeze_encoders()

        self.actor_critic.to(self.device)





    @staticmethod
    def search_dict(ckpt_dict, encoder_name):
        encoder_dict = {}
        for key, value in ckpt_dict['state_dict'].items():
            if encoder_name in key:
                encoder_dict['.'.join(key.split('.')[3:])] = value

        return encoder_dict

    def save_checkpoint(
        self, file_name: str, extra_state=None
    ) -> None:
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        if self.config.RL.PPO.use_belief_predictor:
            checkpoint["belief_predictor"] = self.belief_predictor.state_dict()
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def save_checkpoint_vln(
        self, file_name: str, extra_state=None
    ) -> None:
        checkpoint = {
            "state_dict": self.agent_vln.state_dict(),
            "config": self.config,
        }
        if self.config.RL.PPO.use_belief_predictor:
            checkpoint["belief_predictor"] = self.belief_predictor.state_dict()
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    def try_to_resume_checkpoint(self):
        checkpoints = glob.glob(f"{self.config.CHECKPOINT_FOLDER}/*.pth")
        checkpoints_vln = glob.glob(f"{self.config.CHECKPOINT_FOLDER}/vln/*.pth")
        if len(checkpoints) == 0:
            count_steps = 0
            count_checkpoints = 0
            start_update = 0
        else:
            # Restore option policy weights
            last_ckpt = sorted(checkpoints, key=lambda x: int(x.split(".")[1]))[-1]
            checkpoint_path = last_ckpt
            ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
            self.agent.load_state_dict(ckpt_dict["state_dict"])
            ckpt_id = int(last_ckpt.split("/")[-1].split(".")[1])
            count_steps = ckpt_dict["extra_state"]["step"]
            count_checkpoints = ckpt_id + 1
            start_update = ckpt_dict["config"].CHECKPOINT_INTERVAL * ckpt_id + 1
            print(f"Resuming checkpoint {last_ckpt} at {count_steps} frames")

            # Restore belief_predictor weights
            if self.config.RL.PPO.use_belief_predictor:
                self.belief_predictor.load_state_dict(ckpt_dict["belief_predictor"])

        if len(checkpoints_vln) == 0:
            count_checkpoints_vln = 0
            replay_training_cnt = 0
        else:
            # Restore option policy weights
            last_ckpt = sorted(checkpoints_vln, key=lambda x: int(x.split(".")[1]))[-1]
            checkpoint_path = last_ckpt
            ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
            self.agent_vln.load_state_dict(ckpt_dict["state_dict"])
            ckpt_id = int(last_ckpt.split("/")[-1].split(".")[1])
            count_checkpoints_vln = ckpt_id + 1
            replay_training_cnt = ckpt_dict["extra_state"]["step"]
            print("Resuming vln checkpoint {} at {} model".format(last_ckpt, count_checkpoints_vln-1))

        return count_steps, count_checkpoints, start_update, replay_training_cnt, count_checkpoints_vln

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    def _quat_to_xy_heading(self, quat):
        direction_vector = np.array([0, 0, -1])
        heading_vector = quaternion_rotate_vector(quat, direction_vector)
        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return phi

    def _collect_rollout_step(
        self, rollouts, current_episode_info, running_episode_stats, track_query, track_query_count, tf_ratio=1.0,
    ):
        pth_time = 0.0
        env_time = 0.0
        t_sample_action = time.time()

        is_queried = []
        query_num = []
        cons_reward_list = []

        # ----------------------------------------
        # collect state info
        state_all = self.envs.agent_state()
        curr_rotations = [state[1] for state in state_all]
        scenes = [state[2] for state in state_all]
        views = [state[4] for state in state_all]
        app_points = [state[5] for state in state_all]
        new_episode = self.envs.is_new_episode()
        # dialog pretraining
        sub_instrs = [state[6] for state in state_all]
        current_target_distance = [state[7] for state in state_all]

        with torch.no_grad():
            current_dialog = torch.zeros(self.envs.num_envs, self.max_dialog_len, dtype=torch.long).to(self.device)
            current_query_state = torch.zeros(self.envs.num_envs, self.config.QUERY_COUNT_EMB_SIZE).to(self.device)
            last_query_info = torch.zeros(self.envs.num_envs, self.config.QUERY_COUNT_EMB_SIZE).to(self.device)
            current_agent_step = torch.zeros(self.envs.num_envs,1).to(self.device)

        masks_vln = np.ones((len(new_episode),1))

        if self.config.DIALOG_TRAINING :
            replay_store = None
            # keep info of agent step
            for idx, _ in enumerate(new_episode):
                if new_episode[idx]:
                    track_query[idx]['step'] = 0
                    dialog = sub_instrs[idx]
                    # using clip
                    tokenized_dialog = clip.tokenize(dialog)[0]
                    track_query[idx]['dialog'] = tokenized_dialog
                else:
                    track_query[idx]['step'] += 1

                assert track_query[idx]['step']< self.config.RL.PPO.num_steps, 'track_query[idx][step] crosses num_steps'
                current_dialog[idx,:track_query[idx]['dialog'].size()[0]] = track_query[idx]['dialog']
                current_agent_step[idx].copy_(torch.tensor([track_query[idx]['step']]))

            with torch.no_grad():
                rollouts.agent_step[rollouts.step].copy_(current_agent_step.view(-1))

        with torch.no_grad():
            step_observation = {
                    k: v[rollouts.step] for k, v in rollouts.observations.items()
                }

        with torch.no_grad():
            if self.config.RL.PPO.use_external_memory:
                external_memory_vln = rollouts.external_memory_vln[:, rollouts.step].contiguous()
                external_memory_vln_masks = rollouts.external_memory_vln_masks[rollouts.step]

                if not self.config.DIALOG_TRAINING:
                    external_memory_goal = rollouts.external_memory_goal[:, rollouts.step].contiguous()
                    external_memory_option = rollouts.external_memory_option[:, rollouts.step].contiguous()
                    external_memory_masks = rollouts.external_memory_masks[rollouts.step]

            if self.config.RL.PPO.use_state_memory:
                external_memory_vln_dialog = rollouts.external_memory_vln_dialog[:, rollouts.step].contiguous()
                external_memory_vln_masks = rollouts.external_memory_vln_masks[rollouts.step]

        if not self.config.DIALOG_TRAINING:
            for idx, _ in enumerate(new_episode):
                if new_episode[idx]:
                    track_query[idx]['queried'] = False
                    track_query[idx]['step'] = 0
                    track_query[idx]['total_step'] = 0
                    track_query[idx]['last_query_step'] = 0
                    track_query[idx]['cons_reward'] = 0
                    track_query[idx]['all_step'] = []
                    track_query[idx]['all_reward'] = []

                    track_query[idx]['dialog'] = []
                    track_query_count[idx] = 0
                    diff_step = 150
                else:
                    track_query[idx]['total_step'] += 1
                    if track_query_count[idx]>=2:
                        diff_step = track_query[idx]['total_step']-track_query[idx]['last_query_step']
                    else:
                        diff_step = 150

                with torch.no_grad():
                    current_query_state[idx,:].copy_(self.pe[track_query_count[idx],:])
                    last_query_info[idx,:].copy_(self.pe[diff_step,:])

        with torch.no_grad():

            if self.config.DIALOG_TRAINING:

                (
                    values,
                    actions,
                    actions_log_probs,
                    recurrent_hidden_states,
                    external_memory_features,
                    external_memory_dialog_features,
                    action_prob_vln  # dialog pretraining
                ) = self.actor_critic_vln.act_dialog(
                    step_observation,
                    rollouts.recurrent_hidden_states[rollouts.step],
                    rollouts.prev_actions[rollouts.step],
                    rollouts.masks_vln[rollouts.step],
                    external_memory_vln,
                    external_memory_vln_dialog,
                    external_memory_vln_masks,
                    # -----
                    current_dialog,
                    rollouts.agent_step[rollouts.step],
                    without_dialog=self.config.DIALOG_TRAINING_WITHOUT_DIALOG,

                )
                actions_option = torch.zeros(self.envs.num_envs, 1).long()
                actions_log_probs_option = torch.zeros(self.envs.num_envs, 1).long()
            else:
                # print('external_memory', external_memory.size())
                # sys.exit()
                values, unct, actions_option, actions_log_probs_option, recurrent_hidden_states, external_memory_features_option, action_prob = self.agent.actor_critic.act_option(
                    step_observation,
                    rollouts.recurrent_hidden_states[rollouts.step],
                    rollouts.prev_actions[rollouts.step],
                    rollouts.masks[rollouts.step],
                    external_memory_option,
                    external_memory_masks,
                    rollouts.query_state[rollouts.step],
                    rollouts.last_query_info[rollouts.step]
                )         # option_prob: (b,2), 0 : goal based, 1: vln



                for idx, _ in enumerate(new_episode):
                    if not track_query[idx]['queried'] and actions_option[idx]==1: # and action_prob[idx][0]< action_prob[idx][1]:
                        if self.config.QUERY_WITHIN_RADIUS:
                            track_query[idx]['queried'] = True
                            track_query_count[idx] += 1
                        else:
                            if current_target_distance[idx]>3:
                                track_query[idx]['queried'] = True
                                track_query_count[idx] += 1

                    query_num.append(track_query_count[idx])



        if not self.config.DIALOG_TRAINING:
            replay_store = None
            if self.config.REPLAY_STORE:
                # to store dialog episode information
                replay_store = {}
                replay_store['batch'] = {} # done
                for key in step_observation.keys():
                    replay_store['batch'][key] = []
                replay_store['recurrent_hidden_states'] = [] # done
                replay_store['actions'] = [] # done
                replay_store['actions_option'] = [] # done
                replay_store['actions_log_probs_option'] = [] # done
                replay_store['values'] = [] # done
                replay_store['rewards'] = [] # done
                replay_store['masks'] = [] # done
                replay_store['masks_vln'] = [] # done
                replay_store['external_memory_features'] = [] #done
                replay_store['external_memory_dialog_features'] = [] #done
                replay_store['current_dialog'] = [] # done
                replay_store['o_action'] = [] # done
                replay_store['o_mask'] = [] # done
                replay_store['action_prob'] = [] # done
                replay_store['current_query_state'] = []# done
                replay_store['current_agent_step'] = []   # done
                replay_store['id'] = []         # done

            rl_mask = []
            #actions_option_updated = []
            for idx, _ in enumerate(new_episode):
                track_query[idx]['cons_reward'] = 0
                if track_query[idx]['queried']:
                    #actions_option_updated.append(1)
                    is_queried.append(True)
                    if self.config.REPLAY_STORE:
                        # store in replay buffer
                        with torch.no_grad():
                            replay_store['id'].append(idx)
                            for key, elem in step_observation.items():
                                replay_store['batch'][key].append(elem[idx].cpu().numpy())

                            replay_store['recurrent_hidden_states'].append(rollouts.recurrent_hidden_states[rollouts.step,:,idx,:].cpu().numpy())
                            replay_store['actions_option'].append(actions_option[idx].cpu().numpy())
                            replay_store['actions_log_probs_option'].append(actions_log_probs_option[idx].cpu().numpy())
                            replay_store['values'].append(values[idx].cpu().numpy())
                            replay_store['current_dialog'].append(current_dialog[idx].cpu().numpy())
                            replay_store['masks'].append(rollouts.masks[rollouts.step][idx].cpu().numpy())
                            replay_store['masks_vln'].append(rollouts.masks[rollouts.step][idx].cpu().numpy())


                    if track_query[idx]['step']==0:
                        track_query[idx]['all_step'].append('Q')
                        if track_query_count[idx]>=2:
                            diff_step = (track_query[idx]['total_step']-(track_query[idx]['last_query_step']+2))
                            # track_query[idx]['cons_reward'] = self.config.RL.CONSECUTIVE_REWARD/math.exp(0.6*max(diff_step-1,0))
                            if diff_step>10:
                                track_query[idx]['cons_reward'] = 0
                            else:
                                track_query[idx]['cons_reward'] = self.config.RL.CONSECUTIVE_REWARD/max(diff_step,1)

                            # print('cons_reward', track_query[idx]['cons_reward'], 'total_step', track_query[idx]['total_step'], 'last_query_step', track_query[idx]['last_query_step'])
                        track_query[idx]['last_query_step'] = track_query[idx]['total_step']
                        rl_mask.append(1)
                        if not app_points[idx]:
                            # self.invalid_point_cnt += 1
                            continue
                        heading = self._quat_to_xy_heading(curr_rotations[idx])
                        # get the ground truth approximate path
                        gt_app_path = [app_points[idx][0]]
                        for gt_vln_node in app_points[idx][1:]:
                            if gt_vln_node not in gt_app_path and len(gt_app_path)<3:
                                gt_app_path.append(gt_vln_node)

                        # from GT trajectory, get instruction
                        speaker_entry = {
                                         'heading': heading,
                                         'scene': scenes[idx],
                                         'path': gt_app_path
                                         }
                        # change ----------------------------
                        if SPEAKER:
                            dialog = ' '.join(self.speaker.generate_instr(speaker_entry)[0]['words'])
                        else:
                            dialog='hello'
                        # using clip
                        with torch.no_grad():
                            tokenized_dialog = clip.tokenize(dialog)[0]
                            track_query[idx]['dialog'] = tokenized_dialog



                        with torch.no_grad():
                            if track_query_count[idx]> self.config.RL.NUM_TOTAL_QUERY:
                                current_episode_info['current_episode_query_cnt_thresh'][idx] += 1

                            if current_target_distance[idx] <= 3:
                                current_episode_info['current_episode_query_cnt_radius'][idx] += 1

                            if track_query_count[idx]==1:
                                current_episode_info['current_episode_1st_query'][idx] = track_query[idx]['total_step']

                            if track_query_count[idx]==4:
                                current_episode_info['current_episode_4th_query'][idx] = track_query[idx]['total_step']
                    else:
                        rl_mask.append(0)
                        track_query[idx]['all_step'].append('V')

                    if track_query[idx]['step'] < self.config.NUM_DIALOG_STEPS:
                        # tokenized_dialog_tensor = torch.from_numpy(track_query[idx]['dialog'])
                        with torch.no_grad():
                            current_dialog[idx,:track_query[idx]['dialog'].shape[0]] = track_query[idx]['dialog']
                            current_agent_step[idx].copy_(torch.tensor([track_query[idx]['step']]))
                        track_query[idx]['step'] += 1
                else:
                    is_queried.append(False)
                    rl_mask.append(1)
                    track_query[idx]['all_step'].append('G')
                    #actions_option_updated.append(0)

                cons_reward_list.append(track_query[idx]['cons_reward'])
            with torch.no_grad():
                rollouts.agent_step[rollouts.step].copy_(current_agent_step.view(-1))
                rollouts.query_state[rollouts.step].copy_(current_query_state)
                rollouts.last_query_info[rollouts.step].copy_(last_query_info)

            with torch.no_grad():
                (
                    values_goal,
                    actions_goal,
                    actions_log_probs_goal,
                    recurrent_hidden_states_goal,
                    external_memory_features_goal,
                    action_prob_goal,
                ) = self.actor_critic_goal.act(
                    step_observation,
                    rollouts.recurrent_hidden_states[rollouts.step],
                    rollouts.prev_actions[rollouts.step],
                    rollouts.masks[rollouts.step],
                    external_memory_goal,
                    external_memory_masks,
                )

                (
                    values_vln,
                    actions_vln,
                    actions_log_probs_vln,
                    recurrent_hidden_states_vln,
                    external_memory_features_vln,
                    external_memory_dialog_features,
                    action_prob_vln,
                ) = self.agent_vln.actor_critic.act_dialog(
                    step_observation,
                    rollouts.recurrent_hidden_states[rollouts.step],
                    rollouts.prev_actions[rollouts.step],
                    rollouts.masks_vln[rollouts.step],
                    external_memory_vln,
                    external_memory_vln_dialog,
                    external_memory_vln_masks,
                    # -----
                    current_dialog,
                    rollouts.agent_step[rollouts.step],
                    without_dialog=self.config.DIALOG_TRAINING_WITHOUT_DIALOG,
                )
                # create storage tensors
                # sort out all the element based on selection
                actions = []
                oracle_actions = self.envs.compute_oracle_actions()

                o_action = [a[0] for a in oracle_actions]
                o_action_updated = [a[0] for a in oracle_actions]

                o_mask = []
                ucnt_gt = []
                goal_action_prob_npy = np.sort(action_prob_goal.cpu().numpy())

                for idx, _ in enumerate(new_episode):
                    if goal_action_prob_npy[idx][3]-goal_action_prob_npy[idx][2]<0.1:
                        ucnt_gt.append(1)
                    else:
                        ucnt_gt.append(0)

                    if track_query[idx]['queried']:
                        if o_action_updated[idx]==0:
                            if self.config.ORACLE_WHEN_QUERIED:
                                if not self.config.ALLOW_STOP:
                                    actions.append(actions_vln[idx])
                                    #actions.append(actions_goal[idx])
                                else:
                                    actions.append(torch.tensor([o_action_updated[idx]],dtype=torch.long).to(self.device))
                                    #actions.append(actions_goal[idx])
                            else:
                                actions.append(torch.tensor([o_action_updated[idx]],dtype=torch.long).to(self.device))
                                #actions.append(actions_goal[idx])
                            o_mask.append(0)
                            # collecting info
                            current_episode_info['current_episode_step_stat_vln'][idx, actions_vln[idx]] += 1

                        else:
                            if self.config.ORACLE_WHEN_QUERIED:
                                actions.append(torch.tensor([o_action_updated[idx]],dtype=torch.long).to(self.device))
                                #actions.append(actions_goal[idx])
                                # collecting info
                                current_episode_info['current_episode_step_stat_vln'][idx, o_action_updated[idx]] += 1
                            else:
                                actions.append(actions_vln[idx])
                                #actions.append(actions_goal[idx])
                                # collecting info
                                current_episode_info['current_episode_step_stat_vln'][idx, actions_vln[idx]] += 1

                            o_mask.append(1)

                    # elif action_prob[idx][0]>= action_prob[idx][1] and self.config.QUERY_WITHIN_RADIUS:
                    else:
                        actions.append(actions_goal[idx])
                        o_mask.append(1)
                        # collecting info
                        current_episode_info['current_episode_step_stat_goal'][idx, actions_goal[idx]] += 1

                # print('actions', actions)
                actions = torch.stack(actions, dim=0)
                # print('actions', actions)
                # sys.exit()

        curr_actions = [a[0].item() for a in actions]

        # updated_action considering ground truth action when query is called
        # u_actions = copy.deepcopy(curr_actions)

        # dialog pretraining
        if self.config.DIALOG_TRAINING:
            o_action = [int(a) for a in self.o_actions_updated[rollouts.step, :]]
            o_mask = torch.tensor([a for a in self.o_actions_mask[rollouts.step, :]], dtype=torch.long)
            outputs = self.envs.step(o_action)
            o_action = torch.tensor(o_action, dtype=torch.float)
        else:
            self.envs.set_is_queried(is_queried)
            self.envs.set_query_num(query_num)
            self.envs.set_constraint_reward(cons_reward_list)

            outputs = self.envs.step(curr_actions)
            # for using ground truth
            with torch.no_grad():
                o_action = torch.tensor(o_action, dtype=torch.float)
                o_action_updated = torch.tensor(o_action_updated, dtype=torch.float)
                o_mask = torch.tensor(o_mask, dtype=torch.long)
                rl_mask = torch.tensor(rl_mask, dtype=torch.long)
                #actions_option_updated = torch.unsqueeze(torch.tensor(actions_option_updated, dtype=torch.long),-1).to(device=self.device)
                ucnt_gt = torch.tensor(ucnt_gt, dtype=torch.long)
            # -------------------
            # o_mask = torch.tensor([1]*len(new_episode), dtype=torch.long)

        pth_time += time.time() - t_sample_action
        t_step_env = time.time()


        # logging.info('actions_option: {},  rl_mask: {}'.format(actions_option, rl_mask))
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
        # logger.info('Reward: {}'.format(rewards[0]))

        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(observations, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=current_episode_info['current_episode_reward'].device)
        rewards = rewards.unsqueeze(1)

        if not self.config.DIALOG_TRAINING :
            for idx, _ in enumerate(new_episode):
                track_query[idx]['all_reward'].append(rewards[idx].cpu().numpy()[0])
                if track_query[idx]['queried']:
                    with torch.no_grad():
                        current_episode_info['current_episode_reward_vln'][idx] += rewards[idx]
                        current_episode_info['current_episode_step_vln'][idx] += 1


                    if self.config.REPLAY_STORE:
                        with torch.no_grad():
                            # store
                            replay_store['o_action'].append(o_action[idx].cpu().numpy())
                            replay_store['o_mask'].append(o_mask[idx].cpu().numpy())
                            replay_store['rewards'].append(rewards[idx].cpu().numpy())
                            replay_store['actions'].append(actions[idx].cpu().numpy())
                            replay_store['external_memory_features'].append(external_memory_features_vln[idx].cpu().numpy())
                            replay_store['external_memory_dialog_features'].append(external_memory_dialog_features[idx].cpu().numpy())
                            replay_store['current_query_state'].append(current_query_state[idx].cpu().numpy())
                            replay_store['current_agent_step'].append(current_agent_step.view(-1)[idx].cpu().numpy())
                            replay_store['action_prob'].append(action_prob_vln[idx].cpu().numpy())

                    if track_query[idx]['step']>= self.config.NUM_DIALOG_STEPS:
                        track_query[idx]['queried'] = False
                        track_query[idx]['step'] = 0
                        track_query[idx]['dialog'] = []
                        masks_vln[idx,0] = 0.0
                else:
                    with torch.no_grad():
                        current_episode_info['current_episode_reward_goal'][idx] += rewards[idx]
                        current_episode_info['current_episode_step_goal'][idx] += 1

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones], dtype=torch.float, device=current_episode_info['current_episode_reward'].device
        )

        masks_vln = torch.tensor(
            masks_vln, dtype=torch.float, device=current_episode_info['current_episode_reward'].device
        )


        current_episode_info['current_episode_reward'] += rewards


        #print('current_episode_info', current_episode_info)
        #print('running_episode_stats[vln_ratio]', running_episode_stats['vln_ratio'])

        running_episode_stats["reward"] += (1 - masks) * current_episode_info['current_episode_reward']
        if not self.config.DIALOG_TRAINING:
            with torch.no_grad():
                query_num = torch.tensor(query_num, device=current_episode_info['current_episode_reward'].device).unsqueeze(-1)

            running_episode_stats["reward_goal"] += (1 - masks) * current_episode_info['current_episode_reward_goal']
            running_episode_stats["reward_vln"] += (1 - masks) * current_episode_info['current_episode_reward_vln']
            running_episode_stats['query_count'] += (1 - masks) * query_num
            running_episode_stats['step_count'] += (1 - masks) * (current_episode_info['current_episode_step_goal']+current_episode_info['current_episode_step_vln'])
            running_episode_stats['step_count_goal'] += (1 - masks) * current_episode_info['current_episode_step_goal']
            running_episode_stats['step_count_vln'] += (1 - masks) * current_episode_info['current_episode_step_vln']
            running_episode_stats['forward_step_goal'] += (1 - masks) * current_episode_info['current_episode_step_stat_goal'][:,1].unsqueeze(-1)
            running_episode_stats['left_step_goal'] += (1 - masks) * current_episode_info['current_episode_step_stat_goal'][:,2].unsqueeze(-1)
            running_episode_stats['right_step_goal'] += (1 - masks) * current_episode_info['current_episode_step_stat_goal'][:,3].unsqueeze(-1)
            running_episode_stats['forward_step_vln'] += (1 - masks) * current_episode_info['current_episode_step_stat_vln'][:,1].unsqueeze(-1)
            running_episode_stats['left_step_vln'] += (1 - masks) * current_episode_info['current_episode_step_stat_vln'][:,2].unsqueeze(-1)
            running_episode_stats['right_step_vln'] += (1 - masks) * current_episode_info['current_episode_step_stat_vln'][:,3].unsqueeze(-1)
            running_episode_stats['query_count_thresh'] += (1 - masks) * current_episode_info['current_episode_query_cnt_thresh']
            running_episode_stats['query_count_radius'] += (1 - masks) * current_episode_info['current_episode_query_cnt_radius']
            running_episode_stats['query_step_1st'] += (1 - masks) * current_episode_info['current_episode_1st_query']
            running_episode_stats['query_step_4th'] += (1 - masks) * current_episode_info['current_episode_4th_query']

        running_episode_stats["count"] += 1 - masks
        for k, v in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v, dtype=torch.float, device=current_episode_info['current_episode_reward'].device
            ).unsqueeze(1)
            if k not in running_episode_stats:
                running_episode_stats[k] = torch.zeros_like(
                    running_episode_stats["count"]
                )

            running_episode_stats[k] += (1 - masks) * v

        current_episode_info['current_episode_reward'] *= masks

        if not self.config.DIALOG_TRAINING:
            current_episode_info['current_episode_reward_goal'] *= masks
            current_episode_info['current_episode_reward_vln'] *= masks
            current_episode_info['current_episode_step_goal'] *= masks
            current_episode_info['current_episode_step_vln'] *= masks
            current_episode_info['current_episode_step_stat_goal'] *= masks.repeat(1,4)
            current_episode_info['current_episode_step_stat_vln'] *= masks.repeat(1,4)
            current_episode_info['current_episode_query_cnt_thresh'] *= masks
            current_episode_info['current_episode_query_cnt_radius'] *= masks
            current_episode_info['current_episode_1st_query'] *= masks
            current_episode_info['current_episode_4th_query'] *= masks

        # also insert dialog
        if self.config.DIALOG_TRAINING:
            rollouts.insert(
                batch,
                recurrent_hidden_states,
                actions,
                actions_option,
                actions_log_probs_option,
                values,
                rewards.to(device=self.device),
                masks.to(device=self.device), # proxy
                masks.to(device=self.device),
                external_memory_features, # proxy
                external_memory_features,
                external_memory_features, # proxy
                external_memory_dialog_features,  # [:,:256]
                current_dialog,
                # dialog pretraining
                o_action,
                o_mask,
                o_mask, # proxy for rl_mask
                o_mask, # proxy for ucnt_gt
                action_prob_vln,
                current_query_state,
                current_query_state, # proxy
                current_agent_step.view(-1),
            )
        else:
            rollouts.insert(
                batch,
                recurrent_hidden_states,
                actions,
                actions_option,
                actions_log_probs_option,
                values,
                rewards.to(device=self.device),
                masks.to(device=self.device),
                masks_vln.to(device=self.device),
                external_memory_features_goal,
                external_memory_features_option,
                external_memory_features_vln,
                external_memory_dialog_features,  # [:,:256]
                current_dialog,
                # dialog pretraining
                o_action,
                o_mask,
                rl_mask,
                ucnt_gt,
                action_prob_vln,
                current_query_state,
                last_query_info,
                current_agent_step.view(-1),
            )

        if self.config.RL.PPO.use_belief_predictor:
            step_observation = {k: v[rollouts.step] for k, v in rollouts.observations.items()}
            self.belief_predictor.update(step_observation, dones)
            for sensor in [LocationBelief.cls_uuid, CategoryBelief.cls_uuid]:
                rollouts.observations[sensor][rollouts.step].copy_(step_observation[sensor])

        pth_time += time.time() - t_update_stats
        return pth_time, env_time, self.envs.num_envs, track_query, track_query_count, replay_store



    def assign_to_replay_buffer(self, replay_store):
        # print('len(replay_store[id])',len(replay_store['id']))
        with torch.no_grad():
            for idx, elem in enumerate(replay_store['id']):
                for key in replay_store.keys():
                    if key=='batch':
                        for key2, value in replay_store['batch'].items():
                            # print('value', type(value), len(value), len(replay_store['id']))
                            self.replay_buffer[elem]['batch'][key2].append(value[idx])
                    elif key!='id':
                        self.replay_buffer[elem][key].append(replay_store[key][idx])

    def store_in_rollout(self, rollouts_vln):
        for elem in self.replay_buffer.keys():
            if len(self.replay_buffer[elem]['actions'])==self.config.NUM_DIALOG_STEPS:
                with torch.no_grad():
                    for key in self.replay_buffer[elem].keys():
                        if key=='batch':
                            for key2 in self.replay_buffer[elem]['batch'].keys():
                                self.replay_buffer[elem]['batch'][key2] = torch.from_numpy(np.stack(self.replay_buffer[elem]['batch'][key2], axis=0))
                        else:
                            self.replay_buffer[elem][key] = torch.from_numpy(np.stack(self.replay_buffer[elem][key], axis=0))

                    # send it to rollouts_vln
                    rollouts_vln.insert_replay(
                        self.replay_buffer[elem]['batch'],
                        self.replay_buffer[elem]['recurrent_hidden_states'],
                        self.replay_buffer[elem]['actions'],
                        self.replay_buffer[elem]['actions_option'],
                        self.replay_buffer[elem]['actions_log_probs_option'],
                        self.replay_buffer[elem]['values'],
                        self.replay_buffer[elem]['rewards'], #.to(device=self.device),
                        self.replay_buffer[elem]['masks'], #.to(device=self.device),
                        self.replay_buffer[elem]['masks_vln'], #.to(device=self.device),
                        self.replay_buffer[elem]['external_memory_features'],
                        self.replay_buffer[elem]['external_memory_dialog_features'],  # [:,:256]
                        self.replay_buffer[elem]['current_dialog'],
                        self.replay_buffer[elem]['o_action'],
                        self.replay_buffer[elem]['o_mask'],
                        self.replay_buffer[elem]['action_prob'],
                        self.replay_buffer[elem]['current_query_state'],
                        self.replay_buffer[elem]['current_agent_step'],
                    )
                    self.replay_buffer[elem] = deepcopy(self.store_dict)
                    # print(elem, self.replay_buffer[elem])
                    # sys.exit()

                if rollouts_vln.env_id == self.config.NUM_PROCESSES:
                    rollouts_vln.env_id = 0
                    return True
        return False







    def train_belief_predictor(self, rollouts):
        bp = self.belief_predictor
        num_epoch = 5
        num_mini_batch = 1

        advantages = torch.zeros_like(rollouts.returns)
        value_loss_epoch = 0
        running_regressor_corrects = 0
        num_sample = 0

        for e in range(num_epoch):
            data_generator = rollouts.recurrent_generator(
                advantages, num_mini_batch
            )

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    _,
                    prev_actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                    _,
                    _,
                    external_memory, # should use goal policy external memory
                    _,
                    _,
                    _,
                    _,
                    _,
                    all_dialog_batch,
                    _,
                    _,
                    _,
                ) = sample

                bp.optimizer.zero_grad()

                inputs = obs_batch[SpectrogramSensor.cls_uuid].permute(0, 3, 1, 2)
                preds = bp.cnn_forward(obs_batch)

                masks = (torch.sum(torch.reshape(obs_batch[SpectrogramSensor.cls_uuid],
                        (obs_batch[SpectrogramSensor.cls_uuid].shape[0], -1)), dim=1, keepdim=True) != 0).float()
                gts = obs_batch[IntegratedPointGoalGPSAndCompassSensor.cls_uuid]
                transformed_gts = torch.stack([gts[:, 1], -gts[:, 0]], dim=1)
                masked_preds = masks.expand_as(preds) * preds
                masked_gts = masks.expand_as(transformed_gts) * transformed_gts
                loss = bp.regressor_criterion(masked_preds, masked_gts)

                bp.before_backward(loss)
                loss.backward()
                # self.after_backward(loss)

                bp.optimizer.step()
                value_loss_epoch += loss.item()

                rounded_preds = torch.round(preds)
                bitwise_close = torch.bitwise_and(torch.isclose(rounded_preds[:, 0], transformed_gts[:, 0]),
                                                  torch.isclose(rounded_preds[:, 1], transformed_gts[:, 1]))
                running_regressor_corrects += torch.sum(torch.bitwise_and(bitwise_close, masks.bool().squeeze(1)))
                num_sample += torch.sum(masks).item()

        value_loss_epoch /= num_epoch * num_mini_batch
        if num_sample == 0:
            prediction_accuracy = 0
        else:
            prediction_accuracy = running_regressor_corrects / num_sample

        return value_loss_epoch, prediction_accuracy

    def _update_agent_dialog(self, rollouts):
        t_update_model = time.time()
        dialog_loss = self.agent.update_dialog(rollouts)
        rollouts.after_update()
        return (time.time() - t_update_model, dialog_loss)

    def _update_agent_vln(self, rollouts):
        t_update_model = time.time()
        dialog_loss = self.agent_vln.update_dialog(rollouts)
        return (time.time() - t_update_model, dialog_loss)

    def _update_agent(self, ppo_cfg, rollouts):
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = {
                k: v[-1] for k, v in rollouts.observations.items()
            }
            external_memory = None
            external_memory_masks = None
            if ppo_cfg.use_external_memory:
                external_memory = rollouts.external_memory_option[:, rollouts.step].contiguous()
                external_memory_masks = rollouts.external_memory_masks[rollouts.step]

            # print(f"Used GPU memory after external- {nvidia_smi.getInstance().DeviceQuery('memory.used')}")

            next_value = self.agent.actor_critic.get_value_option(
                last_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
                external_memory,
                external_memory_masks,
                rollouts.query_state[rollouts.step-1],
                rollouts.last_query_info[rollouts.step-1],

            ).detach()
            last_observation=None

        rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )
        # print(f"Used GPU memory after compute returns- {nvidia_smi.getInstance().DeviceQuery('memory.used')}")


        value_loss, action_loss, dist_entropy, values_debug, return_batch_debug, unct_loss = self.agent.update(rollouts)
        # print(f"Used GPU memory after update- {nvidia_smi.getInstance().DeviceQuery('memory.used')}")

        rollouts.after_update()
        # print(f"Used GPU memory after_update call- {nvidia_smi.getInstance().DeviceQuery('memory.used')}")


        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
            values_debug,
            return_batch_debug,
            unct_loss,
        )

    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """
        logger.info(f"config: {self.config}")
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)

        # add_signal_handlers()

        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME), workers_ignore_signals=True
        )

        ppo_cfg = self.config.RL.PPO
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)
        self._setup_actor_critic_agent(ppo_cfg)
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        if ppo_cfg.use_external_memory:
            memory_dim = self.actor_critic.net.memory_dim
        else:
            memory_dim = None

        rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size,
            ppo_cfg.use_external_memory,
            ppo_cfg.SCENE_MEMORY_TRANSFORMER.memory_size + ppo_cfg.num_steps,
            ppo_cfg.SCENE_MEMORY_TRANSFORMER.memory_size,
            memory_dim,
        )
        rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(observations)
        if self.config.RL.PPO.use_belief_predictor:
            self.belief_predictor.update(batch, None)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0
        start_update = 0
        prev_time = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )

        interrupted_state = load_interrupted_state(model_dir=self.config.MODEL_DIR)
        if interrupted_state is not None:
            self.agent.load_state_dict(interrupted_state["state_dict"])
            self.agent.optimizer.load_state_dict(
                interrupted_state["optimizer_state"]
            )
            lr_scheduler.load_state_dict(interrupted_state["lr_scheduler_state"])

            requeue_stats = interrupted_state["requeue_stats"]
            env_time = requeue_stats["env_time"]
            pth_time = requeue_stats["pth_time"]
            count_steps = requeue_stats["count_steps"]
            count_checkpoints = requeue_stats["count_checkpoints"]
            start_update = requeue_stats["start_update"]
            prev_time = requeue_stats["prev_time"]

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            for update in range(start_update, self.config.NUM_UPDATES):
                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )

                if EXIT.is_set():
                    self.envs.close()

                    if REQUEUE.is_set():
                        requeue_stats = dict(
                            env_time=env_time,
                            pth_time=pth_time,
                            count_steps=count_steps,
                            count_checkpoints=count_checkpoints,
                            start_update=update,
                            prev_time=(time.time() - t_start) + prev_time,
                        )
                        save_interrupted_state(
                            dict(
                                state_dict=self.agent.state_dict(),
                                optimizer_state=self.agent.optimizer.state_dict(),
                                lr_scheduler_state=lr_scheduler.state_dict(),
                                config=self.config,
                                requeue_stats=requeue_stats,
                            ),
                            model_dir=self.config.MODEL_DIR
                        )
                        requeue_job()
                    return

                for step in range(ppo_cfg.num_steps):
                    delta_pth_time, delta_env_time, delta_steps = self._collect_rollout_step(
                        rollouts,
                        current_episode_reward,
                        running_episode_stats
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps += delta_steps

                delta_pth_time, value_loss, action_loss, dist_entropy = self._update_agent(
                    ppo_cfg, rollouts
                )
                pth_time += delta_pth_time

                deltas = {
                    k: (
                        (v[-1] - v[0]).sum().item()
                        if len(v) > 1
                        else v[0].sum().item()
                    )
                    for k, v in window_episode_stats.items()
                }
                deltas["count"] = max(deltas["count"], 1.0)

                writer.add_scalar(
                    "Metrics/reward", deltas["reward"] / deltas["count"], count_steps
                )

                # Check to see if there are any metrics
                # that haven't been logged yet
                metrics = {
                    k: v / deltas["count"]
                    for k, v in deltas.items()
                    if k not in {"reward", "count"}
                }
                if len(metrics) > 0:
                    # writer.add_scalars("metrics", metrics, count_steps)
                    for metric, value in metrics.items():
                        writer.add_scalar(f"Metrics/{metric}", value, count_steps)

                writer.add_scalar("Policy/value_loss", value_loss, count_steps)
                writer.add_scalar("Policy/policy_loss", action_loss, count_steps)
                writer.add_scalar("Policy/entropy_loss", dist_entropy, count_steps)
                writer.add_scalar('Policy/learning_rate', lr_scheduler.get_lr()[0], count_steps)

                # log stats
                if update > 0 and update % self.config.LOG_INTERVAL == 0:
                    logger.info(
                        "update: {}\tfps: {:.3f}\t".format(
                            update, count_steps / (time.time() - t_start)
                        )
                    )

                    logger.info(
                        "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                        "frames: {}".format(
                            update, env_time, pth_time, count_steps
                        )
                    )

                    logger.info(
                        "Average window size: {}  {}".format(
                            len(window_episode_stats["count"]),
                            "  ".join(
                                "{}: {:.3f}".format(k, v / deltas["count"])
                                for k, v in deltas.items()
                                if k != "count"
                            ),
                        )
                    )

                # checkpoint model
                if update % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(f"ckpt.{count_checkpoints}.pth")
                    count_checkpoints += 1

            self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0
    ) -> Dict:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """


        #### -----------------------------------------
        # for interactive
        ONLY_GOAL_POLICY = False
        ONLY_VLN_POLICY = False
        ORACLE_WHEN_QUERIED = False
        ORACLE_WITH_STOP = False
        LIMIT_QUERY = True
        HOW_MANY_QUERY = 3
        ENFORCED_GAP = 1
        VIDEO_CHECKING = False


        QS_METHOD = 'ours'  # 'ours' 'random', 'uniform', 'jask'
        if QS_METHOD != 'ours':
            USE_GOAL_BELIEF = True
        else:
            USE_GOAL_BELIEF = True
        if ONLY_GOAL_POLICY:
            USE_GOAL_BELIEF = True

        #### -----------------------------------------
        random_start = 0
        random_end = 30
        random_step = 3

        UNIFORM_STEP_SIZE = 10

        # for dialog training
        TAKE_ORACLE_ACTION = True


        if self.config.DIALOG_TRAINING:
            # to check paiwise disrtribution
            actions = [0,1,2,3] # update if i change action space
            all_seq = list(it.product(actions, repeat=2)) # update if i change num of steps for dialog
            all_seq = [str(i) for i in all_seq]
            all_seq_dict_curr = {i: 0 for i in all_seq}
            all_seq_dict_curr_correct = {i: 0 for i in all_seq}
            all_seq_dict_oracle = {i: 0 for i in all_seq}
        else:
            query_hist = np.zeros(300)
            query_dist = np.zeros(500)
            sound_hist = np.zeros(500)

            sr_hist = np.zeros(10)
            spl_hist = np.zeros(10)
            ratio_count = np.zeros(10)

            success_per_ratio = {}

            # keeping log
            import time
            timestr = time.strftime("%Y%m%d-%H%M%S")
            if not ONLY_GOAL_POLICY:
                test_log_name = "/{}_test_log_{}_{}_seed_{}.txt".format(timestr, QS_METHOD, HOW_MANY_QUERY, self.config.TASK_CONFIG.SEED)
                spr_file_name = '/{}_{}_{}_spr.npz'.format(timestr, QS_METHOD, HOW_MANY_QUERY)
            else:
                test_log_name = "/{}_test_log_only_goal.txt".format(timestr)
                spr_file_name = '/{}_only_goal_spr.npz'.format(timestr)

            if VIDEO_CHECKING:
                test_log_name = "/{}_test_log_video.txt".format(timestr)
                spr_file_name = '/{}_video_spr.npz'.format(timestr)

            spr_file = '/'.join(checkpoint_path.split('/')[:-2]) + spr_file_name
            test_log = open('/'.join(checkpoint_path.split('/')[:-2]) + test_log_name, "w")




        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)

        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG: # true
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        config.defrost()
        self.config.defrost()
        # use the config savi_pretraining_dialog_training.yaml
        if self.config.DIALOG_TRAINING:
            config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
            config.TASK_CONFIG.DATASET.DATA_PATH = 'data/datasets/semantic_audionav_dialog_approx/mp3d/{version}/{split}/{split}.json.gz'
            self.config.NUM_DIALOG_STEPS = config.NUM_DIALOG_STEPS
        else:
            config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
            self.config.RL.PPO.SCENE_MEMORY_TRANSFORMER.pretraining = False
            config.RL.PPO.SCENE_MEMORY_TRANSFORMER.pretraining = False
            # strangely it loads interactive policy type from ckpt!!

        if len(self.config.VIDEO_OPTION) > 0:
            config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = \
                    config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT = 512

        if self.config.DISPLAY_RESOLUTION != config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH:
            model_resolution = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH
            config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = \
                config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT = \
                self.config.DISPLAY_RESOLUTION
        else:
            model_resolution = self.config.DISPLAY_RESOLUTION


        config.freeze()
        self.config.freeze()
        ppo_cfg = config.RL.PPO

        #print(self.config.VISUALIZATION_OPTION, 'self.config.VISUALIZATION_OPTION')
        #sys.exit()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            # edit---------------------------------(facing error for top_down)
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            # config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()
        elif "top_down_map" in self.config.VISUALIZATION_OPTION:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.freeze()

        logger.info(f"env config: {config}")
        # write some of the info in the text file
        # what i need?
        if not self.config.DIALOG_TRAINING:
            # QS_METHOD, heard or unheard or distractor sound, if uniform/random what is the range, enforced gap amount, if only_goal?
            if not ONLY_GOAL_POLICY:
                test_log.write('QS_METHOD: {}\n'.format(QS_METHOD))
            if QS_METHOD == 'random':
                test_log.write('random_start: {}, random_end: {}, random_step: {} \n'.format(random_start, random_end, random_step))
            elif QS_METHOD == 'uniform':
                test_log.write('UNIFORM_STEP_SIZE: {} \n'.format(UNIFORM_STEP_SIZE))
            test_log.write('ONLY_GOAL_POLICY: {}\n'.format(ONLY_GOAL_POLICY))
            test_log.write('With distractor: {}\n'.format(config.TASK_CONFIG.SIMULATOR.AUDIO.HAS_DISTRACTOR_SOUND))
            test_log.write('Sound type: {}\n'.format(self.config.SOUND_TYPE))

            test_log.write('ORACLE_WHEN_QUERIED: {}\n'.format(ORACLE_WHEN_QUERIED))
            test_log.write('ORACLE_WITH_STOP: {}\n'.format(ORACLE_WITH_STOP))
            test_log.write('HOW_MANY_QUERY: {}\n'.format(HOW_MANY_QUERY))
            test_log.write('LIMIT_QUERY: {}\n'.format(LIMIT_QUERY))
            test_log.write('ENFORCED_GAP: {}\n'.format(ENFORCED_GAP))

            test_log.write('USE_GOAL_BELIEF: {}\n'.format(USE_GOAL_BELIEF))

            test_log.write('spr_file_name: {}\n'.format(spr_file_name))
            test_log.write('checkpoint_path: {}\n'.format(checkpoint_path))
            test_log.write('VIDEO_CHECKING: {}\n'.format(VIDEO_CHECKING))
            test_log.write('\n\n')



        self.envs = construct_envs(
            config, get_env_class(config.ENV_NAME)
        )

        if self.config.DISPLAY_RESOLUTION != model_resolution:
            # this condition should not be met
            # and shape attribute cannot be assigned
            observation_space = self.envs.observation_spaces[0]
            observation_space.spaces['depth'].shape = (model_resolution, model_resolution, 1)
            observation_space.spaces['rgb'].shape = (model_resolution, model_resolution, 3)
        else:
            observation_space = self.envs.observation_spaces[0]

        if self.config.DIALOG_TRAINING:
            self._setup_actor_critic_agent(ppo_cfg, observation_space)
            self.agent.load_state_dict(ckpt_dict["state_dict"], strict=False)
            self.actor_critic = self.agent.actor_critic
        else:
            self._setup_actor_critic_agent_interactive(ppo_cfg, observation_space)
            self.agent.load_state_dict(ckpt_dict["state_dict"], strict=False)
            self.actor_critic = self.agent.actor_critic

            # also need to load for vln and goal network
            # let's assign default path
            vln_path = self.config.VLN_CKPT_PATH # 'data/models/savi/data/vln/ckpt.11.pth' # 'data/models/savi1/data/vln/ckpt.6.pth'
            if QS_METHOD!='ours':
                vln_path = "data/pretrained_weights/semantic_audionav/savi/vln/ckpt.29_fix_bp.pth"
            goal_path = self.config.GOAL_CKPT_PATH
            # print('goal_path', goal_path)


            ckpt_dict_vln = self.load_checkpoint(vln_path, map_location="cpu")
            ckpt2load_vln = {}
            for k,v in ckpt_dict_vln["state_dict"].items():
                if k.split('.')[0]=='actor_critic':
                    ckpt2load_vln['.'.join(k.split('.')[1:])] = v
            self.actor_critic_vln.load_state_dict(ckpt2load_vln, strict=False)

            ckpt_dict_goal = self.load_checkpoint(goal_path, map_location="cpu")
            ckpt2load_goal = {}
            for k,v in ckpt_dict_goal["state_dict"].items():
                if k.split('.')[0]=='actor_critic':
                    ckpt2load_goal['.'.join(k.split('.')[1:])] = v

            for k,v in ckpt2load_goal.items():
                if k.split('.')[0] == 'action_distribution':
                    ckpt2load_goal['action_distribution_goal.'+'.'.join(k.split('.')[1:])] = v
                    del ckpt2load_goal[k]
                if k.split('.')[0] == 'critic':
                    ckpt2load_goal['critic_goal.'+'.'.join(k.split('.')[1:])] = v
                    del ckpt2load_goal[k]
            self.actor_critic_goal.load_state_dict(ckpt2load_goal, strict=False)
            total_query_all = 0
            total_stop_taken = 0
            num_episode_debug = 0
            total_query_cycle = 0
            total_query_model = 0
            total_2_query_diff = 0
            total_pair = 0
            error_dict = {'queried_error': 0, 'queried_step': 0, 'goal_error': 0 , 'goal_step': 0}



        if not self.config.DIALOG_TRAINING:
            if self.config.RL.PPO.use_belief_predictor:
                if not USE_GOAL_BELIEF and "belief_predictor" in ckpt_dict:
                    self.belief_predictor.load_state_dict(ckpt_dict["belief_predictor"])
                else:
                    self.belief_predictor.load_state_dict(ckpt_dict_goal["belief_predictor"])
        else:
             self.belief_predictor.load_state_dict(ckpt_dict["belief_predictor"])

        self.metric_uuids = []
        # get name of performance metric, e.g. "spl"
        for metric_name in self.config.TASK_CONFIG.TASK.MEASUREMENTS:
            metric_cfg = getattr(self.config.TASK_CONFIG.TASK, metric_name)
            measure_type = baseline_registry.get_measure(metric_cfg.TYPE)
            assert measure_type is not None, "invalid measurement type {}".format(
                metric_cfg.TYPE
            )
            self.metric_uuids.append(measure_type(sim=None, task=None, config=None)._get_uuid())
            # self.metric_uuids ['distance_to_goal', 'normalized_distance_to_goal', 'success', 'spl', 'softspl', 'na', 'sna', 'sws']


        observations = self.envs.reset()
        if config.DISPLAY_RESOLUTION != model_resolution:
            obs_copy = resize_observation(observations, model_resolution)
        else:
            obs_copy = observations
        batch = batch_obs(obs_copy, self.device, skip_list=['view_point_goals', 'intermediate',
                                                            'oracle_action_sensor'])

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        if self.actor_critic.net.num_recurrent_layers == -1:
            num_recurrent_layers = 1
        else:
            num_recurrent_layers = self.actor_critic.net.num_recurrent_layers

        test_recurrent_hidden_states = torch.zeros(
            num_recurrent_layers,
            self.config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )


        if self.config.DIALOG_TRAINING:
            if ppo_cfg.use_external_memory:
                test_em_vln = ExternalMemory(
                    self.config.NUM_PROCESSES,
                    self.config.NUM_DIALOG_STEPS,
                    self.config.NUM_DIALOG_STEPS,
                    self.actor_critic.net.memory_dim,
                )
                test_em_vln.to(self.device)
            else:
                test_em_vln = None

            if ppo_cfg.use_state_memory:
                test_em_vln_dialog = ExternalMemory(
                    self.config.NUM_PROCESSES,
                    self.config.NUM_DIALOG_STEPS,
                    self.config.NUM_DIALOG_STEPS,
                    self.actor_critic.net._hidden_size,
                )
                test_em_vln_dialog.to(self.device)
            else:
                test_em_vln_dialog = None
        else:
            memoey_dim_goal = self.actor_critic_goal.net.memory_dim # 276# self.actor_critic_option.net.memory_dim-32
            memoey_dim_vln = self.actor_critic_vln.net.memory_dim
            memory_dim_option = self.actor_critic_option.net.memory_dim

            if ppo_cfg.use_external_memory:
                test_em_goal = ExternalMemory(
                    self.config.NUM_PROCESSES,
                    ppo_cfg.SCENE_MEMORY_TRANSFORMER.memory_size,
                    ppo_cfg.SCENE_MEMORY_TRANSFORMER.memory_size,
                    memoey_dim_goal,#self.actor_critic.net.memory_dim,
                )
                test_em_goal.to(self.device)

                test_em_option = ExternalMemory(
                    self.config.NUM_PROCESSES,
                    ppo_cfg.SCENE_MEMORY_TRANSFORMER.memory_size,
                    ppo_cfg.SCENE_MEMORY_TRANSFORMER.memory_size,
                    memory_dim_option,#self.actor_critic.net.memory_dim,
                )
                test_em_option.to(self.device)

                test_em_vln = ExternalMemory(
                    self.config.NUM_PROCESSES,
                    self.config.NUM_DIALOG_STEPS,
                    self.config.NUM_DIALOG_STEPS,
                    memoey_dim_vln, #self.actor_critic.net.memory_dim,
                )
                test_em_vln.to(self.device)
            else:
                test_em = None
                test_em_option = None
                test_em_vln = None

            if ppo_cfg.use_state_memory:
                test_em_vln_dialog = ExternalMemory(
                    self.config.NUM_PROCESSES,
                    self.config.NUM_DIALOG_STEPS,
                    self.config.NUM_DIALOG_STEPS,
                    self.actor_critic_vln.net._hidden_size,
                )
                test_em_vln_dialog.to(self.device)
            else:
                test_em_vln_dialog = None

        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )

        # similar to not done mask, initialize track_perf to track performance of each query
        track_perf = [{'is_correct': True} for idx in range(self.config.NUM_PROCESSES)]
        track_agent_step = [{'dialog': [], 'step':0, 'is_correct': [], 'cur_step':[], 'cur_oracle_step':[]} for idx in range(self.config.NUM_PROCESSES)]

        stats_episodes = dict()  # dict of dicts that stores stats per episode

        if self.config.RL.PPO.use_belief_predictor:
            self.belief_predictor.update(batch, None)

            descriptor_pred_gt = [[] for _ in range(self.config.NUM_PROCESSES)]
            for i in range(len(descriptor_pred_gt)):
                category_prediction = np.argmax(batch['category_belief'].cpu().numpy()[i])
                location_prediction = batch['location_belief'].cpu().numpy()[i]
                category_gt = np.argmax(batch['category'].cpu().numpy()[i])
                location_gt = batch['pointgoal_with_gps_compass'].cpu().numpy()[i]
                geodesic_distance = -1
                pair = (category_prediction, location_prediction, category_gt, location_gt, geodesic_distance)
                if 'view_point_goals' in observations[i]:
                    pair += (observations[i]['view_point_goals'],)
                descriptor_pred_gt[i].append(pair)

        actual_step = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]
        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]
        text_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]
        audios = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]
        all_sub_instrs = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        if self.config.DIALOG_TRAINING:
            self.actor_critic.eval()
        else:
            self.actor_critic.eval()
            self.actor_critic_goal.eval()
            self.actor_critic_vln.eval()

        if self.config.RL.PPO.use_belief_predictor:
            self.belief_predictor.eval()
        t = tqdm(total=self.config.TEST_EPISODE_COUNT)

        track_query = [{'dialog': [], 'step':0, 'queried': False, 'epi_step_num': 0, 'instr': ''} for idx in range(self.config.NUM_PROCESSES)]
        track_query_count = [0 for idx in range(self.config.NUM_PROCESSES)]
        track_episode = [{'name': ' ', 'spl': 0.0, 'step': 0, 'random_step': [], 'uniform_step': [], 'all_step': [], 'query_step_idx': [], 'query_step_distance': [], 'query_ratio': 0, 'prev_actions':[], 'vln_step_status':[], 'last_query_step': 0, 'all_entropy':[], 'success_status': False, 'instr': [], 'action_taken':[] } for idx in range(self.config.NUM_PROCESSES)]
        tq_flag = 0
        all_2_query_diff = []


        if self.config.DIALOG_TRAINING:
            if SPEAKER:
                logger.info('Using speaker to generate dialog')
            vln_result = 0
            vln_cnt = 0
            vln_result_sep = [0 for i in range(5)]
            vln_cnt_sep = [0 for i in range(5)]
            vln_cnt_pred_sep = [0 for i in range(5)]

            vln_result_seq = 0
            vln_cnt_seq = 0

            vln_result_seq_sep = [0 for i in range(10)]
            vln_cnt_seq_sep = [0 for i in range(10)]

            # number of oracle action in each step
            vln_oracle_action_stepwise = {k:[0]*4 for k in range(5)}
            vln_current_action_stepwise = {k:[0]*4 for k in range(5)}
            vln_accuracy_stepwise = {k:[0]*4 for k in range(5)}



        while (
            len(stats_episodes) < self.config.TEST_EPISODE_COUNT
            and self.envs.num_envs > 0
        ):

            current_episodes = self.envs.current_episodes()


            state_all = self.envs.agent_state()
            curr_rotations = [state[1] for state in state_all]
            scenes = [state[2] for state in state_all]
            views = [state[4] for state in state_all]
            app_points = [state[5] for state in state_all]
            new_episode = self.envs.is_new_episode()
            # dialog pretraining
            sub_instrs = [state[6] for state in state_all]
            current_target_distance = [state[7] for state in state_all]

            for i in range(self.envs.num_envs):
                assert not_done_masks[i] == 1-new_episode[i], 'ndm: {}, ne: {}'.format(not_done_masks[i] ,new_episode[i])

            current_dialog = torch.zeros(self.envs.num_envs, self.max_dialog_len, dtype=torch.long).to(self.device)
            current_query_state = torch.zeros(self.envs.num_envs, self.config.QUERY_COUNT_EMB_SIZE).to(self.device)
            last_query_info = torch.zeros(self.envs.num_envs, self.config.QUERY_COUNT_EMB_SIZE).to(self.device)
            current_agent_step = torch.zeros(self.envs.num_envs, 1).to(self.device)


            if self.config.DIALOG_TRAINING:
                # right now sequential, need to update it to batchwise dialog generation
                o_actions = self.envs.compute_oracle_actions()
                oracle_actions = [a[0] for a in o_actions]

                for idx in range(self.envs.num_envs):
                    if SPEAKER:
                        if not app_points[idx]:
                            self.invalid_point_count += 1
                            continue

                        heading = self._quat_to_xy_heading(curr_rotations[idx])
                        gt_app_path = [app_points[idx][0]]
                        for gt_vln_node in app_points[idx][1:]:
                            if gt_vln_node not in gt_app_path and len(gt_app_path)<3:
                                gt_app_path.append(gt_vln_node)

                        # from GT trajectory, get instruction
                        speaker_entry = {
                                         'heading': heading,
                                         'scene': scenes[idx],
                                         'path': gt_app_path
                                        }
                        dialog = ' '.join(self.speaker.generate_instr(speaker_entry)[0]['words'])
                    else:
                        dialog = sub_instrs[idx]

                    # using clip
                    tokenized_dialog = clip.tokenize(dialog)[0]
                    current_dialog[idx,:tokenized_dialog.shape[0]] = tokenized_dialog
                    if new_episode[idx]:
                        track_agent_step[idx]['step'] = 0
                        track_agent_step[idx]['is_correct'] = []
                        track_agent_step[idx]['cur_step'] = []
                        track_agent_step[idx]['cur_oracle_step'] = []
                    else:
                        if track_agent_step[idx]['step']<self.config.NUM_DIALOG_STEPS-1:
                            track_agent_step[idx]['step'] += 1
                    current_agent_step[idx].copy_(torch.tensor([track_agent_step[idx]['step']]).to(self.device))
                    assert track_agent_step[idx]['step']< self.config.NUM_DIALOG_STEPS, 'step number crosses allowed steps'

            else:

                for idx in range(self.envs.num_envs):
                    track_query[idx]['epi_step_num'] += 1

                    if new_episode[idx]:

                        # writing on a text file
                        if track_episode[idx]['step']> 0:
                            test_log.write('###################\n')
                            test_log.write('name:\n')
                            test_log.write(track_episode[idx]['name'])
                            test_log.write('\n')
                            test_log.write('all steps:\n')
                            test_log.write(' '.join(track_episode[idx]['all_step']))
                            test_log.write('\n')
                            test_log.write('action taken:\n')
                            test_log.write(' '.join([str(aa) for aa in track_episode[idx]['action_taken']]))
                            test_log.write('\n')
                            test_log.write('query step index:\n')
                            test_log.write(' '.join([str(aa) for aa in track_episode[idx]['query_step_idx']]))
                            test_log.write('\n')
                            test_log.write('query step distance:\n')
                            test_log.write(' '.join([str(round(aa,2)) for aa in track_episode[idx]['query_step_distance']]))
                            test_log.write('\n')
                            test_log.write('instr:\n')
                            test_log.write(' ; '.join(track_episode[idx]['instr']))
                            test_log.write('\n')
                            test_log.write('vln_step_status:\n')
                            test_log.write(' '.join(track_episode[idx]['vln_step_status']))
                            test_log.write('\n')
                            test_log.write('query ratio:\n')
                            test_log.write(str( len(track_episode[idx]['query_step_idx']) / len(track_episode[idx]['all_step']) ))
                            test_log.write('\n')
                            test_log.write('success status:\n')
                            test_log.write(str(  track_episode[idx]['success_status']))
                            test_log.write('\n')
                            test_log.write('spl:\n')
                            test_log.write(str(round(track_episode[idx]['spl'],2)))
                            test_log.write('\n')
                            test_log.write('all entropy:\n')
                            test_log.write(' ; '.join([' '.join([aa[0], str(round(aa[1],2)), aa[2]]) for aa in track_episode[idx]['all_entropy']]))
                            test_log.write('\n')
                            test_log.write('\n')



                        track_episode[idx]['step'] = 0
                        track_episode[idx]['name'] = ' '
                        track_episode[idx]['spl'] = 0.0
                        track_episode[idx]['random_step'] = random.sample(range(random_start,random_end,random_step),HOW_MANY_QUERY)  #(0,30,3)
                        track_episode[idx]['uniform_step'] = [au*UNIFORM_STEP_SIZE for au in range(HOW_MANY_QUERY)]
                        track_episode[idx]['all_step'] = []
                        track_episode[idx]['query_step_idx'] = []
                        track_episode[idx]['query_step_distance'] = []
                        track_episode[idx]['vln_step_status'] = []
                        track_episode[idx]['action_taken'] = []

                        track_episode[idx]['query_ratio'] = 0
                        track_episode[idx]['prev_actions'] = []
                        track_episode[idx]['last_query_step'] = 0
                        track_episode[idx]['all_entropy'] = []
                        track_episode[idx]['instr'] = []
                        track_episode[idx]['success_status'] = False

                        query_hist[track_query_count[idx]] += 1
                        track_query[idx]['queried'] = False
                        track_query[idx]['step'] = 0
                        track_query[idx]['dialog'] = []
                        track_query[idx]['instr'] = ''
                        track_query_count[idx] = 0
                        track_query[idx]['epi_step_num'] = 0


                        num_episode_debug += 1
                        diff_step = 150
                    else:
                        track_episode[idx]['step'] += 1
                        if track_query_count[idx]>=2:
                            diff_step = track_episode[idx]['step']-track_episode[idx]['last_query_step']
                        else:
                            diff_step = 150

                    with torch.no_grad():
                        current_query_state[idx,:].copy_(self.pe[track_query_count[idx],:])
                        last_query_info[idx,:].copy_(self.pe[diff_step,:])


            with torch.no_grad():
                if self.config.DIALOG_TRAINING:
                    _, actions, _, test_recurrent_hidden_states, test_em_vln_features, test_em_vln_dialog_features, d_prob = self.actor_critic.act_dialog(
                        batch,
                        test_recurrent_hidden_states,
                        prev_actions,
                        not_done_masks,
                        test_em_vln.memory[:, 0] if ppo_cfg.use_external_memory else None,
                        test_em_vln_dialog.memory[:, 0] if ppo_cfg.use_state_memory else None,
                        test_em_vln.masks if ppo_cfg.use_external_memory else None,
                        current_dialog,
                        current_agent_step.view(-1),
                        deterministic=True,
                        without_dialog=self.config.DIALOG_TRAINING_WITHOUT_DIALOG,
                    )
                    oracle_actions_t = torch.Tensor(oracle_actions).to(self.device)
                    prev_actions.copy_(torch.unsqueeze(oracle_actions_t,1))
                else:
                    (
                        _,
                        unct,
                        actions_option,
                        _,
                        test_recurrent_hidden_states,
                        test_em_option_features,
                        action_prob,
                    ) = self.actor_critic.act_option(
                        batch,
                        test_recurrent_hidden_states,
                        prev_actions,
                        not_done_masks,
                        test_em_option.memory[:, 0] if ppo_cfg.use_external_memory else None,
                        test_em_option.masks if ppo_cfg.use_external_memory else None,
                        current_query_state,
                        last_query_info,
                        deterministic=False,
                    )         # option_prob: (b,2), 0 : goal based, 1: vln



                    # doing audio goal navigation early for jask method
                    (
                        _,
                        actions_goal,
                        _,
                        _,
                        test_em_goal_features,
                        goal_action_prob,
                    ) = self.actor_critic_goal.act(
                        batch,
                        test_recurrent_hidden_states,
                        prev_actions,
                        not_done_masks,
                        test_em_goal.memory[:, 0] if ppo_cfg.use_external_memory else None,
                        test_em_goal.masks if ppo_cfg.use_external_memory else None,
                        deterministic=False,
                    )
                    goal_action_prob_npy = np.sort(goal_action_prob.cpu().numpy())

                    for idx, _ in enumerate(new_episode):
                        if QS_METHOD=='ours':

                            # prob = ((1-goal_action_prob_npy[idx][3]+goal_action_prob_npy[idx][2])*5) + action_prob[idx][1]/2
                            # jask_sel = goal_action_prob_npy[idx][3]-goal_action_prob_npy[idx][2]<.1

                            # if not track_query[idx]['queried'] and (action_prob[idx][0]< action_prob[idx][1] or jask_sel) : #prob >=.5: #
                            if not track_query[idx]['queried'] and action_prob[idx][0]< action_prob[idx][1]:
                                # if action_prob[idx][0]< action_prob[idx][1] and track_query_count[idx]<HOW_MANY_QUERY:
                                #    total_query_model += 1

                                if 'Q' not in track_episode[idx]['all_step'][-min(len(track_episode[idx]['all_step']),ENFORCED_GAP):]:

                                    if self.config.QUERY_WITHIN_RADIUS:
                                        track_query[idx]['queried'] = True
                                        if LIMIT_QUERY and track_query_count[idx]>=HOW_MANY_QUERY:
                                            track_query[idx]['queried'] = False

                                    else:
                                        if current_target_distance[idx]>3:
                                            track_query[idx]['queried'] = True
                                            if LIMIT_QUERY and track_query_count[idx]>=HOW_MANY_QUERY:
                                                track_query[idx]['queried'] = False


                        elif QS_METHOD=='pred_unct':

                            if not track_query[idx]['queried'] and unct[idx][0] < unct[idx][1]:
                                # if action_prob[idx][0]< action_prob[idx][1] and track_query_count[idx]<HOW_MANY_QUERY:
                                #    total_query_model += 1

                                if 'Q' not in track_episode[idx]['all_step'][-min(len(track_episode[idx]['all_step']),ENFORCED_GAP):]:

                                    if self.config.QUERY_WITHIN_RADIUS:
                                        track_query[idx]['queried'] = True
                                        if LIMIT_QUERY and track_query_count[idx]>=HOW_MANY_QUERY:
                                            track_query[idx]['queried'] = False

                                    else:
                                        if current_target_distance[idx]>3:
                                            track_query[idx]['queried'] = True
                                            if LIMIT_QUERY and track_query_count[idx]>=HOW_MANY_QUERY:
                                                track_query[idx]['queried'] = False


                        elif QS_METHOD=='random':
                            if not track_query[idx]['queried'] and track_episode[idx]['step'] in track_episode[idx]['random_step']:
                                if self.config.QUERY_WITHIN_RADIUS:
                                    track_query[idx]['queried'] = True
                                    if LIMIT_QUERY and track_query_count[idx]>=HOW_MANY_QUERY:
                                        track_query[idx]['queried'] = False

                                else:
                                    if current_target_distance[idx]>3:
                                        track_query[idx]['queried'] = True
                                        if LIMIT_QUERY and track_query_count[idx]>=HOW_MANY_QUERY:
                                            track_query[idx]['queried'] = False


                        elif QS_METHOD=='uniform':
                            if not track_query[idx]['queried'] and track_episode[idx]['step'] in track_episode[idx]['uniform_step']:
                                if self.config.QUERY_WITHIN_RADIUS:
                                    track_query[idx]['queried'] = True
                                    if LIMIT_QUERY and track_query_count[idx]>=HOW_MANY_QUERY:
                                        track_query[idx]['queried'] = False

                                else:
                                    if current_target_distance[idx]>3:
                                        track_query[idx]['queried'] = True
                                        if LIMIT_QUERY and track_query_count[idx]>=HOW_MANY_QUERY:
                                            track_query[idx]['queried'] = False

                        elif QS_METHOD=='jask':

                            # print('diff', goal_action_prob_npy[idx][3]-goal_action_prob_npy[idx][2])
                            if not track_query[idx]['queried'] and goal_action_prob_npy[idx][3]-goal_action_prob_npy[idx][2]<0.1:
                                if self.config.QUERY_WITHIN_RADIUS:
                                    track_query[idx]['queried'] = True
                                    if LIMIT_QUERY and track_query_count[idx]>=HOW_MANY_QUERY:
                                        track_query[idx]['queried'] = False

                                else:
                                    if current_target_distance[idx]>3:
                                        track_query[idx]['queried'] = True
                                        if LIMIT_QUERY and track_query_count[idx]>=HOW_MANY_QUERY:
                                            track_query[idx]['queried'] = False


                    o_actions = self.envs.compute_oracle_actions()
                    oracle_actions = [a[0] for a in o_actions]

                    for idx in range(self.envs.num_envs):


                        if track_query[idx]['queried']:
                            if track_query[idx]['step']==0:
                                track_query_count[idx] += 1
                                total_query_all += 1
                                track_episode[idx]['last_query_step'] = track_episode[idx]['step']

                                track_episode[idx]['query_step_idx'].append(track_episode[idx]['step'])
                                track_episode[idx]['query_step_distance'].append(current_target_distance[idx])
                                query_dist[track_query[idx]['epi_step_num']]+=1

                                if oracle_actions[idx]!=0:

                                    if not app_points[idx]:
                                        # self.invalid_point_cnt += 1
                                        continue


                                    heading = self._quat_to_xy_heading(curr_rotations[idx])
                                    # get the ground truth approximate path
                                    gt_app_path = [app_points[idx][0]]
                                    for gt_vln_node in app_points[idx][1:]:
                                        if gt_vln_node not in gt_app_path and len(gt_app_path)<3:
                                            gt_app_path.append(gt_vln_node)

                                    # from GT trajectory, get instruction
                                    speaker_entry = {
                                                     'heading': heading,
                                                     'scene': scenes[idx],
                                                     'path': gt_app_path
                                                     }
                                    # change ----------------------------
                                    if SPEAKER:
                                        dialog = ' '.join(self.speaker.generate_instr(speaker_entry)[0]['words'])
                                    else:
                                        dialog = ''
                                else:
                                    dialog = 'stop here'


                                    # using clip
                                tokenized_dialog = clip.tokenize(dialog)[0]
                                track_query[idx]['dialog'] = tokenized_dialog
                                track_query[idx]['instr'] = dialog
                                track_episode[idx]['all_step'].append("Q")
                                track_episode[idx]['instr'].append(dialog)

                                # checking
                                if actions_goal[idx].item() != oracle_actions[idx]:
                                    error_dict['queried_error'] += 1
                                    curr_goal_step = 'W'
                                else:
                                    curr_goal_step = 'R'
                                error_dict['queried_step'] += 1

                            else:
                                track_episode[idx]['all_step'].append("V")
                                if actions_goal[idx].item() != oracle_actions[idx]:
                                    curr_goal_step = 'W'
                                else:
                                    curr_goal_step = 'R'

                            if track_query[idx]['step']<self.config.NUM_DIALOG_STEPS:
                                current_dialog[idx,:track_query[idx]['dialog'].shape[0]] = track_query[idx]['dialog']
                                current_agent_step[idx].copy_(torch.tensor([track_query[idx]['step']]))
                                track_query[idx]['step'] += 1
                        else:
                            track_episode[idx]['all_step'].append("G")


                            # checking
                            if actions_goal[idx].item() != oracle_actions[idx]:
                                error_dict['goal_error'] += 1
                                curr_goal_step = 'W'
                            else:
                                curr_goal_step = 'R'
                            error_dict['goal_step'] += 1

                        curr_entropy = scipy.stats.entropy(goal_action_prob_npy[idx], base=2)
                        track_episode[idx]['all_entropy'].append((track_episode[idx]['all_step'][-1], curr_entropy, curr_goal_step ))


                    (                            _,
                        actions_vln,
                        _,
                        _,
                        test_em_vln_features,
                        test_em_vln_dialog_features,
                        _,
                    ) = self.actor_critic_vln.act_dialog(
                        batch,
                        test_recurrent_hidden_states,
                        prev_actions,
                        not_done_masks,
                        test_em_vln.memory[:, 0] if ppo_cfg.use_external_memory else None,
                        test_em_vln_dialog.memory[:, 0] if ppo_cfg.use_state_memory else None,
                        test_em_vln.masks if ppo_cfg.use_external_memory else None,
                        current_dialog,
                        current_agent_step.view(-1),
                        deterministic=True,
                        without_dialog=self.config.DIALOG_TRAINING_WITHOUT_DIALOG,
                    )



                    oracle_actions_t = torch.Tensor(oracle_actions).to(self.device)

                    if QS_METHOD=='ours':
                        if not ONLY_GOAL_POLICY and not ONLY_VLN_POLICY:
                            # create storage tensors
                            # sort out all the element based on selection
                            actions = []
                            for idx, _ in enumerate(new_episode):

                                if track_query[idx]['queried']:

                                    if oracle_actions_t[idx]==actions_vln[idx]:
                                        track_episode[idx]['vln_step_status'].append('C')
                                    else:
                                        track_episode[idx]['vln_step_status'].append('W')

                                    if not ORACLE_WHEN_QUERIED:
                                        if track_query[idx]['instr'] != 'stop here':
                                            actions.append(actions_vln[idx])
                                        else:
                                            actions.append(torch.tensor([oracle_actions[idx]],dtype=torch.long).to(self.device))
                                            total_stop_taken += 1

                                    else:
                                        if oracle_actions_t[idx]==0:
                                            if ORACLE_WITH_STOP:
                                                actions.append(oracle_actions_t[idx])
                                            else:
                                                actions.append(actions_vln[idx])

                                        else:
                                            actions.append(oracle_actions_t[idx])

                                else:
                                    actions.append(actions_goal[idx])

                                if len(track_episode[idx]['prev_actions'])==6:
                                    track_episode[idx]['prev_actions'] = track_episode[idx]['prev_actions'][1:]
                                track_episode[idx]['prev_actions'].append(actions[-1].item())

                            if not ORACLE_WHEN_QUERIED:
                                actions = torch.stack(actions, dim=0)
                            else:
                                # actions = torch.stack(actions, dim=0)
                                actions = torch.Tensor(actions).to(self.device)
                                actions = torch.unsqueeze(actions,1)

                        elif  ONLY_GOAL_POLICY:
                            actions = actions_goal

                        elif  ONLY_VLN_POLICY:
                            if not ORACLE_WHEN_QUERIED:
                                actions = actions_vln
                            else:
                                actions = []

                                for idx, _ in enumerate(new_episode):
                                    if oracle_actions_t[idx]==0:
                                        if ORACLE_WITH_STOP:
                                            actions.append(oracle_actions_t[idx])
                                        else:
                                            actions.append(actions_vln[idx])
                                actions = torch.Tensor(actions).to(self.device)
                                actions = torch.unsqueeze(actions,1)


                    else:
                        actions = []
                        for idx, _ in enumerate(new_episode):

                            if track_query[idx]['queried']:
                                if not ORACLE_WHEN_QUERIED:
                                    if track_query[idx]['instr'] != 'stop here':
                                        actions.append(actions_vln[idx])
                                    else:
                                        actions.append(torch.tensor([oracle_actions[idx]],dtype=torch.long).to(self.device))
                                        total_stop_taken += 1

                                else:
                                    if oracle_actions_t[idx]==0:
                                        if ORACLE_WITH_STOP:
                                            actions.append(oracle_actions_t[idx])
                                        else:
                                            actions.append(actions_vln[idx])

                                    else:
                                        actions.append(oracle_actions_t[idx])



                            else:
                                actions.append(actions_goal[idx])


                        if not ORACLE_WHEN_QUERIED:
                            actions = torch.stack(actions, dim=0)
                        else:
                            actions = torch.Tensor(actions).to(self.device)
                            actions = torch.unsqueeze(actions,1)

                    prev_actions.copy_(actions)

                    for idx in range(self.envs.num_envs):

                        if track_query[idx]['step']>= self.config.NUM_DIALOG_STEPS:
                            track_query[idx]['queried'] = False
                            track_query[idx]['step'] = 0
                            track_query[idx]['dialog'] = []
                            track_query[idx]['instr'] = ''



            curr_actions = [int(a[0].item()) for a in actions]
            for idx, a in enumerate(curr_actions):
                track_episode[idx]['action_taken'].append(a)

            if self.config.DIALOG_TRAINING:
                for idx, a in enumerate(curr_actions):
                   # assign predicted & oracle action
                   track_agent_step[idx]['cur_step'].append(a)
                   track_agent_step[idx]['cur_oracle_step'].append(oracle_actions[idx])

                   vln_cnt += 1
                   vln_cnt_sep[oracle_actions[idx]] += 1
                   vln_cnt_pred_sep[curr_actions[idx]] += 1

                   if a == oracle_actions[idx]:
                        vln_result += 1
                        vln_result_sep[oracle_actions[idx]] += 1
                        track_agent_step[idx]['is_correct'].append(True)
                   else:
                        track_perf[idx]['is_correct'] = False
                        track_agent_step[idx]['is_correct'].append(False)

                   vln_oracle_action_stepwise[track_agent_step[idx]['step']][oracle_actions[idx]] += 1
                   vln_current_action_stepwise[track_agent_step[idx]['step']][curr_actions[idx]] += 1
                   if oracle_actions[idx]==a:
                       vln_accuracy_stepwise[track_agent_step[idx]['step']][oracle_actions[idx]] += 1

            if self.config.DIALOG_TRAINING:
                if TAKE_ORACLE_ACTION:
                    outputs = self.envs.step(oracle_actions)
                else:
                    outputs = self.envs.step(curr_actions)
                action_taken = curr_actions
            else:
                outputs = self.envs.step(curr_actions)
                # outputs = self.envs.step([int(all_action[track_episode[0]['step']])])

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]





            if config.DISPLAY_RESOLUTION != model_resolution:
                obs_copy = resize_observation(observations, model_resolution)
            else:
                obs_copy = observations
            batch = batch_obs(obs_copy, self.device, skip_list=['view_point_goals', 'intermediate',
                                                                'oracle_action_sensor'])

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            if self.config.DIALOG_TRAINING:
                for idx, status in enumerate(not_done_masks):
                    if status[0]==0.0:
                        vln_cnt_seq += 1
                        if track_perf[idx]['is_correct'] == True:
                            vln_result_seq += 1
                        else:
                            track_perf[idx]['is_correct'] = True

                        curr_state = True
                        for idx2, state in enumerate(track_agent_step[idx]['is_correct']):
                            if idx2 < 10:
                               vln_cnt_seq_sep[idx2] += 1
                               if not state:
                                   curr_state = False
                               if curr_state:
                                   vln_result_seq_sep[idx2] += 1

                        # for pair distribution
                        for i in range(len(track_agent_step[idx]['cur_step'])-1):
                            all_seq_dict_curr[str(tuple(track_agent_step[idx]['cur_step'][i:i+2]))] += 1
                            all_seq_dict_oracle[str(tuple(track_agent_step[idx]['cur_oracle_step'][i:i+2]))] += 1
                            if str(tuple(track_agent_step[idx]['cur_step'][i:i+2])) == str(tuple(track_agent_step[idx]['cur_oracle_step'][i:i+2])):
                                all_seq_dict_curr_correct[str(tuple(track_agent_step[idx]['cur_step'][i:i+2]))] += 1

            # Update external memory
            if ppo_cfg.use_external_memory:
                if self.config.DIALOG_TRAINING:
                    test_em_vln.insert(test_em_vln_features, not_done_masks)
                else:

                    test_em_goal.insert(test_em_goal_features, not_done_masks)
                    test_em_option.insert(test_em_option_features, not_done_masks)
                    test_em_vln.insert(test_em_vln_features, not_done_masks)
            if ppo_cfg.use_state_memory:
                test_em_vln_dialog.insert(test_em_vln_dialog_features, not_done_masks)

            if self.config.RL.PPO.use_belief_predictor:
                self.belief_predictor.update(batch, dones)

                for i in range(len(descriptor_pred_gt)):
                    category_prediction = np.argmax(batch['category_belief'].cpu().numpy()[i])
                    location_prediction = batch['location_belief'].cpu().numpy()[i]
                    category_gt = np.argmax(batch['category'].cpu().numpy()[i])
                    location_gt = batch['pointgoal_with_gps_compass'].cpu().numpy()[i]
                    if dones[i]:
                        geodesic_distance = -1
                    else:
                        geodesic_distance = infos[i]['distance_to_goal']
                    pair = (category_prediction, location_prediction, category_gt, location_gt, geodesic_distance)
                    if 'view_point_goals' in observations[i]:
                        pair += (observations[i]['view_point_goals'],)
                    descriptor_pred_gt[i].append(pair)

            if len(self.config.VIDEO_OPTION) > 0:
                for i in range(self.envs.num_envs):
                    scene = current_episodes[i].scene_id.split('/')[3]
                    sound = current_episodes[i].sound_id.split('/')[1][:-4]
                    curr_epi_name = f"{config.EVAL.SPLIT}_{scene}_{current_episodes[i].episode_id}_{sound}"

                    if self.config.RL.PPO.use_belief_predictor:
                        pred = descriptor_pred_gt[i][-1]
                    else:
                        pred = None
                    curr_text = None
                    if not self.config.DIALOG_TRAINING:

                        if len(track_query[i]['instr'])>0:
                            curr_text = 'Query: '+ track_query[i]['instr']
                        '''
                        # print('track_episode[i]', track_episode[i]['step'])
                        # if str(track_episode[i]['step']) in qs_idx:
                        for idx_qs,qs_idx_curr in enumerate(qs_idx):
                            if qs_idx_curr==str(track_episode[i]['step']):
                                # print('track_episode[i]',track_episode[i]['step'])
                                curr_text = 'Instruction: '+ q_instr[idx_qs]
                                # print(curr_text)
                        '''

                    else:
                        curr_text = sub_instrs[i] + '\n'+ 'cur_oracle_step: ' + str(track_agent_step[i]['cur_oracle_step'])\
                                                 + '\n'+ 'cur_predicted_step: ' + str(track_agent_step[i]['cur_step'])
                    if config.TASK_CONFIG.SIMULATOR.CONTINUOUS_VIEW_CHANGE and 'intermediate' in observations[i]:
                        for observation in observations[i]['intermediate']:
                            frame = observations_to_image(observation, infos[i], pred=pred)
                            rgb_frames[i].append(frame)
                            text_frames[i].append(curr_text)

                        del observations[i]['intermediate']
                    if "rgb" not in observations[i]:
                        observations[i]["rgb"] = np.zeros((self.config.DISPLAY_RESOLUTION,
                                                           self.config.DISPLAY_RESOLUTION, 3))
                    frame = observations_to_image(observations[i], infos[i], pred=pred)
                    rgb_frames[i].append(frame)
                    text_frames[i].append(curr_text)
                    audios[i].append(observations[i]['audiogoal'])


            rewards = torch.tensor(
                rewards, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            for i in range(self.envs.num_envs):
                # pause envs which runs out of episodes
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not_done_masks[i].item() == 0:
                    episode_stats = dict()

                    scene = current_episodes[i].scene_id.split('/')[3]
                    sound = current_episodes[i].sound_id.split('/')[1][:-4]
                    curr_epi_name = f"{config.EVAL.SPLIT}_{scene}_{current_episodes[i].episode_id}_{sound}"
                    track_episode[i]['name'] = f"{config.EVAL.SPLIT}_{scene}_{current_episodes[i].episode_id}_{sound}"

                    for metric_uuid in self.metric_uuids:
                        episode_stats[metric_uuid] = infos[i][metric_uuid]
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats['geodesic_distance'] = current_episodes[i].info['geodesic_distance']
                    episode_stats['euclidean_distance'] = norm(np.array(current_episodes[i].goals[0].position) -
                                                               np.array(current_episodes[i].start_position))
                    episode_stats['audio_duration'] = int(current_episodes[i].duration)
                    episode_stats['gt_na'] = int(current_episodes[i].info['num_action'])

                    if not self.config.DIALOG_TRAINING:
                        # success_per_ratio
                        ss_ratio = int(4*episode_stats['gt_na']/episode_stats['audio_duration'])
                        if ss_ratio not in success_per_ratio:
                            success_per_ratio[ss_ratio] = 0
                        success_per_ratio[ss_ratio] += episode_stats['success']

                    if episode_stats['success']>0:
                        track_episode[i]['success_status'] = True
                    track_episode[i]['spl'] = infos[i]['spl']

                    if len(track_episode[i]['query_step_idx'])>=2:
                        for j in range(1,len(track_episode[i]['query_step_idx'])):
                            all_2_query_diff.append(track_episode[i]['query_step_idx'][j]-track_episode[i]['query_step_idx'][j-1])
                            total_2_query_diff += (track_episode[i]['query_step_idx'][j]-track_episode[i]['query_step_idx'][j-1])
                            total_pair += 1


                    if self.config.RL.PPO.use_belief_predictor:
                        episode_stats['gt_na'] = int(current_episodes[i].info['num_action'])
                        episode_stats['descriptor_pred_gt'] = descriptor_pred_gt[i][:-1]
                        descriptor_pred_gt[i] = [descriptor_pred_gt[i][-1]]
                    logging.debug(episode_stats)
                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    t.update()
                    if not self.config.DIALOG_TRAINING and tq_flag%20 == 0:
                        print('total_query_all', total_query_all, 'total_stop_taken', total_stop_taken, 'total_query_cycle', total_query_cycle)
                    tq_flag += 1


                    if len(self.config.VIDEO_OPTION) > 0:

                        fps = self.config.TASK_CONFIG.SIMULATOR.VIEW_CHANGE_FPS \
                                    if self.config.TASK_CONFIG.SIMULATOR.CONTINUOUS_VIEW_CHANGE else 1
                        if 'sound' in current_episodes[i].info:
                            sound = current_episodes[i].info['sound']
                        else:
                            sound = current_episodes[i].sound_id.split('/')[1][:-4]
                        # print(len(rgb_frames[i]), 'len(rgb_frames[i]') # 5
                        #text = None
                        #if self.config.DIALOG_TRAINING:
                        #    text = sub_instrs[i] + '\n'+ 'cur_oracle_step: ' + str(track_agent_step[i]['cur_oracle_step'])\
                        #                         + '\n'+ 'cur_predicted_step: ' + str(track_agent_step[i]['cur_step'])\
                        #                         + '\n'+ 'actual_step: ' + " ".join(str(x) for x in actual_step[i][:-1])
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i], #[:-1],
                            scene_name=current_episodes[i].scene_id.split('/')[3],
                            sound=sound,
                            sr=self.config.TASK_CONFIG.SIMULATOR.AUDIO.RIR_SAMPLING_RATE,
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metric_name='spl',
                            metric_value=infos[i]['spl'],
                            tb_writer=writer,
                            audios=audios[i],#[:-1],
                            fps=fps,
                            text = text_frames[i],#[:-1],
                            num_steps = self.config.NUM_DIALOG_STEPS-1,
                            qs_method=QS_METHOD,
                        )


                        # observations has been reset but info has not
                        # to be consistent, do not use the last frame
                        rgb_frames[i] = []
                        text_frames[i] = []
                        if self.config.DIALOG_TRAINING:
                            actual_step[i] = []
                        audios[i] = []

                    if "top_down_map" in self.config.VISUALIZATION_OPTION:
                        if self.config.RL.PPO.use_belief_predictor:
                            pred = episode_stats['descriptor_pred_gt'][-1]
                        else:
                            pred = None

                        top_down_map = plot_top_down_map(infos[i],
                                                         dataset=self.config.TASK_CONFIG.SIMULATOR.SCENE_DATASET,
                                                         pred=pred)
                        scene = current_episodes[i].scene_id.split('/')[3]
                        sound = current_episodes[i].sound_id.split('/')[1][:-4]
                        writer.add_image(f"{config.EVAL.SPLIT}_{scene}_{current_episodes[i].episode_id}_{sound}/"
                                         f"{infos[i]['spl']}",
                                         top_down_map,
                                         dataformats='WHC')


            if not self.config.RL.PPO.use_belief_predictor:
                descriptor_pred_gt = None

            if self.config.DIALOG_TRAINING:
                (
                    self.envs,
                    test_recurrent_hidden_states,
                    not_done_masks,
                    test_em_vln,
                    test_em_vln_dialog,
                    current_episode_reward,
                    prev_actions,
                    batch,
                    rgb_frames,
                    text_frames,
                    track_query,
                    track_query_count,
                    track_perf,
                    track_agent_step,
                ) = self._pause_envs(
                    envs_to_pause,
                    self.envs,
                    test_recurrent_hidden_states,
                    not_done_masks,
                    current_episode_reward,
                    prev_actions,
                    batch,
                    rgb_frames,
                    text_frames,
                    track_query,
                    track_query_count,
                    track_perf,
                    track_agent_step,
                    test_em_vln=test_em_vln,
                    test_em_vln_dialog=test_em_vln_dialog,
                    descriptor_pred_gt=descriptor_pred_gt,
                )
            else:
                (
                    self.envs,
                    test_recurrent_hidden_states,
                    not_done_masks,
                    test_em_goal,
                    test_em_option,
                    test_em_vln,
                    test_em_vln_dialog,
                    current_episode_reward,
                    prev_actions,
                    batch,
                    rgb_frames,
                    text_frames,
                    track_query,
                    track_query_count,
                    track_perf,
                    track_agent_step,
                ) = self._pause_envs(
                    envs_to_pause,
                    self.envs,
                    test_recurrent_hidden_states,
                    not_done_masks,
                    current_episode_reward,
                    prev_actions,
                    batch,
                    rgb_frames,
                    text_frames,
                    track_query,
                    track_query_count,
                    track_perf,
                    track_agent_step,
                    test_em_goal=test_em_goal,
                    test_em_option=test_em_option,
                    test_em_vln=test_em_vln,
                    test_em_vln_dialog=test_em_vln_dialog,
                    descriptor_pred_gt=descriptor_pred_gt,

                )

        if not self.config.DIALOG_TRAINING and not ONLY_GOAL_POLICY:
            print('average distance between queries',  total_2_query_diff/max(total_pair,1))

            assert sum(all_2_query_diff)==total_2_query_diff, 'sum is not same'
            var = np.sum(np.square(np.array(all_2_query_diff)-(total_2_query_diff/max(total_pair,1))))/max(total_pair,1)

            print('variance', var)


        # dump stats for each episode
        stats_file = os.path.join(config.TENSORBOARD_DIR,
                                  '{}_stats_{}.json'.format(config.EVAL.SPLIT, config.SEED))
        with open(stats_file, 'w') as fo:
            json.dump({','.join(str(k) for k in key): value for key, value in stats_episodes.items()}, fo, cls=NpEncoder)

        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            if stat_key in ['audio_duration', 'gt_na', 'descriptor_pred_gt', 'view_point_goals']:
                continue
            aggregated_stats[stat_key] = sum(
                [v[stat_key] for v in stats_episodes.values()]
            )
        num_episodes = len(stats_episodes)

        episode_reward_mean = aggregated_stats["reward"] / num_episodes
        episode_metrics_mean = {}
        for metric_uuid in self.metric_uuids:
            episode_metrics_mean[metric_uuid] = aggregated_stats[metric_uuid] / num_episodes

        if not self.config.DIALOG_TRAINING:
            test_log.write('\n')
        logger.info(f"Average episode reward: {episode_reward_mean:.6f}")
        for metric_uuid in self.metric_uuids:
            logger.info(
                f"Average episode {metric_uuid}: {episode_metrics_mean[metric_uuid]:.6f}"
            )
            if not self.config.DIALOG_TRAINING:
                test_log.write(f"Average episode {metric_uuid}: {episode_metrics_mean[metric_uuid]:.6f}\n")


        if not config.EVAL.SPLIT.startswith('test'):
            writer.add_scalar("{}/reward".format(config.EVAL.SPLIT), episode_reward_mean, checkpoint_index)
            for metric_uuid in self.metric_uuids:
                writer.add_scalar(f"{config.EVAL.SPLIT}/{metric_uuid}", episode_metrics_mean[metric_uuid],
                                  checkpoint_index)

        self.envs.close()

        if self.config.DIALOG_TRAINING:
            logger.info('vln accuracy: {}%'.format(100*vln_result/vln_cnt))
            for idx, res in enumerate(vln_result_sep):
                logger.info('vln accuracy for action {}: {}%, actual count: {}, predicted_count: {}'\
                .format(idx, 100*res/vln_cnt_sep[idx] if vln_cnt_sep[idx]!=0 else None, vln_cnt_sep[idx], vln_cnt_pred_sep[idx]))
            logger.info('vln seq accuracy: {}%'.format(100*vln_result_seq/vln_cnt_seq))
            logger.info('vln cnt seq: {}'.format(vln_cnt_seq))
            print('all_seq_dict_curr', all_seq_dict_curr)
            print('all_seq_dict_oracle', all_seq_dict_oracle)
            for key in all_seq_dict_curr_correct:
                if all_seq_dict_oracle[key]!=0:
                    all_seq_dict_curr_correct[key] = (100*all_seq_dict_curr_correct[key])/all_seq_dict_oracle[key]
            print('all_seq_dict_curr_correct', all_seq_dict_curr_correct)

            for idx, res in enumerate(vln_result_seq_sep):
                logger.info('vln_seq accuracy for step {}: {}%, actual count: {}'\
                .format(idx, 100*res/vln_cnt_seq_sep[idx] if vln_cnt_seq_sep[idx]!=0 else None, vln_cnt_seq_sep[idx]))

            logger.info('vln_oracle_action_stepwise: {}'.format(vln_oracle_action_stepwise))
            logger.info('vln_current_action_stepwise: {}'.format(vln_current_action_stepwise))
            for key in vln_accuracy_stepwise.keys():
                for idx, _ in enumerate(vln_accuracy_stepwise[key]):
                    if vln_oracle_action_stepwise[key][idx]!=0:
                        vln_accuracy_stepwise[key][idx] = (100*vln_accuracy_stepwise[key][idx])/vln_oracle_action_stepwise[key][idx]

            logger.info('vln_accuracy_stepwise: {}'.format(vln_accuracy_stepwise))
            logger.info('invalid_point_count: {}'.format(self.invalid_point_count))

        else:

            spr_np = np.zeros((2,len(success_per_ratio)))
            cnt_spr = 0
            for k, v in success_per_ratio.items():
                spr_np[0, cnt_spr] = k
                spr_np[1, cnt_spr] = v
                cnt_spr += 1

            np.savez(spr_file, spr = spr_np, qdist = query_dist)



        result = {
            'episode_reward_mean': episode_reward_mean
        }
        for metric_uuid in self.metric_uuids:
            result['episode_{}_mean'.format(metric_uuid)] = episode_metrics_mean[metric_uuid]

        print('ONLY_GOAL_POLICY:', ONLY_GOAL_POLICY, 'QS_METHOD:', QS_METHOD)
        if not self.config.DIALOG_TRAINING:
            test_log.write('total_query_all: {}, total_stop_taken: {} \n'.format(total_query_all, total_stop_taken))
            test_log.write('vln_path: {}\n'.format(vln_path))
            test_log.write('error_dict: {}\n'.format(error_dict))
            test_log.close()
        return result


def compute_distance_to_pred(pred, sim):
    from habitat.utils.geometry_utils import quaternion_rotate_vector
    import networkx as nx

    current_position = sim.get_agent_state().position
    agent_state = sim.get_agent_state()
    source_position = agent_state.position
    source_rotation = agent_state.rotation

    rounded_pred = np.round(pred)
    direction_vector_agent = np.array([rounded_pred[1], 0, -rounded_pred[0]])
    direction_vector = quaternion_rotate_vector(source_rotation, direction_vector_agent)
    pred_goal_location = source_position + direction_vector.astype(np.float32)
    pred_goal_location[1] = source_position[1]

    try:
        if sim.position_encoding(pred_goal_location) not in sim._position_to_index_mapping:
            pred_goal_location = sim.find_nearest_graph_node(pred_goal_location)
        distance_to_target = sim.geodesic_distance(current_position, [pred_goal_location])
    except nx.exception.NetworkXNoPath:
        distance_to_target = -1
    return distance_to_target

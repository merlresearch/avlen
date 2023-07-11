#!/usr/bin/env python3

# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: CC-BY-4.0

import contextlib
import os
import random
import time
from collections import defaultdict, deque
import sys
from copy import deepcopy
import math

import numpy as np
import torch
import torch.distributed as distrib
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import CosineAnnealingLR

from habitat import Config, logger
from ss_baselines.common.baseline_registry import baseline_registry
from ss_baselines.common.env_utils import construct_envs
from ss_baselines.common.environments import get_env_class
from ss_baselines.savi.models.rollout_storage import RolloutStorage
from ss_baselines.common.tensorboard_utils import TensorboardWriter
from ss_baselines.common.utils import batch_obs, linear_decay
from ss_baselines.savi.ddppo.algo.ddp_utils import (
    EXIT,
    REQUEUE,
    add_signal_handlers,
    init_distrib_slurm,
    load_interrupted_state,
    requeue_job,
    save_interrupted_state,
)

from ss_baselines.savi.ddppo.algo.ddppo import DDPPO
from ss_baselines.savi.models.belief_predictor import BeliefPredictor, BeliefPredictorDDP
from ss_baselines.savi.ppo.ppo_trainer import PPOTrainer
from ss_baselines.savi.ppo.policy import AudioNavSMTPolicy, AudioNavBaselinePolicy, AudioNavDialogPolicy, AudioNavOptionPolicy
sys.path.append('./ss_baselines/savi/dialog/speaker/build/')
sys.path.append('./ss_baselines/savi/dialog/speaker/')
sys.path.append('./ss_baselines/savi/dialog/speaker/tasks/R2R/')
from ss_baselines.savi.dialog.speaker.tasks.R2R.speaker_pipeline import Speaker, SpeakerDDP
from ss_baselines.savi.dialog.ques_gen.utils.train_utils import Vocabulary
vocab_path = './ss_baselines/savi/dialog/ques_gen/processed/vocab_iq_vln.json'

import pynvml
from pynvml.smi import nvidia_smi
pynvml.nvmlInit()

SPEAKER = True #(also change in ppo_trainer.py)
DEBUG = False

@baseline_registry.register_trainer(name="ddppo")
class DDPPOTrainer(PPOTrainer):
    # DD-PPO cuts rollouts short to mitigate the straggler effect
    # This, in theory, can cause some rollouts to be very short.
    # All rollouts contributed equally to the loss/model-update,
    # thus very short rollouts can be problematic.  This threshold
    # limits the how short a short rollout can be as a fraction of the
    # max rollout length
    SHORT_ROLLOUT_THRESHOLD: float = 0.25

    def __init__(self, config=None):
        '''
        #  for the time being stop using interupted state
        # interrupted_state = load_interrupted_state()
        if interrupted_state is not None:
            config = interrupted_state["config"]
        '''
        super().__init__(config)
        self.max_dialog_len = 77 # to match the context ength of clip
        self.vocab = Vocabulary()
        self.vocab.load(vocab_path)
        self.invalid_point_count = 0

    def teacher_forcing_scheduler(self, update):
        tf_ratio = 1.0
        if update > 15000:
            tf_ratio = .7
        if update > 30000:
            tf_ratio = .5
        return tf_ratio

    def _setup_actor_critic_agent(self, ppo_cfg: Config, observation_space=None) -> None:
        r"""Sets up actor critic and agent for DD-PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)
        action_space = self.envs.action_spaces[0]

        self.action_space = action_space

        has_distractor_sound = self.config.TASK_CONFIG.SIMULATOR.AUDIO.HAS_DISTRACTOR_SOUND
        if ppo_cfg.policy_type == 'rnn':
            self.actor_critic = AudioNavBaselinePolicy(
                observation_space=self.envs.observation_spaces[0],
                action_space=self.action_space,
                hidden_size=ppo_cfg.hidden_size,
                goal_sensor_uuid=self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
                extra_rgb=self.config.EXTRA_RGB,
                use_mlp_state_encoder=ppo_cfg.use_mlp_state_encoder
            )

            if ppo_cfg.use_belief_predictor:
                belief_cfg = ppo_cfg.BELIEF_PREDICTOR
                bp_class = BeliefPredictorDDP if belief_cfg.online_training else BeliefPredictor
                self.belief_predictor = bp_class(belief_cfg, self.device, None, None,
                                                 ppo_cfg.hidden_size, self.envs.num_envs, has_distractor_sound
                                                 ).to(device=self.device)
                if belief_cfg.online_training:
                    params = list(self.belief_predictor.predictor.parameters())
                    if belief_cfg.train_encoder:
                        params += list(self.actor_critic.net.goal_encoder.parameters()) + \
                                  list(self.actor_critic.net.visual_encoder.parameters()) + \
                                  list(self.actor_critic.net.action_encoder.parameters())
                    self.belief_predictor.optimizer = torch.optim.Adam(params, lr=belief_cfg.lr)
                self.belief_predictor.freeze_encoders()


        elif ppo_cfg.policy_type == 'smt':
            smt_cfg = ppo_cfg.SCENE_MEMORY_TRANSFORMER
            belief_cfg = ppo_cfg.BELIEF_PREDICTOR
            self.actor_critic = AudioNavSMTPolicy(
                observation_space=self.envs.observation_spaces[0],
                action_space=self.envs.action_spaces[0],
                hidden_size=smt_cfg.hidden_size,
                nhead=smt_cfg.nhead,
                num_encoder_layers=smt_cfg.num_encoder_layers,
                num_decoder_layers=smt_cfg.num_decoder_layers,
                dropout=smt_cfg.dropout,
                activation=smt_cfg.activation,
                use_pretrained=smt_cfg.use_pretrained,
                pretrained_path=smt_cfg.pretrained_path,
                pretraining=smt_cfg.pretraining,
                use_belief_encoding=smt_cfg.use_belief_encoding,
                use_belief_as_goal=ppo_cfg.use_belief_predictor,
                use_label_belief=belief_cfg.use_label_belief,
                use_location_belief=belief_cfg.use_location_belief,
                normalize_category_distribution=belief_cfg.normalize_category_distribution,
                use_category_input=has_distractor_sound,
                query_count_emb_size = self.config.QUERY_COUNT_EMB_SIZE,
            )
            if smt_cfg.freeze_encoders:
                self._static_smt_encoder = True
                self.actor_critic.net.freeze_encoders()

            if ppo_cfg.use_belief_predictor:
                smt = self.actor_critic.net.smt_state_encoder
                bp_class = BeliefPredictorDDP if belief_cfg.online_training else BeliefPredictor
                self.belief_predictor = bp_class(belief_cfg, self.device, smt._input_size, smt._pose_indices,
                                                 smt.hidden_state_size, self.envs.num_envs, has_distractor_sound
                                                 ).to(device=self.device)
                if belief_cfg.online_training:
                    params = list(self.belief_predictor.predictor.parameters())
                    if belief_cfg.train_encoder:
                        params += list(self.actor_critic.net.goal_encoder.parameters()) + \
                                  list(self.actor_critic.net.visual_encoder.parameters()) + \
                                  list(self.actor_critic.net.action_encoder.parameters())
                    self.belief_predictor.optimizer = torch.optim.Adam(params, lr=belief_cfg.lr)
                self.belief_predictor.freeze_encoders()

            # -----------------------------------------------------------------------------
            # add speaker module here
            if SPEAKER:
                self.speaker = Speaker(device=self.device)
            # already sent to cuda and set in eval mode

        elif ppo_cfg.policy_type == 'dialog':

            smt_cfg = ppo_cfg.SCENE_MEMORY_TRANSFORMER
            belief_cfg = ppo_cfg.BELIEF_PREDICTOR
            # new for dialog based
            self.actor_critic_vln = AudioNavDialogPolicy(
                observation_space=self.envs.observation_spaces[0],
                action_space=self.envs.action_spaces[0],
                hidden_size=smt_cfg.hidden_size,
                nhead=smt_cfg.nhead,
                num_encoder_layers=smt_cfg.num_encoder_layers,
                num_decoder_layers=smt_cfg.num_decoder_layers,
                dropout=smt_cfg.dropout,
                activation=smt_cfg.activation,
                use_pretrained=smt_cfg.use_pretrained,
                pretrained_path=smt_cfg.pretrained_path,
                pretraining=smt_cfg.pretraining,
                use_belief_encoding=smt_cfg.use_belief_encoding,
                use_belief_as_goal=ppo_cfg.use_belief_predictor,
                use_label_belief=belief_cfg.use_label_belief,
                use_location_belief=belief_cfg.use_location_belief,
                normalize_category_distribution=belief_cfg.normalize_category_distribution,
                use_category_input=has_distractor_sound,
                num_steps = self.config.NUM_DIALOG_STEPS,
            )

            if smt_cfg.freeze_encoders:
                self._static_smt_encoder = True
                self.actor_critic_vln.net.freeze_encoders()

            if ppo_cfg.use_belief_predictor:
                smt = self.actor_critic_vln.net.smt_state_encoder
                # check, only actor_critic_vln updated when belief predictor updated?
                bp_class = BeliefPredictorDDP if belief_cfg.online_training else BeliefPredictor
                self.belief_predictor = bp_class(belief_cfg, self.device, smt._input_size, smt._pose_indices,
                                                 smt.hidden_state_size, self.envs.num_envs, has_distractor_sound
                                                 ).to(device=self.device)
                '''
                if belief_cfg.online_training:
                    params = list(self.belief_predictor.predictor.parameters())
                    if belief_cfg.train_encoder:
                        params += list(self.actor_critic_vln.net.goal_encoder.parameters()) + \
                                  list(self.actor_critic_vln.net.visual_encoder.parameters()) + \
                                  list(self.actor_critic_vln.net.action_encoder.parameters())
                    self.belief_predictor.optimizer = torch.optim.Adam(params, lr=belief_cfg.lr)

                self.belief_predictor.freeze_encoders()
                '''
            # -----------------------------------------------------------------------------
            # add speaker module here
            if SPEAKER:
                self.speaker = Speaker(device=self.device)
            # already sent to cuda and set in eval mode


        else:
            raise ValueError(f'Policy type {ppo_cfg.policy_type} is not defined!')


        # edit-----------------------
        self.actor_critic_vln.to(self.device)

        # load weights for both actor critic and the encoder
        pretrained_state = torch.load(self.config.GOAL_CKPT_PATH, map_location="cpu")
        self.actor_critic_vln.net.visual_encoder.rgb_encoder.load_state_dict(
            {
                k[len("actor_critic.net.visual_encoder.rgb_encoder."):]: v
                for k, v in pretrained_state["state_dict"].items()
                if "actor_critic.net.visual_encoder.rgb_encoder." in k
            },
        )
        self.actor_critic_vln.net.visual_encoder.depth_encoder.load_state_dict(
            {
                k[len("actor_critic.net.visual_encoder.depth_encoder."):]: v
                for k, v in pretrained_state["state_dict"].items()
                if "actor_critic.net.visual_encoder.depth_encoder." in k
            },
        )
        logger.info('visual encoder loaded')

        self.actor_critic_vln.net.goal_encoder.load_state_dict(
            {
                k[len("actor_critic.net.goal_encoder."):]: v
                for k, v in pretrained_state['state_dict'].items()
                if "actor_critic.net.goal_encoder." in k
            },
        )
        logger.info('goal encoder loaded')

        self.actor_critic_vln.net.action_encoder.load_state_dict(
            {
                k[len("actor_critic.net.action_encoder."):]: v
                for k, v in pretrained_state['state_dict'].items()
                if "actor_critic.net.action_encoder." in k
            },
        )
        logger.info('action_encoder loaded')

        self.belief_predictor.load_state_dict(pretrained_state["belief_predictor"])
        logger.info('belief_predictor loaded loaded')

        if self.config.RL.DDPPO.reset_critic:
            nn.init.orthogonal_(self.actor_critic_vln.critic.fc.weight)
            nn.init.constant_(self.actor_critic_vln.critic.fc.bias, 0)


        self.agent = DDPPO(
            actor_critic=self.actor_critic_vln,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )


    def _setup_actor_critic_agent_interactive(self, ppo_cfg: Config, observation_space=None) -> None:
        logger.add_filehandler(self.config.LOG_FILE)
        action_space = self.envs.action_spaces[0]

        self.action_space = action_space
        has_distractor_sound = self.config.TASK_CONFIG.SIMULATOR.AUDIO.HAS_DISTRACTOR_SOUND

        if ppo_cfg.policy_type == 'interactive':
            smt_cfg = ppo_cfg.SCENE_MEMORY_TRANSFORMER
            belief_cfg = ppo_cfg.BELIEF_PREDICTOR

            self.actor_critic_goal = AudioNavSMTPolicy(
                observation_space=self.envs.observation_spaces[0],
                action_space=self.envs.action_spaces[0],
                hidden_size=smt_cfg.hidden_size,
                nhead=smt_cfg.nhead,
                num_encoder_layers=smt_cfg.num_encoder_layers,
                num_decoder_layers=smt_cfg.num_decoder_layers,
                dropout=smt_cfg.dropout_goal,
                activation=smt_cfg.activation,
                use_pretrained=smt_cfg.use_pretrained,
                pretrained_path=smt_cfg.pretrained_path,
                pretraining=False,
                use_belief_encoding=smt_cfg.use_belief_encoding,
                use_belief_as_goal=ppo_cfg.use_belief_predictor,
                use_label_belief=belief_cfg.use_label_belief,
                use_location_belief=belief_cfg.use_location_belief,
                normalize_category_distribution=belief_cfg.normalize_category_distribution,
                use_category_input=has_distractor_sound,
            )

            self.actor_critic_vln = AudioNavDialogPolicy(
                observation_space=self.envs.observation_spaces[0],
                action_space=self.envs.action_spaces[0],
                hidden_size=smt_cfg.hidden_size,
                nhead=smt_cfg.nhead,
                num_encoder_layers=smt_cfg.num_encoder_layers,
                num_decoder_layers=smt_cfg.num_decoder_layers,
                dropout=smt_cfg.dropout,
                activation=smt_cfg.activation,
                use_pretrained=smt_cfg.use_pretrained,
                pretrained_path=smt_cfg.pretrained_path,
                pretraining=False,
                use_belief_encoding=smt_cfg.use_belief_encoding,
                use_belief_as_goal=ppo_cfg.use_belief_predictor,
                use_label_belief=belief_cfg.use_label_belief,
                use_location_belief=belief_cfg.use_location_belief,
                normalize_category_distribution=belief_cfg.normalize_category_distribution,
                use_category_input=has_distractor_sound,
                num_steps = self.config.NUM_DIALOG_STEPS,
            )

            # need a model for option policy
            self.actor_critic_option = AudioNavOptionPolicy(
                observation_space=self.envs.observation_spaces[0],
                action_space=self.envs.action_spaces[0],
                hidden_size=smt_cfg.hidden_size,
                nhead=smt_cfg.nhead,
                num_encoder_layers=smt_cfg.num_encoder_layers,
                num_decoder_layers=smt_cfg.num_decoder_layers,
                dropout=smt_cfg.dropout,
                activation=smt_cfg.activation,
                use_pretrained=smt_cfg.use_pretrained,
                pretrained_path=smt_cfg.pretrained_path,
                pretraining= smt_cfg.pretraining,
                use_belief_encoding=smt_cfg.use_belief_encoding,
                use_belief_as_goal=ppo_cfg.use_belief_predictor,
                use_label_belief=belief_cfg.use_label_belief,
                use_location_belief=belief_cfg.use_location_belief,
                normalize_category_distribution=belief_cfg.normalize_category_distribution,
                use_category_input=has_distractor_sound,
                # num_steps = self.config.NUM_DIALOG_STEPS,
                query_count_emb_size = self.config.QUERY_COUNT_EMB_SIZE,
            )

            if smt_cfg.freeze_encoders:
                self._static_smt_encoder = True
                self.actor_critic_goal.net.freeze_encoders()

            if ppo_cfg.use_belief_predictor:
                smt = self.actor_critic_goal.net.smt_state_encoder  # we can use smt_state_encoder of actor_critic_vln too
                bp_class = BeliefPredictorDDP if belief_cfg.online_training else BeliefPredictor
                self.belief_predictor = bp_class(belief_cfg, self.device, smt._input_size, smt._pose_indices,
                                                 smt.hidden_state_size, self.envs.num_envs, has_distractor_sound
                                                 ).to(device=self.device)
                # not learning online
                self.belief_predictor.freeze_encoders()

            # -----------------------------------------------------------------------------
            if SPEAKER:
                self.speaker = Speaker(device=self.device)

        else:
            raise ValueError(f'Policy type {ppo_cfg.policy_type} is not defined for this case!')

        self.actor_critic_goal.to(self.device)
        self.actor_critic_vln.to(self.device)
        self.actor_critic_option.to(self.device)

        # for vln
        for name, param in self.actor_critic_vln.named_parameters():
            if 'net.clip' in name:
                param.requires_grad = False

        ckpt_dict_vln = self.load_checkpoint(self.config.VLN_CKPT_PATH, map_location="cpu")
        ckpt2load_vln = {}
        for k,v in ckpt_dict_vln["state_dict"].items():
            if k.split('.')[0]=='actor_critic':
                ckpt2load_vln['.'.join(k.split('.')[1:])] = v
        self.actor_critic_vln.load_state_dict(ckpt2load_vln, strict=False)


        # for goal based policy
        for name, param in self.actor_critic_goal.named_parameters():
            param.requires_grad = False
        ckpt_dict_goal = self.load_checkpoint(self.config.GOAL_CKPT_PATH, map_location="cpu")

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

        if not DEBUG:
            # loading encoders for option policy
            self.actor_critic_option.net.visual_encoder.rgb_encoder.load_state_dict(
                {
                    k[len("actor_critic.net.visual_encoder.rgb_encoder."):]: v
                    for k, v in ckpt_dict_goal['state_dict'].items()
                    if "actor_critic.net.visual_encoder.rgb_encoder." in k
                },
            )
            self.actor_critic_option.net.visual_encoder.depth_encoder.load_state_dict(
                {
                    k[len("actor_critic.net.visual_encoder.depth_encoder."):]: v
                    for k, v in ckpt_dict_goal['state_dict'].items()
                    if "actor_critic.net.visual_encoder.depth_encoder." in k
                },
            )
            logger.info('visual_encoder loaded')
            self.actor_critic_option.net.goal_encoder.load_state_dict(
                {
                    k[len("actor_critic.net.goal_encoder."):]: v
                    for k, v in ckpt_dict_goal['state_dict'].items()
                    if "actor_critic.net.goal_encoder." in k
                },
            )
            logger.info('goal_encoder loaded')
            self.actor_critic_option.net.action_encoder.load_state_dict(
                {
                    k[len("actor_critic.net.action_encoder."):]: v
                    for k, v in ckpt_dict_goal['state_dict'].items()
                    if "actor_critic.net.action_encoder." in k
                },
            )
            logger.info('action_encoder loaded')


        self.belief_predictor.load_state_dict(ckpt_dict_goal["belief_predictor"])
        logger.info('belief_predictor loaded')

        # releasing checkpoints
        ckpt_dict_vln = None
        ckpt2load_vln = None
        ckpt_dict_goal = None
        ckpt2load_goal = None

        if self.config.RL.DDPPO.reset_critic:
            nn.init.orthogonal_(self.actor_critic_goal.critic.fc.weight)
            nn.init.constant_(self.actor_critic_goal.critic.fc.bias, 0)


        self.agent = DDPPO(
            actor_critic=self.actor_critic_option,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )

        self.agent_vln = DDPPO(
            actor_critic=self.actor_critic_vln,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )

        with torch.no_grad():
            max_len = 1000
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.config.QUERY_COUNT_EMB_SIZE, 2) * (-math.log(10000.0) / self.config.QUERY_COUNT_EMB_SIZE))
            self.pe = torch.zeros(max_len, self.config.QUERY_COUNT_EMB_SIZE)
            self.pe[:, 0::2] = torch.sin(position * div_term)
            self.pe[:, 1::2] = torch.cos(position * div_term)


    def train(self) -> None:
        r"""Main method for DD-PPO.

        Returns:
            None
        """

        self.local_rank, tcp_store = init_distrib_slurm(
            self.config.RL.DDPPO.distrib_backend
        )

        add_signal_handlers()
        # Stores the number of workers that have finished their rollout
        num_rollouts_done_store = distrib.PrefixStore(
            "rollout_tracker", tcp_store
        )
        num_rollouts_done_store.set("num_done", "0")

        self.world_rank = distrib.get_rank()
        self.world_size = distrib.get_world_size()

        self.config.defrost()
        self.config.TORCH_GPU_ID = self.local_rank
        self.config.SIMULATOR_GPU_ID = self.local_rank
        # Multiply by the number of simulators to make sure they also get unique seeds
        self.config.TASK_CONFIG.SEED += (
            self.world_rank * self.config.NUM_PROCESSES
        )

        self.config.freeze()

        random.seed(self.config.TASK_CONFIG.SEED)
        np.random.seed(self.config.TASK_CONFIG.SEED)
        torch.manual_seed(self.config.TASK_CONFIG.SEED)
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        # use the config savi_pretraining_dialog_training.yaml
        # this config internally calls task config semantic_audiogoal_dialog.yaml
        if self.config.DIALOG_TRAINING:
            self.config.defrost()
            # NUM_UPDATES should be set based on number of gpus
            self.config.NUM_UPDATES = self.config.NUM_UPDATES_DIALOG
            self.config.CHECKPOINT_INTERVAL = self.config.CHECKPOINT_INTERVAL_DIALOG
            self.config.RL.PPO.num_steps = self.config.NUM_DIALOG_STEPS
            self.config.RL.PPO.SCENE_MEMORY_TRANSFORMER.pretraining = False
            self.config.freeze()

        # constructing env
        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )
        ppo_cfg = self.config.RL.PPO

        if (
            not os.path.isdir(self.config.CHECKPOINT_FOLDER)
            and self.world_rank == 0
        ):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        if self.config.DIALOG_TRAINING:
            self._setup_actor_critic_agent(ppo_cfg)
        else:
            self._setup_actor_critic_agent_interactive(ppo_cfg)

        self.agent.init_distributed(find_unused_params=True)
        if not self.config.DIALOG_TRAINING:
            self.agent_vln.init_distributed(find_unused_params=True)

        if ppo_cfg.use_belief_predictor and ppo_cfg.BELIEF_PREDICTOR.online_training:
            self.belief_predictor.init_distributed(find_unused_params=True)

        if self.world_rank == 0:
            logger.info(
                "agent number of trainable parameters: {}".format(
                    sum(
                        param.numel()
                        for param in self.agent.parameters()
                        if param.requires_grad
                    )
                )
            )
            if not self.config.DIALOG_TRAINING:
                logger.info(
                    "agent_vln number of trainable parameters: {}".format(
                        sum(
                            param.numel()
                            for param in self.agent_vln.parameters()
                            if param.requires_grad
                        )
                    )
                )
            if ppo_cfg.use_belief_predictor:
                logger.info(
                    "belief predictor number of trainable parameters: {}".format(
                        sum(
                            param.numel()
                            for param in self.belief_predictor.parameters()
                            if param.requires_grad
                        )
                    )
                )
            logger.info(f"config: {self.config}")

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        obs_space = self.envs.observation_spaces[0]


        if ppo_cfg.use_external_memory:
            if not self.config.DIALOG_TRAINING:
                memory_dim_option = self.actor_critic_option.net.memory_dim
                memory_dim_goal = self.actor_critic_goal.net.memory_dim
                memory_dim_vln = self.actor_critic_vln.net.memory_dim

            else:
                memory_dim_option = self.actor_critic_vln.net.memory_dim
                memory_dim_goal = self.actor_critic_vln.net.memory_dim
                memory_dim_vln = self.actor_critic_vln.net.memory_dim

        else:
            memory_dim_option = None
            memory_dim_goal = None
            memory_dim_vln = None

        if ppo_cfg.use_state_memory:
            memory_dim_dialog = ppo_cfg.SCENE_MEMORY_TRANSFORMER.hidden_size
        else:
            memory_dim_dialog = None


        rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            obs_space,
            self.action_space,
            ppo_cfg.hidden_size,
            ppo_cfg.use_external_memory,
            ppo_cfg.SCENE_MEMORY_TRANSFORMER.memory_size + ppo_cfg.num_steps, # for goal
            ppo_cfg.SCENE_MEMORY_TRANSFORMER.memory_size,
            ppo_cfg.SCENE_MEMORY_TRANSFORMER.memory_size + ppo_cfg.num_steps, # for query
            ppo_cfg.SCENE_MEMORY_TRANSFORMER.memory_size,
            self.config.NUM_DIALOG_STEPS, # for vln
            self.config.NUM_DIALOG_STEPS,
            memory_dim_goal,
            memory_dim_vln,
            memory_dim_option,
            memory_dim_dialog,
            num_recurrent_layers=self.actor_critic_vln.net.num_recurrent_layers,
            max_dialog_len=self.max_dialog_len,
            use_state_memory = ppo_cfg.use_state_memory,
        )
        rollouts.to(self.device)


        if not self.config.DIALOG_TRAINING and self.config.REPLAY_STORE:

            self.store_dict = {
                          'batch': {},
                          'recurrent_hidden_states':[],
                          'actions':[],
                          'actions_option':[],
                          'actions_log_probs_option':[],
                          'values':[],
                          'rewards':[],
                          'masks': [],
                          'masks_vln': [],
                          'external_memory_features':[],
                          'external_memory_dialog_features': [],
                          'current_dialog': [],
                          'o_action': [],
                          'o_mask': [],
                          'action_prob': [],
                          'current_query_state': [],
                          'current_agent_step': [],
                         }

            for key in batch.keys():
                self.store_dict['batch'][key]=[]

            self.replay_buffer = {idx:deepcopy(self.store_dict) for idx in range(self.config.NUM_PROCESSES)}
            rollouts_vln = RolloutStorage(
                self.config.NUM_DIALOG_STEPS,
                self.envs.num_envs,
                obs_space,
                self.action_space,
                ppo_cfg.hidden_size,
                ppo_cfg.use_external_memory,
                self.config.NUM_DIALOG_STEPS,
                self.config.NUM_DIALOG_STEPS,
                self.config.NUM_DIALOG_STEPS,
                self.config.NUM_DIALOG_STEPS,
                self.config.NUM_DIALOG_STEPS, # for vln
                self.config.NUM_DIALOG_STEPS,
                memory_dim_goal,
                memory_dim_vln,
                memory_dim_option,
                memory_dim_dialog,
                num_recurrent_layers=self.actor_critic_vln.net.num_recurrent_layers,
                max_dialog_len=self.max_dialog_len,
                use_state_memory = ppo_cfg.use_state_memory,
            )
            rollouts_vln.to(self.device)


        # -----------------------------------------
        # set up a dictionary for tracking when the query is triggered
        track_query = {idx: {'dialog': [], 'step':0, 'queried': False, 'cons_reward': 0, 'last_query_step': 0, 'total_step': 0, 'all_step': [], 'all_reward': []} for idx in range(self.config.NUM_PROCESSES)}
        track_query_count = {idx: 0 for idx in range(self.config.NUM_PROCESSES)}

        if self.config.RL.PPO.use_belief_predictor:
            self.belief_predictor.update(batch, None)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        current_episode_info = dict(
            current_episode_reward = torch.zeros(self.envs.num_envs, 1, device=self.device),
            current_episode_reward_goal = torch.zeros(self.envs.num_envs, 1, device=self.device),
            current_episode_reward_vln = torch.zeros(self.envs.num_envs, 1, device=self.device),
            current_episode_step_goal = torch.zeros(self.envs.num_envs, 1, device=self.device),
            current_episode_step_vln = torch.zeros(self.envs.num_envs, 1, device=self.device),
            current_episode_step_stat_goal = torch.zeros(self.envs.num_envs, 4, device=self.device),
            current_episode_step_stat_vln = torch.zeros(self.envs.num_envs, 4, device=self.device),
            current_episode_query_cnt_thresh = torch.zeros(self.envs.num_envs, 1, device=self.device),
            current_episode_query_cnt_radius = torch.zeros(self.envs.num_envs, 1, device=self.device),
            current_episode_1st_query = torch.zeros(self.envs.num_envs, 1, device=self.device),
            current_episode_4th_query = torch.zeros(self.envs.num_envs, 1, device=self.device),

        )
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1, device=self.device),
            reward=torch.zeros(self.envs.num_envs, 1, device=self.device),
            reward_goal=torch.zeros(self.envs.num_envs, 1, device=self.device),
            reward_vln=torch.zeros(self.envs.num_envs, 1, device=self.device),
            query_count=torch.zeros(self.envs.num_envs, 1, device=self.device),
            step_count=torch.zeros(self.envs.num_envs, 1, device=self.device),
            forward_step_goal = torch.zeros(self.envs.num_envs, 1, device=self.device),
            left_step_goal = torch.zeros(self.envs.num_envs, 1, device=self.device),
            right_step_goal = torch.zeros(self.envs.num_envs, 1, device=self.device),
            forward_step_vln = torch.zeros(self.envs.num_envs, 1, device=self.device),
            left_step_vln = torch.zeros(self.envs.num_envs, 1, device=self.device),
            right_step_vln = torch.zeros(self.envs.num_envs, 1, device=self.device),
            step_count_goal = torch.zeros(self.envs.num_envs, 1, device=self.device),
            step_count_vln = torch.zeros(self.envs.num_envs, 1, device=self.device),
            query_count_thresh = torch.zeros(self.envs.num_envs, 1, device=self.device),
            query_count_radius = torch.zeros(self.envs.num_envs, 1, device=self.device),
            query_step_1st = torch.zeros(self.envs.num_envs, 1, device=self.device),
            query_step_4th = torch.zeros(self.envs.num_envs, 1, device=self.device),
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
        replay_training_cnt = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )

        if self.config.DIALOG_TRAINING:
            lr_scheduler_vln = CosineAnnealingLR(self.agent.dialog_optimizer, T_max=30, eta_min=.000001)

        if self.config.RESUME_CHECKPOINT:
            # Try to resume at previous checkpoint (independent of interrupted states)
            count_steps_start, count_checkpoints, start_update, replay_training_cnt, count_checkpoints_vln = self.try_to_resume_checkpoint()
            count_steps = count_steps_start

        else:
            count_steps_start = 0
            count_checkpoints = 0
            count_checkpoints_vln = 0
            start_update = 0

        interrupted_state = load_interrupted_state()
        assert interrupted_state is None, 'shouldnt start from interupted state'

        if interrupted_state is not None:
            self.agent.load_state_dict(interrupted_state["state_dict"])
            if self.config.RL.PPO.use_belief_predictor:
                self.belief_predictor.load_state_dict(interrupted_state["belief_predictor"])
            self.agent.optimizer.load_state_dict(
                interrupted_state["optim_state"]
            )
            lr_scheduler.load_state_dict(interrupted_state["lr_sched_state"])

            requeue_stats = interrupted_state["requeue_stats"]
            env_time = requeue_stats["env_time"]
            pth_time = requeue_stats["pth_time"]
            count_steps = requeue_stats["count_steps"]
            count_checkpoints = requeue_stats["count_checkpoints"]
            start_update = requeue_stats["start_update"]
            prev_time = requeue_stats["prev_time"]

        with (
            TensorboardWriter(
                self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
            )
            if self.world_rank == 0
            else contextlib.suppress()
        ) as writer:
            for update in range(start_update, self.config.NUM_UPDATES):
                if self.config.DIALOG_TRAINING:
                    lr_scheduler_vln.step()

                if ppo_cfg.use_linear_lr_decay: # False
                    lr_scheduler.step()

                if ppo_cfg.use_linear_clip_decay: # False
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )

                if EXIT.is_set():
                    self.envs.close()

                    if REQUEUE.is_set() and self.world_rank == 0:
                        requeue_stats = dict(
                            env_time=env_time,
                            pth_time=pth_time,
                            count_steps=count_steps,
                            count_checkpoints=count_checkpoints,
                            start_update=update,
                            prev_time=(time.time() - t_start) + prev_time,
                        )
                        state_dict = dict(
                                state_dict=self.agent.state_dict(),
                                optim_state=self.agent.optimizer.state_dict(),
                                lr_sched_state=lr_scheduler.state_dict(),
                                config=self.config,
                                requeue_stats=requeue_stats,
                            )
                        if self.config.RL.PPO.use_belief_predictor:
                            state_dict['belief_predictor'] = self.belief_predictor.state_dict()
                        save_interrupted_state(state_dict)

                    requeue_job()
                    return

                count_steps_delta = 0
                self.agent.eval()
                if not self.config.DIALOG_TRAINING:
                    self.agent_vln.eval()
                    self.actor_critic_goal.eval()
                if self.config.RL.PPO.use_belief_predictor:
                    self.belief_predictor.eval()

                # dialog
                if self.config.DIALOG_TRAINING:
                    o_actions = self.envs.compute_oracle_actions()
                    self.o_actions_updated = np.zeros((self.config.NUM_DIALOG_STEPS, self.config.NUM_PROCESSES))
                    self.o_actions_mask = np.ones((self.config.NUM_DIALOG_STEPS, self.config.NUM_PROCESSES))

                    for process_idx in range(self.config.NUM_PROCESSES):
                        if len(o_actions[process_idx])> self.config.NUM_DIALOG_STEPS:
                            self.o_actions_updated[:, process_idx] = o_actions[process_idx][:self.config.NUM_DIALOG_STEPS]
                        else:
                            self.o_actions_updated[:len(o_actions[process_idx]), process_idx] = o_actions[process_idx]
                            self.o_actions_mask[len(o_actions[process_idx]-1):, process_idx] = 0


                for step in range(ppo_cfg.num_steps):
                    (
                        delta_pth_time,
                        delta_env_time,
                        delta_steps,
                        track_query,
                        track_query_count,
                        replay_store
                    ) = self._collect_rollout_step(
                        rollouts, current_episode_info, running_episode_stats,
                        track_query, track_query_count, tf_ratio = self.teacher_forcing_scheduler(update)
                    )

                    if not self.config.DIALOG_TRAINING and self.config.REPLAY_STORE:

                        self.assign_to_replay_buffer(replay_store)
                        storing_done = self.store_in_rollout(rollouts_vln)

                        if storing_done:

                            self.agent_vln.train()
                            (
                            _,
                            ce_loss_replay,
                            ) = self._update_agent_vln(rollouts_vln)
                            # stats = torch.tensor( [ce_loss_replay], device=self.device)
                            # distrib.all_reduce(stats)


                            #if self.world_rank == 0:
                            #    ce_loss_replay = ce_loss_replay #/ self.world_size  #stats[0].item() / self.world_size
                            #    logger.info("replay_training_cnt: {}, cross entropy loss: {}".format(replay_training_cnt, ce_loss_replay)),

                            # checkpoint model

                            if replay_training_cnt % self.config.CHECKPOINT_INTERVAL_DIALOG == 0:
                                self.save_checkpoint_vln(
                                    f"vln/ckpt.{count_checkpoints_vln}.pth",
                                    dict(step=replay_training_cnt),
                                )
                                count_checkpoints_vln += 1

                            replay_training_cnt += 1

                            self.agent_vln.eval()


                        # ce_loss_replay = 0

                    else:
                        ce_loss_replay = 0

                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps_delta += delta_steps

                    # This is where the preemption of workers happens.  If a
                    # worker detects it will be a straggler, it preempts itself!
                    if not self.config.DIALOG_TRAINING:
                        if (
                            step
                            >= ppo_cfg.num_steps * self.SHORT_ROLLOUT_THRESHOLD
                        ) and int(num_rollouts_done_store.get("num_done")) > (
                            self.config.RL.DDPPO.sync_frac * self.world_size
                        ):
                            break

                num_rollouts_done_store.add("num_done", 1)

                self.agent.train()
                if self.config.RL.PPO.use_belief_predictor:
                    # self.belief_predictor.train()
                    self.belief_predictor.set_eval_encoders()
                if self._static_smt_encoder:
                    if not self.config.DIALOG_TRAINING:
                        self.actor_critic_option.net.set_eval_encoders()
                        self.actor_critic_goal.net.set_eval_encoders()
                        self.actor_critic_vln.net.set_eval_encoders()

                    else:
                        self.actor_critic_vln.net.set_eval_encoders()

                '''
                if ppo_cfg.use_belief_predictor and ppo_cfg.BELIEF_PREDICTOR.online_training:
                    location_predictor_loss, prediction_accuracy = self.train_belief_predictor(rollouts)
                else:
                '''
                location_predictor_loss = 0
                prediction_accuracy = 0

                if self.config.DIALOG_TRAINING:

                    (
                        delta_pth_time,
                        ce_loss,
                    ) = self._update_agent_dialog(rollouts)

                    pth_time += delta_pth_time
                    stats = torch.tensor( [ce_loss, count_steps_delta], device=self.device)
                    distrib.all_reduce(stats)

                    observations = self.envs.reset()
                    batch = batch_obs(observations, device=self.device)
                    for sensor in rollouts.observations:
                        rollouts.observations[sensor][0].copy_(batch[sensor])
                    if self.config.RL.PPO.use_belief_predictor:
                        self.belief_predictor.update(batch, None)
                    batch = None
                    observations = None

                    if self.world_rank == 0:
                        num_rollouts_done_store.set("num_done", "0")
                        loss = stats[0].item() / self.world_size
                        count_steps += stats[1].item()

                        writer.add_scalar("Policy/ce_loss", loss, count_steps)

                        if update > 0 and update % self.config.LOG_INTERVAL == 0:
                            logger.info(
                                "update: {}\tfps: {:.3f}\t".format(
                                    update,
                                    (count_steps - count_steps_start)
                                    / ((time.time() - t_start) + prev_time),
                                )
                            )

                            logger.info(
                                "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                                "frames: {}".format(
                                    update, env_time, pth_time, count_steps
                                )
                            )
                            logger.info('dialog training: {}, without_dialog: {}'.format(self.config.DIALOG_TRAINING, self.config.DIALOG_TRAINING_WITHOUT_DIALOG))
                            logger.info("num_process: {}, weighted sequential cross entropy loss: {}".format(self.config.NUM_PROCESSES, loss)),


                        # checkpoint model
                        if update % self.config.CHECKPOINT_INTERVAL == 0:
                            self.save_checkpoint(
                                f"ckpt.{count_checkpoints}.pth",
                                dict(step=count_steps),
                            )
                            count_checkpoints += 1


                if not self.config.DIALOG_TRAINING:
                    (
                        delta_pth_time,
                        value_loss,
                        action_loss,
                        dist_entropy,
                        values_debug, return_batch_debug,
                        unct_loss,
                    ) = self._update_agent(ppo_cfg, rollouts)

                    pth_time += delta_pth_time



                    stats_ordering = list(sorted(running_episode_stats.keys()))
                    stats = torch.stack(
                    [running_episode_stats[k] for k in stats_ordering], 0
                    )
                    distrib.all_reduce(stats)

                    for i, k in enumerate(stats_ordering):
                        window_episode_stats[k].append(stats[i].clone())

                    stats = torch.tensor(
                        [value_loss, action_loss, dist_entropy, location_predictor_loss, prediction_accuracy, count_steps_delta, ce_loss_replay, values_debug, return_batch_debug, unct_loss],
                        device=self.device,
                    )
                    distrib.all_reduce(stats)

                    count_steps += stats[5].item()

                    if self.world_rank == 0:
                        num_rollouts_done_store.set("num_done", "0")

                        losses = [
                            stats[0].item() / self.world_size,
                            stats[1].item() / self.world_size,
                            stats[2].item() / self.world_size,
                            stats[3].item() / self.world_size,
                            stats[4].item() / self.world_size,
                            stats[6].item() / self.world_size,
                            stats[7].item() / self.world_size,
                            stats[8].item() / self.world_size,
                            stats[9].item() / self.world_size,

                        ]


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

                        metrics = {
                            k: v / deltas["count"]
                            for k, v in deltas.items()
                            if k not in {"reward", "count", 'reward_goal', 'reward_vln', 'query_count', 'step_count', 'forward_step_goal', 'left_step_goal', 'right_step_goal', 'forward_step_vln', 'left_step_vln', 'right_step_vln', 'step_count_goal', 'step_count_vln', 'query_count_thresh', 'query_count_radius', 'query_step_1st', 'query_step_4th'}
                        }

                        if len(metrics) > 0:
                            for metric, value in metrics.items():
                                writer.add_scalar(f"Metrics/{metric}", value, count_steps)

                        # for debugging
                        writer.add_scalar("Debug/reward_goal", deltas['reward_goal']/max(deltas['step_count_goal'],1), count_steps)
                        writer.add_scalar("Debug/reward_vln", deltas['reward_vln']/max(deltas['step_count_vln'],1) , count_steps)
                        writer.add_scalar("Debug/window_query_ratio", deltas['query_count']/max(deltas['step_count'],1)  , count_steps)
                        writer.add_scalar("Debug/forward_ratio_goal", deltas['forward_step_goal']/max(deltas['step_count_goal'],1), count_steps)
                        writer.add_scalar('Debug/left_ratio_goal', deltas['left_step_goal']/max(deltas['step_count_goal'],1), count_steps)
                        writer.add_scalar('Debug/right_ratio_goal', deltas['right_step_goal']/max(deltas['step_count_goal'],1), count_steps)
                        writer.add_scalar('Debug/forward_ratio_vln', deltas['forward_step_vln']/max(deltas['step_count_vln'],1), count_steps)
                        writer.add_scalar('Debug/left_ratio_vln', deltas['left_step_vln']/max(deltas['step_count_vln'],1), count_steps)
                        writer.add_scalar('Debug/right_ratio_vln', deltas['right_step_vln']/max(deltas['step_count_vln'],1), count_steps)


                        deltas_v2 = {
                            k: (
                                (v[-1] - v[-2]).sum().item()
                                if len(v) > 1
                                else v[0].sum().item()
                            )
                            for k, v in window_episode_stats.items() if k in {'reward_goal', 'step_count_goal', 'reward_vln', 'step_count_vln', 'count', 'query_count', 'step_count', 'query_count_thresh', 'query_count_radius', 'query_step_1st', 'query_step_4th'}
                        }
                        '''
                        writer.add_scalar("Debug/current_query_ratio", deltas_v2['query_count']/max(deltas_v2['step_count'],1), count_steps)
                        writer.add_scalar("Debug/current_query_ratio_thresh", deltas_v2['query_count_thresh']/max(deltas_v2['step_count'],1), count_steps)
                        writer.add_scalar("Debug/current_query_ratio_radius", deltas_v2['query_count_radius']/max(deltas_v2['step_count'],1), count_steps)
                        writer.add_scalar("Debug/current_query_step_1st", deltas_v2['query_step_1st']/max(deltas_v2['step_count'],1), count_steps)
                        writer.add_scalar("Debug/current_query_step_4th", deltas_v2['query_step_4th']/max(deltas_v2['step_count'],1), count_steps)
                        '''
                        writer.add_scalar("Debug/current_reward_vln", deltas_v2['reward_vln']/max(deltas_v2['step_count_vln'],1) , count_steps)
                        writer.add_scalar("Debug/current_reward_goal", deltas_v2['reward_goal']/max(deltas_v2['step_count_goal'],1) , count_steps)

                        writer.add_scalar("Debug/current_query", deltas_v2['query_count']/max(deltas_v2['count'],1), count_steps)
                        writer.add_scalar("Debug/current_query_thresh", deltas_v2['query_count_thresh']/max(deltas_v2['count'],1), count_steps)
                        writer.add_scalar("Debug/current_query_radius", deltas_v2['query_count_radius']/max(deltas_v2['count'],1), count_steps)
                        writer.add_scalar("Debug/current_query_step_1st", deltas_v2['query_step_1st']/max(deltas_v2['count'],1), count_steps)
                        writer.add_scalar("Debug/current_query_step_4th", deltas_v2['query_step_4th']/max(deltas_v2['count'],1), count_steps)


                        writer.add_scalar("Policy/value_loss", losses[0], count_steps)
                        writer.add_scalar("Policy/policy_loss", losses[1], count_steps)
                        writer.add_scalar("Policy/entropy_loss", losses[2], count_steps)
                        writer.add_scalar("Policy/predictor_loss", losses[3], count_steps)
                        writer.add_scalar("Policy/predictor_accuracy", losses[4], count_steps)
                        writer.add_scalar('Policy/learning_rate', lr_scheduler.get_lr()[0], count_steps)
                        writer.add_scalar('Policy/values', losses[6], count_steps)
                        writer.add_scalar('Policy/returns', losses[7], count_steps)
                        writer.add_scalar('Policy/unct_loss', losses[8], count_steps)

                        # log stats
                        if update > 0 and update % self.config.LOG_INTERVAL == 0:
                            logger.info(
                                "update: {}\tfps: {:.3f}\t".format(
                                    update,
                                    (count_steps - count_steps_start)
                                    / ((time.time() - t_start) + prev_time),
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
                            logger.info("replay_training_cnt: {}, cross entropy loss: {}".format(replay_training_cnt, losses[5])),



                        # checkpoint model
                        if update % self.config.CHECKPOINT_INTERVAL == 0:
                            self.save_checkpoint(
                                f"ckpt.{count_checkpoints}.pth",
                                dict(step=count_steps),
                            )
                            count_checkpoints += 1
                            torch.cuda.empty_cache()


            self.envs.close()

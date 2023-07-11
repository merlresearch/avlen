#!/usr/bin/env python3

# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: CC-BY-4.0

import os
import time
from typing import ClassVar, Dict, List
import glob

import torch

from habitat import Config, logger
from ss_baselines.common.tensorboard_utils import TensorboardWriter
from ss_baselines.common.utils import poll_checkpoint_folder


class BaseTrainer:
    r"""Generic trainer class that serves as a base template for more
    specific trainer classes like RL trainer, SLAM or imitation learner.
    Includes only the most basic functionality.
    """

    supported_tasks: ClassVar[List[str]]

    def train(self) -> None:
        raise NotImplementedError

    def eval(self) -> None:
        raise NotImplementedError

    def save_checkpoint(self, file_name) -> None:
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        raise NotImplementedError


class BaseRLTrainer(BaseTrainer):
    r"""Base trainer class for RL trainers. Future RL-specific
    methods should be hosted here.
    """
    device: torch.device
    config: Config
    video_option: List[str]
    _flush_secs: int

    def __init__(self, config: Config):
        super().__init__()
        assert config is not None, "needs config file to initialize trainer"
        self.config = config
        self._flush_secs = 30

    @property
    def flush_secs(self):
        return self._flush_secs

    @flush_secs.setter
    def flush_secs(self, value: int):
        self._flush_secs = value

    def train(self) -> None:
        raise NotImplementedError

    def eval(self, eval_interval=1, prev_ckpt_ind=-1, use_last_ckpt=False) -> None:
        r"""Main method of trainer evaluation. Calls _eval_checkpoint() that
        is specified in Trainer class that inherits from BaseRLTrainer

        Returns:
            None
        """
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if "tensorboard" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.TENSORBOARD_DIR) > 0
            ), "Must specify a tensorboard directory for video display"
        if "disk" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.VIDEO_DIR) > 0
            ), "Must specify a directory for storing videos on disk"

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            # eval last checkpoint in the folder
            if use_last_ckpt:
                models_paths = list(
                    filter(os.path.isfile, glob.glob(self.config.EVAL_CKPT_PATH_DIR + "/*"))
                )
                models_paths.sort(key=os.path.getmtime)
                self.config.defrost()
                self.config.EVAL_CKPT_PATH_DIR = models_paths[-1]
                self.config.freeze()

            if os.path.isfile(self.config.EVAL_CKPT_PATH_DIR):
                # evaluate single checkpoint
                result = self._eval_checkpoint(self.config.EVAL_CKPT_PATH_DIR, writer)
                return result
            else:
                # evaluate multiple checkpoints in order
                while True:
                    current_ckpt = None
                    while current_ckpt is None:
                        current_ckpt = poll_checkpoint_folder(
                            self.config.EVAL_CKPT_PATH_DIR, prev_ckpt_ind, eval_interval
                        )
                        time.sleep(2)  # sleep for 2 secs before polling again
                    logger.info(f"=======current_ckpt: {current_ckpt}=======")
                    prev_ckpt_ind += eval_interval
                    self._eval_checkpoint(
                        checkpoint_path=current_ckpt,
                        writer=writer,
                        checkpoint_index=prev_ckpt_ind
                    )

    def _setup_eval_config(self, checkpoint_config: Config) -> Config:
        r"""Sets up and returns a merged config for evaluation. Config
            object saved from checkpoint is merged into config file specified
            at evaluation time with the following overwrite priority:
                  eval_opts > ckpt_opts > eval_cfg > ckpt_cfg
            If the saved config is outdated, only the eval config is returned.

        Args:
            checkpoint_config: saved config from checkpoint.

        Returns:
            Config: merged config for eval.
        """

        config = self.config.clone()

        ckpt_cmd_opts = checkpoint_config.CMD_TRAILING_OPTS
        eval_cmd_opts = config.CMD_TRAILING_OPTS

        try:
            config.merge_from_other_cfg(checkpoint_config)
            config.merge_from_other_cfg(self.config)
            config.merge_from_list(ckpt_cmd_opts)
            config.merge_from_list(eval_cmd_opts)
        except KeyError:
            logger.info("Saved config is outdated, using solely eval config")
            config = self.config.clone()
            config.merge_from_list(eval_cmd_opts)

        config.TASK_CONFIG.SIMULATOR.AGENT_0.defrost()
        config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = self.config.SENSORS
        config.freeze()

        return config

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint. Trainer algorithms should
        implement this.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        raise NotImplementedError

    def save_checkpoint(self, file_name) -> None:
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        raise NotImplementedError

    @staticmethod
    def _pause_envs(
        envs_to_pause,
        envs,
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
        test_em_goal=None,
        test_em_option=None,
        test_em_vln=None,
        test_em_vln_dialog=None,
        descriptor_pred_gt=None,
    ):

        track_query_new = track_query
        track_query_count_new = track_query_count
        track_perf_new = track_perf
        track_agent_step_new = track_agent_step

        # pausing self.envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)
                if test_em_goal is not None:
                    test_em_goal.pop_at(idx)
                if test_em_option is not None:
                    test_em_option.pop_at(idx)
                if test_em_vln is not None:
                    test_em_vln.pop_at(idx)
                if test_em_vln_dialog is not None:
                    test_em_vln_dialog.pop_at(idx)
                if descriptor_pred_gt is not None:
                    descriptor_pred_gt.pop(idx)

            # indexing along the batch dimensions
            test_recurrent_hidden_states = test_recurrent_hidden_states[
                :, state_index
            ]
            not_done_masks = not_done_masks[state_index]
            current_episode_reward = current_episode_reward[state_index]
            prev_actions = prev_actions[state_index]


            track_query_new = []
            track_query_count_new = []
            track_perf_new = []
            track_agent_step_new = []
            for idx in state_index:
                track_query_new.append(track_query[idx])
                track_query_count_new.append(track_query_count[idx])
                track_perf_new.append(track_perf[idx])
                track_agent_step_new.append(track_agent_step[idx])

            for k, v in batch.items():
                batch[k] = v[state_index]

            rgb_frames = [rgb_frames[i] for i in state_index]
            text_frames = [text_frames[i] for i in state_index]

        if test_em_option is None:
            return (
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                test_em_vln,
                test_em_vln_dialog,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
                text_frames,
                track_query_new,
                track_query_count_new,
                track_perf_new,
                track_agent_step_new
            )
        else:
            return (
                envs,
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
                track_query_new,
                track_query_count_new,
                track_perf_new,
                track_agent_step_new,
            )

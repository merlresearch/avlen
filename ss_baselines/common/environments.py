#!/usr/bin/env python3

# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: CC-BY-4.0

r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@baseline_registry.register_env(name="myEnv")` for reusability
"""

from typing import Optional, Type
import logging
import sys
import numpy as np
import math

import habitat
from habitat import Config, Dataset
from ss_baselines.common.baseline_registry import baseline_registry


def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.
    Args:
        env_name: name of the environment.
    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return baseline_registry.get_env(env_name)


@baseline_registry.register_env(name="AudioNavRLEnv")
class AudioNavRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG

        self._previous_target_distance = None
        self._previous_action = None
        self._new_episode = True
        self._episode_distance_covered = None
        self._success_distance = self._core_env_config.TASK.SUCCESS_DISTANCE
        # ------
        self.query_num = None
        self.env_idx = None
        self.is_queried = None
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._previous_action = None
        self._new_episode = True
        # ------
        self.query_num = 0
        self.is_queried = False
        self.env_idx = None
        observations = super().reset()
        logging.debug(super().current_episode)

        self._previous_target_distance = self.habitat_env.current_episode.info[
            "geodesic_distance"
        ]
        return observations

    #------

    def set_query_num(self, query_num):
        self.query_num = query_num

    def set_idx(self, env_idx):
        self.env_idx = env_idx

    def set_is_queried(self, is_queried):
        self.is_queried = is_queried

    def set_constraint_reward(self, cons_reward):
        self.cons_reward = cons_reward

    def compute_oracle_actions(self):
        return self._env.sim.compute_oracle_actions()

    def step(self, *args, **kwargs):
        self._new_episode = False
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = 0

        if self._rl_config.WITH_TIME_PENALTY:
            reward += self._rl_config.SLACK_REWARD

        if self._rl_config.WITH_DISTANCE_REWARD:
            current_target_distance = self._distance_target()
            # if current_target_distance < self._previous_target_distance:
            reward += (self._previous_target_distance - current_target_distance) * self._rl_config.DISTANCE_REWARD_SCALE
            self._previous_target_distance = current_target_distance

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD
            logging.debug('Reaching goal!')

        if self._rl_config.WITH_QUERY_CONSTRAINT and self.is_queried:
            if self.query_num<=self._rl_config.NUM_TOTAL_QUERY:
                if self._rl_config.SOFT_QUERY_REWARD:
                    # reward += (self.query_num/self._rl_config.NUM_TOTAL_QUERY)*(max(self._rl_config.QUERY_REWARD/2, self._rl_config.SOFT_QUERY_REWARD_MAX))
                    # taking max as negative value
                    reward += (self.query_num/self._rl_config.NUM_TOTAL_QUERY)*(math.exp(-self._rl_config.NUM_TOTAL_QUERY)+self._rl_config.QUERY_REWARD)
            else:
                reward += math.exp(-self.query_num)+self._rl_config.QUERY_REWARD

            if self._rl_config.CONSECUTIVE_CONSTRAINT_REWARD:
                reward += self.cons_reward

        if self._rl_config.WITH_DISTANCE_CONSTRAINT and self.is_queried:
            if self._rl_config.DISTANCE_DISTRIBUTION_TYPE=='gaussian':
                samp_val = np.random.normal(self._rl_config.MEAN,self._rl_config.SD, 1)[0]
            if self._rl_config.DISTANCE_DISTRIBUTION_TYPE=='beta':
                samp_val = np.random.beta(self._rl_config.ALPHA, self._rl_config.BETA, 1)[0]
            if current_target_distance*samp_val <=3:
                reward += self._rl_config.QUERY_REWARD_DISTANCE


        return reward

    # ------------------------------
    def agent_state(self):
        position = self._env.sim.get_agent_state().position.tolist()
        rotation = self._env.sim.get_agent_state().rotation.tolist()
        receiver_node = self._env.sim._receiver_position_index
        source_node = self._env.sim._source_position_index
        scene = self._env.sim._current_scene.split('/')[-2]
        view = self._env.sim._node2view[scene][str(receiver_node)]
        # dialog pretraining
        sub_instr = self._env.sim._sub_instr
        current_target_distance = self._distance_target()

        appro_next_points = []
        if receiver_node in self._env.sim.paths.keys():
            if source_node in self._env.sim.paths[receiver_node].keys():
                gt_next_points = self._env.sim.paths[receiver_node][source_node][:4]

                for point in gt_next_points:
                    appro_next_points.append(self._env.sim._node2view[scene][str(point)])

        return position, rotation, scene, receiver_node, view, appro_next_points, sub_instr, current_target_distance

    def _distance_target(self):
        current_position = self._env.sim.get_agent_state().position.tolist()
        target_positions = [goal.position for goal in self._env.current_episode.goals]
        distance = self._env.sim.geodesic_distance(
            current_position, target_positions
        )
        return distance

    def _episode_success(self):
        if (
                self._env.task.is_stop_called
                # and self._distance_target() < self._success_distance
                and self._env.sim.reaching_goal
        ):
            return True
        return False

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    # for data collection
    def get_current_episode_id(self):
        return self.habitat_env.current_episode.episode_id

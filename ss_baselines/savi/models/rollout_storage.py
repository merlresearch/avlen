#!/usr/bin/env python3

# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: CC-BY-4.0

import pdb
from collections import defaultdict
import torch
import sys


class RolloutStorage:
    r"""Class for storing rollout information for RL trainers.

    """

    def __init__(
        self,
        num_steps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        use_external_memory,
        external_memory_size,
        external_memory_capacity,
        external_memory_option_size,
        external_memory_option_capacity,
        external_memory_vln_size,
        external_memory_vln_capacity,
        external_memory_dim_goal,
        external_memory_dim_vln,
        external_memory_dim_option,
        external_memory_dim_dialog,
        num_recurrent_layers=1,
        max_dialog_len=20,
        query_count_emb_size = 32,
        use_state_memory = False,
    ):
        '''
        print('external_memory_size', external_memory_size)
        print('external_memory_capacity', external_memory_capacity)
        print('external_memory_vln_size', external_memory_vln_size)
        print('external_memory_vln_capacity', external_memory_vln_capacity)
        external_memory_size 300
        external_memory_capacity 150
        external_memory_vln_size 3
        external_memory_vln_capacity 3
        '''

        self.num_steps = num_steps
        self.observations = {}
        self.num_envs = num_envs
        for sensor in observation_space.spaces:
            self.observations[sensor] = torch.zeros(
                num_steps + 1,
                num_envs,
                *observation_space.spaces[sensor].shape
            )
        # This is introduced to handle an edge case where the
        # SMT policy returns -1 for num_recurrent_layers.
        if num_recurrent_layers < 1:
            num_recurrent_layers = 1
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1,
            num_recurrent_layers,
            num_envs,
            recurrent_hidden_state_size,
        )

        # -----------------
        # tensor for token index with maxlength 20 for each dialog
        # assign dialog when query action is triggered
        self.all_dialog = torch.zeros(num_steps, num_envs, max_dialog_len, dtype=torch.long)

        ## query count encoder
        self.query_state = torch.zeros(num_steps, num_envs, query_count_emb_size)
        self.last_query_info = torch.zeros(num_steps, num_envs, query_count_emb_size)
        self.agent_step = torch.zeros(num_steps, num_envs)

        self.rewards = torch.zeros(num_steps, num_envs, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_envs, 1)
        self.returns = torch.zeros(num_steps + 1, num_envs, 1)

        self.action_log_probs = torch.zeros(num_steps, num_envs, 1)
        if action_space.__class__.__name__ == "ActionSpace":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]


        self.actions = torch.zeros(num_steps, num_envs, action_shape)
        self.actions_option = torch.zeros(num_steps, num_envs, action_shape)


        self.prev_actions = torch.zeros(num_steps + 1, num_envs, action_shape)
        # self.prev_actions_option = torch.zeros(num_steps + 1, num_envs, action_shape)
        if action_space.__class__.__name__ == "ActionSpace":
            self.actions = self.actions.long()
            self.prev_actions = self.prev_actions.long()
            # self.prev_actions_option = self.prev_actions_option.long()
            self.actions_option = self.actions_option.long()

        self.masks = torch.zeros(num_steps + 1, num_envs, 1)
        self.masks_vln = torch.zeros(num_steps + 1, num_envs, 1)

        # ----------------- for dialog pretraining
        # o_action: action predicted by oracle
        self.o_actions = torch.zeros(num_steps, num_envs)
        # o_mask: masking actions that won't be used for training
        self.o_masks = torch.zeros((num_steps, num_envs), dtype=torch.long)
        self.ucnt_gt = torch.zeros((num_steps, num_envs), dtype=torch.long)
        self.rl_masks = torch.zeros((num_steps, num_envs), dtype=torch.long)
        # action_prob: agent action probabilities
        self.action_probs = torch.zeros(num_steps, num_envs, 4)

        self.use_external_memory = use_external_memory
        self.use_state_memory = use_state_memory

        self.em_size = external_memory_size
        self.em_capacity = external_memory_capacity

        self.em_option_size = external_memory_option_size
        self.em_option_capacity = external_memory_option_capacity

        self.em_vln_size = external_memory_vln_size
        self.em_vln_capacity = external_memory_vln_capacity

        self.em_dim_goal = external_memory_dim_goal
        self.em_dim_vln = external_memory_dim_vln
        self.em_dim_dialog = external_memory_dim_dialog
        self.em_dim_option = external_memory_dim_option

        # sys.exit()
        # This is kept outside for backward compatibility with _collect_rollout_step
        # em_mask for goal policy
        # em_vln_mask for vln policy
        self.em_masks = torch.zeros(num_steps + 1, num_envs, self.em_size)
        self.em_vln_masks = torch.zeros(num_steps + 1, num_envs, self.em_vln_size)

        if use_external_memory:
            # self.em will be used for goal based policy
            # self.em_vln will be used for vln policy
            self.em = ExternalMemory(
                num_envs, self.em_size, self.em_capacity,
                self.em_dim_goal, num_copies=num_steps + 1, num_steps=num_steps,
            )
            self.em_option = ExternalMemory(
                num_envs, self.em_option_size, self.em_option_capacity,
                self.em_dim_option, num_copies=num_steps + 1, num_steps=num_steps,
            )
            # self.em_vln will be used for vln policy
            self.em_vln = ExternalMemory(
                num_envs, self.em_vln_size, self.em_vln_capacity,
                self.em_dim_vln, num_copies=num_steps + 1, num_steps=num_steps,
            )
        else:
            self.em = None
            self.em_vln = None
            self.em_option = None

        if use_state_memory:
            # em_vln_dialog will be used for vln policy
            self.em_vln_dialog = ExternalMemory(
                num_envs, self.em_vln_size, self.em_vln_capacity,
                self.em_dim_dialog, num_copies=num_steps + 1, num_steps=num_steps,
            )
        else:
            self.em_vln_dialog = None

        self.num_steps = num_steps
        self.step = 0
        self.env_id = 0

    def to(self, device):
        for sensor in self.observations:
            self.observations[sensor] = self.observations[sensor].to(device)

        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.actions_option = self.actions_option.to(device)
        self.prev_actions = self.prev_actions.to(device)
        # self.prev_actions_options = self.prev_actions_option.to(device)
        self.masks = self.masks.to(device)
        self.masks_vln = self.masks_vln.to(device)
        self.em_masks = self.em_masks.to(device) # for goal based policy
        self.em_vln_masks = self.em_vln_masks.to(device) # for vln policy
        # dialog pretraining
        self.o_masks = self.o_masks.to(device)
        self.ucnt_gt = self.ucnt_gt.to(device)
        self.rl_masks = self.rl_masks.to(device)
        self.o_actions = self.o_actions.to(device)
        self.action_probs = self.action_probs.to(device)
        # -----------
        self.all_dialog = self.all_dialog.to(device)
        self.query_state = self.query_state.to(device)
        self.last_query_info = self.last_query_info.to(device)
        self.agent_step = self.agent_step.to(device)
        if self.use_external_memory:
            self.em.to(device)
            self.em_vln.to(device)
            self.em_option.to(device)
        if self.use_state_memory:
            self.em_vln_dialog.to(device)

    def insert(
        self,
        observations,
        recurrent_hidden_states,
        actions,
        actions_option,
        action_log_probs,
        value_preds,
        rewards,
        not_done_masks,
        not_done_masks_vln,
        em_features,
        em_features_option,
        em_features_vln,
        # --------
        em_features_dialog,
        # -----
        all_dialog,
        # ----- dialog pretraining
        o_action,
        o_mask,
        rl_masks,
        ucnt_gt,
        action_prob,
        query_state,
        last_query_info,
        agent_step,

    ):
        for sensor in observations:
            self.observations[sensor][self.step + 1].copy_(
                observations[sensor]
            )
        self.recurrent_hidden_states[self.step + 1].copy_(
            recurrent_hidden_states
        )
        # ------
        self.all_dialog[self.step].copy_(
            all_dialog
        )
        self.query_state[self.step].copy_(
           query_state
        )
        self.last_query_info[self.step].copy_(
           last_query_info
        )
        self.agent_step[self.step].copy_(
           agent_step
        )

        # ------ dialog pretraining
        if o_action != None:
            self.o_masks[self.step].copy_(o_mask)
            self.ucnt_gt[self.step].copy_(ucnt_gt)
            self.rl_masks[self.step].copy_(rl_masks)
            self.o_actions[self.step].copy_(o_action)
            self.action_probs[self.step].copy_(action_prob)

        self.actions[self.step].copy_(actions)
        if actions_option != None:
            self.actions_option[self.step].copy_(actions_option)
            # self.prev_actions_option[self.step + 1].copy_(actions_option)

        self.prev_actions[self.step + 1].copy_(actions)

        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(not_done_masks)
        self.masks_vln[self.step + 1].copy_(not_done_masks_vln)
        if self.use_external_memory:
            self.em.insert(em_features, not_done_masks)
            self.em_masks[self.step + 1].copy_(self.em.masks)

            self.em_option.insert(em_features_option, not_done_masks)

            self.em_vln.insert(em_features_vln, not_done_masks_vln)
            self.em_vln_masks[self.step + 1].copy_(self.em_vln.masks)

        if self.use_state_memory:
            self.em_vln_dialog.insert(em_features_dialog, not_done_masks_vln)
            self.em_vln_masks[self.step + 1].copy_(self.em_vln.masks)

        self.step = self.step + 1


    def insert_replay(
        self,
        observations,
        recurrent_hidden_states,
        actions,
        actions_option,
        action_log_probs,
        value_preds,
        rewards,
        not_done_masks,
        not_done_masks_vln,
        em_features,
        # --------
        em_features_dialog,
        # -----
        all_dialog,
        # ----- dialog pretraining
        o_action,
        o_mask,
        action_prob,
        query_state,
        agent_step,
    ):

        for sensor in observations:
            self.observations[sensor][:self.num_steps, self.env_id].copy_(
                observations[sensor]
            )
        self.recurrent_hidden_states[:self.num_steps,:, self.env_id,:].copy_(
            recurrent_hidden_states
        )
        # ------
        self.all_dialog[:self.num_steps, self.env_id].copy_(
            all_dialog
        )
        self.query_state[:self.num_steps, self.env_id].copy_(
           query_state
        )
        self.agent_step[:self.num_steps, self.env_id].copy_(
           agent_step
        )

        # ------ dialog pretraining
        if o_action != None:
            self.o_masks[:self.num_steps, self.env_id].copy_(o_mask)
            self.o_actions[:self.num_steps, self.env_id].copy_(o_action)
            self.action_probs[:self.num_steps, self.env_id].copy_(action_prob)

        self.actions[:self.num_steps, self.env_id].copy_(actions)
        if actions_option != None:
            self.actions_option[:self.num_steps, self.env_id].copy_(actions_option)

        self.prev_actions[:self.num_steps, self.env_id].copy_(actions)
        self.action_log_probs[:self.num_steps, self.env_id].copy_(action_log_probs)
        self.value_preds[:self.num_steps, self.env_id].copy_(value_preds)
        self.rewards[:self.num_steps, self.env_id].copy_(rewards)
        self.masks[:self.num_steps, self.env_id].copy_(not_done_masks)
        self.masks_vln[:self.num_steps, self.env_id].copy_(not_done_masks_vln)
        if self.use_external_memory:
            self.em.insert_replay(em_features)
            self.em_masks[:self.num_steps, self.env_id,:].copy_(self.em.masks_replay[:,self.env_id,:])

            self.em_vln.insert_replay(em_features)
            self.em_vln_masks[:self.num_steps, self.env_id,:].copy_(self.em_vln.masks_replay[:,self.env_id,:])

        if self.use_state_memory:
            self.em_vln_dialog.insert_replay(em_features_dialog)
            self.em_vln_masks[:self.num_steps, self.env_id, :].copy_(self.em_vln.masks_replay[:,self.env_id,:])

        self.env_id = self.env_id + 1
        self.step = self.num_steps


    def after_update(self):
        for sensor in self.observations:
            self.observations[sensor][0].copy_(
                self.observations[sensor][self.step]
            )

        self.recurrent_hidden_states[0].copy_(
            self.recurrent_hidden_states[self.step]
        )
        # all_dialog, query_state, agent_step are not copied since they comes with episode information (track query)

        self.masks[0].copy_(self.masks[self.step])
        self.masks_vln[0].copy_(self.masks_vln[self.step])
        self.prev_actions[0].copy_(self.prev_actions[self.step])
        if self.use_external_memory:
            self.em_masks[0].copy_(self.em_masks[self.step])
            self.em_vln_masks[0].copy_(self.em_vln_masks[self.step])
        if self.use_state_memory:
            self.em_vln_masks[0].copy_(self.em_vln_masks[self.step])
        self.step = 0

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[self.step] = next_value
            gae = 0
            for step in reversed(range(self.step)):
                delta = (
                    self.rewards[step]
                    + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                    - self.value_preds[step]
                )
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[self.step] = next_value
            for step in reversed(range(self.step)):
                self.returns[step] = (
                    self.returns[step + 1] * gamma * self.masks[step + 1]
                    + self.rewards[step]
                )

    def dialog_batching(self):

        observations_batch = defaultdict(list)
        recurrent_hidden_states_batch = []
        actions_batch = []
        prev_actions_batch = []
        value_preds_batch = []
        return_batch = []
        masks_batch = []
        old_action_log_probs_batch = []
        # --------------
        all_dialog = []
        # --------------
        query_state = []
        agent_step = []

        if self.use_external_memory:
            em_store_batch = []
            em_masks_batch = []

            em_vln_store_batch = []
            em_vln_masks_batch = []
        else:
            em_store_batch = None
            em_masks_batch = None

            em_vln_store_batch = None
            em_vln_masks_batch = None

        if self.use_state_memory:
            em_vln_dialog_store_batch = []
            if not self.use_external_memory:
                em_vln_masks_batch = []
        else:
            em_vln_dialog_store_batch = None
            if not self.use_external_memory:
                em_vln_masks_batch = None


        for ind in range(self.num_envs):

            for sensor in self.observations:
                observations_batch[sensor].append(
                    self.observations[sensor][: self.step, ind]
                )
            recurrent_hidden_states_batch.append(
                self.recurrent_hidden_states[0, :, ind]
            )

            # -------------------
            all_dialog.append(self.all_dialog[: self.step, ind])

            # -------------------
            query_state.append(self.query_state[: self.step, ind])
            agent_step.append(self.agent_step[: self.step, ind])

            actions_batch.append(self.actions[: self.step, ind])
            prev_actions_batch.append(self.prev_actions[: self.step, ind])
            value_preds_batch.append(self.value_preds[: self.step, ind])
            return_batch.append(self.returns[: self.step, ind])
            masks_batch.append(self.masks[: self.step, ind])
            old_action_log_probs_batch.append(
                self.action_log_probs[: self.step, ind]
            )
            if self.use_external_memory:
                em_store_batch.append(self.em.memory[:, : self.step, ind])
                em_masks_batch.append(self.em_masks[: self.step, ind])

                em_vln_store_batch.append(self.em_vln.memory[:, : self.step, ind])
                em_vln_masks_batch.append(self.em_vln_masks[: self.step, ind])

            if self.use_state_memory:
                em_vln_dialog_store_batch.append(self.em_vln_dialog.memory[:, : self.step, ind])
                if not self.use_external_memory:
                    em_vln_masks_batch.append(self.em_vln_masks[: self.step, ind])

        T, N = self.step, self.num_envs

        # These are all tensors of size (T, N, -1)
        for sensor in observations_batch:
            observations_batch[sensor] = torch.stack(
                observations_batch[sensor], 1
            )

        # ------------------
        all_dialog = torch.stack(all_dialog,1)
        # ------------------

        query_state = torch.stack(query_state,1)
        agent_step= torch.stack(agent_step,1)

        actions_batch = torch.stack(actions_batch, 1)
        prev_actions_batch = torch.stack(prev_actions_batch, 1)
        value_preds_batch = torch.stack(value_preds_batch, 1)
        return_batch = torch.stack(return_batch, 1)
        masks_batch = torch.stack(masks_batch, 1)
        old_action_log_probs_batch = torch.stack(
            old_action_log_probs_batch, 1
        )
        if self.use_external_memory:
            # This is a (em_size, num_steps, bs, em_dim) tensor
            em_store_batch = torch.stack(em_store_batch, 2)
            # This is a (num_steps, bs, em_size) tensor
            em_masks_batch = torch.stack(em_masks_batch, 1)

            em_vln_store_batch = torch.stack(em_vln_store_batch, 2)
            # This is a (num_steps, bs, em_size) tensor
            em_vln_masks_batch = torch.stack(em_vln_masks_batch, 1)

        if self.use_state_memory:
            # This is a (em_size, num_steps, bs, em_dim) tensor
            em_vln_dialog_store_batch = torch.stack(em_vln_dialog_store_batch, 2)
            if not self.use_external_memory:
                # This is a (num_steps, bs, em_size) tensor
                em_vln_masks_batch = torch.stack(em_vln_masks_batch, 1)

        # States is just a (num_recurrent_layers, N, -1) tensor
        recurrent_hidden_states_batch = torch.stack(
            recurrent_hidden_states_batch, 1
        )

        # Flatten the (T, N, ...) tensors to (T * N, ...)
        for sensor in observations_batch:
            observations_batch[sensor] = self._flatten_helper(
                T, N, observations_batch[sensor]
            )
        # --------
        all_dialog_batch = self._flatten_helper(T, N, all_dialog)
        # --------
        query_state_batch = self._flatten_helper(T, N, query_state)
        agent_step_batch = self._flatten_helper(T, N, agent_step)

        actions_batch = self._flatten_helper(T, N, actions_batch)
        prev_actions_batch = self._flatten_helper(T, N, prev_actions_batch)
        value_preds_batch = self._flatten_helper(T, N, value_preds_batch)
        return_batch = self._flatten_helper(T, N, return_batch)
        masks_batch = self._flatten_helper(T, N, masks_batch)
        old_action_log_probs_batch = self._flatten_helper(
            T, N, old_action_log_probs_batch
        )
        if self.use_external_memory:
            em_store_batch = em_store_batch.view(-1, T * N, self.em_dim_goal)
            em_masks_batch = self._flatten_helper(T, N, em_masks_batch)

            em_vln_store_batch = em_vln_store_batch.view(-1, T * N, self.em_dim_vln)
            em_vln_masks_batch = self._flatten_helper(T, N, em_vln_masks_batch)

        if self.use_state_memory:
            em_vln_dialog_store_batch = em_vln_dialog_store_batch.view(-1, T * N, self.em_dim_dialog)
            if not self.use_external_memory:
                em_vln_masks_batch = self._flatten_helper(T, N, em_vln_masks_batch)

        return (
                observations_batch,
                recurrent_hidden_states_batch,
                actions_batch,
                prev_actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                old_action_log_probs_batch,
                em_store_batch,
                em_vln_store_batch,
                em_vln_dialog_store_batch,
                em_masks_batch,
                em_vln_masks_batch,
                # -----------------
                all_dialog_batch,
                # -----------------
                # query_state_batch,
                agent_step_batch,
                # -------------------- to provide num_step and num_env
                self.num_steps,
                self.num_envs,
            )


    def recurrent_generator(self, advantages, num_mini_batch):

        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "Trainer requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(num_processes, num_mini_batch)
        )

        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)

        # print(num_processes, num_envs_per_batch)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = defaultdict(list)

            recurrent_hidden_states_batch = []
            actions_batch = []
            actions_option_batch = []
            prev_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            rl_masks_batch = []
            ucnt_gt_batch = []
            # -------
            all_dialog = []
            # -------
            query_state = []
            last_query_info = []
            agent_step = []

            if self.use_external_memory:
                em_store_batch = []
                em_option_store_batch = []
                em_masks_batch = []

                em_vln_store_batch = []
                em_vln_masks_batch = []
            else:
                em_store_batch = None
                em_option_store_batch = None
                em_masks_batch = None

                em_vln_store_batch = None
                em_vln_masks_batch = None

            if self.use_state_memory:
                em_vln_store_dialog_batch = []
                if not self.use_external_memory:
                    em_vln_masks_batch = []
            else:
                em_vln_store_dialog_batch = None
                if not self.use_external_memory:
                    em_vln_masks_batch = None

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]

                for sensor in self.observations:
                    observations_batch[sensor].append(
                        self.observations[sensor][: self.step, ind]
                    )

                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0, :, ind]
                    # why only the index 0?
                )

                # -------------------
                all_dialog.append(self.all_dialog[: self.step, ind])
                # -------------------
                query_state.append(self.query_state[: self.step, ind])
                last_query_info.append(self.last_query_info[: self.step, ind])
                agent_step.append(self.agent_step[: self.step, ind])

                actions_batch.append(self.actions[: self.step, ind])
                actions_option_batch.append(self.actions_option[: self.step, ind])
                prev_actions_batch.append(self.prev_actions[: self.step, ind])
                # prev_actions_option_batch.append(self.prev_actions_option[: self.step, ind])
                value_preds_batch.append(self.value_preds[: self.step, ind])
                return_batch.append(self.returns[: self.step, ind])
                masks_batch.append(self.masks[: self.step, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[: self.step, ind]
                )
                adv_targ.append(advantages[: self.step, ind])
                rl_masks_batch.append(self.rl_masks[: self.step, ind])
                ucnt_gt_batch.append(self.ucnt_gt[: self.step, ind])
                if self.use_external_memory:
                    em_store_batch.append(self.em.memory[:, : self.step, ind])
                    em_option_store_batch.append(self.em_option.memory[:, : self.step, ind])
                    em_masks_batch.append(self.em_masks[: self.step, ind])

                    em_vln_store_batch.append(self.em_vln.memory[:, : self.step, ind])
                    em_vln_masks_batch.append(self.em_vln_masks[: self.step, ind])

                if self.use_state_memory:
                    em_vln_store_dialog_batch.append(self.em_vln_dialog.memory[:, : self.step, ind])
                    if not self.use_external_memory:
                        em_vln_masks_batch.append(self.em_vln_masks[: self.step, ind])

            T, N = self.step, num_envs_per_batch

            # print(T, N)

            # These are all tensors of size (T, N, -1)
            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(
                    observations_batch[sensor], 1
                )

            # ------------------
            all_dialog = torch.stack(all_dialog,1)
            # ------------------
            query_state = torch.stack(query_state,1)
            last_query_info = torch.stack(last_query_info,1)
            agent_step = torch.stack(agent_step,1)

            actions_batch = torch.stack(actions_batch, 1)
            actions_option_batch = torch.stack(actions_option_batch, 1)
            prev_actions_batch = torch.stack(prev_actions_batch, 1)
            # prev_actions_option_batch = torch.stack(prev_actions_option_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1
            )
            adv_targ = torch.stack(adv_targ, 1)
            rl_masks_batch = torch.stack(rl_masks_batch, 1)
            ucnt_gt_batch = torch.stack(ucnt_gt_batch, 1)
            if self.use_external_memory:
                # This is a (em_size, num_steps, bs, em_dim) tensor
                em_store_batch = torch.stack(em_store_batch, 2)
                em_option_store_batch = torch.stack(em_option_store_batch, 2)
                # This is a (num_steps, bs, em_size) tensor
                em_masks_batch = torch.stack(em_masks_batch, 1)

                em_vln_store_batch = torch.stack(em_vln_store_batch, 2)
                # This is a (num_steps, bs, em_size) tensor
                em_vln_masks_batch = torch.stack(em_vln_masks_batch, 1)

            if self.use_state_memory:
                em_vln_store_dialog_batch = torch.stack(em_vln_store_dialog_batch, 2)
                if not self.use_external_memory:# This is a (num_steps, bs, em_size) tensor
                    em_vln_masks_batch = torch.stack(em_vln_masks_batch, 1)

            # States is just a (num_recurrent_layers, N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1
            )

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            for sensor in observations_batch:
                observations_batch[sensor] = self._flatten_helper(
                    T, N, observations_batch[sensor]
                )
            # --------
            all_dialog_batch = self._flatten_helper(T, N, all_dialog)
            # --------
            query_state_batch = self._flatten_helper(T, N, query_state)
            last_query_info_batch = self._flatten_helper(T, N, last_query_info)
            agent_step_batch = self._flatten_helper(T, N, agent_step)

            actions_batch = self._flatten_helper(T, N, actions_batch)
            actions_option_batch = self._flatten_helper(T, N, actions_option_batch)
            prev_actions_batch = self._flatten_helper(T, N, prev_actions_batch)
            # prev_actions_option_batch = self._flatten_helper(T, N, prev_actions_option_batch)
            value_preds_batch = self._flatten_helper(T, N, value_preds_batch)
            return_batch = self._flatten_helper(T, N, return_batch)
            masks_batch = self._flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = self._flatten_helper(
                T, N, old_action_log_probs_batch
            )
            adv_targ = self._flatten_helper(T, N, adv_targ)
            rl_masks_batch = self._flatten_helper(T, N, rl_masks_batch)
            ucnt_gt_batch = self._flatten_helper(T, N, ucnt_gt_batch)
            if self.use_external_memory:
                em_store_batch = em_store_batch.view(-1, T * N, self.em_dim_goal)
                em_option_store_batch = em_option_store_batch.view(-1, T * N, self.em_dim_option)
                em_masks_batch = self._flatten_helper(T, N, em_masks_batch)

                em_vln_store_batch = em_vln_store_batch.view(-1, T * N, self.em_dim_vln)
                em_vln_masks_batch = self._flatten_helper(T, N, em_vln_masks_batch)

            if self.use_state_memory:
                em_vln_store_dialog_batch = em_vln_store_dialog_batch.view(-1, T * N, self.em_dim_dialog)
                if not self.use_external_memory:
                    em_vln_masks_batch = self._flatten_helper(T, N, em_vln_masks_batch)

            yield (
                observations_batch,
                recurrent_hidden_states_batch,
                actions_batch,
                actions_option_batch,
                prev_actions_batch,
                # prev_actions_option_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                rl_masks_batch,
                ucnt_gt_batch,
                em_store_batch,
                em_option_store_batch,
                em_vln_store_batch,
                em_vln_store_dialog_batch,
                em_masks_batch,
                em_vln_masks_batch,
                # -----------------
                all_dialog_batch,
                # -----------------
                query_state_batch,
                last_query_info_batch,
                agent_step_batch,
            )

    @staticmethod
    def _flatten_helper(t: int, n: int, tensor: torch.Tensor) -> torch.Tensor:
        r"""Given a tensor of size (t, n, ..), flatten it to size (t*n, ...).

        Args:
            t: first dimension of tensor.
            n: second dimension of tensor.
            tensor: target tensor to be flattened.

        Returns:
            flattened tensor of size (t*n, ...)
        """
        return tensor.view(t * n, *tensor.size()[2:])

    @property
    def external_memory_goal(self):
        return self.em.memory

    @property
    def external_memory_option(self):
        return self.em_option.memory

    @property
    def external_memory_masks(self):
        return self.em_masks

    @property
    def external_memory_goal_idx(self):
        return self.em.idx

    @property
    def external_memory_option_idx(self):
        return self.em_option.idx

    @property
    def external_memory_vln(self):
        return self.em_vln.memory

    @property
    def external_memory_vln_idx(self):
        return self.em_vln.idx

    @property
    def external_memory_vln_masks(self):
        return self.em_vln_masks

    @property
    def external_memory_vln_dialog(self):
        return self.em_vln_dialog.memory

    @property
    def external_memory_vln_dialog_idx(self):
        return self.em_vln_dialog.idx


class RolloutStorageVariedExternal(RolloutStorage):
    r"""Class for storing rollout information for RL trainers.
    For the case of external_memory, it maintains a vector of em_idxes instead of just
    one. This allows more versatility in storage for hierarchical policy training.
    """

    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if self.use_external_memory:
            num_envs = self.rewards.size(1)
            self.em = ExternalMemoryVaried(
                num_envs, self.em_size, self.em_capacity,
                self.em_dim, num_copies = self.num_steps + 1
            )
        else:
            self.em = None

    def get_em_store_and_mask(self, i, si, ei):
        """Given a start and end idx for a process, obtain the corresponding memory
        values and the masks at the current step.

        Outputs:
            feats - (L, num_steps+1, feat_dim)
            masks - (L, )
        """
        assert(ei != si)
        if ei > si:
            feats = self.em.memory[si:ei, :, i]
            masks = self.em_masks[self.step, i, si:ei]
        else:
            feats = torch.cat([self.em.memory[si:, :, i], self.em.memory[:ei, :, i]], 0)
            masks = torch.cat([self.em_masks[self.step, i, si:],
                               self.em_masks[self.step, i, :ei]], 0)
        return feats, masks


class ExternalMemory:
    def __init__(self, num_envs, total_size, capacity, dim, num_copies=1, num_steps=150):
        r"""An external memory that keeps track of observations over time.

        Inputs:
            num_envs - number of parallel environments
            capacity - total capacity of the memory per episode
            total_size - capacity + additional buffer size for rollout updates
            dim - size of observations
            num_copies - number of copies of the data to maintain for efficient training
        """
        self.num_envs = num_envs
        self.total_size = total_size
        self.capacity = capacity
        self.dim = dim
        self.masks = torch.zeros(num_envs, self.total_size)
        self.masks_replay = torch.zeros(num_steps, num_envs, self.total_size)
        self.memory = torch.zeros(self.total_size, num_copies, num_envs, self.dim)
        self.idx = 0
        self.env_id = 0
        self.num_steps = num_steps


    def insert(self, em_features, not_done_masks):

        # Update memory storage and add new memory as a valid entry
        self.memory[self.idx].copy_(em_features.unsqueeze(0))
        # Account for overflow capacity
        capacity_overflow_flag = self.masks.sum(1) == self.capacity
        assert(not torch.any(self.masks.sum(1) > self.capacity))
        self.masks[capacity_overflow_flag, self.idx - self.capacity] = 0.0
        self.masks[:, self.idx] = 1.0
        # Mask out the entire memory for the next observation if episode done
        self.masks *= not_done_masks
        self.idx = (self.idx + 1) % self.total_size

    def insert_replay(self, em_features):
        self.masks_replay[:,self.env_id,:] = 0.0
        # Update memory storage and add new memory as a valid entry
        self.memory[:em_features.size()[0],:,self.env_id,:].copy_(em_features.unsqueeze(1))
        assert(not torch.any(self.masks.sum(1) > self.capacity))
        for i in range(1,self.num_steps):
            self.masks_replay[i, self.env_id, :i] = 1.0
        # Mask out the entire memory for the next observation if episode done
        self.env_id = (self.env_id + 1) % self.num_envs


    def pop_at(self, idx):
        self.masks = torch.cat([self.masks[:idx, :], self.masks[idx+1:, :]], dim=0)
        self.memory = torch.cat([self.memory[:, :, :idx, :], self.memory[:, :, idx+1:, :]], dim=2)

    def to(self, device):
        self.masks = self.masks.to(device)
        self.memory = self.memory.to(device)


class ExternalMemoryVaried(ExternalMemory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_envs = self.memory.size(2)
        self.idx = torch.zeros(self.num_envs).long()

    def insert(self, em_features, not_done_masks):
        # Update memory storage and add new memory as a valid entry
        for i in range(self.num_envs):
            # To account for cases where memory is too small to accommodate all data.
            feat_size = min(em_features[i].size(0), self.capacity)
            em_feats_i = em_features[i][-feat_size:]
            si = self.idx[i].item()
            ei = (si + feat_size) % self.total_size
            self._write_em_store_and_mask(i, si, ei, em_feats_i)
            self.idx[i] = ei
        # Mask out the entire memory for the next observation if episode done
        self.masks *= not_done_masks

    def _write_em_store_and_mask(self, i, si, ei, feats):
        """Given features for process i and the corresponding start and end idxes,
        write the features and the masks. This needs to take care of circular wrapping.

        Args:
            i - process ID
            si - start index between 0 and total_size - 1
            ei - end index between 0 and total_size - 1
            feats - (L, em_dim)

        Note: ei may be less than si if the indices wrap around to the beginning of
        the em_store buffer.
        """
        if ei == si:
            assert(self.total_size == 1 and ei == 0)
            ei += 1
        if ei > si:
            self.memory[si:ei, :, i, :].copy_(feats.unsqueeze(1))
            self.masks[i, si:ei] = 1.0
        else:
            mi = self.total_size - si
            self.memory[si:, :, i, :].copy_(feats[:mi].unsqueeze(1))
            self.memory[:ei, :, i, :].copy_(feats[mi:].unsqueeze(1))
            self.masks[i, si:] = 1.0
            self.masks[i, :ei] = 1.0
        # Handle capacity overflow
        overflow_value = int(self.masks[i].sum().item()) - self.capacity
        if overflow_value > 0:
            osi = (ei - self.capacity - overflow_value) % self.total_size
            oei = (ei - self.capacity) % self.total_size
            self._write_em_mask(i, osi, oei, 0.0)
        assert(self.masks[i].sum().item() <= self.capacity)

    def _write_em_mask(self, i, si, ei, value):
        if ei > si:
            self.masks[i, si:ei] = value
        else:
            self.masks[i, si:] = value
            self.masks[i, :ei] = value

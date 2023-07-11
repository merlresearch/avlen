#!/usr/bin/env python3

# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: CC-BY-4.0

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import itertools as it

import sys

import pynvml
from pynvml.smi import nvidia_smi
pynvml.nvmlInit()


EPS_PPO = 1e-5


class PPO(nn.Module):
    def __init__(
        self,
        actor_critic,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        value_loss_coef,
        entropy_coef,
        lr=None,
        eps=None,
        max_grad_norm=None,
        use_clipped_value_loss=True,
        use_normalized_advantage=True,
        unct_coef = 0.5,
    ):

        super().__init__()

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.unct_coef = unct_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        self.dialog_optimizer = optim.Adam(actor_critic.parameters(), lr=.00001, eps=eps)# weight_decay= .5e-5
        self.unct_criterion = nn.CrossEntropyLoss()

        self.device = next(actor_critic.parameters()).device

        self.use_normalized_advantage = use_normalized_advantage
        # dialog pretraining
        weight_type = 'balanced' # None: no weight, scale: independent of count, count: based on count

        if weight_type == None:
            self.dialog_criterion = nn.CrossEntropyLoss()
        elif weight_type == 'balanced':
            weight_scale = torch.tensor([0, .33, .33, .33]).to(self.device)
            self.dialog_criterion = nn.CrossEntropyLoss(weight=weight_scale)
        elif weight_type == 'scale':
            weight_scale = torch.tensor([0, .25, .45, .3]).to(self.device)
            self.dialog_criterion = nn.CrossEntropyLoss(weight=weight_scale)
        elif weight_type == 'count':
            weight_count = torch.tensor([0.0, 0.09, 0.61, 0.29]).to(self.device)
            self.dialog_criterion = nn.CrossEntropyLoss(weight=weight_count)

        self.seq_criterion = nn.CrossEntropyLoss()


    def forward(self, *x):
        raise NotImplementedError

    def get_advantages(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        if not self.use_normalized_advantage:
            return advantages

        return (advantages - advantages.mean()) / (advantages.std() + EPS_PPO)


    # dialog pretraining
    def update_dialog(self, rollouts):
        (
             obs_batch,
             recurrent_hidden_states_batch,
             actions_batch,
             prev_actions_batch,
             _,
             _,
             masks_batch,
             _,
             _,
             external_memory,
             external_memory_dialog,
             external_memory_masks,
             external_memory_vln_masks,
             all_dialog_batch,
             agent_step_state_batch,
             num_steps,
             num_envs,
        ) = rollouts.dialog_batching()

        (
            _,
            _,
            _,
            _,
            _,
            _,
            logits,
        ) = self.actor_critic.evaluate_actions_dialog(
            obs_batch,
            recurrent_hidden_states_batch,
            prev_actions_batch,
            masks_batch,
            actions_batch,
            external_memory,
            external_memory_dialog,
            external_memory_vln_masks,
            all_dialog_batch,
            agent_step_state_batch.detach(),
        )

        mask_flat = rollouts.o_masks.view(-1)
        o_actions = rollouts.o_actions.view(-1)[torch.nonzero(mask_flat).squeeze(-1)]
        logits = logits[torch.nonzero(mask_flat).squeeze(-1),:]
        assert logits.size()[0]==o_actions.size()[0], 'logits.size(): {}, o_actions.size(): {}'.format(logits.size(), o_actions.size())
        dialog_loss = self.dialog_criterion(logits, o_actions.long())



        self.dialog_optimizer.zero_grad()
        self.before_backward(dialog_loss)
        dialog_loss.backward()
        self.dialog_optimizer.step()

        return dialog_loss


    def update(self, rollouts):
        advantages = self.get_advantages(rollouts)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        unct_loss_epoch = 0

        values_debug_epoch = 0
        return_batch_debug_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch
            )

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    actions_option_batch,
                    prev_actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                    rl_masks_batch,
                    unct_gt_batch,
                    _,
                    external_memory,
                    _,
                    _,
                    external_memory_masks,
                    external_memory_masks_vln,
                    _,
                    query_state_batch,
                    last_query_info,
                    _,
                ) = sample

                (
                    values,
                    unct,
                    action_log_probs,
                    dist_entropy,
                    _,
                    _,
                    _,
                ) = self.actor_critic.evaluate_actions_option(
                    obs_batch,
                    recurrent_hidden_states_batch,
                    prev_actions_batch,
                    masks_batch,
                    actions_option_batch,
                    external_memory,
                    external_memory_masks,
                    query_state_batch,
                    last_query_info,
                )

                ratio = torch.exp(
                    action_log_probs - old_action_log_probs_batch
                )


                surr1 = ratio * adv_targ * rl_masks_batch.unsqueeze(1)


                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    * adv_targ * rl_masks_batch.unsqueeze(1)
                )

                action_loss = -torch.min(surr1, surr2).sum()/torch.sum(rl_masks_batch)# .mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (
                        values - value_preds_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch
                    ).pow(2)
                    value_loss = (
                        0.5
                        * torch.max(value_losses, value_losses_clipped).mean()
                    )
                    values_debug = values.mean()
                    return_batch_debug = return_batch.mean()

                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                unct_loss = self.unct_criterion(unct, unct_gt_batch.long())

                self.optimizer.zero_grad()
                total_loss = (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                    + self.unct_coef * unct_loss
                )

                self.before_backward(total_loss)
                total_loss.backward()
                self.after_backward(total_loss)

                self.before_step()
                self.optimizer.step()
                self.after_step()



                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                unct_loss_epoch += unct_loss.item()

                values_debug_epoch += values_debug.item()
                return_batch_debug_epoch += return_batch_debug.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        unct_loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, values_debug_epoch, return_batch_debug_epoch, unct_loss_epoch

    def before_backward(self, loss):
        pass

    def after_backward(self, loss):
        pass

    def before_step(self):
        nn.utils.clip_grad_norm_(
            self.actor_critic.parameters(), self.max_grad_norm
        )

    def after_step(self):
        pass

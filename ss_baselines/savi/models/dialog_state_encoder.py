#!/usr/bin/env python3

# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: CC-BY-4.0

import torch
import torch.nn as nn
from torch import Tensor
import math

from typing import Tuple, Optional


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Args:
            # x: Tensor, shape [seq_len, batch_size, embedding_dim]
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
            y: Tensor, shape [batch_size]
        """
        # x = x + self.pe[:x.size(0)]
        x = x + self.pe[y.long(),0,:].unsqueeze(0)
        return self.dropout(x)


class DialogStateEncoder(nn.Module):
    """
    The core Scene Memory Transformer block from https://arxiv.org/abs/1903.03878
    """
    def __init__(
        self,
        input_size: int,
        nhead: int = 8,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "relu",
        # pose_indices: Optional[Tuple[int, int]] = None,
        pretraining: bool = False
    ):
        r"""A Transformer for encoding the dialog state in RL and decoding features based on
        the state found using SMT StateEncoder.

        Supports masking the hidden state during various timesteps in the forward pass

        Args:
            input_size: The input size of the SMT
            nhead: The number of encoding and decoding attention heads
            num_encoder_layers: The number of encoder layers
            num_decoder_layers: The number of decoder layers
            dim_feedforward: The hidden size of feedforward layers in the transformer
            dropout: The dropout value after each attention layer
            activation: The activation to use after each linear layer
        """

        super().__init__()
        self._input_size = input_size
        self._nhead = nhead
        self._num_encoder_layers = num_encoder_layers
        self._num_decoder_layers = num_decoder_layers
        self._dim_feedforward = dim_feedforward
        self._dropout = dropout
        self._activation = activation
        self._pretraining = pretraining

        self.fusion_encoder = nn.Sequential(
            nn.Linear(input_size, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_feedforward),
        )

        self.dialog_transformer = nn.Transformer(
            d_model=dim_feedforward,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )

        self.pos_encode = PositionalEncoding(d_model=self._dim_feedforward, dropout=0.0, max_len=100)  # update later


    def _convert_masks_to_transformer_format(self, memory_masks):
        r"""The memory_masks is a FloatTensor with
            -   zeros for invalid locations, and
            -   ones for valid locations.

        The required format is a BoolTensor with
            -   True for invalid locations, and
            -   False for valid locations
        """
        return (1 - memory_masks) > 0

    def single_forward(self, x, memory_state, memory_masks, d_emb, agent_step, goal=None):
        r"""Forward for a non-sequence input

        Args:
            x: (N, input_size) Tensor
            memory: The memory of encoded observations in the episode. It is a
                (M, N, input_size) Tensor.
            memory_state: Should be the memory of state encodings: (M,N,dim_feedforward)
            memory_masks: The masks indicating the set of valid memory locations
                for the current observations. It is a (N, M) Tensor.
            # i don't need seperate memory mask for dialog, i can use the same mask from state encoder
            goal: (N, goal_dims) Tensor (optional)
            step_emb: (N, step_emb_dims) (not option) (step_emb_dims requires to be same as dim_feedforward?) (not used yet)
        """
        # If memory_masks is all zeros for a data point, x_att will be NaN.
        # In these cases, just set memory_mask to ones and replace x_att with x.
        # all_zeros_mask = (memory_masks.sum(dim=1) == 0).float().unsqueeze(1)
        # memory_masks = 1.0 * all_zeros_mask + memory_masks * (1 - all_zeros_mask)

        memory_masks = torch.cat([memory_masks, torch.ones([memory_masks.shape[0], 1], device=memory_masks.device)],
                                     dim=1)
        memory_state = torch.cat([memory_state, x.unsqueeze(0)])
        M, bs = memory_state.shape[:2]

        if d_emb != None:
            memory_state = torch.cat([memory_state, d_emb.unsqueeze(0).repeat(M,1,1)], dim=-1)
            memory_state = self.fusion_encoder(memory_state.view(M*bs, -1)).view(M, bs, -1)

        memory_state = self.pos_encode(memory_state, agent_step)

        # Transformer operations
        t_masks = self._convert_masks_to_transformer_format(memory_masks)
        assert goal is not None
        x_att_dialog = self.dialog_transformer(
                memory_state,
                goal.unsqueeze(0), # step_emb.unsqueeze(0),
                src_key_padding_mask=t_masks,
                memory_key_padding_mask=t_masks,
            )[-1]


        return x_att_dialog

    @property
    def hidden_state_size(self):
        return self._dim_feedforward

    def forward(self, x_att, memory_state, memory_mask, d_emb, agent_step, *args, **kwargs):
        """
        Single input case:
            Inputs:
                x_att - (N, em_dim)
                memory_state - (M, N, em_dim)
                memory_masks - (N, M)
                d_emb - (N, dim_emb_size)
        Sequential input case:
            Inputs:
                x - (T*N, em_dim)
                memory - (M, N, em_dim)
                memory_masks - (T*N, M)
        """
        assert x_att.size(0) == memory_state.size(1)
        return self.single_forward(x_att, memory_state, memory_mask, d_emb, agent_step, *args, **kwargs)

    def _encode_pose(self, agent_pose, memory_pose):
        """
        Args:
            agent_pose: (bs, 4) Tensor containing x, y, heading, time
            memory_pose: (M, bs, 4) Tensor containing x, y, heading, time
        """
        agent_xyh, agent_t = agent_pose[..., :3], agent_pose[..., 3:4]
        memory_xyh, memory_t = memory_pose[..., :3], memory_pose[..., 3:4]

        # Compute relative poses
        agent_rel_xyh = self._compute_relative_pose(agent_xyh, agent_xyh)
        agent_rel_pose = torch.cat([agent_rel_xyh, agent_t], -1)
        memory_rel_xyh = self._compute_relative_pose(agent_xyh.unsqueeze(0), memory_xyh)
        memory_rel_pose = torch.cat([memory_rel_xyh, memory_t], -1)

        # Format pose
        agent_pose_formatted = self._format_pose(agent_rel_pose)
        memory_pose_formatted = self._format_pose(memory_rel_pose)

        # Encode pose
        agent_pose_encoded = self.pose_encoder(agent_pose_formatted)
        M, bs = memory_pose_formatted.shape[:2]
        memory_pose_encoded = self.pose_encoder(
            memory_pose_formatted.view(M * bs, -1)
        ).view(M, bs, -1)

        return agent_pose_encoded, memory_pose_encoded

    def _compute_relative_pose(self, pose_a, pose_b):
        """
        Computes the pose_b - pose_a in pose_a's coordinates.

        Args:
            pose_a: (..., 3) Tensor of x in meters, y in meters, heading in radians
            pose_b: (..., 3) Tensor of x in meters, y in meters, heading in radians

        Expected pose format:
            At the origin, x is forward, y is rightward,
            and heading is measured from x to -y.
        """
        # Negate the heading to get angle from x to y
        heading_a = -pose_a[..., 2]
        heading_b = -pose_b[..., 2]
        # Compute relative pose
        r_ab = torch.norm(pose_a[..., :2] - pose_b[..., :2], dim=-1)
        phi_ab = torch.atan2(pose_b[..., 1] - pose_a[..., 1], pose_b[..., 0] - pose_a[..., 0])
        phi_ab = phi_ab - heading_a
        x_ab = r_ab * torch.cos(phi_ab)
        y_ab = r_ab * torch.sin(phi_ab)
        heading_ab = heading_b - heading_a
        # Normalize angles to lie between -pi to pi
        heading_ab = torch.atan2(torch.sin(heading_ab), torch.cos(heading_ab))
        # Negate the heading to get angle from x to -y
        heading_ab = -heading_ab

        return torch.stack([x_ab, y_ab, heading_ab], -1) # (..., 3)

    def _format_pose(self, pose):
        """
        Args:
            pose: (..., 4) Tensor containing x, y, heading, time
        """
        x, y, heading, time = torch.unbind(pose, dim=-1)
        cos_heading, sin_heading = torch.cos(heading), torch.sin(heading)
        e_time = torch.exp(-time)
        formatted_pose = torch.stack([x, y, cos_heading, sin_heading, e_time], -1)
        return formatted_pose

    @property
    def pose_indices(self):
        return self._pose_indices

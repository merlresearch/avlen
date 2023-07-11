#!/usr/bin/env python3

# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: CC-BY-4.0

import abc
import logging
import itertools

import torch
import torch.nn as nn
from torchsummary import summary
import sys
import clip

from soundspaces.tasks.nav import PoseSensor, SpectrogramSensor, LocationBelief, CategoryBelief, Category
from ss_baselines.common.utils import CategoricalNet
from ss_baselines.av_nav.models.rnn_state_encoder import RNNStateEncoder
from ss_baselines.savi.models.visual_cnn import VisualCNN
from ss_baselines.savi.models.audio_cnn import AudioCNN
from ss_baselines.savi.models.smt_state_encoder import SMTStateEncoder
from ss_baselines.savi.models.dialog_state_encoder import DialogStateEncoder
from ss_baselines.savi.models.smt_cnn import SMTCNN

# --------------------------------------
from ss_baselines.savi.models.dialog_encoder import DialogEncoder

import pynvml
from pynvml.smi import nvidia_smi
pynvml.nvmlInit()

DUAL_GOAL_DELIMITER = ','


class Policy(nn.Module):
    def __init__(self, net, dim_actions, dim_actions_option=2):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions
        self.dim_actions_option = dim_actions_option

        self.action_distribution_option = CategoricalNet(
            self.net.output_size, self.dim_actions_option
        )

        self.action_distribution_goal = CategoricalNet(
            self.net.output_size, self.dim_actions
        )

        self.action_distribution_vln = CategoricalNet(
            self.net.output_size, self.dim_actions
        )

        self.critic_goal = CriticHead(self.net.output_size)
        self.critic_option = CriticHead(self.net.output_size)
        self.uncertainty_option = CriticHead2(self.net.output_size)
        self.critic_vln = CriticHead(self.net.output_size)
        '''
        # for sequential
        self.seq_proj = nn.Linear(25, 3125)
        '''

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        ext_memory,
        ext_memory_masks,
        deterministic=False,

    ):
        features, rnn_hidden_states, ext_memory_feats = self.net(
            observations, rnn_hidden_states, prev_actions, masks, ext_memory, ext_memory_masks
        )


        distribution, _ = self.action_distribution_goal(features)
        value = self.critic_goal(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states, ext_memory_feats, distribution.probs

    def act_option(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        ext_memory,
        ext_memory_masks,
        query_state,
        last_query_info,
        deterministic=False,

    ):
        features, rnn_hidden_states, ext_memory_feats = self.net(
            observations, rnn_hidden_states, prev_actions, masks, ext_memory, ext_memory_masks, query_state, last_query_info,
        )


        distribution, _ = self.action_distribution_option(features)
        value = self.critic_option(features)
        unct = self.uncertainty_option(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, unct, action, action_log_probs, rnn_hidden_states, ext_memory_feats, distribution.probs


    def act_dialog(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        ext_memory,
        ext_memory_dialog,
        ext_memory_masks,
        all_dialog,
        agent_step,
        deterministic=False,
        without_dialog=False,
    ):
        if without_dialog:
            all_dialog=None

        features, rnn_hidden_states, ext_memory_feats, ext_memory_dialog_feats  = self.net(
            observations, rnn_hidden_states, prev_actions, masks, ext_memory, ext_memory_dialog, ext_memory_masks, all_dialog, agent_step
        )

        distribution, _ = self.action_distribution_vln(features)
        value = self.critic_vln(features)

        # action = distribution.mode()
        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states, ext_memory_feats, ext_memory_dialog_feats, distribution.probs

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks, ext_memory, ext_memory_masks):
        features, _, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks, ext_memory, ext_memory_masks
        )
        return self.critic_goal(features)

    def get_value_option(self, observations, rnn_hidden_states, prev_actions, masks, ext_memory, ext_memory_masks, query_state, last_query_info):
        features, _, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks, ext_memory, ext_memory_masks, query_state, last_query_info,
        )
        return self.critic_option(features)


    def get_value_dialog(self, observations, rnn_hidden_states, prev_actions, masks, ext_memory, ext_memory_masks, all_dialog): # , query_state
        features, _, _, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks, ext_memory, ext_memory_dialog, ext_memory_masks, all_dialog
        )
        return self.critic_vln(features)

    def evaluate_actions(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        action,
        ext_memory,
        ext_memory_masks,

    ):

        features, rnn_hidden_states, ext_memory_feats = self.net(
            observations, rnn_hidden_states, prev_actions,
            masks, ext_memory, ext_memory_masks
        )
        distribution, _ = self.action_distribution_goal(features)
        value = self.critic_goal(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states, ext_memory_feats

    def evaluate_actions_option(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        action,
        ext_memory,
        ext_memory_masks,
        query_state,
        last_query_info,

    ):
        # print(f"Used GPU memory inside evaluate before self.net- {nvidia_smi.getInstance().DeviceQuery('memory.used')}")

        features, rnn_hidden_states, ext_memory_feats = self.net(
            observations, rnn_hidden_states, prev_actions,
            masks, ext_memory, ext_memory_masks, query_state, last_query_info,
        )
        # print(f"Used GPU memory inside evaluate after self.net- {nvidia_smi.getInstance().DeviceQuery('memory.used')}")

        distribution, _ = self.action_distribution_option(features)
        value = self.critic_option(features)
        unct = self.uncertainty_option(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, unct, action_log_probs, distribution_entropy, rnn_hidden_states, ext_memory_feats, distribution.probs


    def evaluate_actions_dialog(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        action,
        ext_memory,
        ext_memory_dialog,
        ext_memory_masks,
        all_dialog,
        # query_state,
        # agent_step_state
        agent_step,
        without_dialog=False,
    ):
        '''
        features, rnn_hidden_states, ext_memory_feats, ext_memory_dialog_feats = self.net(
            observations, rnn_hidden_states, prev_actions,
            masks, ext_memory, ext_memory_dialog, ext_memory_masks, all_dialog, # query_state,
            agent_step_state
        )
        '''
        if without_dialog:
            all_dialog=None

        features, rnn_hidden_states, ext_memory_feats, ext_memory_dialog_feats = self.net(
            observations, rnn_hidden_states, prev_actions,
            masks, ext_memory, ext_memory_dialog, ext_memory_masks, all_dialog, # query_state,
            agent_step
        )
        distribution, logit = self.action_distribution_vln(features)
        # value = self.critic(features)
        value = None

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states, ext_memory_feats, ext_memory_dialog_feats, logit


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)

class CriticHead2(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 2)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)

class AudioNavBaselinePolicy(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        goal_sensor_uuid,
        hidden_size=512,
        extra_rgb=False,
        use_mlp_state_encoder=False
    ):
        super().__init__(
            AudioNavBaselineNet(
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
                extra_rgb=extra_rgb,
                use_mlp_state_encoder=use_mlp_state_encoder
            ),
            action_space.n,
        )


class AudioNavSMTPolicy(Policy):
    def __init__(self, observation_space, action_space, hidden_size=128, **kwargs):
        super().__init__(
            AudioNavSMTNet(
                observation_space,
                action_space,
                hidden_size=hidden_size,
                **kwargs
            ),
            action_space.n
        )


class AudioNavDialogPolicy(Policy):
    def __init__(self, observation_space, action_space, hidden_size=128, **kwargs):
        super().__init__(
            AudioNavDialogNet(
                observation_space,
                action_space,
                hidden_size=hidden_size,
                **kwargs
            ),
            action_space.n
        )

class AudioNavOptionPolicy(Policy):
    def __init__(self, observation_space, action_space, hidden_size=128, **kwargs):
        super().__init__(
            AudioNavOptionNet(
                observation_space,
                action_space,
                hidden_size=hidden_size,
                **kwargs
            ),
            2 # number of action
        )

class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class AudioNavBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, observation_space, hidden_size, goal_sensor_uuid, extra_rgb=False, use_mlp_state_encoder=False):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        self._audiogoal = False
        self._pointgoal = False
        self._n_pointgoal = 0
        self._label = 'category' in observation_space.spaces

        # for goal descriptors
        self._use_label_belief = False
        self._use_location_belief = False
        self._use_mlp_state_encoder = use_mlp_state_encoder

        if DUAL_GOAL_DELIMITER in self.goal_sensor_uuid:
            goal1_uuid, goal2_uuid = self.goal_sensor_uuid.split(DUAL_GOAL_DELIMITER)
            self._audiogoal = self._pointgoal = True
            self._n_pointgoal = observation_space.spaces[goal1_uuid].shape[0]
        else:
            if 'pointgoal_with_gps_compass' == self.goal_sensor_uuid:
                self._pointgoal = True
                self._n_pointgoal = observation_space.spaces[self.goal_sensor_uuid].shape[0]
            else:
                self._audiogoal = True

        self.visual_encoder = VisualCNN(observation_space, hidden_size, extra_rgb)
        if self._audiogoal:
            if 'audiogoal' in self.goal_sensor_uuid:
                audiogoal_sensor = 'audiogoal'
            elif 'spectrogram' in self.goal_sensor_uuid:
                audiogoal_sensor = 'spectrogram'
            self.audio_encoder = AudioCNN(observation_space, hidden_size, audiogoal_sensor)

        rnn_input_size = (0 if self.is_blind else self._hidden_size) + \
                         (self._n_pointgoal if self._pointgoal else 0) + \
                         (self._hidden_size if self._audiogoal else 0) + \
                         (observation_space.spaces['category'].shape[0] if self._label else 0) + \
                         (observation_space.spaces[CategoryBelief.cls_uuid].shape[0] if self._use_label_belief else 0) + \
                         (observation_space.spaces[LocationBelief.cls_uuid].shape[0] if self._use_location_belief else 0)
        if not self._use_mlp_state_encoder:
            self.state_encoder = RNNStateEncoder(rnn_input_size, self._hidden_size)
        else:
            self.state_encoder = nn.Linear(rnn_input_size, self._hidden_size)

        if not self.visual_encoder.is_blind:
            summary(self.visual_encoder.cnn, self.visual_encoder.input_shape, device='cpu')
        if self._audiogoal:
            audio_shape = observation_space.spaces[audiogoal_sensor].shape
            summary(self.audio_encoder.cnn, (audio_shape[2], audio_shape[0], audio_shape[1]), device='cpu')

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        if self._use_mlp_state_encoder:
            return 1
        else:
            return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks, ext_memory=None, ext_memory_masks=None):
        x = []

        if self._pointgoal:
            x.append(observations[self.goal_sensor_uuid.split(DUAL_GOAL_DELIMITER)[0]])
        if self._audiogoal:
            x.append(self.audio_encoder(observations))
        if not self.is_blind:
            x.append(self.visual_encoder(observations))
        if self._label:
            x.append(observations['category'].to(device=x[0].device))

        if self._use_label_belief:
            x.append(observations[CategoryBelief.cls_uuid])
        if self._use_location_belief:
            x.append(observations[LocationBelief.cls_uuid])

        x1 = torch.cat(x, dim=1)
        if self._use_mlp_state_encoder:
            x2 = self.state_encoder(x1)
            rnn_hidden_states1 = x2
        else:
            x2, rnn_hidden_states1 = self.state_encoder(x1, rnn_hidden_states, masks)

        assert not torch.isnan(x2).any().item()

        return x2, rnn_hidden_states1, None

    def get_features(self, observations, prev_actions):
        x = []

        if self._pointgoal:
            x.append(observations[self.goal_sensor_uuid.split(DUAL_GOAL_DELIMITER)[0]])
        if self._audiogoal:
            x.append(self.audio_encoder(observations))
        if not self.is_blind:
            x.append(self.visual_encoder(observations))
        if self._label:
            x.append(observations['category'].to(device=x[0].device))

        if self._use_label_belief:
            x.append(observations[CategoryBelief.cls_uuid])
        if self._use_location_belief:
            x.append(observations[LocationBelief.cls_uuid])

        x = torch.cat(x, dim=1)

        return x


class AudioNavSMTNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN. Implements the
    policy from Scene Memory Transformer: https://arxiv.org/abs/1903.03878
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size=128,
        use_pretrained=False,
        pretrained_path='',
        use_belief_as_goal=True,
        use_label_belief=True,
        use_location_belief=True,
        use_belief_encoding=False,
        normalize_category_distribution=False,
        use_category_input=False,
        **kwargs
    ):
        super().__init__()
        self._use_action_encoding = True
        self._use_residual_connection = False
        self._use_belief_as_goal = use_belief_as_goal
        self._use_label_belief = use_label_belief
        self._use_location_belief = use_location_belief
        self._hidden_size = hidden_size
        self._action_size = action_space.n
        self._use_belief_encoder = use_belief_encoding
        self._normalize_category_distribution = normalize_category_distribution
        self._use_category_input = use_category_input

        assert SpectrogramSensor.cls_uuid in observation_space.spaces
        self.goal_encoder = AudioCNN(observation_space, 128, SpectrogramSensor.cls_uuid)
        audio_feature_dims = 128

        self.visual_encoder = SMTCNN(observation_space)
        if self._use_action_encoding:
            self.action_encoder = nn.Linear(self._action_size, 16)
            action_encoding_dims = 16
        else:
            action_encoding_dims = 0
        nfeats = self.visual_encoder.feature_dims + action_encoding_dims + audio_feature_dims

        if self._use_category_input:
            nfeats += 21

        # Add pose observations to the memory
        assert PoseSensor.cls_uuid in observation_space.spaces
        if PoseSensor.cls_uuid in observation_space.spaces:
            pose_dims = observation_space.spaces[PoseSensor.cls_uuid].shape[0]
            # Specify which part of the memory corresponds to pose_dims
            pose_indices = (nfeats, nfeats + pose_dims)
            nfeats += pose_dims
        else:
            pose_indices = None

        self._feature_size = nfeats


        self.smt_state_encoder = SMTStateEncoder(
            nfeats,
            dim_feedforward=hidden_size,
            pose_indices=pose_indices,
            **kwargs
        )

        self.state_size = self.smt_state_encoder.hidden_state_size
        if self._use_residual_connection:
            self.state_size += self._feature_size

        if self._use_belief_encoder:
            self.belief_encoder = nn.Linear(self._hidden_size, self._hidden_size)

        if use_pretrained:
            assert(pretrained_path != '')
            self.pretrained_initialization(pretrained_path)

        self.train()

    @property
    def memory_dim(self):
        return self._feature_size


    @property
    def output_size(self):
        size = self.smt_state_encoder.hidden_state_size
        if self._use_residual_connection:
            size += self._feature_size
        return size

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return -1

    def forward(self, observations, rnn_hidden_states, prev_actions, masks, ext_memory, ext_memory_masks):
        x = self.get_features(observations, prev_actions)

        if self._use_belief_as_goal:
            belief = torch.zeros((x.shape[0], self._hidden_size), device=x.device)
            if self._use_label_belief:
                if self._normalize_category_distribution:
                    belief[:, :21] = nn.functional.softmax(observations[CategoryBelief.cls_uuid], dim=1)
                else:
                    belief[:, :21] = observations[CategoryBelief.cls_uuid]

            if self._use_location_belief:
                belief[:, 21:23] = observations[LocationBelief.cls_uuid]

            if self._use_belief_encoder:
                belief = self.belief_encoder(belief)
        else:
            belief = None

        x_att = self.smt_state_encoder(x, ext_memory, ext_memory_masks, goal=belief)
        if self._use_residual_connection:
            x_att = torch.cat([x_att, x], 1)

        return x_att, rnn_hidden_states, x


    def _get_one_hot(self, actions):
        if actions.shape[1] == self._action_size:
            return actions
        else:
            N = actions.shape[0]
            actions_oh = torch.zeros(N, self._action_size, device=actions.device)
            actions_oh.scatter_(1, actions.long(), 1)
            return actions_oh

    def pretrained_initialization(self, path):
        logging.info(f'AudioNavSMTNet ===> Loading pretrained model from {path}')
        state_dict = torch.load(path)['state_dict']
        cleaned_state_dict = {
            k[len('actor_critic.net.'):]: v for k, v in state_dict.items()
            if 'actor_critic.net.' in k
        }
        self.load_state_dict(cleaned_state_dict, strict=False)

    def freeze_encoders(self):
        """Freeze goal, visual and fusion encoders. Pose encoder is not frozen."""
        logging.info(f'AudioNavSMTNet ===> Freezing goal, visual, fusion encoders!')
        params_to_freeze = []
        params_to_freeze.append(self.goal_encoder.parameters())
        params_to_freeze.append(self.visual_encoder.parameters())
        if self._use_action_encoding:
            params_to_freeze.append(self.action_encoder.parameters())
        for p in itertools.chain(*params_to_freeze):
            p.requires_grad = False

    def set_eval_encoders(self):
        """Sets the goal, visual and fusion encoders to eval mode."""
        self.goal_encoder.eval()
        self.visual_encoder.eval()

    def get_features(self, observations, prev_actions):
        x = []
        x.append(self.visual_encoder(observations))
        x.append(self.action_encoder(self._get_one_hot(prev_actions)))
        x.append(self.goal_encoder(observations))
        if self._use_category_input:
            x.append(observations[Category.cls_uuid])

        x.append(observations[PoseSensor.cls_uuid])

        x = torch.cat(x, dim=1)

        return x




class AudioNavDialogNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN. Implements the
    policy from Scene Memory Transformer: https://arxiv.org/abs/1903.03878
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size=128,
        use_pretrained=False,
        pretrained_path='',
        use_belief_as_goal=True,
        use_label_belief=True,
        use_location_belief=True,
        use_belief_encoding=False,
        normalize_category_distribution=False,
        use_category_input=False,
        # query_count_emb_size = 32,
        # agent_step_emb_size = 128,
        num_steps = 5,
        **kwargs
    ):
        super().__init__()
        self._use_action_encoding = True
        self._use_residual_connection = False
        self._use_belief_as_goal = use_belief_as_goal
        self._use_label_belief = use_label_belief
        self._use_location_belief = use_location_belief
        self._hidden_size = hidden_size
        self._action_size = action_space.n
        self._use_belief_encoder = use_belief_encoding
        self._normalize_category_distribution = normalize_category_distribution
        self._use_category_input = use_category_input
        # self._query_count_emb_size = query_count_emb_size
        # self._agent_step_emb_size = agent_step_emb_size
        self._num_steps = num_steps

        assert SpectrogramSensor.cls_uuid in observation_space.spaces
        self.goal_encoder = AudioCNN(observation_space, 128, SpectrogramSensor.cls_uuid)
        audio_feature_dims = 128

        self.visual_encoder = SMTCNN(observation_space)
        if self._use_action_encoding:
            self.action_encoder = nn.Linear(self._action_size, 16)
            action_encoding_dims = 16
        else:
            action_encoding_dims = 0
        nfeats = self.visual_encoder.feature_dims + action_encoding_dims + audio_feature_dims

        '''
        if self._use_category_input:
            nfeats += 21
        '''


        # Add pose observations to the memory
        # assert PoseSensor.cls_uuid in observation_space.spaces
        if PoseSensor.cls_uuid in observation_space.spaces:
            pose_dims = observation_space.spaces[PoseSensor.cls_uuid].shape[0]
            # Specify which part of the memory corresponds to pose_dims
            pose_indices = (nfeats, nfeats + pose_dims)
            nfeats += pose_dims
            self.pose_dims = pose_dims
        else:
            pose_indices = None
        '''
        # also adding query state
        nfeats += self._query_count_emb_size
        '''
        self._feature_size = nfeats

        self.smt_state_encoder = SMTStateEncoder(
            nfeats,
            dim_feedforward=hidden_size,
            pose_indices=pose_indices,
            **kwargs
        )

        # self.dialog_encoder = DialogEncoder()
        # clip
        self.clip, _ = clip.load("ViT-B/32")  #, device='cpu'
        self.dialog_layer = nn.Linear(self.clip.transformer.width, self._hidden_size)

        self.dialog_state_encoder = DialogStateEncoder(
            hidden_size + self._hidden_size, # hope it matches
            dim_feedforward=hidden_size,
            **kwargs
        )

        self.state_size = self.smt_state_encoder.hidden_state_size
        if self._use_residual_connection:
            self.state_size += self._feature_size
        # self.state_modifier = nn.Linear(self.state_size+self.dialog_encoder.encoded_dim, self.state_size)

        # for agent_step_state
        # self.agent_step_emb = nn.Embedding(self._num_steps, self._agent_step_emb_size)
        # self.agent_step_emb.weight.data.uniform_(-1, 1)

        if self._use_belief_encoder:
            self.belief_encoder = nn.Linear(self._hidden_size, self._hidden_size)

        if use_pretrained:
            assert(pretrained_path != '')
            self.pretrained_initialization(pretrained_path)

        self.train()

    @property
    def memory_dim(self):
        return self._feature_size

    @property
    def output_size(self):
        size = self.smt_state_encoder.hidden_state_size
        if self._use_residual_connection:
            size += self._feature_size
        return size

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return -1

    def forward(self, observations, rnn_hidden_states, prev_actions, masks, ext_memory, ext_memory_dialog, ext_memory_masks, all_dialog, agent_step): # step_emb
        x = self.get_features(observations, prev_actions)
        # ---------------------------
        # x = torch.cat([x, query_state], 1)

        if self._use_belief_as_goal:
            belief = torch.zeros((x.shape[0], self._hidden_size), device=x.device)
            if self._use_label_belief:
                if self._normalize_category_distribution:
                    belief[:, :21] = nn.functional.softmax(observations[CategoryBelief.cls_uuid], dim=1)
                else:
                    belief[:, :21] = observations[CategoryBelief.cls_uuid]

            if self._use_location_belief:
                belief[:, 21:23] = observations[LocationBelief.cls_uuid]

            if self._use_belief_encoder:
                belief = self.belief_encoder(belief)
        else:
            belief = None


        '''
        if self._use_category_input:
            ext_memory = torch.cat([ext_memory[:,:,:272], ext_memory[:,:,293:]],  dim=-1)
        '''
        x_att = self.smt_state_encoder(x, ext_memory, ext_memory_masks, goal=belief)
        '''
        if self._use_residual_connection:  # it needs to be modified for dialog based navigation
            x_att = torch.cat([x_att, x], 1)
        '''
        # -------------------------------------
        # add the dialog part
        # dialog: (B, max_seq_length)
        # dialog_emb: (B, self.dialog_encoder.encoded_dim)


        if all_dialog!=None:
            # dialog_emb = self.dialog_encoder(all_dialog)
            # clip
            with torch.no_grad():
                dialog_emb = self.clip.encode_text(all_dialog).float()
            dialog_emb = self.dialog_layer(dialog_emb)
        else:
            dialog_emb = None

        '''
        # change   # why this??---------
        dialog_encoding = dialog_encoding[:x_att_dialog.size()[0],:]
        '''
        # x_att_dialog = self.dialog_state_encoder(x_att, ext_memory_dialog, ext_memory_masks, dialog_emb, goal=step_emb)  #  should not be goal ------------------
        x_att_dialog = self.dialog_state_encoder(x_att, ext_memory_dialog, ext_memory_masks, dialog_emb, agent_step, goal=belief)

        '''
        if self._use_residual_connection: # it needs to be modified for dialog based navigation
            x_att_dialog = torch.cat([x_att_dialog, x], 1)
        '''

        return x_att_dialog, rnn_hidden_states, x, x_att_dialog
        # return x_att, rnn_hidden_states, x


    def _get_one_hot(self, actions):
        if actions.shape[1] == self._action_size:
            return actions
        else:
            N = actions.shape[0]
            actions_oh = torch.zeros(N, self._action_size, device=actions.device)
            actions_oh.scatter_(1, actions.long(), 1)
            return actions_oh

    def pretrained_initialization(self, path):
        logging.info(f'AudioDialogNet ===> Loading pretrained model from {path}')
        state_dict = torch.load(path)['state_dict']
        cleaned_state_dict = {
            k[len('actor_critic.net.'):]: v for k, v in state_dict.items()
            if 'actor_critic.net.' in k
        }
        self.load_state_dict(cleaned_state_dict, strict=False)

    def freeze_encoders(self):
        """Freeze goal, visual and fusion encoders. Pose encoder is not frozen."""
        logging.info(f'AudioDialogNet ===> Freezing goal, visual, fusion encoders!')
        params_to_freeze = []
        params_to_freeze.append(self.goal_encoder.parameters())
        params_to_freeze.append(self.visual_encoder.parameters())
        if self._use_action_encoding:
            params_to_freeze.append(self.action_encoder.parameters())
        for p in itertools.chain(*params_to_freeze):
            p.requires_grad = False

    def set_eval_encoders(self):
        """Sets the goal, visual and fusion encoders to eval mode."""
        self.goal_encoder.eval()
        self.visual_encoder.eval()

    def get_features(self, observations, prev_actions):
        x = []
        x.append(self.visual_encoder(observations))
        x.append(self.action_encoder(self._get_one_hot(prev_actions)))
        x.append(self.goal_encoder(observations))
        '''
        if self._use_category_input:
            x.append(observations[Category.cls_uuid])
        '''
        x.append(observations[PoseSensor.cls_uuid])

        x = torch.cat(x, dim=1)

        return x


class AudioNavOptionNet(Net):

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size=128,

        # configuration
        use_pretrained=False,
        pretrained_path='',
        use_belief_as_goal=True,
        use_label_belief=True,
        use_location_belief=True,
        use_belief_encoding=False,
        normalize_category_distribution=False,
        use_category_input=False,
        query_count_emb_size = 32,
        **kwargs
    ):
        super().__init__()
        self._use_action_encoding = True
        self._use_residual_connection = False
        self._use_belief_as_goal = use_belief_as_goal
        self._use_label_belief = use_label_belief
        self._use_location_belief = use_location_belief
        self._hidden_size = hidden_size
        self._action_size = action_space.n
        self._use_belief_encoder = use_belief_encoding
        self._normalize_category_distribution = normalize_category_distribution
        self._use_category_input = use_category_input
        self._query_count_emb_size = query_count_emb_size

        assert SpectrogramSensor.cls_uuid in observation_space.spaces
        self.goal_encoder = AudioCNN(observation_space, 128, SpectrogramSensor.cls_uuid)
        audio_feature_dims = 128

        self.visual_encoder = SMTCNN(observation_space)
        if self._use_action_encoding:
            self.action_encoder = nn.Linear(self._action_size, 16)
            action_encoding_dims = 16
        else:
            action_encoding_dims = 0
        nfeats = self.visual_encoder.feature_dims + action_encoding_dims + audio_feature_dims

        if self._use_category_input:
            nfeats += 21

        # Add pose observations to the memory
        assert PoseSensor.cls_uuid in observation_space.spaces
        if PoseSensor.cls_uuid in observation_space.spaces:
            pose_dims = observation_space.spaces[PoseSensor.cls_uuid].shape[0]
            # Specify which part of the memory corresponds to pose_dims
            pose_indices = (nfeats, nfeats + pose_dims)
            nfeats += pose_dims
        else:
            pose_indices = None

        # also adding query state
        nfeats += self._query_count_emb_size

        self._feature_size = nfeats


        self.smt_state_encoder = SMTStateEncoder(
            nfeats,
            dim_feedforward=hidden_size,
            pose_indices=pose_indices,
            use_query_count=False,
            **kwargs
        )
        self.state_size = self.smt_state_encoder.hidden_state_size
        if self._use_residual_connection:
            self.state_size += self._feature_size
        self.policy_selector = nn.Linear(self.state_size, 2)

        # for query state
        self._qcnt_emb = nn.Embedding(2, self._query_count_emb_size)

        if self._use_belief_encoder:
            self.belief_encoder = nn.Linear(self._hidden_size, self._hidden_size)

        if use_pretrained:
            assert(pretrained_path != '')
            self.pretrained_initialization(pretrained_path)

        self.train()

    @property
    def memory_dim(self):
        return self._feature_size

    @property
    def qcnt_emb(self):
        return self._qcnt_emb


    @property
    def output_size(self):
        size = self.smt_state_encoder.hidden_state_size
        if self._use_residual_connection:
            size += self._feature_size
        return size

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return -1

    def forward(self, observations, rnn_hidden_states, prev_actions, masks, ext_memory, ext_memory_masks, query_state, last_query_info):
        x = self.get_features(observations, prev_actions)


        with torch.no_grad():
            x_query = torch.cat([x, query_state], 1)
            # ext_memory_query = torch.cat([ext_memory, query_state.unsqueeze(0).repeat(ext_memory.size()[0], 1, 1)], 2)


        if self._use_belief_as_goal:
            belief = torch.zeros((x.shape[0], self._hidden_size), device=x.device)
            if self._use_label_belief:
                if self._normalize_category_distribution:
                    belief[:, :21] = nn.functional.softmax(observations[CategoryBelief.cls_uuid], dim=1)
                else:
                    belief[:, :21] = observations[CategoryBelief.cls_uuid]

            if self._use_location_belief:
                belief[:, 21:23] = observations[LocationBelief.cls_uuid]

            if self._use_belief_encoder:
                belief = self.belief_encoder(belief)
        else:
            belief = None

        # x_att = self.smt_state_encoder(x_query, ext_memory_query, ext_memory_masks, goal=belief)
        x_att = self.smt_state_encoder(x_query, ext_memory, ext_memory_masks, goal=belief)
        # x_att = self.smt_state_encoder(x, ext_memory, ext_memory_masks, goal=belief)
        if self._use_residual_connection:
            x_att = torch.cat([x_att, x], 1)

        with torch.no_grad():
            x_for_memory = torch.cat([x, last_query_info], 1)

        return x_att, rnn_hidden_states, x_for_memory


    def _get_one_hot(self, actions):
        if actions.shape[1] == self._action_size:
            return actions
        else:
            N = actions.shape[0]
            actions_oh = torch.zeros(N, self._action_size, device=actions.device)
            actions_oh.scatter_(1, actions.long(), 1)
            return actions_oh

    def pretrained_initialization(self, path):
        logging.info(f'AudioOptionNet ===> Loading pretrained model from {path}')
        state_dict = torch.load(path)['state_dict']
        cleaned_state_dict = {
            k[len('actor_critic.net.'):]: v for k, v in state_dict.items()
            if 'actor_critic.net.' in k
        }
        self.load_state_dict(cleaned_state_dict, strict=False)

    def freeze_encoders(self):
        """Freeze goal, visual and fusion encoders. Pose encoder is not frozen."""
        logging.info(f'AudioNavOptionNet ===> Freezing goal, visual, fusion encoders!')
        params_to_freeze = []
        params_to_freeze.append(self.goal_encoder.parameters())
        params_to_freeze.append(self.visual_encoder.parameters())
        if self._use_action_encoding:
            params_to_freeze.append(self.action_encoder.parameters())
        for p in itertools.chain(*params_to_freeze):
            p.requires_grad = False

    def set_eval_encoders(self):
        """Sets the goal, visual and fusion encoders to eval mode."""
        self.goal_encoder.eval()
        self.visual_encoder.eval()

    def get_features(self, observations, prev_actions):
        x = []
        x.append(self.visual_encoder(observations))
        x.append(self.action_encoder(self._get_one_hot(prev_actions)))
        x.append(self.goal_encoder(observations))
        if self._use_category_input:
            x.append(observations[Category.cls_uuid])

        x.append(observations[PoseSensor.cls_uuid])

        x = torch.cat(x, dim=1)

        return x

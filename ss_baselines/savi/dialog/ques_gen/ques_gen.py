# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: CC-BY-4.0

from torchvision import transforms
import os
import torch
import sys
import numpy as np
# sys.path.append('./utils/')
import torch.nn as nn
import torchvision.models as models

from ss_baselines.savi.dialog.ques_gen.models.iq_vln import IQ_VLN
from ss_baselines.savi.dialog.ques_gen.utils import load_vocab
from ss_baselines.savi.dialog.ques_gen.utils import get_glove_embedding

# need to add logger too

# set up the parameters
def set_params():
    params = {}
    params['model_path'] = './ss_baselines/savi/dialog/ques_gen/weights/vqg-tf-50.pkl'
    params['max_length'] = 20
    params['hidden_size'] = 512
    params['embedding_dim'] = 300
    params['num_layers'] = 1
    params['rnn_cell'] = 'LSTM'
    params['dropout_p'] = 0.3
    params['input_dropout_p'] = 0.3
    params['encoder_max_len'] = 4  # ??
    params['num_att_layers'] = 2
    params['z_size'] = 100
    params['no_answer_recon'] = True
    params['no_image_recon'] = True
    params['no_category_space'] = True
    params['vocab_path'] = './ss_baselines/savi/dialog/ques_gen/processed/vocab_iq_vln.json'
    params['embedding_name'] = '6B'

    return params


class DecentralizedDistributedMixinQuesGen:
    def init_distributed(self, find_unused_params: bool = True) -> None:
        r"""Initializes distributed training for the model
        1. Broadcasts the model weights from world_rank 0 to all other workers
        2. Adds gradient hooks to the model
        :param find_unused_params: Whether or not to filter out unused parameters
                                   before gradient reduction.  This *must* be True if
                                   there are any parameters in the model that where unused in the
                                   forward pass, otherwise the gradient reduction
                                   will not work correctly.
        """
        # NB: Used to hide the hooks from the nn.Module,
        # so they don't show up in the state_dict
        class Guard:
            def __init__(self, model, device):
                if torch.cuda.is_available():
                    self.ddp = torch.nn.parallel.DistributedDataParallel(
                        model, device_ids=[device], output_device=device
                    )
                else:
                    self.ddp = torch.nn.parallel.DistributedDataParallel(model)

        self._ddp_hooks = Guard(self, self.device)

        self.reducer = self._ddp_hooks.ddp.reducer
        self.find_unused_params = find_unused_params

    def before_backward(self, loss):
        if self.find_unused_params:
            self.reducer.prepare_for_backward([loss])
        else:
            self.reducer.prepare_for_backward([])


class QuesGen:
    def __init__(self, device):
        params = set_params()
        self.device = device
        # getting float with range (0,255) and also tensor
        # divide it with 255 and normalize
        # dividing it in ques_out()
        # need to make it consistant with ques generation training
        self.transform = transforms.Compose([transforms.Normalize(mean=[0.3853, 0.3853, 0.3855],
                                                                  std=[0.1050, 0.1050, 0.1047])
                                             ])

        self.vocab = load_vocab(params['vocab_path'])

        # should always use glove(or any other learned) embedding
        embedding = get_glove_embedding(params['embedding_name'],
                                        params['embedding_dim'],
                                        self.vocab)

        # Initialize model
        self.vqg = IQ_VLN(len(self.vocab), params['max_length'], params['hidden_size'], params['embedding_dim'],
                     self.vocab(self.vocab.SYM_SOQ), self.vocab(self.vocab.SYM_EOS),
                     num_layers=params['num_layers'],
                     rnn_cell=params['rnn_cell'],
                     dropout_p=params['dropout_p'],
                     input_dropout_p=params['input_dropout_p'],
                     encoder_max_len=params['encoder_max_len'],
                     embedding=embedding,
                     num_att_layers=params['num_att_layers'],
                     z_size=params['z_size'],
                     no_answer_recon=params['no_answer_recon'],
                     no_image_recon=params['no_image_recon'],
                     no_category_space=params['no_category_space'])

        self.vqg.load_state_dict(torch.load(params['model_path']))
        self.vqg.to(device=self.device)
        '''
        if torch.cuda.is_available():
            self.vqg.cuda()
        self.vqg.eval()
        '''

    def ques_out(self, image):
        # as getting float with range (0, 255)
        # need to make it consistant with training
        image = image/255.0
        image = self.transform(image)
        output = self.vqg.predict_from_image(image)
        # let's consider only single image
        # if multiple image then next line needs to be updated
        ques = self.vocab.tokens_to_words(output[0])
        return ques


class QuesGenDDP(QuesGen, DecentralizedDistributedMixinQuesGen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

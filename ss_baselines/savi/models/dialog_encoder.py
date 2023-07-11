# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import gzip
import json

import torch
import torch.nn as nn
from habitat import Config
# don't need config, can do it later

from ss_baselines.savi.dialog.ques_gen.utils import get_glove_embedding
from ss_baselines.savi.dialog.ques_gen.utils import load_vocab

# set up some params
def setup_params_dialog():
    params = dict()
    params['embedding_name'] = '6B'
    params['embedding_dim'] = 300
    params['rnn_type'] = "GRU" # else 'LSTM' # let's keep it to GRU
    params['bidirectional'] = False
    params['hidden_size'] = 64
    params['final_state_only'] = True
    params['vocab_path'] = './ss_baselines/savi/dialog/ques_gen/processed/vocab_iq_vln.json'
    
    return params




class DialogEncoder(nn.Module):
    def __init__(self):
        r"""An encoder that uses RNN to encode dialog. Returns
        the final hidden state after processing the instruction sequence.
        Args:
            config: must have
                vocab_size: number of words in the vocabulary  
                embedding_size: The dimension of each embedding vector
                use_pretrained_embeddings:
                embedding_file:
                fine_tune_embeddings:
                dataset_vocab:
                hidden_size: The hidden (output) size
                rnn_type: The RNN cell type.  Must be GRU or LSTM
                final_state_only: Whether or not to return just the final state
        """
        super().__init__()
        # self.config = config
        self.params = setup_params_dialog()
        
        # utilizing vocabulary word2idx mapping from ques generation 
        # vocabulary object
        self.vocab = load_vocab(self.params['vocab_path'])
        # add two extra token for questioner and answerer
        self.vocab.SYM_QUES = '<ques>'
        self.vocab.SYM_ANS = '<ans>'
        # also add Yes
        self.vocab.yes = 'Yes'
        self.vocab.add_word(self.vocab.yes)
        self.vocab.add_word(self.vocab.SYM_QUES)
        self.vocab.add_word(self.vocab.SYM_ANS)
        self.encoded_dim = self.params['hidden_size']
        
        
        # should always use glove(or any other learned) embedding
        # will directly use embedding from .vector_cache folder (also used by ques generator)
        embedding_weight = get_glove_embedding(self.params['embedding_name'],
                                               self.params['embedding_dim'],
                                               self.vocab)
        self.embedding_layer = nn.Embedding(len(self.vocab), self.params['embedding_dim'])
        self.embedding_layer.weight.data = embedding_weight        
         
        '''
        if self.config.use_pretrained_embeddings:
            self.embedding_layer = nn.Embedding.from_pretrained(
                embeddings=self._load_embeddings(),
                freeze=not self.config.fine_tune_embeddings,
            )
        else:  # each embedding initialized to sampled Gaussian
            self.embedding_layer = nn.Embedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.embedding_size,
                padding_idx=0,
            )
        '''

        rnn = nn.GRU if self.params['rnn_type'] == "GRU" else nn.LSTM
        self.bidir = self.params['bidirectional']
        self.encoder_rnn = rnn(
            input_size=self.params['embedding_dim'],
            hidden_size=self.params['hidden_size'],
            bidirectional=self.bidir,
        )
        self.final_state_only = self.params['final_state_only']
        

    @property
    def output_size(self):
        return self.params['hidden_size'] * (2 if self.bidir else 1)


    def forward(self, dialog):
        # sudipta
        # dialog: b x seq_len
        
        """
        Tensor sizes after computation:
            instruction: [batch_size x seq_length]
            lengths: [batch_size]
            hidden_state: [batch_size x hidden_size]
        """
        instruction = dialog.long()

        lengths = (instruction != 0.0).long().cpu().sum(dim=1)
        lengths = lengths.clamp(min=1)
        embedded = self.embedding_layer(instruction)
        packed_seq = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )

        output, final_state = self.encoder_rnn(packed_seq)

        if self.params['rnn_type'] == "LSTM":
            final_state = final_state[0]

        if self.final_state_only:
            return final_state.squeeze(0)
        else:
            return nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[
                0
            ].permute(0, 2, 1)

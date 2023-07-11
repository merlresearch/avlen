# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2019, Ranjay Krishna
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT

"""Contains code for the IQ model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder_cnn import EncoderCNN
from .decoder_rnn import DecoderRNN
from .mlp import MLP
import sys


class IQ_VLN(nn.Module):
    """Information Maximization question generation.
    """
    def __init__(self, vocab_size, max_len, hidden_size, embedding_dim,
                 sos_id, eos_id, num_layers=1, rnn_cell='LSTM',
                 bidirectional=False, input_dropout_p=0, dropout_p=0,
                 encoder_max_len=None, num_att_layers=2, att_ff_size=512,
                 embedding=None, z_size=20, no_answer_recon=False,
                 no_image_recon=False, no_category_space=False):
        """Constructor for IQ.

        Args:
            vocab_size: Number of words in the vocabulary.
            max_len: The maximum length of the answers we generate.
            hidden_size: Number of dimensions of RNN hidden cell.
            sos_id: Vocab id for <start>.
            eos_id: Vocab id for <end>.
            num_layers: The number of layers of the RNNs.
            rnn_cell: LSTM or RNN or GRU.
            bidirectional: Whether the RNN is bidirectional.
            input_dropout_p: Dropout applied to the input question words.
            dropout_p: Dropout applied internally between RNN steps.
            encoder_max_len: Maximum length of encoder.
            num_att_layers: Number of stacked attention layers.
            att_ff_size: Dimensions of stacked attention.
            embedding (vocab_size, hidden_size): Tensor of embeddings or
                None. If None, embeddings are learned.
            z_size: Dimensions of noise epsilon.
        """
        super(IQ_VLN, self).__init__()
        self.hidden_size = hidden_size
        if encoder_max_len is None:
            encoder_max_len = max_len
        self.num_layers = num_layers

        # Setup image encoder.
        self.encoder_cnn = EncoderCNN(hidden_size)
        self.image_proj = MLP(hidden_size, att_ff_size, hidden_size,
                               num_layers=num_att_layers)
        # self.bn = nn.BatchNorm1d(self.hidden_size)
        self.decoder = DecoderRNN(vocab_size, max_len, hidden_size, embedding_dim,
                                  sos_id=sos_id,
                                  eos_id=eos_id,
                                  n_layers=num_layers,
                                  rnn_cell=rnn_cell,
                                  input_dropout_p=input_dropout_p,
                                  dropout_p=dropout_p,
                                  embedding=embedding)


    # needed
    def flatten_parameters(self):
        if hasattr(self, 'decoder'):
            self.decoder.rnn.flatten_parameters()
        if hasattr(self, 'encoder'):
            self.encoder.rnn.flatten_parameters()


    # needed
    def generator_parameters(self):
        params = self.parameters()
        params = filter(lambda p: p.requires_grad, params)
        return params


    # needed
    def modify_hidden(self, func, hidden, rnn_cell):
        """Applies the function func to the hidden representation.
        This method is useful because some RNNs like LSTMs have a tuples.
        Args:
            func: A function to apply to the hidden representation.
            hidden: A RNN (or LSTM or GRU) representation.
            rnn_cell: One of RNN, LSTM or GRU.
        Returns:
            func(hidden).
        """
        if rnn_cell is nn.LSTM:
            return (func(hidden[0]), func(hidden[1]))
        return func(hidden)


    # needed
    def encode_images(self, images):
        """Encodes images.
        Args:
            images: Batch of image Tensors.
        Returns:
            Batch of image features.
        """
        images = self.encoder_cnn(images)
        images = self.image_proj(images)
        # images = self.bn(images)
        return images


    # needed
    def decode_questions(self, image_features, questions=None,
                         teacher_forcing_ratio=0, decode_function=F.log_softmax):
        """Decodes the question from the latent space.
        Args:
            image_features: Batch of image features.
            questions: Batch of question Variables.
            teacher_forcing_ratio: Whether to predict with teacher forcing.
            decode_function: What to use when choosing a word from the
                distribution over the vocabulary.
        """
        batch_size = image_features.size(0)

        # Reshape encoder_hidden (NUM_LAYERS * N * HIDDEN_SIZE).
        hiddens = image_features.view((1, batch_size, self.hidden_size))
        hiddens = hiddens.expand((self.num_layers, batch_size,
                                  self.hidden_size)).contiguous()
        if self.decoder.rnn_cell is nn.LSTM:
            hiddens = (hiddens, hiddens)
        result = self.decoder(inputs=questions,
                              encoder_hidden=hiddens,
                              function=decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        return result

    '''
    # not needed
    def forward(self, images, questions=None,
                teacher_forcing_ratio=0, decode_function=F.log_softmax):
        """Passes the image and the question through a model and generates answers.
        Args:
            images: Batch of image Variables.
            questions: Batch of question Variables.
            teacher_forcing_ratio: Whether to predict with teacher forcing.
            decode_function: What to use when choosing a word from the
                distribution over the vocabulary.
        Returns:
            - outputs: The output scores for all steps in the RNN.
            - hidden: The hidden states of all the RNNs.
            - ret_dict: A dictionary of attributes. See DecoderRNN.py for details.
        """
        # features is (N * HIDDEN_SIZE)
        image_features = self.encode_images(images)
        result = self.decode_questions(image_features, questions=questions,
                                       decode_function=decode_function,
                                       teacher_forcing_ratio=teacher_forcing_ratio)

        return result
    '''


    def parse_outputs_to_tokens(self, outputs):
        """Converts model outputs to tokens.
        Args:
            outputs: Model outputs.
        Returns:
            A tensor of batch_size X max_len.
        """
        # Take argmax for each timestep
        # Output is list of MAX_LEN containing BATCH_SIZE * VOCAB_SIZE.

        # BATCH_SIZE * VOCAB_SIZE -> BATCH_SIZE
        # outputs = [o.max(1)[1] for o in outputs]

        # sanity check
        outputs = [o.max(1)[1] for o in outputs]

        outputs = torch.stack(outputs)  # Tensor(max_len, batch)
        outputs = outputs.transpose(0, 1)  # Tensor(batch, max_len)
        return outputs


    def predict_from_image(self, images, questions=None, teacher_forcing_ratio=0,
                            decode_function=F.log_softmax):
        """Outputs the predicted vocab tokens for the answers in a minibatch.
        Args:
            images: Batch of image Tensors.
            teacher_forcing_ratio: Whether to predict with teacher forcing.
            decode_function: What to use when choosing a word from the
                distribution over the vocabulary.
        Returns:
            A tensor with BATCH_SIZE X MAX_LEN where each element is the index
            into the vocab word.
        """
        image_features = self.encode_images(images)
        outputs, _, _ = self.decode_questions(image_features, questions=questions,
                                       decode_function=decode_function,
                                       teacher_forcing_ratio=teacher_forcing_ratio)
        return self.parse_outputs_to_tokens(outputs)

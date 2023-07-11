# Copyright (C) 2019, Ranjay Krishna
#
# SPDX-License-Identifier: MIT

"""Genearates a representation for an image input.
"""

import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    """Generates a representation for an image input.
    """

    def __init__(self, output_size):
        """Load the pretrained ResNet-152 and replace top fc layer.
        """
        super(EncoderCNN, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, output_size)
        self.relu = nn.ReLU()
        # self.bn = nn.BatchNorm1d(output_size, momentum=0.01)
        self.init_weights()

    def init_weights(self):

        # self.cnn.fc.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_normal_(self.cnn.fc.weight.data)
        # self.cnn.fc.bias.data.fill_(0)

    def forward(self, images):

        features = self.relu(self.cnn(images))
        return features
        # output = self.bn(features)
        # return output

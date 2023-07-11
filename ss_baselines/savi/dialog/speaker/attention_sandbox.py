# BSD 2-Clause License

# Copyright (c) 2018, Daniel Fried, Ronghang Hu, Volkan Cirik, Anna Rohrbach,
# Jacob Andreas, Louis-Philippe Morency, Taylor Berg-Kirkpatrick, Kate Saenko,
# Dan Klein, Trevor Darrell

# All rights reserved.

batch_size = 20
feature_size = 10
h = 5
w = 4
context_size = 15
hidden_size = 6

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

#mechanism = SimpleImageAttention(feature_size, context_size, hidden_size)
def forward(mechanism):
    feature = torch.zeros(batch_size, feature_size, h, w)
    context = torch.zeros(batch_size, context_size)
    return mechanism.forward(Variable(feature), Variable(context))

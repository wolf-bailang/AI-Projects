import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO(Lab-02): Complete the network model.
class PolicyNet(nn.Module):  # Actor
    ##################################################################################################
    def __init__(self, input_size=23, hidden_size=512, output_size=2):
        ##################################################################################################
        super(PolicyNet, self).__init__()
        ##################################################################################################
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)
        ##################################################################################################

    def forward(self, s):
        ##################################################################################################
        x = F.relu(self.linear1(s), inplace=True)
        x = F.relu(self.linear2(x), inplace=True)
        x = F.relu(self.linear3(x), inplace=True)
        x = torch.tanh(self.linear4(x))

        #x = nn.ReLU(self.linear1(s), inplace=True)
        #x = nn.ReLU(self.linear2(x), inplace=True)
        #x = nn.ReLU(self.linear3(x), inplace=True)
        #x = nn.Tanh(self.linear4(x))

        return x
        # pass
        ##################################################################################################


class QNet(nn.Module):  # Critic
    ##################################################################################################
    def __init__(self, input_size=23, hidden_size=512, output_size=1):
        ##################################################################################################
        super(QNet, self).__init__()
        ##################################################################################################
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size + 2, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)
        ##################################################################################################

    def forward(self, s, a):
        ##################################################################################################
        x = F.relu(self.linear1(s), inplace=True)
        x = torch.cat([x, a], 1)
        x = F.relu(self.linear2(x), inplace=True)
        x = F.relu(self.linear3(x), inplace=True)
        x = self.linear4(x)
        
        #x = nn.ReLU(self.linear1(s), inplace=True)
        #x = torch.cat([x, a], 1)
        #x = nn.ReLU(self.linear2(x), inplace=True)
        #x = nn.ReLU(self.linear3(x), inplace=True)
        #x = self.linear4(x)
        
        return x

        # pass
        ##################################################################################################

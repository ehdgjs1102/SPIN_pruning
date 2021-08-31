import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import os

import config

from models import  densenet121, comp_resnet50, comp_dense, final_model

from torchinfo import summary

compress_rate = [ [0.35], [0.35]*6, [0.35]*12, [0.35]*24, [0.35]*16 ]
model = comp_dense(config.SMPL_MEAN_PARAMS, compress_rate)

# compress_rate = [0.2] + [0.2]*4 + [0.2]*16
# model = comp_resnet50(config.SMPL_MEAN_PARAMS, compress_rate)

summary(model, (1, 3, 224, 224))
# state_dict = model.state_dict()
# for key in enumerate(state_dict.keys()):
#     print(key)
# print(state_dict['fc1.bias'].shape)


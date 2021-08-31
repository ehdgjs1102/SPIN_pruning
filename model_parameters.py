import torch
from torch.utils.data import DataLoader
import numpy as np

import os

import config

from models import my_model, densenet121, comp_resnet50, comp_dense, final_model
# from results import my_model
from thop import profile

def count_parameters_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# stage_rate = [0.2, 0.2, 0.3, 0.8] 
# covn2_rate = [0.2, 0.2, 0.2] 
# conv3_rate = [0.3, 0.3, 0.3, 0.3]
# conv4_rate = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
# conv5_rate = [0.8, 0.8, 0.8]
# compress_rate = stage_rate + covn2_rate + conv3_rate + conv4_rate + conv5_rate
# compress_rate = [0] + [0]*3 + [0.5]*16 

# print(compress_rate)

compress_rate = [ [0.35], [0.35]*6, [0.35]*12, [0.35]*24, [0.35]*16 ]
model = comp_dense(config.SMPL_MEAN_PARAMS, compress_rate).to(device)
# model = comp_resnet50(config.SMPL_MEAN_PARAMS, compress_rate).to(device)

# num_trainable_parameters = count_parameters_trainable(model)
# num_parameters = count_parameters(model)

# print('compressed_model : ', num_trainable_parameters)
# print('compressed_model : ', num_parameters)


# model = hmr(config.SMPL_MEAN_PARAMS)
# num_trainable_parameters = count_parameters_trainable(model)
# num_parameters = count_parameters(model)

# print('original_model : ', num_trainable_parameters)
# print('original_model : ', num_parameters)

input_image_size = 224
input_image = torch.randn(1, 3, input_image_size, input_image_size).to(device)
flops, params = profile(model, inputs=(input_image,))


print('Params: %.2f'%(params))
print('Flops: %.2f'%(flops))

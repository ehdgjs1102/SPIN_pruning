import torch
from torch import nn
# import torch.nn.utils.prune as prune
import torch.nn.functional as F


import os

import config
import constants
from models import hmr, SMPL, densenet121


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def count_parameters_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())




model = densenet121(config.SMPL_MEAN_PARAMS)
checkpoint = torch.load('logs/densenet/checkpoints/2021_08_01-11_23_09.pt')
model.load_state_dict(checkpoint['model'], strict=False)

# Transfer model to the GPU
model.to(device)
pruning_modules = []

# for name, module in model.named_modules():
#     # 모든 2D-conv 층의 20% 연결에 대해 가지치기 기법을 적용
#     if isinstance(module, torch.nn.Conv2d):
#         pruning_modules.append((module, 'weight'))
#     # 모든 선형 층의 40% 연결에 대해 가지치기 기법을 적용
#     elif isinstance(module, torch.nn.Linear):
#         pruning_modules.append((module, 'weight'))

# prune.global_unstructured(
#     tuple(pruning_modules),
#     pruning_method=prune.L1Unstructured,
#     amount=0.2,
# )

# for (module, name) in pruning_modules:
#     prune.remove(module, name)

checkpoint = {}
checkpoint['model'] = model.state_dict()
torch.save(checkpoint, 'temp.pt')



num_trainable_parameters = count_parameters_trainable(model)
num_parameters = count_parameters(model)

print(num_trainable_parameters)
print(num_parameters)
print(list(model.named_parameters()))
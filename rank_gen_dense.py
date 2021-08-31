import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
import argparse
import json
from collections import namedtuple
from tqdm import tqdm
import torchgeometry as tgm

import config
import constants
from models import SMPL, densenet121,comp_dense, final_model
from datasets import BaseDataset
from utils.imutils import uncrop
from utils.pose_utils import reconstruction_error
from utils.part_utils import PartRenderer
from utils import CheckpointDataLoader, CheckpointSaver
from utils import TrainOptions
import datetime
from datasets import MixedDataset


options = TrainOptions().parse_args()

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='/home/urp10/SPIN/pruned_model/densenet_0.35+re_spinfits.pt', help='Path to network checkpoint')
parser.add_argument('--dataset', default='3dpw', choices=['h36m-p1', 'h36m-p2', 'lsp', '3dpw', 'mpi-inf-3dhp'], help='Choose evaluation dataset')
parser.add_argument('--log_freq', default=50, type=int, help='Frequency of printing intermediate results')
parser.add_argument('--batch_size', default=32, help='Batch size for testing')
parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data')
parser.add_argument('--num_workers', default=8, type=int, help='Number of processes for data loading')
parser.add_argument('--result_file', default=None, help='If set, save detections to a .npz file')
parser.add_argument('--limit', type=int, default=3, help='The num of batch to get rank.')
parser.add_argument('--name', default=None, help='Name of the experiment')

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

compress_rate = [ [0.35], [0.35]*6, [0.35]*12, [0.35]*24, [0.35]*16 ]
model = final_model(config.SMPL_MEAN_PARAMS, compress_rate)

checkpoint = torch.load(args.checkpoint)
model.load_state_dict(checkpoint['model'])
model.to(device)

relu_test = nn.ReLU(inplace=True)

train_ds = MixedDataset(options, ignore_3d=options.ignore_3d, is_train=True)
train_data_loader = CheckpointDataLoader(train_ds,checkpoint=None,
                                                     batch_size=args.batch_size,
                                                     num_workers=options.num_workers,
                                                     pin_memory=options.pin_memory,
                                                     shuffle=options.shuffle_train)

feature_result = torch.tensor(0.)
total = torch.tensor(0.)            


def get_feature_hook(self, input, output):
    global feature_result
    global entropy
    global total
    output = relu_test(output)
    a = output.shape[0]
    b = output.shape[1]
    c = torch.tensor([torch.matrix_rank(output[i,j,:,:]).item() for i in range(a) for j in range(b)])

    c = c.view(a, -1).float()
    c = c.sum(0)
    feature_result = feature_result * total + c
    total = total + a
    feature_result = feature_result / total


def inference():
    model.eval()
    limit = args.limit
    
    with torch.no_grad():
         for step, batch in enumerate(tqdm(train_data_loader, desc='Computing Rank',
                                              total=len(train_ds) // args.batch_size,
                                              initial=train_data_loader.checkpoint_batch_idx),
                                         train_data_loader.checkpoint_batch_idx):
            #use the first 5 batches to estimate the rank.
            if step >= limit:
               break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
            images = batch['img']
        
            outputs = model(images)



# densenet121 ranking generating
cov_layer = model.features.pool0
print(cov_layer)

handler = cov_layer.register_forward_hook(get_feature_hook)
inference()
handler.remove()

pre = 'rank_densenet0.35+re_no_retrain'


if not os.path.isdir(pre + '/densenet121_limit%d'%(args.limit)):
    os.mkdir(pre + '/densenet121_limit%d'%(args.limit))
np.save(pre + '/densenet121_limit%d'%(args.limit) + '/rank_conv%d' % (1) + '.npy', feature_result.numpy())
feature_result = torch.tensor(0.)
total = torch.tensor(0.)

# Densenet per Denselayer
cnt=1
model_block = [model.features.denseblock1, model.features.denseblock2, model.features.denseblock3, model.features.denseblock4]
trainsition_block = [model.features.transition1, model.features.transition2, model.features.transition3]
number_blocks = [6,12,24,16]
for i in range(4):
    block = model_block[i]
    for j in range(number_blocks[i]):
    
        cov_layer = block[j].conv1

        print(i, j, cov_layer)

        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()
        np.save(pre + '/densenet121_limit%d'%(args.limit) + '/rank_conv%d'%(cnt+1)+'.npy',
                feature_result.numpy())
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)
        cnt+=1
 
            
        cov_layer = block[j].conv2

        print(i, j, cov_layer)

        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()
        np.save(pre + '/densenet121_limit%d' % (args.limit) + '/rank_conv%d' % (cnt + 1) + '.npy',
                feature_result.numpy())
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)
        cnt += 1
        

    if i != 3:
      
        t_block = trainsition_block[i]
        cov_layer = t_block.pool

        print(i, cov_layer)

        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()
        np.save(pre + '/densenet121_limit%d' % (args.limit) + '/rank_conv%d' % (cnt + 1) + '.npy',
                feature_result.numpy())
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

        cnt += 1
            

       
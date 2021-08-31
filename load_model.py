import torch
import numpy as np
import os
import config
from models import densenet121, comp_resnet50
import torch.nn as nn

def load_resnet_model(model, oristate_dict):
    cfg = {'resnet18': [2, 2, 2, 2],
           'resnet34': [3, 4, 6, 3],
           'resnet_50': [3, 4, 6, 3],
           'resnet101': [3, 4, 23, 3],
           'resnet152': [3, 8, 36, 3]}

    state_dict = model.state_dict()

    current_cfg = cfg['resnet_50']
    last_select_index = None

    all_honey_conv_weight = []

    bn_part_name=['.weight','.bias','.running_mean','.running_var']#,'.num_batches_tracked']
    prefix = 'rank_conv/resnet_50_limit5/rank_conv'
    subfix = '.npy'
    cnt=1

    conv_weight_name = 'conv1.weight'
    all_honey_conv_weight.append(conv_weight_name)
    oriweight = oristate_dict[conv_weight_name]
    curweight = state_dict[conv_weight_name]
    orifilter_num = oriweight.size(0)
    currentfilter_num = curweight.size(0)

    if orifilter_num != currentfilter_num:
        print('loading rank from: ' + prefix + str(cnt) + subfix)
        rank = np.load(prefix + str(cnt) + subfix)
        select_index = np.argsort(rank)[orifilter_num - currentfilter_num:]  # preserved filter id
        select_index.sort()
        for index_i, i in enumerate(select_index):
            state_dict[conv_weight_name][index_i] = oristate_dict[conv_weight_name][i]
            for bn_part in bn_part_name:
                state_dict['bn1' + bn_part][index_i] = oristate_dict['bn1' + bn_part][i]
        last_select_index = select_index
    else:
        state_dict[conv_weight_name] = oriweight
        for bn_part in bn_part_name:
            state_dict['bn1' + bn_part] = oristate_dict['bn1'+bn_part]

    state_dict['bn1' + '.num_batches_tracked'] = oristate_dict['bn1' + '.num_batches_tracked']

    cnt+=1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'

        for k in range(num):
        
            iter = 3
            if k==0:
                down_last_index = last_select_index
                iter +=1
            for l in range(iter):
                record_last=True
                if k==0 and l==2:
                    before_conv_out = last_select_index
                    conv_name = layer_name + str(k) + '.downsample.0'
                    bn_name = layer_name + str(k) + '.downsample.1'
                    last_select_index = down_last_index
                    record_last=False
  
                elif k==0 and l==3:
                    conv_name = layer_name + str(k) + '.conv' + str(l)
                    bn_name = layer_name + str(k) + '.bn' + str(l)
                else:
                    conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                    bn_name = layer_name + str(k) + '.bn' + str(l + 1)

                conv_weight_name = conv_name + '.weight'
                all_honey_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)
                # print(conv_weight_name)
                # print(orifilter_num)
                # print(currentfilter_num)
                # print(len(last_select_index))

                if orifilter_num != currentfilter_num:
                    

            
                    rank = np.load(prefix + str(cnt) + subfix)
                    select_index = np.argsort(rank)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()

                    
                    if last_select_index is not None:
                        print('(input&output)loading rank from: ' + prefix + str(cnt) + subfix)
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[conv_weight_name][index_i][index_j] = oristate_dict[conv_weight_name][i][j]

                            for bn_part in bn_part_name:
                                state_dict[bn_name + bn_part][index_i] = oristate_dict[bn_name + bn_part][i]

                    else:
                        print('(output)loading rank from: ' + prefix + str(cnt) + subfix)
                        for index_i, i in enumerate(select_index):
                            state_dict[conv_weight_name][index_i] = oristate_dict[conv_weight_name][i]

                            for bn_part in bn_part_name:
                                state_dict[bn_name + bn_part][index_i] = oristate_dict[bn_name + bn_part][i]

                    if record_last:
                        last_select_index = select_index
                    else : 
                        last_select_index = before_conv_out


                elif last_select_index is not None:
                    print('(input)loading rank from: ' + prefix + str(cnt) + subfix)
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[conv_weight_name][index_i][index_j] = oristate_dict[conv_weight_name][index_i][j]

                    for bn_part in bn_part_name:
                        state_dict[bn_name + bn_part] = oristate_dict[bn_name + bn_part]

                    if record_last:
                        last_select_index = None
                    else :
                        last_select_index = before_conv_out


                else:
                    state_dict[conv_weight_name] = oriweight
                    for bn_part in bn_part_name:
                        state_dict[bn_name + bn_part] = \
                            oristate_dict[bn_name + bn_part]
                    if record_last:
                        last_select_index = None
                    else :
                        last_select_index = before_conv_out


                state_dict[ bn_name + '.num_batches_tracked'] = oristate_dict[bn_name + '.num_batches_tracked']
                cnt+=1
    for name, module in model.named_modules():
        name = name.replace('module.', '')
        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if conv_name not in all_honey_conv_weight:
                print('*******************error********************')
                state_dict[conv_name] = oristate_dict[conv_name]

        if isinstance(module, nn.Linear):
            print('fill fc layer ' + str(name))
            if name == 'fc1':
                if last_select_index is not None :
                    orifilter_num = state_dict[name + '.weight'].size(0)
                    for index_j, j in enumerate(last_select_index):
                        for index_i in range(orifilter_num):
                            state_dict[name + '.weight'][index_i][index_j] = oristate_dict[name + '.weight'][index_i][j]

                    state_dict[name + '.bias'] = oristate_dict[name + '.bias']
                    
                else: 
                    state_dict[name + '.weight'] = oristate_dict[name + '.weight']
                    state_dict[name + '.bias'] = oristate_dict[name + '.bias']
            else:
                state_dict[name + '.weight'] = oristate_dict[name + '.weight']
                state_dict[name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)
    checkpoint = {}
    checkpoint['model'] = model.state_dict()
    torch.save(checkpoint, 'pruned_model/resnet_0.2_with_spinfits' + '.pt' ) 
  




device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


compress_rate = [0.2] + [0.2]*4 + [0.2]*16
model = comp_resnet50(config.SMPL_MEAN_PARAMS, compress_rate).to(device)

compress_rate = [0] + [0]*4 + [0]*16
origin_model = comp_resnet50(config.SMPL_MEAN_PARAMS, compress_rate).to(device)

checkpoint = torch.load('/home/urp10/SPIN/logs/resnet50_with_spinfits/checkpoints/2021_08_29-23_33_45.pt')
origin_model.load_state_dict(checkpoint['model'])
oristate_dict = origin_model.state_dict()
load_resnet_model(model, oristate_dict)
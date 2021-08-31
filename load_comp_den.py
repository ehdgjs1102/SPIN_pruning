import torch
import numpy as np
import os
import config
from models import densenet121, comp_resnet50, comp_dense, final_model
import torch.nn as nn

def load_densenet_model(model, oristate_dict):
    cfg = {'dense121': [6, 12, 24, 16]}

    state_dict = model.state_dict()

    current_cfg = cfg['dense121']
    last_select_index = None
    last_concat_index = None

    all_honey_conv_weight = []

    bn_part_name=['.weight','.bias','.running_mean','.running_var']#,'.num_batches_tracked']
    prefix = 'rank_densenet0.35_spinfits/densenet121_limit3/rank_conv'
    subfix = '.npy'
    cnt=1

    conv_weight_name = 'features.conv0.weight'
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
                state_dict['features.norm0' + bn_part][index_i] = oristate_dict['features.norm0' + bn_part][i]
        last_select_index = select_index
    else:
        state_dict[conv_weight_name] = oriweight
        for bn_part in bn_part_name:
            state_dict['features.norm0' + bn_part] = oristate_dict['features.norm0'+bn_part]
        last_select_index = np.array(range(int(64*0.65)))

    state_dict['features.norm0' + '.num_batches_tracked'] = oristate_dict['features.norm0' + '.num_batches_tracked']

    model_block = [model.features.denseblock1, model.features.denseblock2, model.features.denseblock3, model.features.denseblock4]
    trainsition_block = [model.features.transition1, model.features.transition2, model.features.transition3]
    cnt+=1
    print(len(last_select_index))
    for layer, num in enumerate(current_cfg):
        if last_select_index is not None:
            last_concat_index = last_select_index
        else : 
            # last_concat_index = np.array(range(int((2**(layer+6)))))
            
            if layer == 0:
                last_concat_index = np.array(range(int(64*0.65)))
            elif layer == 1:
                last_concat_index = np.array(range(80))
            elif layer == 2:
                last_concat_index = np.array(range(160))
            elif layer == 3:
                last_concat_index = np.array(range(320))
            
        
        block_name = 'features.' + 'denseblock' + str(layer + 1) + '.'
        for k in range(num):
            layer_name = block_name + 'denselayer' + str(k+1) + '.'

            for num_conv in range(1,3):
                if num_conv == 1:
                    last_select_index = last_concat_index
                bn_name = layer_name + 'norm' + str(num_conv)
                conv_name = layer_name + 'conv' + str(num_conv)
                
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
                        # print('select_index : ', select_index)
                        # print('Last_select_index : ', last_select_index)
                        for index_j, j in enumerate(last_select_index):
                            for index_i, i in enumerate(select_index):
                                state_dict[conv_weight_name][index_i][index_j] = oristate_dict[conv_weight_name][i][j]

                            for bn_part in bn_part_name:
                                state_dict[bn_name + bn_part][index_j] = oristate_dict[bn_name + bn_part][j]

                    else:
                        print('(output)loading rank from: ' + prefix + str(cnt) + subfix)
                        # print('select_index : ', select_index)
                        for index_i, i in enumerate(select_index):
                            state_dict[conv_weight_name][index_i] = oristate_dict[conv_weight_name][i]

                        for bn_part in bn_part_name:
                            state_dict[bn_name + bn_part] = oristate_dict[bn_name + bn_part]

             
                    last_select_index = select_index


                elif last_select_index is not None:
                    print('(input)loading rank from: ' + prefix + str(cnt) + subfix)
                    # print('Last_select_index : ', last_select_index)
                    for index_j, j in enumerate(last_select_index):
                        for index_i in range(orifilter_num):
                            state_dict[conv_weight_name][index_i][index_j] = oristate_dict[conv_weight_name][index_i][j]

                        for bn_part in bn_part_name:
                            state_dict[bn_name + bn_part][index_j] = oristate_dict[bn_name + bn_part][j]

                  
                    last_select_index = None
            

                else:
                    print('(no_change)loading rank from: ' + prefix + str(cnt) + subfix)
                    state_dict[conv_weight_name] = oriweight
                    for bn_part in bn_part_name:
                        state_dict[bn_name + bn_part] = oristate_dict[bn_name + bn_part]
                 
                    last_select_index = None


                state_dict[ bn_name + '.num_batches_tracked'] = oristate_dict[bn_name + '.num_batches_tracked']
             
                cnt+=1
                if num_conv ==2 :
                    if layer == 0:
                        ori_out_num = int(64*0.65) + int(32*0.65) *k
                    elif layer == 1:
                        ori_out_num = 80 + int(32*0.65) *k
                    elif layer == 2:
                        ori_out_num = 160 + int(32*0.65) *k
                    elif layer == 3:
                        ori_out_num = 320 + int(32*0.65) *k
                    # ori_out_num = int((2**(layer+6)))  + 32 * k
                    if last_select_index is not None:
                        change = last_select_index + ori_out_num
                        last_concat_index = np.concatenate((last_concat_index, change), axis=0 )
                    else :
                        no_change = np.array(range(int(32*0.65))) 
                        no_change = no_change + ori_out_num
                        last_concat_index = np.concatenate((last_concat_index, no_change), axis=0 )
                print(len(last_concat_index))



        if layer != 3:
            block_name = 'features.' + 'transition' + str(layer + 1) + '.'
            bn_name = block_name + 'norm' 
            conv_name = block_name + 'conv' 
            conv_weight_name = conv_name + '.weight'
            all_honey_conv_weight.append(conv_weight_name)
            oriweight = oristate_dict[conv_weight_name]
            curweight = state_dict[conv_weight_name]
            orifilter_num = oriweight.size(0)
            currentfilter_num = curweight.size(0)
            last_select_index = last_concat_index
        

            if orifilter_num != currentfilter_num:
                
                rank = np.load(prefix + str(cnt) + subfix)
                select_index = np.argsort(rank)[orifilter_num - currentfilter_num:]  # preserved filter id
                select_index.sort()

                if last_select_index is not None:
                    print('(input&output)loading rank from: ' + prefix + str(cnt) + subfix)
                    # print('select_index : ', select_index)
                    # print('Last_select_index : ', last_select_index)
                    for index_j, j in enumerate(last_select_index):
                        for index_i, i in enumerate(select_index):
                            state_dict[conv_weight_name][index_i][index_j] = oristate_dict[conv_weight_name][i][j]

                        for bn_part in bn_part_name:
                            state_dict[bn_name + bn_part][index_j] = oristate_dict[bn_name + bn_part][j]

                else:
                    print('(output)loading rank from: ' + prefix + str(cnt) + subfix)
                    # print('select_index : ', select_index)
                    for index_i, i in enumerate(select_index):
                        state_dict[conv_weight_name][index_i] = oristate_dict[conv_weight_name][i]

                    for bn_part in bn_part_name:
                        state_dict[bn_name + bn_part] = oristate_dict[bn_name + bn_part]

                last_select_index = select_index


            elif last_select_index is not None:
                print('(input)loading rank from: ' + prefix + str(cnt) + subfix)
                # print('Last_select_index : ', last_select_index)
                for index_j, j in enumerate(last_select_index):
                    for index_i in range(orifilter_num):
                        state_dict[conv_weight_name][index_i][index_j] = oristate_dict[conv_weight_name][index_i][j]

                    for bn_part in bn_part_name:
                        state_dict[bn_name + bn_part][index_j] = oristate_dict[bn_name + bn_part][j]

      
                last_select_index = None
        

            else:
                print('(no_change)loading rank from: ' + prefix + str(cnt) + subfix)
                state_dict[conv_weight_name] = oriweight
                for bn_part in bn_part_name:
                    state_dict[bn_name + bn_part] = \
                        oristate_dict[bn_name + bn_part]
         
                last_select_index = None


            state_dict[ bn_name + '.num_batches_tracked'] = oristate_dict[bn_name + '.num_batches_tracked']
     
            cnt+=1
            print(len(last_concat_index))

    bn_name = 'features.norm5'  
    last_select_index = last_concat_index
    if last_select_index is not None:
        print('(input)norm5')
        for index_j, j in enumerate(last_select_index):
    
            for bn_part in bn_part_name:
                state_dict[bn_name + bn_part][index_j] = oristate_dict[bn_name + bn_part][j]

    else: 
        print('(no)norm5')
        for bn_part in bn_part_name:
            state_dict[bn_name + bn_part] = oristate_dict[bn_name + bn_part]

    state_dict[ bn_name + '.num_batches_tracked'] = oristate_dict[bn_name + '.num_batches_tracked']

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
    torch.save(checkpoint, 'pruned_model/densenet(0.35+re)_spinfits' + '.pt' ) 
  
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


compress_rate = [ [0.35], [0.35]*6, [0.35]*12, [0.35]*24, [0.35]*16 ]
model = final_model(config.SMPL_MEAN_PARAMS, compress_rate).to(device)

compress_rate = [ [0.35], [0.35]*6, [0.35]*12, [0.35]*24, [0.35]*16 ]
origin_model =  comp_dense(config.SMPL_MEAN_PARAMS, compress_rate).to(device)

checkpoint = torch.load('/home/urp10/SPIN/logs/densenet_0.35_with_spinfits2/checkpoints/2021_08_30-05_40_27_best.pt')
origin_model.load_state_dict(checkpoint['model'])
oristate_dict = origin_model.state_dict()
load_densenet_model(model, oristate_dict)
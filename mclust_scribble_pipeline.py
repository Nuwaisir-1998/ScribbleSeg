# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import argparse
import sys
import numpy as np
import pandas as pd
import torch.nn.init
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import os
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.pairwise import euclidean_distances
import math
from scipy import spatial
import json
import random
from os.path import exists

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


import json

import argparse

parser = argparse.ArgumentParser(description='ScribbleSeg expert annotation pipeline')
parser.add_argument('--params', help="The input parameters json file path", required=True)

args = parser.parse_args()

with open(args.params) as f:
   params = json.load(f)
test_folder_base_name = params['test_folder_base_name']
dataset = params['dataset']
n_pcs = params['n_pcs']
scribble = params['scribble']
expert_scribble = params['expert_scribble']
mclust_pre_heat = params['mclust_pre_heat']
pre_heat_max_iter = params['pre_heat_max_iter']
nChannel = params['nChannel']
max_iter = params['max_iter']
nConv = params['nConv']
visualize = params['visualize']
use_background_scribble = params['use_background_scribble']
added_layers = params['added_layers']
last_layer_channel_count = params['last_layer_channel_count']
hyper_sum_division = params['hyper_sum_division']
seed_options = params['seed_options']
sim_options = params['sim_options']
miu_options = params['miu_options']
niu_options = params['niu_options']
lr_options = params['lr_options']

use_cuda = torch.cuda.is_available()

if use_cuda:
    print("GPU available")
else:
    print("GPU not available")

mclust_scribble = not expert_scribble
use_background_scribble = False
if scribble:
    if expert_scribble: scheme = 'Expert_scribble'
    elif mclust_scribble:
        if mclust_pre_heat:
            scheme = 'Mclust_scribble_with_pre_heating'
        else: scheme = 'Mclust_scribble_normal'
    else: scheme = 'Other_scribble'
else: scheme = 'No_scribble'

added_layers = 0
intermediate_channels = n_pcs # was n_pcs
hyper_sum_division = True
meta_data_index = ['test_name', 'seed', 'dataset', 'sample', 'n_pcs', 'scribble', 'max_iter', 'sim', 'miu', 'niu', 'scheme', 'lr', 'nConv', 'no_of_scribble_layers', 'intermediate_channels', 'added_layers', 'last_layer_channel_count', 'hyper_sum_division']

test_name = f'{test_folder_base_name}_itr_{max_iter}'
# seed_options = pd.read_csv('./Data/seed_list.csv')['seeds'].values

samples = params['samples']

models = []
for sample in samples:
    for seed in seed_options:
        for sim in sim_options:
            for miu in miu_options:
                for niu in niu_options:
                    for lr in lr_options:
                        models.append(
                            {
                                'seed': seed,
                                'stepsize_sim': sim,
                                'stepsize_con': miu,
                                'stepsize_scr': niu,
                                'lr': lr,
                                'sample': sample,
                            }
                        )

# %%
for model in tqdm(models):
    seed = model['seed']
    lr = model['lr']
    stepsize_sim = model['stepsize_sim']
    stepsize_con = model['stepsize_con']
    stepsize_scr = model['stepsize_scr']
    sample = model['sample']

    print("************************************************")
    print('Model description:')
    print(f'Sample: {sample}')
    print(f'seed: {seed}')
    print(f'lr: {lr}')
    print(f'sim: {stepsize_sim}')
    print(f'miu: {stepsize_con}')
    print(f'niu: {stepsize_scr}')

    npz_path = f'Algorithms/Unsupervised_Segmentation/Approaches/With_Scribbles/Local_Data/{dataset}/{sample}/Npzs'
    npy_path = f'Algorithms/Unsupervised_Segmentation/Approaches/With_Scribbles/Local_Data/{dataset}/{sample}/Npys'
    pickle_path = f'Algorithms/Unsupervised_Segmentation/Approaches/With_Scribbles/Local_Data/{dataset}/{sample}/Pickles'
    coordinates_file_name = 'coordinates.csv'

    # %%
    def make_directory_if_not_exist(path):
        if not os.path.exists(path):
            os.makedirs(path)

    scribble_img = f'Algorithms/Unsupervised_Segmentation/Approaches/With_Scribbles/Local_Data/{dataset}/{sample}/Scribble/manual_scribble_1.npy'
    if mclust_scribble:
        scribble_img = f'Algorithms/Unsupervised_Segmentation/Approaches/With_Scribbles/Local_Data/{dataset}/{sample}/Scribble/mclust_scribble.npy'
    local_data_folder_path = './Algorithms/Unsupervised_Segmentation/Approaches/With_Scribbles/Local_Data'

    input = f'{npy_path}/mapped_{n_pcs}.npy'
    inv_xy = f'{pickle_path}/inv_spot_xy.pickle'
    border = npz_path+'/borders.npz'
    background = npy_path+'/backgrounds.npy'
    foreground = npy_path+'/foregrounds.npy'
    indices_arg = npy_path+'/indices.npy'
    pixel_barcode_map_path = pickle_path+'/pixel_barcode_map.pickle'
    coordinate_file = f'Data/{dataset}/{sample}/{coordinates_file_name}'
    map_pixel_to_grid_spot_file_path = f'{local_data_folder_path}/{dataset}/{sample}/Jsons/map_pixel_to_grid_spot.json'
    pixel_barcode_file_path = f'{local_data_folder_path}/{dataset}/{sample}/Npys/pixel_barcode.npy'
    manual_annotation_file_path = f'./Data/{dataset}/{sample}/manual_annotations.csv'

    output_folder_path = f'./Outputs/{test_name}/{dataset}/{sample}'
    leaf_output_folder_path = f'{output_folder_path}/{scheme}/{n_pcs}_pcs/Seed_{seed}/Lr_{lr}/Hyper_{stepsize_sim}_{stepsize_con}_{stepsize_scr}'
    labels_per_itr_folder_path = f'{leaf_output_folder_path}/Labels_per_itr/'
    image_per_itr_folder_path = f'{leaf_output_folder_path}/Image_per_itr/'
    path_to_pre_heat_model_folder = f'{leaf_output_folder_path}/Models'
    # %%
    pixel_barcode = np.load(pixel_barcode_file_path)
    pixel_rows_cols = np.argwhere(pixel_barcode != '')
    df_man = pd.read_csv(manual_annotation_file_path, index_col=0)
    manual_annotation_labels = df_man['label'].values
    ari_per_itr = []
    loss_per_itr = []
    df_barcode_labels_per_itr = pd.DataFrame(index = pixel_barcode[pixel_barcode != ''])
    mclust_barcode_label = pd.read_csv(f'./Data/{dataset}/{sample}/mclust_result.csv', index_col = 0)
    backgrounds = np.load(background)
    foregrounds = np.load(foreground)

    # %%
    make_directory_if_not_exist(output_folder_path)
    make_directory_if_not_exist(labels_per_itr_folder_path)
    make_directory_if_not_exist(image_per_itr_folder_path)
    make_directory_if_not_exist(path_to_pre_heat_model_folder)

    # %%
    with open(map_pixel_to_grid_spot_file_path, 'r') as f:
        map_pixel_to_grid_spot = json.load(f)

    # %%
    def make_str(x):
        return f'({x[0]}, {x[1]})'

    def get_grid_spots_from_pixels(pixels, colors):
        grid_spots = np.array([map_pixel_to_grid_spot[make_str(pixel)] for pixel in pixels if make_str(pixel) in map_pixel_to_grid_spot])
        predicted_colors = [colors[i] for i in range(len(pixels)) if make_str(pixels[i]) in map_pixel_to_grid_spot]
        return grid_spots, predicted_colors

    # %%
    def calc_ari(df_1, df_2):
        df_merged = pd.merge(df_1, df_2, left_index=True, right_index=True).dropna()
        cols = df_merged.columns
        for col in cols:
            df_merged[col] = df_merged[col].values.astype('int')
        return adjusted_rand_score(df_merged[cols[0]].values, df_merged[cols[1]].values)

    # %%
    torch.manual_seed(seed)
    np.random.seed(seed)

    no_of_scribble_layers = 0

    # CNN model
    class MyNet(nn.Module):
        def __init__(self,input_dim):
            super(MyNet, self).__init__()
            self.conv1 = nn.Conv2d(input_dim, intermediate_channels, kernel_size=3, stride=1, padding=1 )
            self.bn1 = nn.BatchNorm2d(intermediate_channels)
            self.conv2 = nn.ModuleList()
            self.bn2 = nn.ModuleList()
            for i in range(nConv-1):
                self.conv2.append( nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=1, padding=1 ) )
                self.bn2.append( nn.BatchNorm2d(intermediate_channels) )

            r = last_layer_channel_count

            print('last layer size:', r)
            self.conv3 = nn.Conv2d(intermediate_channels, r, kernel_size=1, stride=1, padding=0 )
            self.bn3 = nn.BatchNorm2d(r)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu( x )
            x = self.bn1(x)
            for i in range(nConv-1):
                x = self.conv2[i](x)
                x = F.relu( x )
                x = self.bn2[i](x)
            x = self.conv3(x)
            x = self.bn3(x)
            return x

    # %%
    im = np.load(input)
    im.shape

    data = torch.from_numpy( np.array([im.transpose( (2, 0, 1) ).astype('float32')]) ) # z, y, x
    data.shape

    # %%
    if use_cuda:
        data = data.cuda()
    data = Variable(data)
    data.shape

    # %%
    def relabel_mask(mask, background_val):
        row, col = mask.shape
        mask = mask.reshape(-1)
        values = np.unique(mask[mask != background_val])
        lookup = {k: v for v, k in enumerate(dict.fromkeys(values))}
        lookup[background_val] = background_val
        mask = np.array([lookup[i] for i in mask])
        return mask.reshape(row, col)

    # %%
    def compare_with_mclust(mclust_labels, labels):
        assert(len(mclust_labels) == len(labels))
        z = mclust_labels - labels
        matched_count = len(z[z == 0])
        return matched_count / len(labels)


    # %%
    # load scribble
    if scribble:
        mask = np.load(scribble_img)
        foreground_val = 1000
        background_val = 255
        mask = relabel_mask(mask.copy(), background_val)
        if len(mask[mask != background_val]) == 0:
            print('Expecting some scribbles, but no scribbles are found!')
            last_layer_channel_count = 100 + added_layers
            nChannel = last_layer_channel_count
        else:
            
            mask_foreground = mask.copy()
            mask_foreground[foregrounds[:, 0], foregrounds[:, 1]] = foreground_val
            
            mx_label_num = mask[mask != background_val].max()
            if use_background_scribble:
                mask[backgrounds[:, 0], backgrounds[:, 1]] = mx_label_num + 1 # Assuming that scribble labels increase by 1
            mask = mask.reshape(-1)
            scr_idx = np.where(mask != 255)[0]
            mask_foreground = mask_foreground.reshape(-1)
            mask_foreground_and_scr = mask_foreground.copy()

            mask_inds = np.unique(mask)
            mask_inds = np.delete( mask_inds, np.argwhere(mask_inds==background_val) )

            for i in range(1, len(mask_inds)):
                if mask_inds[i] - mask_inds[i-1] != 1:
                    print("Problem in scribble labels. Not increasing by 1.")

            # # Take the non-scribbled foreground
            mask_foreground[scr_idx] = background_val
            inds_sim = torch.from_numpy( np.where( mask_foreground == foreground_val )[ 0 ] ) 

            inds_sim_for_mclust = torch.from_numpy( np.where( mask_foreground_and_scr == foreground_val )[ 0 ] )

            inds_scr = torch.from_numpy( np.where( mask != background_val )[ 0 ] )
            inds_scr_array = [None for _ in range(mask_inds.shape[0])]

            for i in range(mask_inds.shape[0]):
                inds_scr_array[i] = torch.from_numpy( np.where( mask == mask_inds[i] )[ 0 ] )

            target_scr = torch.from_numpy( mask.astype(np.int64) )

            if use_cuda:
                inds_sim = inds_sim.cuda()
                inds_scr = inds_scr.cuda()
                target_scr = target_scr.cuda()


            target_scr = Variable( target_scr ) # *************** Why? **************

            minLabels = len(mask_inds)
            # nChannel = minLabels + 1
            nChannel = minLabels + added_layers # ************ Change ************ 

            no_of_scribble_layers = minLabels # **************** Addition *****************
            last_layer_channel_count = no_of_scribble_layers + added_layers
    else:
        last_layer_channel_count = 100 + added_layers
        nChannel = last_layer_channel_count


    # %%
    data.shape

    # %%
    # train
    model = MyNet( data.size(1) )
    if use_cuda:
        model.cuda()
    model.train()

    # %%
    # similarity loss definition
    loss_fn = torch.nn.CrossEntropyLoss()

    # scribble loss definition
    loss_fn_scr = torch.nn.CrossEntropyLoss()

    # continuity loss definition
    loss_hpy = torch.nn.L1Loss(reduction='mean')
    loss_hpz = torch.nn.L1Loss(reduction='mean')
    # loss for the diagonal neighbour
    loss_hp_diag = torch.nn.L1Loss(reduction='mean')

    HPy_target = torch.zeros(im.shape[0] - 1, im.shape[1], nChannel)
    HPz_target = torch.zeros(im.shape[0], im.shape[1] - 1, nChannel)
    #extra
    HP_diag_target = torch.zeros(im.shape[0] - 1, im.shape[1] - 1, nChannel)
    if use_cuda:
        HPy_target = HPy_target.cuda()
        HPz_target = HPz_target.cuda()
        #extra
        HP_diag_target = HP_diag_target.cuda()
        
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    label_colours = np.random.randint(255,size=(255,3))

    label_colours[0,:] = [255,255,255]
    label_colours[1,:] = [0,255,0]
    label_colours[2,:] = [255,0,0]
    label_colours[3,:] = [255,255,0]
    label_colours[4,:] = [0,255,255]
    label_colours[5,:] = [255,0,255]
    label_colours[6,:] = [0,0,0]
    label_colours[7,:] = [73,182,255]

    loss_comparison = 0

    # %%
    borders = np.load(border)

    right_border = borders['right_border']
    left_border = borders['left_border']
    up_border = borders['up_border']
    down_border = borders['down_border']
    nw_border = borders['nw_border']
    se_border = borders['se_border']

    import warnings
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    stepsize_sim_orig = stepsize_sim
    stepsize_con_orig = stepsize_con
    stepsize_scr_orig = stepsize_scr
    max_iter_orig = max_iter

    def train_model(train_type):
        assert(train_type == 'pre-heat' or train_type == 'post-heat')
        if train_type == 'pre-heat':
            stepsize_scr = 1
            stepsize_sim = 0
            stepsize_con = 0
            max_iter = pre_heat_max_iter
        else:
            stepsize_sim = stepsize_sim_orig
            stepsize_con = stepsize_con_orig
            # stepsize_scr = 0
            stepsize_scr = stepsize_scr_orig
            max_iter = max_iter_orig

        loss_list = []
        # loss_comparison_list = []
        loss_without_hyperparam_list = []
        mclust_comp = []

        const_factor = 1000.0
        end_pre_heating = False

        for batch_idx in (range(max_iter)):

            # forwarding
            optimizer.zero_grad()   # ******************** check ********************

            output = model( data )[ 0 ]
            output[:, backgrounds[:, 0], backgrounds[:, 1]] = 0 # Big problem, as all these 1s will be normalized

            output = output.permute( 1, 2, 0 )
            output = output.contiguous().view( -1, nChannel )

            outputHP = output.reshape( (im.shape[0], im.shape[1], nChannel) )


            HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
            HPy[up_border[:, 0] - 1, up_border[:, 1], :] = 0
            HPy[down_border[:, 0], down_border[:, 1], :] = 0

            HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
            HPz[left_border[:, 0], left_border[:, 1] - 1, :] = 0
            HPz[right_border[:, 0], right_border[:, 1], :] = 0
            
            HP_diag = outputHP[1:,1:, :] - outputHP[0:-1, 0:-1, :]
            HP_diag[nw_border[:, 0] - 1, nw_border[:, 1] - 1, :] = 0
            HP_diag[se_border[:, 0], se_border[:, 1], :] = 0


            lhpy = loss_hpy(HPy, HPy_target)
            lhpz = loss_hpz(HPz, HPz_target)
            lhp_diag = 0
            lhp_diag = loss_hp_diag(HP_diag, HP_diag_target)
            

            ignore, target = torch.max( output, 1 )


            im_target = target.data.cpu().numpy()
            nLabels = len(np.unique(im_target))

            im_ari = im_target.reshape(im.shape[0], im.shape[1])

            
            im_cluster_num = im_target.reshape(im.shape[0], im.shape[1])
            labels = im_cluster_num[pixel_rows_cols[:, 0], pixel_rows_cols[:, 1]]
            df_labels = pd.DataFrame({'label': labels}, index=pixel_barcode[pixel_barcode != ''])
            ari_per_itr.append(calc_ari(df_man, df_labels))
            df_barcode_labels_per_itr[f'itr_{batch_idx}'] = labels

            mclust_labels = mask.reshape(im.shape[0], im.shape[1])[pixel_rows_cols[:, 0], pixel_rows_cols[:, 1]]

            if train_type == 'pre-heat':
                mclust_closeness = compare_with_mclust(mclust_labels, labels)
                mclust_comp.append(mclust_closeness)
                if mclust_closeness >= 0.95:
                    end_pre_heating = True
            
            # if len(np.unique(labels)) < no_of_scribble_layers:
            #     print(f"Lesser amount of labels detected at iteration {batch_idx}!")
            # elif len(np.unique(labels)) > no_of_scribble_layers:
            #     print(f"Higher amount of labels detected at iteration {batch_idx}!")
            

            if visualize and (batch_idx<10 or batch_idx%10 == 0) and train_type == 'post-heat':
            
                im_cluster_num = im_target.reshape(im.shape[0], im.shape[1])
                labels = im_cluster_num[pixel_rows_cols[:, 0], pixel_rows_cols[:, 1]]
                grid_spots, colors = get_grid_spots_from_pixels(pixel_rows_cols, labels)
                if dataset == 'Custom': rad = 700
                else: rad = 10
                plt.figure(figsize=(5.5,5))
                plt.scatter(grid_spots[:, 1], 1000 - grid_spots[:, 0], c=colors, s=rad)
                plt.axis('off')
                plt.savefig(f'{image_per_itr_folder_path}/{train_type}_itr_{batch_idx}.png',format='png',dpi=1200,bbox_inches='tight',pad_inches=0)
                plt.close('all')

            # loss 
            if scribble:
        
                loss_lr = 0
                for i in range(mask_inds.shape[0]):
                    loss_lr += loss_fn_scr(output[ inds_scr_array[i] ], target_scr[ inds_scr_array[i] ])

                # loss_sim = loss_fn(output[ inds_sim ], target[ inds_sim ])
                loss_sim = loss_fn(output[ inds_sim_for_mclust ], target[ inds_sim_for_mclust ])
                hyper_sum = stepsize_sim + stepsize_scr + stepsize_con

                sim_multiplier = 1
                con_multiplier = 1
                scr_multiplier = 1
                L_sim = stepsize_sim * loss_sim * sim_multiplier
                L_scr = stepsize_scr * loss_lr * scr_multiplier

                L_con = stepsize_con * (lhpy + lhpz + lhp_diag) * con_multiplier

                loss_without_hyperparam = loss_sim + loss_lr + (lhpy + lhpz + lhp_diag)

                if hyper_sum_division:
                    loss = (L_sim + L_con + L_scr) / hyper_sum
                else:
                    loss = (L_sim + L_con + L_scr)

            else:
                loss = (stepsize_sim * loss_fn(output, target) + stepsize_con * (lhpy + lhpz + lhp_diag))
                # loss = (stepsize_sim * loss_fn(output, target) + stepsize_con * (lhpy + lhpz + lhp_diag)) / (stepsize_sim + stepsize_scr + stepsize_con )


            loss_without_hyperparam_list.append(loss_without_hyperparam.data.cpu().numpy())
            loss_per_itr.append(loss.data.cpu().numpy())
            

            loss.backward()
            optimizer.step()

            if end_pre_heating:
                print(f'ended pre-heating at {batch_idx}')
                break


        if len(mclust_comp) > 0:
            mclust_closeness = pd.DataFrame({'closeness': mclust_comp, 'itr': list(range(len(mclust_comp)))})
            mclust_closeness.to_csv(f'{leaf_output_folder_path}/mclust_closeness.csv')

            
        output = model( data )[ 0 ]
        output = output.permute( 1, 2, 0 ).contiguous().view( -1, nChannel )
        ignore, target = torch.max( output, 1 )
        im_target = target.data.cpu().numpy()
        im_target_rgb = np.array([label_colours[ c % nChannel ] for c in im_target])
        im_target_rgb = im_target_rgb.reshape( np.array([im.shape[0],im.shape[1],3]).astype( np.uint8 ))
        im_cluster_num = im_target.reshape(im.shape[0], im.shape[1])
        f = im_cluster_num
        s = np.argwhere(f != 110) # not a good way
        colors = f.flatten()
        plt.figure(figsize = (4, 4))
        if dataset == 'Custom': rad = 1500
        else: rad = 10
        plt.scatter(s[:, 1], 1000 - s[:, 0], c=colors, s = rad)


        labels = im_cluster_num[pixel_rows_cols[:, 0], pixel_rows_cols[:, 1]]
        grid_spots, colors = get_grid_spots_from_pixels(pixel_rows_cols, labels)

        df_ari_per_itr = pd.DataFrame({f'{train_type}_ARI': ari_per_itr})
        df_ari_per_itr.to_csv(f'{leaf_output_folder_path}/{train_type}_ari_per_itr.csv')

        df_loss_per_itr = pd.DataFrame({f'{train_type}_Loss': loss_per_itr})
        df_loss_per_itr.to_csv(f'{leaf_output_folder_path}/{train_type}_loss_per_itr.csv')

        df_loss_without_hyperparam_per_itr = pd.DataFrame({f'{train_type}_Loss_without_hyperparam': loss_without_hyperparam_list})
        df_loss_without_hyperparam_per_itr.to_csv(f'{leaf_output_folder_path}/{train_type}_loss_without_hyperparam_per_itr.csv')

        df_labels = pd.DataFrame({f'{train_type}_label': labels}, index=pixel_barcode[pixel_barcode != ''])
        df_labels.to_csv(f'{leaf_output_folder_path}/{train_type}_final_barcode_labels.csv')

        df_final_metrics = pd.DataFrame({f'{train_type}_ARI': df_ari_per_itr[f'{train_type}_ARI'].values[-1:], f'{train_type}_Loss': df_loss_per_itr[f'{train_type}_Loss'].values[-1:], f'{train_type}_Loss_without_hyperparam': df_loss_without_hyperparam_per_itr[f'{train_type}_Loss_without_hyperparam'].values[-1:]})
        df_final_metrics.to_csv(f'{leaf_output_folder_path}/{train_type}_final_metrics.csv')

        df_barcode_labels_per_itr.to_csv(f'{leaf_output_folder_path}/{train_type}_barcode_labels_per_itr.csv')

        print(f"{train_type}_ARI:", calc_ari(df_man, df_labels))
        print(f"{train_type}_L_sim: {L_sim}, L_con: {L_con}, L_scr: {L_scr}")
        print(f"{train_type}_L_sim + L_con + L_scr: {L_sim + L_con + L_scr}")
        print(f"{train_type}_Total loss: {loss_per_itr[-1]}")
        print(f"{train_type}_Loss without hyperparam: {loss_without_hyperparam_list[-1]}")

        meta_data_value = [test_name, seed, dataset, sample, n_pcs, scribble, max_iter, stepsize_sim, stepsize_con, stepsize_scr, scheme, lr, nConv, no_of_scribble_layers, intermediate_channels, added_layers, last_layer_channel_count, hyper_sum_division]
        df_meta_data = pd.DataFrame(index=meta_data_index, columns=[f'{train_type}_value'])
        df_meta_data[f'{train_type}_value'][meta_data_index] = meta_data_value
        df_meta_data.to_csv(f'{leaf_output_folder_path}/{train_type}_meta_data.csv')

        if dataset == 'Custom': rad = 700
        else: rad = 10
        plt.figure(figsize=(5.5,5))
        plt.axis('off')
        plt.scatter(grid_spots[:, 1], 1000 - grid_spots[:, 0], c=colors, s=rad)
        plt.savefig(f'{leaf_output_folder_path}/{train_type}_seg_{stepsize_sim}_{stepsize_con}_{stepsize_scr}_seed_{seed}_pcs_{n_pcs}.png',format='png',dpi=1200,bbox_inches='tight',pad_inches=0)
        plt.savefig(f'{leaf_output_folder_path}/{train_type}_seg_{stepsize_sim}_{stepsize_con}_{stepsize_scr}_seed_{seed}_pcs_{n_pcs}.eps',format='eps',dpi=1200,bbox_inches='tight',pad_inches=0)
        plt.close('all')
    

    train_model('pre-heat')
    train_model('post-heat')


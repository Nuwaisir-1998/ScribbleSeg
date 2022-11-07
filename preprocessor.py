# %%
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import os
import math
import scanpy
import json
import argparse

parser = argparse.ArgumentParser(description='ScribbleSeg Preprocessor')
parser.add_argument('--scheme', help="'expert' for expert scribbles, 'mclst' for mclust label initialization", required=True)

args = parser.parse_args()


# %%

# dataset = 'Custom'
# sample = 'Small_2'
# h5_path = f'./Data/{dataset}/{sample}/reading_h5/'
# n_pcs = 2
# h5_file = 'adata.h5'
# adata = scanpy.read_h5ad(f'{h5_path}/{h5_file}')
# adata

dataset = 'Human_DLPFC'
samples = ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674', '151675', '151676']
for sample in samples:
    h5_path = f'./Data/{dataset}/{sample}/reading_h5/'
    n_pcs = 15
    h5_file = f'{sample}_filtered_feature_bc_matrix.h5'
    adata = scanpy.read_visium(path = h5_path, count_file = h5_file)
    adata.var_names_make_unique()
    

    mclust_scr = False
    if args.scheme == 'mclust':mclust_scr=True

    


    # %%
    local_data_folder_path = './Algorithms/Unsupervised_Segmentation/Approaches/With_Scribbles/Local_Data'
    mapped_pc_file_path = f'./Algorithms/Unsupervised_Segmentation/Approaches/With_Scribbles/Local_Data/{dataset}/{sample}/Npys/mapped_{n_pcs}.npy'
    backgrounds_file_path = f'{local_data_folder_path}/{dataset}/{sample}/Npys/backgrounds.npy'
    foregrounds_file_path = f'{local_data_folder_path}/{dataset}/{sample}/Npys/foregrounds.npy'
    pixel_barcode_file_path = f'{local_data_folder_path}/{dataset}/{sample}/Npys/pixel_barcode.npy'
    map_pixel_to_grid_spot_file_path = f'{local_data_folder_path}/{dataset}/{sample}/Jsons/map_pixel_to_grid_spot.json'

    pc_csv_path = f'./Data/{dataset}/{sample}/Principal_Components/CSV/pcs_{n_pcs}_from_bayesSpace_top_2000_HVGs.csv'
    scr_csv_path = f'./Data/{dataset}/{sample}/manual_scribble.csv'
    scr_file_path = f'./Algorithms/Unsupervised_Segmentation/Approaches/With_Scribbles/Local_Data/{dataset}/{sample}/Scribble/manual_scribble_1.npy'

    if mclust_scr:
        scr_csv_path = f'./Data/{dataset}/{sample}/mclust_result.csv'
        scr_file_path = f'./Algorithms/Unsupervised_Segmentation/Approaches/With_Scribbles/Local_Data/{dataset}/{sample}/Scribble/mclust_scribble.npy'


    # %%
    plt.axis('off')
    plt.scatter(adata.obs['array_col'], 1000 - adata.obs['array_row'], s = 1000)

    # %%
    def make_directory_if_not_exist(path):
        if not os.path.exists(path):
            os.makedirs(path)

    # %%
    make_directory_if_not_exist(f'{local_data_folder_path}/{dataset}/{sample}/Jsons')
    make_directory_if_not_exist(f'{local_data_folder_path}/{dataset}/{sample}/Npys')
    make_directory_if_not_exist(f'{local_data_folder_path}/{dataset}/{sample}/Npzs')
    make_directory_if_not_exist(f'{local_data_folder_path}/{dataset}/{sample}/Scribble')

    # %%
    def make_grid_idx(adata):
        n = adata.obs['array_row'].max() + 1
        m = adata.obs['array_col'].max() + 1
        # barcode_grid = np.empty([n, m], dtype='<U100')
        grid_idx = np.zeros((n, m), dtype='int') - 1
        spot_rows = adata.obs['array_row']
        spot_cols = adata.obs['array_col']
        # barcode_grid[spot_rows, spot_cols] = adata.obs.index
        grid_idx[spot_rows, spot_cols] = range(len(adata.obs.index))
        return grid_idx

    # %%
    def make_grid_barcode(adata):
        n = adata.obs['array_row'].max() + 1
        m = adata.obs['array_col'].max() + 1
        grid_barcode = np.empty([n, m], dtype='<U100')
        # grid_idx = np.zeros((n, m), dtype='int') - 1
        spot_rows = adata.obs['array_row']
        spot_cols = adata.obs['array_col']
        grid_barcode[spot_rows, spot_cols] = adata.obs.index
        # grid_idx[spot_rows, spot_cols] = range(len(adata.obs.index))
        return grid_barcode

    # %%
    def check_grid_validity_return_starting_pos(grid):
        '''
        Check if the grid is valid or not, valid if (i + j) % 2 for all non -1s are equal where i and j can be row and col
        Returns (i + j) % 2 of any 1 present in grid
        '''

        # Efficiency can be improved I think

        n = grid.shape[0]
        m = grid.shape[1]
        parity = -1
        started = False
        for i in range(n):
            for j in range(m):
                if grid[i, j] != -1 and not started:
                    parity = (i + j) % 2
                    started = True
                if grid[i, j] != -1 and parity != (i + j) % 2:
                    print("Invalid grid structure!")
                    return -1
        return parity

    # %%
    def refine(grid):pass

    # %%
    grid_idx = make_grid_idx(adata)
    grid_barcode = make_grid_barcode(adata)
    parity = check_grid_validity_return_starting_pos(grid_idx)

    # %%
    def make_grid_pixel_coor(grid_idx, parity):

        n = grid_idx.shape[0]
        m = grid_idx.shape[1]

        grid_pixel_coor = np.zeros((n, m + 2, 2), dtype=int) - 1
        # print(grid_pixel_coor)

        n = grid_pixel_coor.shape[0]
        m = grid_pixel_coor.shape[1]

        for i in range(n):
            for j in range(m):
                if (i + j) % 2 == parity:
                    if i == 0 and j <= 1:
                        grid_pixel_coor[0, j, :] = 0
                    else:
                        if j <= 1:
                            grid_pixel_coor[i, j, :] = grid_pixel_coor[i - 1, j + 1, :] + [1, 0]
                        else:
                            grid_pixel_coor[i, j, :] = grid_pixel_coor[i, j - 2, :] + [0, 1]
                    

        return grid_pixel_coor[:,:-2]

    # %%
    def get_pixel_to_grid_spot_map(grid_pixel_coor, grid_idx):
        n = grid_pixel_coor.shape[0]
        m = grid_pixel_coor.shape[1]
        map_pixel_to_grid_spot = {}
        for i in range(n):
            for j in range(m):
                if grid_idx[i, j] != -1:
                    map_from = f'({grid_pixel_coor[i, j, 0]}, {grid_pixel_coor[i, j, 1]})'
                    map_to = (i, j)
                    map_pixel_to_grid_spot[map_from] = map_to
        return map_pixel_to_grid_spot


    # %%
    grid_pixel_coor = make_grid_pixel_coor(grid_idx, parity)

    # %%
    map_pixel_to_grid_spot = get_pixel_to_grid_spot_map(grid_pixel_coor, grid_idx)

    # %%
    with open(map_pixel_to_grid_spot_file_path, "w") as outfile:
        json.dump(map_pixel_to_grid_spot, outfile)

    # %%
    pixel_coor = grid_pixel_coor.reshape((-1, 2))

    # %%
    mx = pixel_coor[:, 0].max()
    color = grid_idx.flatten()

    # %%
    color[color != -1] = 1
    if dataset == 'Custom': rad = 2000
    else: rad = 10

    # %%
    # np.where(color==1)[0]

    # %%
    # plt.figure(figsize = (7, 7))
    # plt.scatter(pixel_coor[:, 1], mx - pixel_coor[:, 0], c=color, s=rad)


    # %%
    def make_grid_pc(grid_pixel_coor, grid_barcode, map_barcode_pc):
        mx_row = grid_pixel_coor[:, :, 0].max()
        mx_col = grid_pixel_coor[:, :, 1].max()
        grid_pc = np.zeros((mx_row + 1, mx_col + 1, n_pcs))
        # grid_idx_binary = grid_idx.copy()
        # grid_idx_binary[grid_idx_binary != -1] = 1
        # grid_idx_binary[grid_idx_binary == -1] = 0
        idx_rows_cols = np.argwhere(grid_barcode != '')
        idx_rows = idx_rows_cols[:, 0]
        idx_cols = idx_rows_cols[:, 1]
        pixel_rows_cols = grid_pixel_coor[idx_rows, idx_cols]
        barcode_sequence = grid_barcode[grid_barcode != '']
        # idxs_sequence = grid_idx[grid_idx != -1]
        grid_pc[pixel_rows_cols[:, 0], pixel_rows_cols[:, 1], :] = [map_barcode_pc[barcode] for barcode in barcode_sequence]
        return grid_pc

    # %%
    def make_grid_scr(grid_pixel_coor, grid_barcode, df_barcode_scr):
        mx_row = grid_pixel_coor[:, :, 0].max()
        mx_col = grid_pixel_coor[:, :, 1].max()
        grid_scr = np.zeros((mx_row + 1, mx_col + 1), dtype='int') + 255
        idx_rows_cols = np.argwhere(grid_barcode != '')
        idx_rows = idx_rows_cols[:, 0]
        idx_cols = idx_rows_cols[:, 1]
        pixel_rows_cols = grid_pixel_coor[idx_rows, idx_cols]
        barcode_sequence = grid_barcode[grid_barcode != '']
        # print(barcode_sequence)
        grid_scr[pixel_rows_cols[:, 0], pixel_rows_cols[:, 1]] = [df_barcode_scr['cluster.init'][barcode] for barcode in barcode_sequence]
        return grid_scr

    # %%
    def make_pixel_barcode(grid_pixel_coor, grid_barcode):
        mx_row = grid_pixel_coor[:, :, 0].max()
        mx_col = grid_pixel_coor[:, :, 1].max()
        pixel_barcode = np.empty([mx_row + 1, mx_col + 1], dtype='<U100')
        idx_rows_cols = np.argwhere(grid_barcode != '')
        idx_rows = idx_rows_cols[:, 0]
        idx_cols = idx_rows_cols[:, 1]
        pixel_rows_cols = grid_pixel_coor[idx_rows, idx_cols]
        barcode_sequence = grid_barcode[grid_barcode != '']
        # print(barcode_sequence)
        pixel_barcode[pixel_rows_cols[:, 0], pixel_rows_cols[:, 1]] = barcode_sequence
        return pixel_barcode

    # %%
    pixel_barcode = make_pixel_barcode(grid_pixel_coor, grid_barcode)
    np.save(pixel_barcode_file_path, pixel_barcode)

    # %%
    def make_map_barcode_pc(csv_path):
        df_pc = pd.read_csv(csv_path, index_col=0)
        map_barcode_pc = dict(zip(df_pc.index, df_pc.values))
        return map_barcode_pc
    # map_barcode_to_pc = {}
    # map_barcode_to_pc[]

    # %%
    map_barcode_pc = make_map_barcode_pc(pc_csv_path)

    # %%
    grid_idx = make_grid_idx(adata)
    grid_barcode = make_grid_barcode(adata)
    parity = check_grid_validity_return_starting_pos(grid_idx)
    if parity == -1: refine(grid_idx)

    # grid_01 = make_grid01(grid_idx)
    grid_pixel_coor = make_grid_pixel_coor(grid_idx, parity)
    grid_pc = make_grid_pc(grid_pixel_coor, grid_barcode, map_barcode_pc)

    # %%
    # grid_pc

    # %%
    df_barcode_scr = pd.read_csv(scr_csv_path, index_col=0).fillna(255).astype('int')
    grid_scr = make_grid_scr(grid_pixel_coor, grid_barcode, df_barcode_scr)
    # grid_scr

    # %%
    # np.unique(df_barcode_scr['cluster.init'].values)

    # %%
    # df_barcode_scr['cluster.init'].values

    # %%
    np.save(scr_file_path, grid_scr)
    np.save(mapped_pc_file_path, grid_pc)

    # %%
    def find_boundary_coor(grid_pixel_coor, grid_barcode):
        pixel_coor = grid_pixel_coor.reshape((-1, 2))
        n = pixel_coor[:, 0].max() + 1
        m = pixel_coor[:, 1].max() + 1
        grid_binary = np.zeros((n, m), dtype='int')

        idx_rows_cols = np.argwhere(grid_barcode != '')
        idx_rows = idx_rows_cols[:, 0]
        idx_cols = idx_rows_cols[:, 1]
        pixel_rows_cols = grid_pixel_coor[idx_rows, idx_cols]

        grid_binary[pixel_rows_cols[:, 0], pixel_rows_cols[:, 1]] = 1
        # print(grid_binary)
        right_border = []
        left_border = []
        up_border = []
        down_border = []
        nw_border = []
        se_border = []
        for row_col in pixel_rows_cols:
            i = row_col[0]
            j = row_col[1]
            if j + 1 < m and grid_binary[i, j] == 1 and grid_binary[i, j + 1] == 0:
                right_border.append([i, j])
            if j - 1 >= 0 and grid_binary[i, j] == 1 and grid_binary[i, j - 1] == 0:
                left_border.append([i, j])
            if i + 1 < n and grid_binary[i, j] == 1 and grid_binary[i + 1, j] == 0:
                down_border.append([i, j])
            if i - 1 >= 0 and grid_binary[i, j] == 1 and grid_binary[i - 1, j] == 0:
                up_border.append([i, j])
            if i - 1 >= 0 and j - 1 >= 0 and grid_binary[i, j] == 1 and grid_binary[i - 1, j - 1] == 0:
                nw_border.append([i, j])
            if i + 1 < n and j + 1 < m and grid_binary[i, j] == 1 and grid_binary[i + 1, j + 1] == 0:
                se_border.append([i, j])
        borders = {}
        borders['right_border'] = np.array(right_border).reshape(-1, 2)
        borders['left_border'] = np.array(left_border).reshape(-1, 2)
        borders['up_border'] = np.array(up_border).reshape(-1, 2)
        borders['down_border'] = np.array(down_border).reshape(-1, 2)
        borders['nw_border'] = np.array(nw_border).reshape(-1, 2)
        borders['se_border'] = np.array(se_border).reshape(-1, 2)
        return borders


    # %%
    def find_backgrounds(grid_pixel_coor, grid_barcode):
        pixel_coor = grid_pixel_coor.reshape((-1, 2))
        n = pixel_coor[:, 0].max() + 1
        m = pixel_coor[:, 1].max() + 1
        grid_binary = np.zeros((n, m), dtype='int')
        # grid_binary[pixel_coor[:, 0], pixel_coor[:, 1]] = 1
        # backgrounds = np.argwhere(grid_binary == 0)
        idx_rows_cols = np.argwhere(grid_barcode != '')
        idx_rows = idx_rows_cols[:, 0]
        idx_cols = idx_rows_cols[:, 1]
        pixel_rows_cols = grid_pixel_coor[idx_rows, idx_cols]

        grid_binary[pixel_rows_cols[:, 0], pixel_rows_cols[:, 1]] = 1
        background = np.argwhere(grid_binary == 0)
        foreground = np.argwhere(grid_binary == 1)
        return background, foreground

    # %%
    background, foreground = find_backgrounds(grid_pixel_coor, grid_barcode)


    # %%
    np.save(backgrounds_file_path, background)
    np.save(foregrounds_file_path, foreground)

    # %%
    # def find_backgrounds():

    # %%
    borders = find_boundary_coor(grid_pixel_coor, grid_barcode)
    # borders['right_border']


    # %%
    npz_folder_path = f'./Algorithms/Unsupervised_Segmentation/Approaches/With_Scribbles/Local_Data/{dataset}/{sample}/Npzs'
    np.savez(f'{npz_folder_path}/borders.npz', right_border = borders['right_border'], up_border = borders['up_border'], left_border = borders['left_border'], down_border = borders['down_border'], nw_border = borders['nw_border'], se_border = borders['se_border'])

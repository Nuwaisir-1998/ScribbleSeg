import os
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np

def make_directory_if_not_exist(path):
        if not os.path.exists(path):
            os.makedirs(path)

def make_str(x):
        return f'({x[0]}, {x[1]})'

def calc_ari(df_1, df_2):
    df_merged = pd.merge(df_1, df_2, left_index=True, right_index=True).dropna()
    cols = df_merged.columns
    for col in cols:
        df_merged[col] = df_merged[col].values.astype('int')
    return adjusted_rand_score(df_merged[cols[0]].values, df_merged[cols[1]].values)


def get_grid_spots_from_pixels(pixels, colors, map_pixel_to_grid_spot):
    grid_spots = np.array([map_pixel_to_grid_spot[make_str(pixel)] for pixel in pixels if make_str(pixel) in map_pixel_to_grid_spot])
    predicted_colors = [colors[i] for i in range(len(pixels)) if make_str(pixels[i]) in map_pixel_to_grid_spot]
    return grid_spots, predicted_colors

def relabel_mask(mask, background_val):
    row, col = mask.shape
    mask = mask.reshape(-1)
    values = np.unique(mask[mask != background_val])
    lookup = {k: v for v, k in enumerate(dict.fromkeys(values))}
    lookup[background_val] = background_val
    mask = np.array([lookup[i] for i in mask])
    return mask.reshape(row, col)
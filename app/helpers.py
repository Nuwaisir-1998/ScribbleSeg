import os
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score


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
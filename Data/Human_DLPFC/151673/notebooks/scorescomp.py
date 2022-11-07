
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import *
import os

outputbasefolder = '../../../../Algorithms/Unsupervised_Segmentation/Approaches/With_Scribble/Outputs/Human_DLPFC/Report'

co_ordinates = pd.read_csv('../Coordinates/coordinates.csv',index_col=0,header=0)
manual_annotation = pd.read_csv('../manual_annotations.csv',index_col=0,header=0)
pc_s = pd.read_csv('../Principal_Components/CSV/15_pcs.csv',index_col=0,header=0)
pr_dir  = '../../../../Algorithms/Unsupervised_Segmentation/Approaches/With_Scribble/Outputs/Human_DLPFC/151673/'

df = pd.DataFrame(columns=['filename','manual_score','manual_score_distance','ours_score','ours_distance'])

df_schillote = pd.DataFrame(columns=['filename','manual_score','manual_score_distance','ours_score','ours_distance'])

for init_file in os.listdir(pr_dir):
    print(init_file)
    # fname = f'../../../../Algorithms/Unsupervised_Segmentation/Approaches/With_Scribble/Outputs/Human_DLPFC/151673/{init_file}/Outputs/component_labels.csv'
    fname = '../components_manual_annotations.csv'

    if not os.path.exists(fname):
        continue
    mclust_init = pd.read_csv(fname,index_col=0,header=0)

    if mclust_init['label'].nunique() < 1:
        continue

    c2 = (co_ordinates - co_ordinates.mean())*10/ co_ordinates.std()
    # c2 = co_ordinates

    combined_df = pd.concat([c2,manual_annotation,mclust_init,pc_s],axis=1)
    combined_df['label'].fillna(-1,inplace=True)
    # combined_df = combined_df.dropna(subset=['label'])
    # combined_df['label'] = combined_df['manual_annotation']
    combined_df.dropna(inplace=True)
    # combined_df['label'] = combined_df['label'].astype(int)
    # combined_df.drop(combined_df[combined_df['label']==40].index,inplace=True)



    label_map = {
        0: 'WM',
        6: 'L1',
        5: 'L2',
        4: 'L3',
        3: 'L4',
        2: 'L5',
        1: 'L6',
        7: 'L7',
        -1: 'undetected',
    }
    # combined_df['label'] = combined_df['label'].map(label_map)


    combined_df


    features = combined_df[pc_s.columns].values
    features_coordinate = combined_df[['imagerow','imagecol']].values
    labels = combined_df['manual_annotation'].values
    features_cor = combined_df[pc_s.columns.to_list()+['imagerow','imagecol']].values
    labels_ours = combined_df['label'].values



    Calinski_Harabaz_score = calinski_harabasz_score(features,labels)
    #print('Calinski Harabaz Score: ',Calinski_Harabaz_score)
    calinski_harabasz_score_features_coordinates = calinski_harabasz_score(features_cor,labels)
    #print('Calinski Harabaz Score for features coordinates: ',calinski_harabasz_score_features_coordinates)
    calinski_harabasz_score_ours = calinski_harabasz_score(features,labels_ours)
    #print('Calinski Harabaz Score ours: ',calinski_harabasz_score_ours)
    calinski_harabasz_score_features_coordinates_ours = calinski_harabasz_score(features_cor,labels_ours)
    #print('Calinski Harabaz Score for features coordinates ours: ',calinski_harabasz_score_features_coordinates_ours)

    df = df.append({'filename':init_file,'manual_score':Calinski_Harabaz_score,'manual_score_distance':calinski_harabasz_score_features_coordinates,'ours_score':calinski_harabasz_score_ours,'ours_distance':calinski_harabasz_score_features_coordinates_ours},ignore_index=True)

    silhouette_cosine = silhouette_score(features,labels,metric='cosine')
    silhouette_all = silhouette_score(features_coordinate,labels,metric='cosine')
    silhouette_cosine_ours = silhouette_score(features,labels_ours,metric='cosine')
    silhouette_all_ours = silhouette_score(features_cor,labels_ours,metric='cosine')
    
    df_schillote = df_schillote.append({'filename':init_file,'manual_score':silhouette_cosine,'manual_score_distance':silhouette_all,'ours_score':silhouette_cosine_ours,'ours_distance':silhouette_all_ours},ignore_index=True)


df.to_csv(f'{outputbasefolder}/calinski_harabasz_score.csv',index=False)
df_schillote.to_csv(f'{outputbasefolder}/schillote_score.csv',index=False)








import os
import random
import numpy  as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, TweetTokenizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import tree

#Functions to load dataset features
    #Funtcions to load OpenFace extracted features
  
def extract_OpenFace_features(csv, criterion='all', N_chunk=20):
    #FAUs id with Intensisty and Presense
    AU_id      = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45] 
    gaze_names = [' gaze_0_x',' gaze_0_y', ' gaze_0_z',
                       ' gaze_1_x',' gaze_1_y', ' gaze_1_z',
                       ' gaze_angle_x',' gaze_angle_y']
    head_names = [' pose_Tx',' pose_Ty',' pose_Tz',
                  ' pose_Rx',' pose_Ry',' pose_Rz']
    AUc_names  = [" AU{0:>02d}_c".format(i) for i in AU_id]
    AUr_names  = [" AU{0:>02d}_r".format(i) for i in AU_id]
    
    AU_vec   = []
    gaze_vec = []
    head_vec = []
    if csv.shape[0] == 0:
        return None, None, None
    
    if criterion == 'all': #jus take the average over each selected row in current csv
        g_vec = csv[gaze_names].to_numpy(dtype=np.float32)
        h_vec = csv[head_names].to_numpy(dtype=np.float32)
        f_vec = csv[AUr_names].to_numpy(dtype=np.float32)
        #At this time we have a matrix for each feature
        AU_vec   = np.mean(f_vec, axis=0)
        gaze_vec = np.mean(g_vec, axis=0)
        head_vec = np.mean(h_vec, axis=0)
        
        return AU_vec, gaze_vec, head_vec
    
    step     = csv.shape[0]//N_chunk
    faces_20 = [i*step for i in range(N_chunk)]
        
    #print("step/faces: {}/{}".format(step, sub_csv.shape[0]), end="\t")
    for i in faces_20:
        sample = csv.iloc[i]
        g_vec  = np.zeros(len(sample[gaze_names]), dtype=np.float32)
        h_vec  = np.zeros(len(sample[head_names]), dtype=np.float32)
        f_vec  = np.zeros(len(sample[AUr_names]),  dtype=np.float32)
        
        if criterion == 'single':
            g_vec = sample[gaze_names].to_numpy(dtype=np.float32)
            h_vec = sample[head_names].to_numpy(dtype=np.float32)
            f_vec = sample[AUr_names].to_numpy(dtype=np.float32)
        
        elif criterion == 'sum' or criterion=='mean':
            g_vec = np.sum(csv.iloc[list(range(i,i+step))][gaze_names].to_numpy(dtype=np.float32) ,axis=0)
            h_vec = np.sum(csv.iloc[list(range(i,i+step))][head_names].to_numpy(dtype=np.float32) ,axis=0)
            f_vec = np.sum(csv.iloc[list(range(i,i+step))][AUr_names].to_numpy(dtype=np.float32) ,axis=0)
            
            if criterion=='mean':
                g_vec /= step
                h_vec /= step
                f_vec /= step
                
        AU_vec.append(f_vec)
        gaze_vec.append(g_vec)
        head_vec.append(h_vec)
        
    return np.array(AU_vec), np.array(gaze_vec), np.array(head_vec)

def get_OpenFace_feat(filtered_df, criterion='single', N_chunk=20, 
                      dataset1_OF_path=os.path.normpath(r"data/dataset1_OpenFace_clean_emb")):
    dataset1_label = []
    dataset1_AU    = []
    dataset1_gaze  = []
    dataset1_head  = []
    for n in tqdm(filtered_df['id']):
        sub_dir = 'Deceptive' if 'lie' in n else "Truthful"
        dataset1_label.append(1 if 'lie' in n else 0)
        sam_df  = pd.read_csv(os.path.join(dataset1_OF_path,sub_dir,n.replace(".mp4","_emb.csv")))
        sam_arr = extract_OpenFace_features(sam_df, criterion=criterion, N_chunk=N_chunk)
        dataset1_AU.append(sam_arr[0])
        dataset1_gaze.append(sam_arr[1])
        dataset1_head.append(sam_arr[2])

    dataset1_label = np.array(dataset1_label)
    dataset1_AU    = np.array(dataset1_AU)
    dataset1_gaze  = np.array(dataset1_gaze)
    dataset1_head  = np.array(dataset1_head)
    
    return dataset1_AU, dataset1_gaze, dataset1_head, dataset1_label
  
  

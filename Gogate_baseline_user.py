import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from PIL import Image

import time
import shutil
import argparse
import functools
import copy
import random
import nltk
from nltk.tokenize import word_tokenize, TweetTokenizer
nltk.data.path.append("/home/est_posgrado_abdiel.beltran/ntlk_data")

#functions
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def apply_Znorm(x):
    m = np.mean(x)
    s = np.mean(x)
    return (x-m)/s

#create single feature extractors
def get_video_model(conv_act='relu'):
    chan_format = 'channels_last'
    input = keras.Input((10,128,128,1),dtype='float32')
    x = keras.layers.Conv3D(16,2, data_format=chan_format,
                                  activation=conv_act)(input)
    x = keras.layers.Conv3D(32,2, data_format=chan_format,
                                  activation=conv_act)(x)
    x = keras.layers.MaxPool3D((1,2,2), data_format=chan_format)(x)
    x = keras.layers.Conv3D(64,2, data_format=chan_format,
                                  activation=conv_act)(x)
    x = keras.layers.MaxPool3D(2, data_format=chan_format)(x)
    x = keras.layers.Conv3D(64,2, data_format=chan_format,
                                  activation=conv_act)(x)
    x = keras.layers.MaxPool3D((1,2,2), data_format=chan_format)(x)
    x = keras.layers.Flatten()(x)

    return keras.Model(inputs=input, outputs=x)

def get_text_model(conv_act='relu'):
    chan_format = 'channels_first' #format [features, steps]
    input = keras.Input((300,100),dtype='float32')
    x     = keras.layers.Conv1D(15,2, data_format=chan_format,
                                      activation=conv_act)(input)
    x     = keras.layers.MaxPool1D(2, data_format=chan_format)(x)
    x     = keras.layers.Conv1D(15,2, data_format=chan_format,
                                      activation=conv_act)(x)
    x     = keras.layers.MaxPool1D(2, data_format=chan_format)(x)
    x     = keras.layers.Conv1D(15,2, data_format=chan_format,
                                      activation=conv_act)(x)
    x     = keras.layers.MaxPool1D(2, data_format=chan_format)(x)
    x     = keras.layers.Conv1D(15,2, data_format=chan_format,
                                      activation=conv_act)(x)
    x     = keras.layers.MaxPool1D(2, data_format=chan_format)(x)
    x     = keras.layers.Flatten()(x)

    return keras.Model(inputs=input, outputs=x)


#models
def do_5FCV(args):
    conv_activation = args.conv_activation
    mlp_activation  = args.mlp_activation
    #Load predefined data splits
    i = 0
    fold_acc, fold_rec, fold_pre, fold_f1 = [],[],[],[]
    for i in range(5):
        i += 1
        train_idx, test_idx = [], []
        with open( os.path.join(args.fold_path, f"fold_{i}_train.txt"),'r') as f:
            tr_id = [int(v) for v in f.readlines()]
        with open( os.path.join(args.fold_path, f"fold_{i}_test.txt"),"r") as f:
            ts_id = [int(v) for v in f.readlines()]
        
        tr_id = np.array(tr_id)  # user id for train
        ts_id = np.array(ts_id)  # user id for test
  
        # Now save sample idx for users in each partition
        train_idx = np.array([ idx for idx,s_id in enumerate(args.sample_subject) if s_id in tr_id], dtype=np.int32)
        test_idx  = np.array([ idx for idx,s_id in enumerate(args.sample_subject) if s_id in ts_id], dtype=np.int32)
        
        # create dataset partitions
        if args.model == 'early':
            train_f = []
            test_f  = []
            if 'T' in args.modalities:
                train_f.append(args.feat_T[train_idx])
                test_f.append( args.feat_T[test_idx])
            if 'A' in args.modalities:
                train_f.append(args.feat_A[train_idx])
                test_f.append( args.feat_A[test_idx])
            if 'V' in args.modalities:
                train_f.append(args.feat_V[train_idx])
                test_f.append( args.feat_V[test_idx])
            
            train_l = args.label[train_idx]
            test_l  = args.label[test_idx]
        else:
            train_f = args.feat[train_idx]
            train_l = args.label[train_idx]
        
            test_f = args.feat[test_idx]
            test_l = args.label[test_idx]
        
        # Choose model
        if args.model=='textCNN':
            chan_format = 'channels_first' # [features, step]
            model = keras.Sequential([
                        keras.layers.InputLayer(input_shape=(300,100),dtype='float32'),
                        keras.layers.Conv1D(15,2, data_format=chan_format,
                                                  activation=conv_activation),
                        keras.layers.MaxPool1D(2, data_format=chan_format),
                        keras.layers.Conv1D(15,2, data_format=chan_format,
                                                  activation=conv_activation),
                        keras.layers.MaxPool1D(2, data_format=chan_format),
                        keras.layers.Conv1D(15,2, data_format=chan_format,
                                                  activation=conv_activation),
                        keras.layers.MaxPool1D(2, data_format=chan_format),
                        keras.layers.Conv1D(15,2, data_format=chan_format,
                                                  activation=conv_activation),
                        keras.layers.MaxPool1D(2, data_format=chan_format),
                        keras.layers.Flatten(), 
                        keras.layers.Dense(5000,activation=mlp_activation),
                        keras.layers.Dense(500, activation=mlp_activation),
                        keras.layers.Dense(2,   activation='softmax')
                     ])
        elif args.model=='audioMLP':
            model = keras.Sequential([
                         keras.layers.InputLayer(6373,dtype='float32'),
                         keras.layers.Dense(5000,activation='relu'),
                         keras.layers.Dense(500, activation='relu'),
                         keras.layers.Dense(2,   activation='softmax')
                    ])

        elif args.model=='videoCNN':
            chan_format = 'channels_last'
            model = keras.Sequential([
                         keras.layers.InputLayer((10,128,128,1),dtype='float32'),
                         keras.layers.Conv3D( 16, (2,2,2), data_format=chan_format,
                                                           activation=conv_activation),
                         keras.layers.Conv3D( 32, (2,2,2), data_format=chan_format,
                                                           activation=conv_activation),
                         keras.layers.MaxPool3D( (1,2,2),  data_format=chan_format),
                         keras.layers.Conv3D(64, (2,2,2),  data_format=chan_format,
                                                           activation=conv_activation),
                         keras.layers.MaxPool3D( (2,2,2),  data_format=chan_format),
                         keras.layers.Conv3D(64,(2,2,2),   data_format=chan_format,
                                                           activation=conv_activation),
                         keras.layers.MaxPool3D((1,2,2),   data_format=chan_format),
                         keras.layers.Flatten(),
                         keras.layers.Dense(5000,activation=mlp_activation),
                         keras.layers.Dense(500, activation=mlp_activation),
                         keras.layers.Dense(2,   activation='softmax')
                     ])
	
        elif args.model=='early':
            comb_out = []
            comb_inp = []
            if 'T' in args.modalities:
                T_enc = get_text_model(conv_activation)
                comb_out.append(T_enc.output)
                comb_inp.append(T_enc.input)
            if 'A' in args.modalities:
                A_in = keras.Input(shape=(6373))
                comb_out.append(A_in)
                comb_inp.append(A_in)
            if 'V' in args.modalities:
                V_enc = get_video_model(conv_activation)
                comb_out.append(V_enc.output)
                comb_inp.append(V_enc.input)
            
            comb = keras.layers.concatenate(comb_out)
            mlp = keras.layers.Dense(5000,activation='relu')(comb)
            mlp = keras.layers.Dense(1000, activation='relu')(mlp)
            mlp = keras.layers.Dense(100,  activation='relu')(mlp)
            mlp = keras.layers.Dense(2,   activation='softmax')(mlp)

            model = keras.Model(inputs=comb_inp, outputs=mlp)



        model.compile(optimizer= args.optimizer, loss='sparse_categorical_crossentropy',
                       metrics = ['accuracy'])
        
        set_seed(args.s)
        model.fit(train_f, train_l, epochs=args.n_epochs, batch_size=args.b_sz, shuffle=True, verbose=0)

        test_loss, test_acc = model.evaluate(test_f, test_l, verbose=0)
        
        fold_acc.append(test_acc)
        #compute also recall, precision and F1 score
        test_pred = model(test_f, training=False).numpy()
        #calculate argmax to get predicted class
        test_pred = np.argmax(test_pred, axis=1)
        fold_pre.append(precision_score(test_l, test_pred, zero_division=0))
        fold_rec.append(recall_score(test_l, test_pred, zero_division=0))
        fold_f1.append(f1_score(test_l, test_pred, zero_division=0))
    #Add the average of each metric 
    fold_acc.append(np.mean(fold_acc))
    fold_pre.append(np.mean(fold_pre))
    fold_rec.append(np.mean(fold_rec))
    fold_f1.append(np.mean(fold_f1))
    df_metric = pd.DataFrame.from_dict({'fold_acc':fold_acc,'fold_pre':fold_pre,'fold_rec':fold_rec,'fold_f1':fold_f1})
    #clear memory in gpu
    keras.backend.clear_session()
    return df_metric

#Load features
dataset1_subject_id_df = pd.read_excel("data/Dataset1_subject_id.xlsx")

#load dataset text and labels
dataset1_data_path = "data/Real-life_Deception_Detection_2016"
dataset1_text_path    = os.path.join(dataset1_data_path,"Transcription")
dataset1_compare_path = "data/Dataset1_trial_OpenSmile_compare"
dataset1_csv  = pd.read_csv(os.path.join(dataset1_data_path,"Annotation","All_Gestures_Deceptive and Truthful.csv"))

skip_samples   = ["trial_lie_053","trial_lie_055","trial_truth_041","trial_truth_017"]
sample_names   = list(dataset1_csv['id'])
remain_samples = [i for i,n in enumerate(sample_names) if n.replace(".mp4","") not in skip_samples]

filtered_df = dataset1_csv.iloc[remain_samples]

fold_path   = "user_5FCV_split"

dataset1_text = []
dataset1_label = []
dataset1_compare_fun = []
dataset1_sample_subject = []

for idx in range(filtered_df.shape[0]):
    sample = filtered_df.iloc[idx]
    lab = 0
    if sample['class']=='deceptive': lab=1
    dataset1_label.append(lab)
    transcript_id = sample['id'].replace('.mp4','.txt')
    sample_type = 'Truthful'
    if lab: sample_type='Deceptive'

    #add subject
    sample_subject = dataset1_subject_id_df.loc[dataset1_subject_id_df['clip']==sample['id'].replace(".mp4","")]
    dataset1_sample_subject.append(list(sample_subject['id'])[0])

    # store audio functionals
    with open(os.path.join(dataset1_text_path,sample_type,transcript_id),'r',encoding='utf-8') as f:
        dataset1_text.append(f.read().lower())

    audio_csv = pd.read_csv(os.path.join(dataset1_compare_path,sample_type,transcript_id.replace(".txt","_functionals.csv")))
    audio_col = list(audio_csv.columns[3:])
    dataset1_compare_fun.append(audio_csv.iloc[0][audio_col].to_numpy())

dataset1_label = np.array(dataset1_label, dtype= np.int32)
dataset1_compare_fun = np.array(dataset1_compare_fun, dtype=np.float32)

glove_dict = {}
with open("glove.6B.300d.txt","r",encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:],"float32")
        glove_dict[word] = vector

dataset1_glove_100concat = []
for t in dataset1_text:
    toks = word_tokenize(t)
    tok_len = len(toks)
    zero_pad = True         #Flag to use zero-padding
    emb = np.zeros((100,300),dtype=np.float32)
    if tok_len>=100:
        toks = toks[:100]
    else:
        toks += ['pad' for i in range(100 - tok_len)]
    for i,w in enumerate(toks):
        emb[i] = glove_dict[w] if w in glove_dict else glove_dict['unk']
    # at this poin we have the sequence (100, 300)
    if zero_pad and tok_len<100:
        emb[tok_len:,:] = 0
    dataset1_glove_100concat.append(emb)

dataset1_glove_100concat = np.array(dataset1_glove_100concat)
dataset1_glove_100concat = np.transpose(dataset1_glove_100concat,(0,2,1))  #reshape to 300x100 

#load 10 images per sample
dataset1_img_path = "/home/est_posgrado_abdiel.beltran/dataset1_3dcnn_10frames_clean"

dataset1_10frames = []
for idx in range(filtered_df.shape[0]):
    sample = filtered_df.iloc[idx]
    sample_name = sample['id'].replace(".mp4","")
    sample_type = 'Truthful'
    if sample['class'] == 'deceptive':  sample_type = 'Deceptive'
    frames = np.load( os.path.join(dataset1_img_path,sample_type,sample_name+"_10face.npy" ) )
    dataset1_10frames.append(frames)

dataset1_10frames = np.array(dataset1_10frames, dtype=np.float32)

#Set args 
general_seed = 1234
args = argparse.Namespace()
args.fold_path = fold_path
args.sample_subject = dataset1_sample_subject
args.modalities = 'TAV'
args.feat_T   = dataset1_glove_100concat
args.feat_A   = dataset1_compare_fun
args.feat_V   = dataset1_10frames
args.label    = dataset1_label
args.model  = 'early'
args.b_sz   = 15
args.n_epochs = 50
args.optimizer = 'Adam'
args.s        = general_seed
#for different combinations of conv_activation and mlp_activation in unimodal models
args.conv_activation = None
args.mlp_activation  = 'relu'

save_dir = "gogate_user_result"
if os.path.exists(save_dir)==False:
    os.mkdir(save_dir)

args.modalities = 'V'
#args.feat  = dataset1_glove_100concat
#args.model = 'textCNN'

#args.feat  = dataset1_compare_fun
#args.model = 'audioMLP'

#args.feat  =  dataset1_10frames
#args.model = 'videoCNN'

args.model = 'early'

for try_opt in ['RMSprop','Adam']:
    args.optimizer = try_opt
    for try_mod in ['TA','TV','AV','TAV']:
        args.modalities = try_mod
        df = do_5FCV(args)
        if args.model == 'early':
            df.to_csv(os.path.join(save_dir,"{}_{}_Bsz{}_Nep{}_opt_{}_{}.csv".format(args.model,args.modalities,args.b_sz,args.n_epochs,args.optimizer,args.conv_activation)),index=False)
        else:
            df.to_csv(os.path.join(save_dir,"{}_Bsz{}_Nep{}_opt_{}_{}.csv".format(args.model,args.b_sz, args.n_epochs, args.optimizer,args.conv_activation)),index=False)



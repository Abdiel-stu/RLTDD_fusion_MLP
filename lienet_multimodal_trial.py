#Import libraries
import math
import sys
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt


import random
import shutil
import functools
from collections import Counter
import argparse
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split

#For logger
#import logging
import time
#from datetime import timedelta

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
#import torchvision.transforms.functional as TF

from argparse import Namespace

#Define LieNet model
class LieNet_v1(nn.Module):
    def __init__(self, params):
        super(LieNet_v1, self).__init__()
        self.input_dim = params.input_dim
        H, W, in_chan = self.input_dim
        #Conv1 compute over input image
        self.conv1 = nn.Conv2d(in_chan, 16, kernel_size=(3,3), stride=(1,1), padding=0)
        #First Block, compute over conv1 output, as we have padding=same  (height, width) are preserved
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(7,7), stride=(1,1), padding=0)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(1,1), stride=(1,1), padding=0)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=(3,3), stride=(1,1), padding=0)
        self.conv5 = nn.Conv2d(16, 32, kernel_size=(1,1), stride=(1,1), padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=(3,3), stride=(1,1), padding=0)

        #Block 2 compute over block_1's outputs  (as all layer have padding=same height and width are preserved)
        self.conv6 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=(1,1), padding=0)
        self.conv7 = nn.Conv2d(32, 64, kernel_size=(5,5), stride=(1,1), padding=0)
        self.conv8 = nn.Conv2d(16, 64, kernel_size=(1,1), stride=(1,1), padding=0)

        #This work over C2 output will downsize images (stride = 2)
        self.pool2 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=0)     
        #Pool3 work over [C5,C6,C7,C8] concatenated (H,W, 3*64 + 32)
        self.pool3 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=0)

        #Conv9 will work over [Pool2, Pool3] concateanted   
        self.conv9 = nn.Conv2d(64*3+32+32, 64, kernel_size=(1,1), stride=(1,1), padding=0)

        #Pool4 will work over conv9 output
        self.pool4 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=0)

        #Compute H_mid, W_mid to calculate input_dim for FC1
        H_mid = int( (H-3)/2 + 1 )
        W_mid = int( (W-3)/2 + 1 )

        H_flat = int( (H_mid-3)/2 + 1 )
        W_flat = int( (W_mid-3)/2 + 1 )

        #print("FC1 with input: ",H_flat*W_flat*64)
        self.FC1 = nn.Linear( H_flat*W_flat*64, 128)
        self.FC2 = nn.Linear(128, 256)
        self.FC3 = nn.Linear(256,2)                     #Output layer

        #Padding for convolutions following TensorFlow format    
        self.c1_Hpad = ((H-1)+3-H)//2
        self.c1_Wpad = ((W-1)+3-W)//2

        self.c2_Hpad = ((H-1)+7-H)//2
        self.c2_Wpad = ((W-1)+7-W)//2

        self.c3_Hpad = ((H-1)+1-H)//2
        self.c3_Wpad = ((W-1)+1-W)//2

        self.c4_Hpad = ((H-1)+3-H)//2
        self.c4_Wpad = ((W-1)+3-W)//2

        self.c5_Hpad = ((H-1)+1-H)//2
        self.c5_Wpad = ((W-1)+1-W)//2

        self.c6_Hpad = ((H-1)+3-H)//2
        self.c6_Wpad = ((W-1)+3-W)//2

        self.c7_Hpad = ((H-1)+5-H)//2
        self.c7_Wpad = ((W-1)+5-W)//2

        self.c8_Hpad = ((H-1)+1-H)//2
        self.c8_Wpad = ((W-1)+1-W)//2
        
        self.p1_Hpad = ((H-1)+3-H)//2
        self.p1_Wpad = ((W-1)+3-W)//2
        
        self.p2_Hpad = (2*(H-1)+3-H)//2
        self.p2_Wpad = (2*(W-1)+3-W)//2
        #initialize all weights with xavier_uniform
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.init.xavier_uniform_(self.conv5.weight)
        nn.init.xavier_uniform_(self.conv6.weight)
        nn.init.xavier_uniform_(self.conv7.weight)
        nn.init.xavier_uniform_(self.conv8.weight)
        nn.init.xavier_uniform_(self.conv9.weight)
       
    def forward(self, x):
        """  Assume that x is of size [Batch_size, Channels, Height, Width]
        """
        #First conv
        x = F.relu(self.conv1(nn.functional.pad(x,(self.c1_Wpad, self.c1_Wpad, self.c1_Hpad, self.c1_Hpad))))
        #First block of convs and 1 pool
        c2 = F.relu(self.conv2(nn.functional.pad(x,(self.c2_Wpad, self.c2_Wpad, self.c2_Hpad, self.c2_Hpad))))
        c3 = F.relu(self.conv3(nn.functional.pad(x,(self.c3_Wpad, self.c3_Wpad, self.c3_Hpad, self.c3_Hpad))))
        c4 = F.relu(self.conv4(nn.functional.pad(x,(self.c4_Wpad, self.c4_Wpad, self.c4_Hpad, self.c4_Hpad))))
        p1 = self.pool1(nn.functional.pad(x,(self.p1_Wpad, self.p1_Wpad, self.p1_Hpad, self.p1_Hpad),value=1e-9))
        c5 = F.relu(self.conv5(nn.functional.pad(x,(self.c5_Wpad, self.c5_Wpad, self.c5_Hpad, self.c5_Hpad))))
        #Second block of convs and 1 pool
        p2 = self.pool2(c2)
        c6 = F.relu(self.conv6(nn.functional.pad(c3,(self.c6_Wpad, self.c6_Wpad, self.c6_Hpad, self.c6_Hpad))))
        c7 = F.relu(self.conv7(nn.functional.pad(c4,(self.c7_Wpad, self.c7_Wpad, self.c7_Hpad, self.c7_Hpad))))
        c8 = F.relu(self.conv8(nn.functional.pad(p1,(self.c8_Wpad, self.c8_Wpad, self.c8_Hpad, self.c8_Hpad))))
        #Concat [C5,C6,C7,C8] and pass to pool3
        convs = torch.cat([c5,c6,c7,c8], dim=1)
        p3 = self.pool3(convs)
        #Concat [P2,P3] and pass to conv9->pool4-> FC1 -> FC2 -> out_layer
        pools = torch.cat([p2,p3], dim=1)
        c9    = F.relu(self.conv9(pools))
        p4    = self.pool4(c9)
        p4    = torch.flatten(p4, start_dim=1)

        f1 = self.FC1(p4)
        f2 = self.FC2(f1)
        out = self.FC3(f2)
        return out

#Define dataset and helpers functions
def get_labels_and_frequencies(path):
    label_freqs = Counter()
    data_csv = pd.read_csv(path)
    data_labels = list(data_csv["class"])
    n_labels = len(data_labels)
    n_lies   = np.sum(data_labels)          #Lie is coded as 1, truth as 0
    label_freqs.update({'lie':n_lies})
    label_freqs.update({'truth': n_labels - n_lies})
    return list(label_freqs.keys()), label_freqs

class csvDataset(Dataset):
  def __init__(self, data_path, tokenizer, vocab, args):
    self.data = pd.read_csv(data_path)
    self.data_dir = os.path.dirname(data_path)
    self.tokenizer = tokenizer
    self.args = args
    self.vocab = vocab
    self.n_classes = args.n_classes
    self.modality = args.modality
    self.augmentation = args.augmentation

  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, index):
    csv_sample = self.data.iloc[index]
    label = torch.LongTensor( [csv_sample['class']] )
    if label==1:
        class_path = os.path.join(self.data_dir,"Deceptive")
    else:
        class_path = os.path.join(self.data_dir,"Truthful")

    if self.args.modality == 'A':
        img_path = os.path.join(class_path,csv_sample['audio'])
    elif self.args.modality=='V':
        img_path = os.path.join(class_path,csv_sample['video'])

    img=cv2.imread(img_path)
    if self.args.modality == 'A':
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = torch.Tensor(img).permute((2,0,1))  #Change order to (C, H, W)

    return label, img

#Collate_fn to compact in batches Dataset output
def collate_fn(batch, args):
    img_tensor = torch.stack([row[1] for row in batch])
    tgt_tensor = torch.cat([row[0] for row in batch]).long()   #to use BCEwithLogits add .float()
    return tgt_tensor, img_tensor 

def get_data_loaders(args):
    tokenizer = None

    args.labels, args.label_freqs = get_labels_and_frequencies(
        args.train_csv
    )

    vocab = None
    args.vocab = None
    args.vocab_sz = 0
    args.n_classes = 1

    train = csvDataset(
        args.train_csv,
        tokenizer,
        vocab,
        args,
    )

    args.train_data_len = len(train)

    #Not use data augmentation in val/test sets
    args.augmentation = False
    dev = csvDataset(
        args.val_csv,
        tokenizer,
        vocab,
        args,
    )

    test_set = csvDataset(
        args.test_csv,
        tokenizer,
        vocab,
        args,
    )

    args.augmentation = True
    collate = functools.partial(collate_fn, args=args)

    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )

    args.augmentation = False
    collate = functools.partial(collate_fn, args=args)
    val_loader = DataLoader(
        dev,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    test_loader = DataLoader( test_set, batch_size=args.batch_size, shuffle=False, 
                              collate_fn=collate)

    test = {"test": test_loader}

    return train_loader, val_loader, test

def get_criterion(args):
    #criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss() 
    return criterion

def get_optimizer(model, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    return optimizer

def store_preds_to_disk(tgts, preds, args):
    if args.task_type == "multilabel":
        with open(os.path.join(args.savedir, "test_labels_pred.txt"), "w") as fw:
            fw.write(
                "\n".join([" ".join(["1" if x else "0" for x in p]) for p in preds])
            )
        with open(os.path.join(args.savedir, "test_labels_gold.txt"), "w") as fw:
            fw.write(
                "\n".join([" ".join(["1" if x else "0" for x in t]) for t in tgts])
            )
        with open(os.path.join(args.savedir, "test_labels.txt"), "w") as fw:
            fw.write(" ".join([l for l in args.labels]))

    else:
        with open(os.path.join(args.savedir, "test_labels_pred.txt"), "w") as fw:
            fw.write("\n".join([str(x) for x in preds]))
        with open(os.path.join(args.savedir, "test_labels_gold.txt"), "w") as fw:
            fw.write("\n".join([str(x) for x in tgts]))
        with open(os.path.join(args.savedir, "test_labels.txt"), "w") as fw:
            fw.write(" ".join([str(l) for l in args.labels]))

def model_eval(i_epoch, data, model, args, criterion, store_preds=False):
    with torch.no_grad():
        losses, preds, tgts = [], [], []
        for batch in data:
            loss, out, tgt = model_forward(i_epoch, model, args, criterion, batch)
            losses.append(loss.item())
            
            #pred = torch.sigmoid(out).cpu().detach().numpy() > 0.5
            pred = torch.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()
            preds.append(pred)
            tgt = tgt.cpu().detach().numpy()
            tgts.append(tgt)

    metrics = {"loss": np.mean(losses)}
    tgts = [l for sl in tgts for l in sl]
    preds = [l for sl in preds for l in sl]
    metrics["acc"] = accuracy_score(tgts, preds)
    metrics['pre'] = precision_score(tgts, preds)
    metrics['rec'] = recall_score(tgts, preds)
    metrics['f1']  = f1_score(tgts, preds, zero_division=0) 

    if store_preds:
        store_preds_to_disk(tgts, preds, args)

    return metrics


def model_forward(i_epoch, model, args, criterion, batch):
    tgt, img = batch
    img = img.cuda()
  
    out = model(img)

    tgt = tgt.cuda()
    
    #out = out.view(tgt.size())
    loss = criterion(out, tgt)

    return loss, out, tgt

#Utils function for train
MODELS = {"LieNet":LieNet_v1}

def get_model(args):
    return MODELS[args.model](args)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(state, is_best, checkpoint_path, filename="checkpoint.pt"):
    filename = os.path.join(checkpoint_path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_path, "model_best.pt"))

def load_checkpoint(model, path):
    best_checkpoint = torch.load(path)
    model.load_state_dict(best_checkpoint["state_dict"])

#In LieNet paper, authors used validation set to fine-tuning weights for late fusion
#So, we will save best model with respec to to accuracy in training
def train(args):
    train_metrics = {'acc':[], 'pre':[], 'rec':[], 'f1':[]}
    validation_metrics = {'acc':[], 'pre':[], 'rec':[], 'f1':[]}

    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)

    train_loader, val_loader, test_loaders = get_data_loaders(args)

    model = get_model(args)
    criterion = get_criterion(args)
    optimizer = get_optimizer(model, args)

    #logger = create_logger("%s/logfile.log" % args.savedir, args)
    #logger.info(model)
    model.cuda()

    torch.save(args, os.path.join(args.savedir, "args.pt"))

    start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, -np.inf

    if os.path.exists(os.path.join(args.savedir, "checkpoint.pt")):
        checkpoint = torch.load(os.path.join(args.savedir, "checkpoint.pt"))
        start_epoch = checkpoint["epoch"]
        n_no_improve = checkpoint["n_no_improve"]
        best_metric = checkpoint["best_metric"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    best_val_acc = 0.0
    #best_acc = 0.0
    n_ep = 1
    #logger.info("Training..")
    #print("Training..")
    if args.txt_log is not None:
      log_txt = open(args.txt_log,"a")
    for i_epoch in range(start_epoch, args.max_epochs):
        train_losses, train_preds, train_tgts = [], [], []
        model.train()
        optimizer.zero_grad()

        #for batch in train_loader:
        for batch in train_loader:
            loss, out, tgt = model_forward(i_epoch, model, args, criterion, batch)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            train_losses.append(loss.item())
            #pred = torch.sigmoid(out).cpu().detach().numpy() > 0.5
            pred = torch.softmax(out,dim=1).argmax(dim=1).cpu().detach().numpy()
            train_preds.append(pred)
            tgt = tgt.cpu().detach().numpy()
            train_tgts.append(tgt)

            loss.backward()
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        tgts = [l for sl in train_tgts for l in sl]
        preds = [l for sl in train_preds for l in sl]
        tr_metrics = {}
        tr_metrics = {"loss": np.mean(train_losses)}
        tr_metrics["acc"] = accuracy_score(tgts, preds)
        tr_metrics['pre'] = precision_score(tgts, preds)
        tr_metrics['rec'] = recall_score(tgts, preds)
        tr_metrics['f1']  = f1_score(tgts, preds, zero_division=0)

        #Save epoch metric in lists to make csv performance
        train_metrics['acc'].append(tr_metrics['acc'])
        train_metrics['pre'].append(tr_metrics['pre'])
        train_metrics['rec'].append(tr_metrics['rec'])
        train_metrics['f1'].append(tr_metrics['f1'])
        
        #Evaluate validation set for early stopping
        model.eval()
        val_metrics = model_eval(i_epoch, val_loader, model, args, criterion)
        tuning_metric = val_metrics['acc']

        validation_metrics['acc'].append(val_metrics['acc'])
        validation_metrics['pre'].append(val_metrics['pre'])
        validation_metrics['rec'].append(val_metrics['rec'])
        validation_metrics['f1'].append(val_metrics['f1'])

        is_improvement = tuning_metric > best_metric
        if is_improvement:
            best_metric = tuning_metric
            n_no_improve = 0
            best_val_acc = tuning_metric
        else:
            n_no_improve += 1

        save_checkpoint(
            {
                "epoch": i_epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "n_no_improve": n_no_improve,
                "best_metric": best_metric,
            },
            is_improvement,
            args.savedir,
        )
        n_ep+=1
        if n_no_improve >= args.patience:
            #logger.info("No improvement. Breaking out of loop.")
            #print("No improvement. Breaking out of loop")
            break
        if args.txt_log is not None:
          if (i_epoch+1)%5 == 0:
            log_txt.write(f"\n epoch_{i+1}")
    
    model.eval()
    for test_name, test_loader in test_loaders.items():
        last_test_metrics = model_eval(np.inf, test_loader, model, args, criterion, store_preds=False)
    #Load best val_acc model
    load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
    model.eval()
    for test_name, test_loader in test_loaders.items():
        test_metrics = model_eval(
            np.inf, test_loader, model, args, criterion, store_preds=False
        )

    if args.txt_log is not None:
      log_txt.close()

    return train_metrics, validation_metrics, last_test_metrics, test_metrics, [n_ep, best_val_acc]

def get_sub_csv(df_csv, ids):
    csv_idx = []
    for idx in ids:
        user_idx = list(df_csv['usernum']==idx)
        csv_idx += [ i for i in range(df_csv.shape[0]) if user_idx[i]]
        sub_df = df_csv.iloc[csv_idx]
    return sub_df

def get_augmented_csv(df):
    label, audio, video = [],[],[]
    for idx in range(df.shape[0]):
        #Each sample have 5 augmented version, save original sample first
        lab  = df.iloc[idx]['class']
        au_p = df.iloc[idx]['audio']
        vi_p = df.iloc[idx]['video']
        #start with original
        label.append(lab)
        audio.append(au_p)
        video.append(vi_p)
        #Add augmented versions for this sample
        for i in range(1,6):
            label.append(lab)
            audio.append(au_p.replace('.png',f'_aug_{i}.png'))
            video.append(vi_p.replace('.png',f'_aug_{i}.png'))
    #Return augmented csv
    aug_csv = pd.DataFrame.from_dict({'class':label,'audio':audio, 'video':video})
    return aug_csv

#New dataset to return two modalities
class multDataset(Dataset):
  def __init__(self, data_path, args):
    self.data = pd.read_csv(data_path)
    self.data_dir = os.path.dirname(data_path)
    self.args = args
    self.modalities = args.modalities

  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, index):
    csv_sample = self.data.iloc[index]
    label = torch.LongTensor( [csv_sample['class']] )
    if label==1:
        class_path = os.path.join(self.data_dir,"Deceptive")
    else:
        class_path = os.path.join(self.data_dir,"Truthful")
    
    a_img = None
    v_img = None
    if 'A' in self.args.modalities:
        img_path = os.path.join(class_path,csv_sample['audio'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        a_img = torch.Tensor(img).permute((2,0,1))

    if 'V' in self.args.modalities:
        img_path = os.path.join(class_path,csv_sample['video'])
        img = cv2.imread(img_path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        v_img = torch.Tensor(img).permute((2,0,1))

    
    #This verison 2 will make data augmnetation in csv file directly
    return label, a_img, v_img

def collate_mult(batch, args):
    a_tensor, v_tensor, = None, None
    if 'A' in args.modalities:
        a_tensor = torch.stack( [row[1] for row in batch] )
    if 'V' in args.modalities:
        v_tensor = torch.stack( [row[2] for row in batch] )

    tgt_tensor = torch.stack( [row[0] for row in batch] )

    return tgt_tensor, a_tensor, v_tensor


#Args for audio lienet
#Make manual Namespace for notebook training
a_args = argparse.Namespace()
a_args.batch_size  = 1
a_args.data_path = "/home/est_posgrado_abdiel.beltran/Trial_LieNet_img"
a_args.model = 'LieNet'
a_args.Tsz = 2
a_args.input_dim = (256,256,3) #For audio (H,W,Chan)
a_args.augmentation = False
a_args.modality = 'A'
a_args.modalities = 'AV'

#Args for video lienet
v_args = argparse.Namespace()
v_args.batch_size  = 1
v_args.data_path = "/home/est_posgrado_abdiel.beltran/Trial_LieNet_img"
v_args.model = 'LieNet'
v_args.Tsz = 2
v_args.input_dim = (256,1024,3) #For audio (H,W,Chan)
v_args.augmentation = False
v_args.modality = 'V'
v_args.modalities = 'AV'


bol_csv = pd.read_csv("/home/est_posgrado_abdiel.beltran/Trial_LieNet_img/Annotations.csv")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_index = np.array(list(range(bol_csv.shape[0])))
all_label = np.array(list(bol_csv['class']))


#construct lienet models
a_model = LieNet_v1(a_args)
v_model = LieNet_v1(v_args)


fold_acc   = []
fold_a_err = []
fold_v_err = []

i=0
for tr_idx, idx20 in skf.split(all_index, all_label):
    val_idx, test_idx = train_test_split(all_index[idx20], stratify=all_label[idx20], test_size=0.5, random_state=42, shuffle=True)

    val_csv   = bol_csv.iloc[val_idx]
    test_csv  = bol_csv.iloc[test_idx]
    
    #Make augmented version for train samples
    val_csv   = get_augmented_csv(val_csv)
    test_csv  = get_augmented_csv(test_csv)

    val_csv.to_csv(os.path.join(a_args.data_path,"dev_a.csv"), index=False)
    test_csv.to_csv(os.path.join(a_args.data_path,"test_a.csv"), index=False)

    val_csv_path  = os.path.join(a_args.data_path,"dev_a.csv")
    test_csv_path = os.path.join(a_args.data_path,"test_a.csv")
    #Construct datasets manually
    val_ds  = multDataset(val_csv_path, a_args)
    test_ds = multDataset(test_csv_path, a_args)

    collate = functools.partial(collate_mult, args=a_args)
    val_loader  =DataLoader(val_ds, batch_size=a_args.batch_size, shuffle=False, collate_fn=collate)
    test_loader =DataLoader(test_ds, batch_size=a_args.batch_size, shuffle=False, collate_fn=collate)

    #path to load best models
    a_ckp = f"/home/est_posgrado_abdiel.beltran/LieNet/dataset_1/5F/{a_args.modality}/fold_{i}/model_best.pt"
    v_ckp =  f"/home/est_posgrado_abdiel.beltran/LieNet/dataset_1/5F/{v_args.modality}/fold_{i}/model_best.pt"

    load_checkpoint(a_model, a_ckp)
    load_checkpoint(v_model, v_ckp)

    a_model.eval()
    v_model.eval()

    a_model.cuda()
    v_model.cuda()

    a_preds, v_preds, tgts = [], [], []
    for batch in val_loader:
        tgt, a_img, v_img = batch
        a_img, v_img = a_img.cuda(), v_img.cuda()
        #a_img, v_img, e_img =  a_img.cuda(), v_img.cuda(), e_img.cuda()
        a_out = a_model(a_img)
        v_out = v_model(v_img)
        #e_out = e_model(e_img)

        a_pred = torch.softmax(a_out, dim=1).argmax(dim=1).cpu().detach().numpy()
        a_preds.append(a_pred)

        v_pred = torch.softmax(v_out, dim=1).argmax(dim=1).cpu().detach().numpy()
        v_preds.append(v_pred)

        #e_pred = torch.softmax(e_out, dim=1).argmax(dim=1).cpu().detach().numpy()
        #e_preds.append(e_pred)

        tgt = tgt.detach().numpy()
        tgts.append(tgt)
    
    #At this point we have tgts and preds for validation set. Calulate error_rate
    tgts    = [l for sl in tgts for l in sl]
    a_preds = [l for sl in a_preds for l in sl]
    v_preds = [l for sl in v_preds for l in sl]
    #e_preds = [l for sl in e_preds for l in sl]
   
    a_err = np.sum([1 if a_preds[i]!=tgts[i] else 0 for i in range(len(tgts))])/len(tgts)
    v_err = np.sum([1 if v_preds[i]!=tgts[i] else 0 for i in range(len(tgts))])/len(tgts)
    #e_err = np.sum([1 if e_preds[i]!=tgts[i] else 0 for i in range(len(tgts))])/len(tgts)

    a_w   = 0.5*np.log((1.0 - a_err)/a_err)
    v_w   = 0.5*np.log((1.0 - v_err)/v_err)
    #e_w   = 0.5*np.log((1.0 - e_err)/e_err)
    
    fold_a_err.append(a_err)
    fold_v_err.append(v_err)
    #e_ws.append(e_err)
    #Now evaluate in test set with weighted sum for final classification
    eval_preds, eval_tgts  = [], []
    
    for batch in test_loader:
        tgt, a_img, v_img = batch
        a_img, v_img = a_img.cuda(), v_img.cuda()
        
        a_out = a_model(a_img)
        v_out = v_model(v_img)
       
        tgt = tgt.detach().numpy()
        a_prob = torch.softmax(a_out, dim=1).cpu().detach()
        v_prob = torch.softmax(v_out, dim=1).cpu().detach()

        pred = (a_w*a_prob + v_w*v_prob).argmax(dim=1).numpy()
        #pred = (a_w*a_prob + v_w*v_prob + e_w*e_prob).argmax(dim=1).numpy()
        eval_preds.append(pred)
        eval_tgts.append(tgt)

    tgts =  [l for sl in eval_tgts for l in sl]
    preds = [l for sl in eval_preds for l in sl]
    test_acc = accuracy_score(tgts, preds)
    fold_acc.append(test_acc)

    i = i+1

csvPath = "/home/est_posgrado_abdiel.beltran/LieNet/dataset_1/5F/mult_summary.csv"
test_df = pd.DataFrame.from_dict({'Fold':list(range(1,6)),'test_acc':fold_acc,'audio_err':fold_a_err,'video_err':fold_v_err})
test_df.to_csv(csvPath, index=False)

import numpy  as np
import time
import torch
from torch.utils.data import Dataset,DataLoader
import functools
import argparse
import random
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#Dataset for simple_MLP model
class dummy_dataset(Dataset):
    def __init__(self, features, labels):
        self.data   = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).float()
        
    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

#Dataset for multimodal MLP      
class dummy_Multimodal_Dataset(Dataset):
    def __init__(self, args):
        self.labels      = args.labels
        self.modalities  = args.modalities
        self.data_T      = torch.Tensor(args.data_T).float() if 'T' in self.modalities else None
        self.data_A      = torch.Tensor(args.data_A).float() if 'A' in self.modalities else None
        self.data_V      = torch.Tensor(args.data_V).float() if 'V' in self.modalities else None
        
    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, idx):
        tgt   = torch.LongTensor([self.labels[idx]])
        vec_T = None
        vec_A = None
        vec_V = None
        
        if 'T' in self.modalities: vec_T = self.data_T[idx]
        if 'A' in self.modalities: vec_A = self.data_A[idx]
        if 'V' in self.modalities: vec_V = self.data_V[idx]
        
        return vec_T, vec_A, vec_V, tgt
    
def collate_fn(batch, args):
    T_tensor, A_tensor, V_tensor = None, None, None
    
    tgt_tensor =  torch.cat([row[3] for row in batch]).float() #to use BCEwithLogits
    if 'T' in args.modalities: T_tensor = torch.stack([row[0] for row in batch])
    if 'A' in args.modalities: A_tensor = torch.stack([row[1] for row in batch])
    if 'V' in args.modalities: V_tensor = torch.stack([row[2] for row in batch])
    
    return T_tensor, A_tensor, V_tensor, tgt_tensor
  
def get_DataLoader(feats, label, tr_id, ts_id, args):
    s        = args.s
    z_norm   = args.z_norm
    b_sz     = args.b_sz
    
    if args.model == 'uni_MLP':
        x_train = feats[tr_id,:]
        x_test  = feats[ts_id,:]
        y_train = label[tr_id]
        y_test  = label[ts_id]
        
        train_ds = dummy_dataset(x_train, y_train)
        test_ds  = dummy_dataset(x_test, y_test)
    
        train_loader = DataLoader(train_ds, shuffle=True , batch_size=b_sz)
        test_loader  = DataLoader(test_ds , shuffle=False, batch_size=b_sz)
        
    elif args.model == 'concat_MMLP':
        xt_train  = feats[0][tr_id,:] if 'T' in args.modalities else None
        xa_train  = feats[1][tr_id,:] if 'A' in args.modalities else None
        xv_train  = feats[2][tr_id,:] if 'V' in args.modalities else None
        lab_train = label[tr_id]
        
        xt_test  = feats[0][ts_id,:] if 'T' in args.modalities else None
        xa_test  = feats[1][ts_id,:] if 'A' in args.modalities else None
        xv_test  = feats[2][ts_id,:] if 'V' in args.modalities else None
        lab_test = label[ts_id]
        
        #create train dataloader
        data_args = argparse.Namespace()
        data_args.modalities = args.modalities
        data_args.labels     = lab_train
        data_args.data_T     = xt_train
        data_args.data_A     = xa_train
        data_args.data_V     = xv_train
        
        train_ds = dummy_Multimodal_Dataset(data_args)
        collate  = functools.partial(collate_fn, args=data_args)
        
        train_loader = DataLoader(train_ds, shuffle=True , batch_size=b_sz, collate_fn=collate)
        
        #create test dataloader
        data_args = argparse.Namespace()
        data_args.modalities = args.modalities
        data_args.labels     = lab_test
        data_args.data_T     = xt_test
        data_args.data_A     = xa_test
        data_args.data_V     = xv_test
        
        test_ds = dummy_Multimodal_Dataset(data_args)
        collate = functools.partial(collate_fn, args=data_args)
        
        test_loader = DataLoader(test_ds, shuffle=True , batch_size=b_sz, collate_fn=collate)
        
    return train_loader, test_loader

  
def model_eval(data, model, args):
    gpu = args.use_gpu
    model_type = args.model
    with torch.no_grad():
        y_p, y_t = [], []
        for batch in data:
            if model_type == 'uni_MLP' or model_type == 'uni_CNN':
                X,y = batch
                if gpu:
                    X = X.cuda()
                out = model(X)
            elif model_type == 'concat_MMLP':
                x_t, x_a, x_v, y = batch
                if gpu:
                    if 'T' in args.modalities:  x_t = x_t.cuda()
                    if 'A' in args.modalities:  x_a = x_a.cuda()
                    if 'V' in args.modalities:  x_v = x_v.cuda()
                out,_ = model(x_t, x_a, x_v)
            
            y_pred = torch.sigmoid(out.detach()).cpu().numpy() > 0.5
            y_true = y.cpu().numpy()
            
            y_p.append(y_pred)
            y_t.append(y_true)
    
    y_p = [i for j in y_p for i in j]
    y_t = [i for j in y_t for i in j]
    
    metrics = {'acc': accuracy_score(y_t,y_p),
               'pre': precision_score(y_t, y_p, zero_division=0),
               'rec': recall_score(y_t, y_p, zero_division=0),
               'f1':  f1_score(y_t, y_p, zero_division=0)}
    
    return metrics

def model_train(model, train_loader, test_loader, optimizer, criterion, epochs=25, use_gpu=False, verbose=0,
               model_type='uni_MLP',ret_model=False, general_seed=1234):
    acc_history  = []
    loss_history = []
    set_seed(general_seed)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss_epoch = []
        acc_epoch  = []
        start_time = time.time()
        
        for batch in train_loader:
            if model_type == 'uni_MLP':
                X,y = batch
                if use_gpu:
                    X = X.cuda()
                    label = y.cuda()
                label=label.unsqueeze(1)
                out = model(X)
            elif model_type == 'concat_MMLP':
                x_t, x_a, x_v, y = batch
                if use_gpu:
                    if 'T' in args.modalities: x_t = x_t.cuda()
                    if 'A' in args.modalities: x_a = x_a.cuda()
                    if 'V' in args.modalities: x_v = x_v.cuda()
                    label = y.cuda()
                label=label.unsqueeze(1)
                out,_ = model(x_t, x_a, x_v)
            
            loss = criterion(out,label)
            loss_epoch.append(loss.item())
            
            #compute acc
            pred = torch.sigmoid(out.detach()).cpu().numpy() > 0.5
            acc_epoch.append(accuracy_score(label.cpu().numpy(),pred))
            
            #backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        end_time = time.time() 
        #Report Loss and acc in train
        mean_acc  = np.mean(acc_epoch)
        acc_history.append(mean_acc)
        mean_loss = np.mean(loss_epoch)
        loss_history.append(mean_loss)
        
        #Report Loss and acc in test
        model.eval()
        test_dic  = model_eval(test_loader, model, args)
        test_acc  = test_dic['acc']
        
        if verbose:
            print("Epoch [{0:>03d}/{1:>03d}] -\t train_acc: {2:.4f} -\t train_loss: {3:.4f} -\t time:{4:.2f} s | test_acc: {5:.2f}".format(
            epoch+1,epochs,mean_acc,mean_loss,end_time-start_time,test_acc))
    
    if ret_model:
        return test_dic, model
    else:
        return test_dic  

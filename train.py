import os
import random
import numpy  as np
import pandas as pd
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize, TweetTokenizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import tree

from models.fusion import biGMU, triGMU_sigmoid, triGMU_softmax, triGMU_hierarchical, DMRN, TMRN
from models.mlp    import simpleMLP, concat_simpleMLP
from models.utils  import get_DataLoader, model_eval, model_train
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
  
  
def construct_bow(text, vocab, tok,  
                  ngram = (1,1), weight='freq', l2=False, default_vec = False):
    if default_vec == False:
        #construct BoW with same tokenizer as tok
        if weight=='tfidf':  vec = TfidfVectorizer(ngram_range=ngram, tokenizer=tok)
        elif weight=='freq': vec = CountVectorizer(ngram_range=ngram, tokenizer=tok)
        elif weight=='bin':  vec = CountVectorizer(ngram_range=ngram, tokenizer=tok, binary=True)
    else:
        if weight=='tfidf':  vec = TfidfVectorizer(ngram_range=ngram)
        elif weight=='freq': vec = CountVectorizer(ngram_range=ngram)
        elif weight=='bin':  vec = CountVectorizer(ngram_range=ngram, binary=True)
    bow = vec.fit_transform(text).toarray()

    if weight=='freq':  #make normalization for frequency counting
        bow = bow*1.0
        doc_freq = np.sum(bow, axis=0)
        for i in range(bow.shape[1]):
            if doc_freq[i]>0:
                bow[:,i] = bow[:,i]/doc_freq[i]
    
    if l2:
        bow = normalize(bow)
        
    return bow, vec.vocabulary_

def tweet_tok_fn(text):
    tt = TweetTokenizer()
    return tt.tokenize(text)

def mask_unk(text, tokenizer, vocab):
    #ignore unk tokens
    toks = tokenizer(text)
    text_unk = [ w for w in toks if w in vocab]
    return " ".join(text_unk)

def get_vocab(texts, tokenizer='word', use_stem = False):
    tweet_tok = TweetTokenizer()
    porter = nltk.PorterStemmer()
    vocab = {}
    words = []
    for t in texts:
        if tokenizer == 'word':
            toks = word_tokenize(t)
        elif tokenizer == 'tweet':
            toks = tweet_tok.tokenize(t)
        if use_stem:
            toks = [porter.stem(w) for w in toks]
        
        for tok in toks:
            if tok not in words:
                words.append(tok)
    
    #Now construct vocabulary on sorted words
    words = sorted(words)
    for i,w in enumerate(words):
        vocab[w] = i
    return vocab

#Functions to make 5FCV
def bow_user_5FCV(all_text, labels, sample_subject, vocabs,
             weight='binary', norm_bow = False, bow_ngram=(1,1), bow_default = False,
             method='SVM',general_seed=general_seed, extra_feat=None, name=None, use_bow=True,
             fold_path=os.path.normpath("user_5FCV_split"),
             nfolds=5):
    
    tweet_tok    = TweetTokenizer()
    avg_acc, avg_rec, avg_pre, avg_f1 = [],[],[],[]
    fold_acc, fold_rec, fold_pre, fold_f1 = [],[],[],[]
    for i in range(1,nfolds+1):
        #Read freezed train/test split indexes
        with open(os.path.join(fold_path,f'fold_{i}_train.txt'), 'r') as f:
            tr_id = np.array([int(v) for v in f.readlines()],dtype=np.int32)
                    
        with open(os.path.join(fold_path,f'fold_{i}_test.txt'), 'r') as f:
            test_id = np.array([int(v) for v in f.readlines()],dtype=np.int32)
            
        train_sample_idx = np.array([ idx for idx,s_id in enumerate(sample_subject) if s_id in tr_id])
        test_sample_idx  = np.array([ idx for idx,s_id in enumerate(sample_subject) if s_id in test_id])
        
        vocab    = vocabs[i-1]
        
        
        
        #mask unknown words in test samples
        text_unk    = [mask_unk(txt, tweet_tok.tokenize, vocab) 
                       if j in test_sample_idx else txt for j,txt in enumerate(all_text)]
        
        bow, bow_vocab = construct_bow(text_unk, vocab, tweet_tok_fn, weight=weight, 
                            l2=norm_bow, ngram=bow_ngram, default_vec=bow_default)
        
        #with bow flavour, make classification
        #Select method
        if method == 'DT':
            clf = tree.DecisionTreeClassifier(random_state=general_seed)
        elif method == 'SVM':
            clf = svm.LinearSVC(class_weight='balanced', random_state=general_seed, max_iter=2000)
        elif method == 'RF':
            clf = RandomForestClassifier(random_state=general_seed, n_jobs=4)
        
        if extra_feat is None:
            X = bow
        else:
            if use_bow:
                X = np.concatenate([bow]+extra_feat, axis=1)
            else:
                X = np.concatenate(extra_feat, axis=1)
        
        #print(X.shape)
        
        x_train = X[train_sample_idx,:]
        x_test  = X[test_sample_idx,:]
        y_train = labels[train_sample_idx]
        y_test  = labels[test_sample_idx]
        
        #set seed
        set_seed(general_seed)
        clf.fit(x_train,y_train)
        y_pred = clf.predict(x_test)
        fold_acc.append( accuracy_score(y_test, y_pred) ) 
        fold_pre.append( precision_score(y_test, y_pred,zero_division=0) )    
        fold_rec.append( recall_score(y_test, y_pred, zero_division=0) )    
        fold_f1.append(  f1_score(y_test, y_pred, zero_division=0) )
    
    avg_acc = np.mean(fold_acc)
    avg_pre = np.mean(fold_pre)
    avg_rec = np.mean(fold_rec)
    avg_f1 = np.mean(fold_f1)
    if name is None:
        clf_str = method
    else:
        clf_str = name
    df = pd.DataFrame.from_dict({'clf':[clf_str],'acc':avg_acc, 'pre':avg_pre, 'rec':avg_rec, 'f1':avg_f1})
    return(df)

def user_kFCV_DNN(all_feat, label, sample_subject, args,
                  fold_path=os.path.normpath("user_5FCV_split"),
                  save_model=None,nfolds=5):
    
    kF       = args.kF
    b_sz     = args.b_sz
    lr       = args.lr
    n_epochs = args.n_epochs
    use_gpu  = args.use_gpu
    s        = args.s
    verbose  = args.verbose
    modalities = args.modalities
    tweet_tok    = TweetTokenizer()
    
    #args to construct BoW
    if 'T' in args.modalities:
        vocabs      = args.vocabs
        weight      = args.weight
        norm_bow    = args.norm_bow
        bow_ngram   = args.bow_ngram
        bow_default = args.bow_defalut
        all_text    = args.all_text
    
    fold_acc, fold_pre, fold_rec, fold_f1 = [], [], [], []
    copy_feat = copy.deepcopy(all_feat)
    for i in range(1,nfolds+1):
        #Read freezed train/test split indexes
        with open(os.path.join(fold_path,f'fold_{i}_train.txt'), 'r') as f:
            tr_id = np.array([int(v) for v in f.readlines()],dtype=np.int32)
                    
        with open(os.path.join(fold_path,f'fold_{i}_test.txt'), 'r') as f:
            ts_id = np.array([int(v) for v in f.readlines()],dtype=np.int32)
    
        
        train_sample_idx = np.array([ idx for idx,s_id in enumerate(sample_subject) if s_id in tr_id], dtype=np.int32)
        test_sample_idx  = np.array([ idx for idx,s_id in enumerate(sample_subject) if s_id in ts_id], dtype=np.int32)
        
        #Copy features
        fold_feat  = []  #for uni_MLP
        fold_feat2 = [None,None,None]  #for MMLP
        #construct fold BoW if needed
        if 'T' in modalities:
            if args.use_bow:
                vocab       = vocabs[i-1]
                text_unk    = [mask_unk(txt, tweet_tok.tokenize, vocab) 
                               if j in test_sample_idx else txt for j,txt in enumerate(all_text)]
                
                bow, vocab = construct_bow(text_unk, None, tweet_tok_fn, weight=weight, 
                                    l2=norm_bow, ngram=bow_ngram, default_vec=bow_default)
         
                fold_feat.append(bow)
                fold_feat2[0] = bow
            else:
                fold_feat.append(all_feat[0])
                fold_feat2[0] = all_feat[0]

        if 'A' in modalities:
            fold_feat.append(all_feat[1])
            fold_feat2[1] = all_feat[1]
                
        if 'V' in modalities:
            fold_feat.append(all_feat[2])
            fold_feat2[2] = all_feat[2]
        
        if args.model == 'uni_MLP':
            copy_feat     = np.concatenate(fold_feat, axis=1)
            args.data     = copy_feat
            args.input_sz = args.data.shape[1]
                
        elif args.model == 'concat_MMLP':
            copy_feat = fold_feat2
            if 'T' in modalities:
                args.data_T      = copy_feat[0]
                args.T_input_sz  = args.data_T.shape[1]
            if 'A' in modalities:
                args.data_A      = copy_feat[1]
                args.A_input_sz  = args.data_A.shape[1]
            if 'V' in modalities:
                args.data_V      = copy_feat[2]
                args.V_input_sz  = args.data_V.shape[1]
                
        #Make loaders
        train_loader, test_loader = get_DataLoader(copy_feat, label,train_sample_idx, test_sample_idx, args)
        
        if args.model   == 'uni_MLP':
            cnn = simpleMLP(args)
        elif args.model == 'concat_MMLP':
            cnn = concat_simpleMLP(args)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)

        if use_gpu:
            cnn.cuda()
        
        if verbose:
            print("-"*100)
            print(f"Fold {i}")
            print("-"*100)
        record = model_train(cnn, train_loader, test_loader, optimizer, criterion, 
                             epochs=n_epochs, use_gpu=use_gpu, verbose=verbose, model_type=args.model)
        fold_acc.append(record['acc'])
        fold_pre.append(record['pre'])
        fold_rec.append(record['rec'])
        fold_f1.append(record['f1'])
        
        if save_model is not None:
            torch.save(cnn.state_dict(),os.path.join(save_model,f"fold_{i}",f"{args.model}_{args.modalities}_{args.fusion_type}.pt"))
        
    return pd.DataFrame.from_dict({'acc':fold_acc, 'pre':fold_pre, 'rec':fold_rec, 'f1':fold_f1})


# 5FCV example with Stacked_GMU 
#First, load the dataset
subject_id_file = os.path.normpath("data/Dataset1_subject_id.xlsx")
dataset1_subject_id_df = pd.read_excel(subject_id_file)

dataset1_data_path    = os.path.normpath("data/Real-life_Deception_Detection_2016")
dataset1_compare_path = os.path.normpath("data/dataset1_trial_OpenSmile_compare")
dataset1_egemaps_path = os.path.normpath("data/dataset1_trial_OpenSmile_egemaps")
dataset1_vggvox_path  = os.path.normpath("data/dataset1_vggvox_emb")
dataset1_text_path    = os.path.join(dataset1_data_path,"Transcription")

dataset1_csv  = pd.read_csv(os.path.join(dataset1_data_path,"Annotation","All_Gestures_Deceptive and Truthful.csv"))
gesture_name  = list(dataset1_csv.columns)[1:-1]
print("all dataset: ",dataset1_csv.shape)

#To use OpenFace features
skip_samples   = ["trial_lie_053","trial_lie_055","trial_truth_041","trial_truth_017"]
sample_names   = list(dataset1_csv['id'])
remain_samples = [i for i,n in enumerate(sample_names) if n.replace(".mp4","") not in skip_samples]

filtered_df = dataset1_csv.iloc[remain_samples]

dataset1_text        = []
dataset1_label       = []
dataset1_gesture     = []
dataset1_compare_fun = []
dataset1_egemaps_fun = []
dataset1_vggvox_emb  = []
dataset1_sample_subject = []
for idx,sample in filtered_df.iterrows():
    #extract labels and transcriptions
    lab = 0
    if sample['class']=='deceptive':
        lab = 1
    dataset1_label.append(lab)
    dataset1_gesture.append(sample[gesture_name].to_numpy())
    transcript_id = sample['id'].replace(".mp4",".txt")
    sample_subject = dataset1_subject_id_df.loc[dataset1_subject_id_df['clip']==sample['id'].replace('.mp4',"")]
    dataset1_sample_subject.append(int(sample_subject['id']))
    if lab:
        txt_path = os.path.join(dataset1_text_path,"Deceptive",transcript_id)
        sample_type = "Deceptive"
    else:
        txt_path = os.path.join(dataset1_text_path,"Truthful",transcript_id)
        sample_type = 'Truthful'
    with open(txt_path,"r",encoding='utf-8') as f:
        dataset1_text.append(f.read().lower())
    #add functionals 1-vector for corresponding sample
    audio_csv = pd.read_csv(os.path.join(dataset1_compare_path,sample_type,transcript_id.replace(".txt","_functionals.csv")))
    audio_col = list(audio_csv.columns[3:])
    dataset1_compare_fun.append(audio_csv.iloc[0][audio_col].to_numpy())
    
    audio_csv = pd.read_csv(os.path.join(dataset1_egemaps_path,sample_type,transcript_id.replace(".txt","_functionals.csv")))
    audio_col = list(audio_csv.columns[3:])
    dataset1_egemaps_fun.append(audio_csv.iloc[0][audio_col].to_numpy())
    
    #add audio embedding (1 per sample)
    audio_emb = np.load( os.path.join(dataset1_vggvox_path,sample_type,transcript_id.replace(".txt","_emb.npy")) )
    dataset1_vggvox_emb.append(audio_emb)
    
faus_avg, gaze_avg, head_avg, _ = get_OpenFace_feat(filtered_df, criterion='all')
dataset1_OpenFace_avg = np.concatenate([faus_avg, gaze_avg, head_avg], axis=1)


dataset1_label       = np.array(dataset1_label,             dtype=np.int32)
dataset1_gesture     = np.array(dataset1_gesture,           dtype=np.int32 )
dataset1_compare_fun = np.array(dataset1_compare_fun,       dtype=np.float32)
dataset1_egemaps_fun = np.array(dataset1_egemaps_fun,       dtype=np.float32)
dataset1_vggvox_emb  = np.array(dataset1_vggvox_emb,        dtype=np.float32)
dataset1_sample_subject = np.array(dataset1_sample_subject, dtype=np.int32)

#Make BoW vocabulary with train data in each fold
fold_split_path=os.path.normpath("dataset1_user_5FCV_split")

fold_vocab = []
for i in range(1,6):
    with open(os.path.join(fold_split_path,f"fold_{i}_train.txt"),"r") as f:
        tr_id = [int(line) for line in f.readlines()]
        
    with open(os.path.join(fold_split_path,f"fold_{i}_test.txt"),"r") as f:
        ts_id  = [int(line) for line in f.readlines()]
    
    train_sample_idx = np.array([ idx for idx,s_id in enumerate(dataset1_sample_subject) if s_id in tr_id], dtype=np.int32)
    test_sample_idx  = np.array([ idx for idx,s_id in enumerate(dataset1_sample_subject) if s_id in ts_id], dtype=np.int32)
    
    
    train_lie = np.sum(dataset1_label[train_sample_idx])
    test_lie  = np.sum(dataset1_label[test_sample_idx])
    print("train sample: {} \t lie: {} \t truth: {}\t users: {}".format(train_sample_idx.shape[0],
                                                           train_lie,train_sample_idx.shape[0]-train_lie ,
                                                             len(tr_id)))
    print("test  sample: {} \t lie: {} \t truth: {}\t users: {}".format(test_sample_idx.shape[0],
                                                           test_lie,test_sample_idx.shape[0]-test_lie ,
                                                             len(ts_id)))
    print(ts_id)
    #Construct vocab with train samples
    train_text = [dataset1_text[idx] for idx in train_sample_idx]
    vocab = get_vocab(train_text, 'tweet')
    fold_vocab.append(vocab)
    print(f"vocab fold_{i}", len(vocab))
    print("-"*100)
    
#------------------------------------------------------- 5FCV example --------------------------------------------------------------------------------
args = argparse.Namespace()
args.model = 'concat_MMLP'

args.labels      = dataset1_label

#Args to construct BoW if needed
args.vocabs      = fold_vocab
args.weight      = 'tfidf'
args.norm_bow    = False
args.bow_ngram   = (1,1)
args.bow_defalut = False #use tweet_tokenizer
args.all_text    = dataset1_text

#Args for MMLP construction
args.data_T      = None
args.T_input_sz  = None
args.T_hidden_sz = [256] 

args.data_A      = dataset1_vggvox_emb
args.A_input_sz  = args.data_A.shape[1]
args.A_hidden_sz = [64, 32]     

args.data_V      = dataset1_OpenFace_avg
args.V_input_sz  = args.data_V.shape[-1]
args.V_hidden_sz = [64, 32]   

args.fusion_type   = 'biGMU'
args.shared_dim    = 1024   #For GMU
args.out_hidden_sz = None

args.kF              = 5
args.b_sz            = 16
args.lr              = 1e-4
args.n_epochs        = 50
args.use_gpu         = True
args.s               = general_seed
args.z_norm          = False
args.verbose         = False


try_mod = ['TV','AV','TAV','TVA','AVT']
for mod in try_mod:
    args.modalities = mod
    print(f"\n\t\t {mod}")
    avg_acc, avg_rec, avg_pre, avg_f1 = [], [], [], []
    
    df =  user_kFCV_DNN([None, dataset1_vggvox_emb, dataset1_OpenFace_avg], args.labels,
                            dataset1_sample_subject,
                            args,
                            fold_path=fold_split_path)
    avg_acc.append(np.mean(df['acc']))
    avg_pre.append(np.mean(df['pre']))
    avg_rec.append(np.mean(df['rec']))
    avg_f1.append(np.mean(df['f1']))
    
avg_df = pd.DataFrame.from_dict({'modality':try_mod,'avg_acc':avg_acc,
                                 'avg_pre':avg_pre,'avg_rec':avg_rec,'avg_f1':avg_f1})
print(avg_df)

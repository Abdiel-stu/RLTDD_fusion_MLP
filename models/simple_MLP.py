import torch
import torch.nn as nn
from models.fusion import biGMU, triGMU_sigmoid, triGMU_softmax, triGMU_hierarchical, DMRN, TMRN

class simpleMLP(nn.Module):
    def __init__(self,args):
        super(simpleMLP,self).__init__()
        #Create a list of hidden layers from a list of hidden units
        self.input_sz  = args.input_sz
        self.hidden_sz = args.hidden_sz
        
        self.layers = []
        
        #All layer will have Fully Connected -> BatchNorm -> ReLU activation
        for i in range(len(self.hidden_sz)):
            if i==0:
                self.layers.append( nn.Linear(self.input_sz, self.hidden_sz[i]) )
                self.layers.append( nn.BatchNorm1d(self.hidden_sz[i]) )
            else:
                self.layers.append( nn.Linear( self.hidden_sz[i-1], self.hidden_sz[i] ) )
                self.layers.append( nn.BatchNorm1d(self.hidden_sz[i]) )
            self.layers.append( nn.ReLU() )
        
        self.N_layers = len(self.layers)
        
        if len(self.hidden_sz)>0:
            self.layers.append( nn.Linear( self.hidden_sz[-1], 1 ) )  #Output layer
        else:
            self.layers.append( nn.Linear( self.input_sz, 1) )
        
        self.layers = nn.ModuleList(self.layers)
        
        
    def forward(self, x):
        #Expect that x have shape [batch, N_channels], this MLP will work on one tensor per sample
        for i in range(self.N_layers):
            x = self.layers[i](x)
        
        return self.layers[-1](x)
    
class auxMLP(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super(auxMLP,self).__init__()
        layers = []
        
        for i in range(len(hidden_sz)):
            if i==0:
                layers.append( nn.Linear(input_sz, hidden_sz[i]) )
            else:
                layers.append( nn.Linear(hidden_sz[i-1], hidden_sz[i] ) )
                
            layers.append( nn.BatchNorm1d(hidden_sz[i]) )
            layers.append( nn.ReLU() )
        
        self.mlp = nn.ModuleList(layers)
    
    def forward(self, x):
        for l in self.mlp:
            x = l(x)
        return x    
    
class concat_simpleMLP(nn.Module):
    """ Intermediate fusion network for Text, Audio, and Video modalities.
        Each branch contains a MLP, followed by one fusion modlue and an optional out MLP
    """
    def __init__(self, args):
        super(concat_simpleMLP,self).__init__()
        self.modalities  = args.modalities
        
        self.T_input_sz  = args.T_input_sz
        self.T_hidden_sz = args.T_hidden_sz
        
        self.A_input_sz  = args.A_input_sz
        self.A_hidden_sz = args.A_hidden_sz
        
        self.V_input_sz  = args.V_input_sz
        self.V_hidden_sz = args.V_hidden_sz
        
        self.out_hidden_sz = args.out_hidden_sz
        
        #Construct a simple MLP for each modality
        self.T_mlp   = None
        self.A_mlp   = None
        self.V_mlp   = None
        self.out_mlp = None
        #For biGMU fusion if needed
        self.fusion_type = args.fusion_type
        self.shared_dim  = args.shared_dim
        #Stacked_GMU made inside forward
        self.biGMU    = None
        self.biGMU_l2 = None
        self.mod_dim = [None,None,None ]
        #for triGMU if needed
        self.kGMU   = None
        
        #This will have concatenated dim or shared_dim parameter for GMU
        self.comb_dim = 0
        
        if 'T' in self.modalities:  
            self.T_mlp      = auxMLP(self.T_input_sz, self.T_hidden_sz)
            self.comb_dim  += self.T_hidden_sz[-1]
            self.mod_dim[0] = self.T_hidden_sz[-1]
        if 'A' in self.modalities:  
            self.A_mlp      = auxMLP(self.A_input_sz, self.A_hidden_sz)
            self.comb_dim  += self.A_hidden_sz[-1]
            self.mod_dim[1] = self.A_hidden_sz[-1]
        if 'V' in self.modalities:  
            self.V_mlp      = auxMLP(self.V_input_sz, self.V_hidden_sz)
            self.comb_dim  += self.V_hidden_sz[-1]
            self.mod_dim[2] = self.V_hidden_sz[-1]
        
        if self.fusion_type is not None:
            self.comb_dim = self.shared_dim
            if self.fusion_type == 'biGMU' or self.fusion_type == 'DMRN':
                # Stacked version of GMU and DMRN
                #GMU modules in pairs [mod_1, mod_2 ] -> mod_12   ->  [mod_3, mod_12] -> fused_feature_vec
                #Define modality id for forward combination
                if len(args.modalities) == 2:
                    if   self.modalities[0] == 'T': self.x1_id = 0
                    elif self.modalities[0] == 'A': self.x1_id = 1
                    elif self.modalities[0] == 'V': self.x1_id = 2
                
                    if   self.modalities[1] == 'T': self.x2_id = 0
                    elif self.modalities[1] == 'A': self.x2_id = 1
                    elif self.modalities[1] == 'V': self.x2_id = 2
                    
                    if self.fusion_type == 'biGMU':
                        self.biGMU = biGMU(self.comb_dim, self.mod_dim[self.x1_id], self.mod_dim[self.x2_id] )
        
                    else:
                        self.DMRN = DMRN(self.comb_dim, self.mod_dim[self.x1_id], self.mod_dim[self.x2_id] )
                    
                if len(args.modalities) == 3:
                    if   self.modalities[0] == 'T': self.x1_id = 0
                    elif self.modalities[0] == 'A': self.x1_id = 1
                    elif self.modalities[0] == 'V': self.x1_id = 2
                
                    if   self.modalities[1] == 'T': self.x2_id = 0
                    elif self.modalities[1] == 'A': self.x2_id = 1
                    elif self.modalities[1] == 'V': self.x2_id = 2
                
                    if   self.modalities[2] == 'T': self.x3_id = 0
                    elif self.modalities[2] == 'A': self.x3_id = 1
                    elif self.modalities[2] == 'V': self.x3_id = 2
                    
                    if self.fusion_type == 'biGMU':
                        self.biGMU      = biGMU(self.comb_dim, self.mod_dim[self.x1_id], self.mod_dim[self.x2_id] )
                        self.biGMU_l2   = biGMU(self.comb_dim, self.mod_dim[self.x3_id], self.comb_dim)
                    else:
                        self.DMRN     = DMRN(self.comb_dim, self.mod_dim[self.x1_id], self.mod_dim[self.x2_id] )
                        self.DMRN_l2  = DMRN(self.comb_dim, self.mod_dim[self.x3_id], self.comb_dim)
            
            elif self.fusion_type == 'soft_GMU':
                self.kGMU = triGMU_softmax(self.comb_dim, self.mod_dim[0], self.mod_dim[1], self.mod_dim[2]) 
            
            elif self.fusion_type == 'sigm_GMU':
                self.kGMU = triGMU_sigmoid(self.comb_dim, self.mod_dim[0], self.mod_dim[1], self.mod_dim[2]) 
                
            elif self.fusion_type == 'hier_GMU':
                self.kGMU = triGMU_hierarchical( self.comb_dim, self.mod_dim[0], self.mod_dim[1], self.mod_dim[2] )
                
            elif self.fusion_type == 'TMRN':
                self.TMRN = TMRN(self.comb_dim, self.mod_dim[0], self.mod_dim[1], self.mod_dim[2])
                
        #output layer
        if self.out_hidden_sz is not None:
            self.out_mlp   = auxMLP( self.comb_dim, self.out_hidden_sz )
            self.out_layer = nn.Linear( self.out_hidden_sz[-1], 1 )
        else:
            self.out_layer = nn.Linear( self.comb_dim, 1 )
        
    def forward(self,x_t, x_a, x_v):
        x_concat = []
        x_mod = [None, None, None] #for GMU if needed
        extra_info = []
        if x_t is not None:
            x_t = self.T_mlp(x_t)
            x_concat.append(x_t)
            x_mod[0] = x_t
        if x_a is not None:
            x_a = self.A_mlp(x_a)
            x_concat.append(x_a)
            x_mod[1] = x_a
        if x_v is not None:
            x_v = self.V_mlp(x_v)
            x_concat.append(x_v)
            x_mod[2] = x_v
        
        if self.fusion_type is None: #default case with concatenation
            x = torch.cat(x_concat, dim=1)
        elif self.fusion_type == 'biGMU':
            gate = []
            if len(self.modalities) == 2:
                x,gate1 = self.biGMU(x_mod[self.x1_id], x_mod[self.x2_id])
                gate.append(gate1)
            if len(self.modalities) == 3:
                x_pair,gate1 = self.biGMU(x_mod[self.x1_id], x_mod[self.x2_id])
                x,gate2      = self.biGMU_l2(x_mod[self.x3_id], x_pair )
                gate.append(gate1)
                gate.append(gate2)
            extra_info.append(gate)
        elif self.fusion_type == 'DMRN':
            if len(self.modalities) == 2:
                x = self.DMRN(x_mod[self.x1_id], x_mod[self.x2_id])
        
            if len(self.modalities) == 3:
                x_pair = self.DMRN(x_mod[self.x1_id], x_mod[self.x2_id])
                x      = self.DMRN_l2(x_mod[self.x3_id], x_pair )
        elif self.fusion_type == 'soft_GMU':
            x, gate = self.kGMU(x_mod[0], x_mod[1], x_mod[2])
            extra_info.append(gate)
        elif self.fusion_type == 'sigm_GMU':
            x, gate = self.kGMU(x_mod[0], x_mod[1], x_mod[2])
            extra_info.append(gate)
        elif self.fusion_type == 'hier_GMU':
            x, gate = self.kGMU(x_mod[0], x_mod[1], x_mod[2])
            extra_info.append(gate)
        elif self.fusion_type == 'TMRN':
            x = self.TMRN(x_mod[0], x_mod[1], x_mod[2])
        
        if self.out_mlp is not None:
            x = self.out_mlp(x)
        
        last_feat = x.detach().cpu().numpy()
        extra_info.append(last_feat)
        
        return self.out_layer(x), extra_info

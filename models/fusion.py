import torch
import torch.nn as nn
import torch.nn.functional as F

class biGMU(nn.Module):
    """Bimodal Gated Multimodal Unit layer based on "Gated multimodal units for information fusion, J. Arevalo et al." (https://arxiv.org/abs/1702.01992)
         Inputs:
                shared_dim: common dimension to make fusion
                x1_dim: input dimension of modality x1
                x2_dim: input dimension of modality x2
         Output:
                [ fused feature, gate activation ]
    """
    def __init__(self, shared_dim, x1_dim, x2_dim, activation=None, gate_activation=None):
        super(biGMU, self).__init__()
        self.dim = shared_dim
        self.x1_dim = x1_dim
        self.x2_dim = x2_dim
        self.x1_activation = nn.Tanh()
        self.x2_activation = nn.Tanh()
        self.gate_activation = nn.Sigmoid()
        
        if activation is not None:      
            self.x1_activation = activation
            self.x2_activation = activation
        if gate_activation is not None: self.gate_activation = gate_activation
        
        #If one or both features dont match with desired common dim, then add a projection layer
        self.x1_proj = None
        self.x2_proj = None
        if self.x1_dim != self.dim:
            self.x1_proj = nn.Linear(self.x1_dim, self.dim, bias=False)
        
        if self.x2_dim != self.dim:
            self.x2_proj = nn.Linear(self.x2_dim, self.dim, bias=False)
            
        self.gate = nn.Linear(2*self.dim, self.dim, bias=False)
        
    def forward(self, x_1,  x_2):
        if self.x1_proj is not None:
            x_1 = self.x1_proj(x_1)
        if self.x2_proj is not None:
            x_2 = self.x2_proj(x_2)
        
        x   = torch.cat([x_1, x_2], axis=1)
        h_1 = self.x1_activation(x_1)
        h_2 = self.x2_activation(x_2)
        z   = self.gate_activation( self.gate(x) )
        #return z*h_1 + (1.0 - z)*h_2
        fusion = z*h_1 + (1.0 - z)*h_2
        return fusion, z.detach().cpu().numpy()  #Return [fused feature, gate activation]
 
class triGMU_sigmoid(nn.Module):
  """Trimodal Gated Multimodal Unit layer based on "Gated multimodal units for information fusion, J. Arevalo et al." (https://arxiv.org/abs/1702.01992) 
       Inputs:
               shared_dim: common dimension to make fusion
               x1_dim: input dimension of modality x1
               x2_dim: input dimension of modality x2
               x3_dim: input dimension of modality x3
        Output:
               [ fused feature, [sigmoid activation for x1, sigmoid activation for x2, sigmoid activation for x3] ]
  """
    def __init__(self, shared_dim, x1_dim, x2_dim, x3_dim, activation=None, gate_activation=None):
        super(triGMU_sigmoid, self).__init__()
        self.dim = shared_dim
        self.x1_dim = x1_dim
        self.x2_dim = x2_dim
        self.x3_dim = x3_dim
        self.x1_activation = nn.Tanh() 
        self.x2_activation = nn.Tanh() 
        self.x3_activation = nn.Tanh() 
        self.x1_gate_activation = nn.Sigmoid()
        self.x2_gate_activation = nn.Sigmoid()
        self.x3_gate_activation = nn.Sigmoid()
        
        if activation is not None:      
            self.x1_activation = activation
            self.x2_activation = activation
            self.x3_activation = activation
        if gate_activation is not None: 
            self.x1_gate_activation = gate_activation
            self.x2_gate_activation = gate_activation
            self.x3_gate_activation = gate_activation
        
        #If one or both features dont math with desired common dim, then add a projection layer
        self.x1_proj = None
        self.x2_proj = None
        self.x3_proj = None
        if self.x1_dim != self.dim:
            self.x1_proj = nn.Linear(self.x1_dim, self.dim, bias=False)
        
        if self.x2_dim != self.dim:
            self.x2_proj = nn.Linear(self.x2_dim, self.dim, bias=False)
        
        if self.x3_dim != self.dim:
            self.x3_proj = nn.Linear(self.x3_dim, self.dim, bias=False)
        
        #Now make one gate activation for each modality
        self.gate_1 = nn.Linear(self.dim*3, self.dim, bias=False)
        self.gate_2 = nn.Linear(self.dim*3, self.dim, bias=False)
        self.gate_3 = nn.Linear(self.dim*3, self.dim, bias=False)
        
    def forward(self, x_1,  x_2, x_3):
        if self.x1_proj is not None:
            x_1 = self.x1_proj(x_1)
        if self.x2_proj is not None:
            x_2 = self.x2_proj(x_2)
        if self.x3_proj is not None:
            x_3 = self.x3_proj(x_3)
        
        #Concatenate vector in one to make gate_i activations
        x   = torch.cat([x_1, x_2, x_3], dim=1)
        
        h_1 = self.x1_activation(x_1)
        h_2 = self.x2_activation(x_2)
        h_3 = self.x3_activation(x_3)
        
        z_1 = self.x1_gate_activation( self.gate_1(x) )
        z_2 = self.x2_gate_activation( self.gate_2(x) )
        z_3 = self.x3_gate_activation( self.gate_3(x) )
        
        fusion = z_1*h_1 + z_2*h_2 + z_3*h_3
        gate = [z_1.detach().cpu().numpy(), z_2.detach().cpu().numpy(), z_3.detach().cpu().numpy()]
        return fusion, gate
      
class triGMU_softmax(nn.Module):
  """ Trimodal GMU adaptation using Softmax instead of Sigmoid for gate activation
        Inputs:
               shared_dim: common dimension to make fusion
               x1_dim: input dimension of modality x1
               x2_dim: input dimension of modality x2
               x3_dim: input dimension of modality x3
        Output:
               [ fused feature, matrix of softmax gate's activation  ]
  """
    def __init__(self, shared_dim, x1_dim, x2_dim, x3_dim, activation=None, gate_activation=None):
        super(triGMU_softmax, self).__init__()
        self.dim = shared_dim
        self.x1_dim = x1_dim
        self.x2_dim = x2_dim
        self.x3_dim = x3_dim
        self.x1_activation = nn.Tanh() 
        self.x2_activation = nn.Tanh() 
        self.x3_activation = nn.Tanh() 
        self.gate_activation = nn.Softmax(dim=1)
        
        if activation is not None:      
            self.x1_activation = activation
            self.x2_activation = activation
            self.x3_activation = activation
        if gate_activation is not None: self.gate_activation = gate_activation
        
        #If one or both features dont match with desired common dim, then add a projection layer
        self.x1_proj = None
        self.x2_proj = None
        self.x3_proj = None
        if self.x1_dim != self.dim:
            self.x1_proj = nn.Linear(self.x1_dim, self.dim, bias=False)
        
        if self.x2_dim != self.dim:
            self.x2_proj = nn.Linear(self.x2_dim, self.dim, bias=False)
        
        if self.x3_dim != self.dim:
            self.x3_proj = nn.Linear(self.x3_dim, self.dim, bias=False)
            
        #Now make one gate activation for each modality
        self.gate_1 = nn.Linear(self.dim*3, self.dim, bias=False)
        self.gate_2 = nn.Linear(self.dim*3, self.dim, bias=False)
        self.gate_3 = nn.Linear(self.dim*3, self.dim, bias=False)
        
    def forward(self, x_1,  x_2, x_3):
        if self.x1_proj is not None:
            x_1 = self.x1_proj(x_1)
        if self.x2_proj is not None:
            x_2 = self.x2_proj(x_2)
        if self.x3_proj is not None:
            x_3 = self.x3_proj(x_3)
        
        #Concatenate vector in one to make gate_i activations
        x   = torch.cat([x_1, x_2, x_3], dim=1)
        
        y_1 = self.gate_1(x)
        y_2 = self.gate_2(x)
        y_3 = self.gate_3(x)
        
        Y   = torch.stack([y_1, y_2, y_3], dim=1)  #Create a matrix [B, 3, dim]
        
        h_1 = self.x1_activation(x_1)
        h_2 = self.x2_activation(x_2)
        h_3 = self.x3_activation(x_3)
        h   = torch.stack([h_1, h_2, h_3], dim=1)  #Create a matrix [B, 3, dim]
        
        z   = self.gate_activation( Y )
        
        fusion = torch.sum(z*h, dim=1) 
        
        return fusion, z.detach().cpu().numpy()
      
class triGMU_hierarchical(nn.Module):
   """ Trimodal GMU adaptation using sigmoid gates and recursive convex sum of modalities
        Inputs:
               shared_dim: common dimension to make fusion
               x1_dim: input dimension of modality x1
               x2_dim: input dimension of modality x2
               x3_dim: input dimension of modality x3
        Output:
               [ fused feature, [weight of x1, weight of x2, weight of x3]  ]
    """
    def __init__(self, shared_dim, x1_dim, x2_dim, x3_dim, 
                 activation=None, gate_activation=None):
        super(triGMU_hierarchical, self).__init__()
        self.dim = shared_dim
        self.x1_dim = x1_dim
        self.x2_dim = x2_dim
        self.x3_dim = x3_dim
        
        self.x1_activation = nn.Tanh() 
        self.x2_activation = nn.Tanh() 
        self.x3_activation = nn.Tanh()
        
        self.gate1_activation = nn.Sigmoid()
        self.gate2_activation = nn.Sigmoid() #We will use k-1 gates
        
        #If one or both features dont math with desired common dim, then add a projection layer
        self.x1_proj = None
        self.x2_proj = None
        self.x3_proj = None
        if self.x1_dim != self.dim:  self.x1_proj = nn.Linear(self.x1_dim, self.dim, bias=False)
        if self.x2_dim != self.dim:  self.x2_proj = nn.Linear(self.x2_dim, self.dim, bias=False)   
        if self.x3_dim != self.dim:  self.x3_proj = nn.Linear(self.x3_dim, self.dim, bias=False)
            
        self.gate1 = nn.Linear(3*self.dim, self.dim, bias=False)
        self.gate2 = nn.Linear(3*self.dim, self.dim, bias=False)
        
    def forward(self, x_1,  x_2, x_3):
        if self.x1_proj is not None:
            x_1 = self.x1_proj(x_1)
        if self.x2_proj is not None:
            x_2 = self.x2_proj(x_2)
        if self.x3_proj is not None:
            x_3 = self.x3_proj(x_3)
        
        x   = torch.cat([x_1, x_2, x_3], axis=1)
        h_1 = self.x1_activation(x_1)
        h_2 = self.x2_activation(x_2)
        h_3 = self.x3_activation(x_3)
        
        z_1   = self.gate1_activation( self.gate1(x) )
        z_2   = self.gate2_activation( self.gate2(x) )
        
        w_2   = (1.0 - z_1)*z_2
        w_3   = (1.0 - z_1)*(1.0 - z_2)
        
        fusion = z_1*h_1 + w_2*h_2 + w_3*h_3
        
        gates = [ z_1.detach().cpu().numpy(),
                  w_2.detach().cpu().numpy(),
                  w_3.detach().cpu().numpy()]
        
        return fusion, gates
      
class DMRN(nn.Module):
  """ Dual Multimodal Residual Network (DMRN) layer based on "Audio-Visual Event Localization in Unconstrained Videos, Y. Tian et al. 
   (https://openaccess.thecvf.com/content_ECCV_2018/html/Yapeng_Tian_Audio-Visual_Event_Localization_ECCV_2018_paper.html)"
   
     Inputs:
            shared_dim: common dimension to make fusion
            x1_dim: input dimension of modality x1
            x2_dim: input dimension of modality x2
     Output:
            fused feature
  """
    def __init__(self, shared_dim, x1_dim, x2_dim):
        super(DMRN,self).__init__()
        #Projection layers to have common dimension in both modalities
        self.x1_proj = None if x1_dim==shared_dim else nn.Linear(x1_dim, shared_dim, bias=False)
        self.x2_proj = None if x2_dim==shared_dim else nn.Linear(x2_dim, shared_dim, bias=False)
        
        #Blocks in DMRN
        self.U_1 = nn.Sequential( nn.Linear(shared_dim, shared_dim),
                                  nn.Tanh(),
                                  nn.Linear(shared_dim, shared_dim),
                                )
        
        self.U_2 = nn.Sequential( nn.Linear(shared_dim, shared_dim),
                                  nn.Tanh(),
                                  nn.Linear(shared_dim, shared_dim),
                                )
        
        self.tanh_x1 = nn.Tanh()
        self.tanh_x2 = nn.Tanh()
        
    def forward(self, x1, x2):
        #Project to common dim if needed
        if self.x1_proj is not None:
            x1 = self.x1_proj(x1)
        if self.x2_proj is not None:
            x2 = self.x2_proj(x2)
        #Save residual connections
        x1_res = x1
        x2_res = x2
        #Apply nonlinear transformation and merge modalities
        x1 = self.U_1(x1)
        x2 = self.U_2(x2)
        merged = torch.mul(x1 + x2, 0.5) 
        #Update each modality with merged information
        x1_update = self.tanh_x1( x1_res + merged ) 
        x2_update = self.tanh_x2( x2_res + merged )
        #Apply additive fusion
        fusion = torch.mul(x1_update + x2_update, 0.5)
        return fusion
      
class TMRN(nn.Module):
  """" Trimodal extension of Dual Multimodal Residual Network (DMRN) by adding a third branch
       Inputs:
              shared_dim: common dimension to make fusion
              x1_dim: input dimension of modality x1
              x2_dim: input dimension of modality x2
              x3_dim: input dimension of modality x3
       Output:
              fused feature
  """
    def __init__(self, shared_dim, x1_dim, x2_dim, x3_dim):
        super(TMRN,self).__init__()
        #Projection layers to have common dimension in both modalities
        self.x1_proj = None if x1_dim==shared_dim else nn.Linear(x1_dim, shared_dim, bias=False)
        self.x2_proj = None if x2_dim==shared_dim else nn.Linear(x2_dim, shared_dim, bias=False)
        self.x3_proj = None if x3_dim==shared_dim else nn.Linear(x3_dim, shared_dim, bias=False)
        
        #Blocks in DMRN
        self.U_1 = nn.Sequential( nn.Linear(shared_dim, shared_dim),
                                  nn.Tanh(),
                                  nn.Linear(shared_dim, shared_dim),
                                )
        
        self.U_2 = nn.Sequential( nn.Linear(shared_dim, shared_dim),
                                  nn.Tanh(),
                                  nn.Linear(shared_dim, shared_dim),
                                )
        
        self.U_3 = nn.Sequential( nn.Linear(shared_dim, shared_dim),
                                  nn.Tanh(),
                                  nn.Linear(shared_dim, shared_dim),
                                )
        
        self.tanh_x1 = nn.Tanh()
        self.tanh_x2 = nn.Tanh()
        self.tanh_x3 = nn.Tanh()
        
    def forward(self, x1, x2, x3):
        #Project to common dim if needed
        if self.x1_proj is not None:
            x1 = self.x1_proj(x1)
        if self.x2_proj is not None:
            x2 = self.x2_proj(x2)
        if self.x3_proj is not None:
            x3 = self.x3_proj(x3)
        #Save residual connections
        x1_res = x1
        x2_res = x2
        x3_res = x3
        #Apply nonlinear transformation and merge modalities
        x1 = self.U_1(x1)
        x2 = self.U_2(x2)
        x3 = self.U_3(x3)
        merged = torch.mul(x1 + x2 + x3, 0.3) 
        #Update each modality with merged information
        x1_update = self.tanh_x1( x1_res + merged ) 
        x2_update = self.tanh_x2( x2_res + merged )
        x3_update = self.tanh_x3( x3_res + merged )
        #Apply additive fusion
        fusion = torch.mul(x1_update + x2_update + x3_update, 0.3)
        return fusion      

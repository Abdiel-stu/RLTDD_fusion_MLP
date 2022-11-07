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

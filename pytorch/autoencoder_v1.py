class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_layers = []
        self.decoder_layers = []
        self.input_shape = kwargs["input_shape"]
        self.deep = kwargs["deep"]
        
        self.last = kwargs["last"]
        
        in_ = kwargs["input_shape"]
        out_ = kwargs["input_shape"]//2
        
        for i in range(0,kwargs["deep"]):
            self.encoder_layers.append(nn.Linear(
                in_features=in_, out_features=out_
            ))
            in_ = out_
            out_ = out_//2
            if out_ == 0:
                break
        
            
        self.middle_rapp = nn.Linear(
                in_features=in_, out_features=in_
            )

        for i in range(0,kwargs["deep"]):
            in_ = in_*2
            if in_ >= kwargs["input_shape"]:
                in_ = kwargs["input_shape"]
            if i==kwargs["deep"] -1:
                in_ = kwargs["input_shape"]
            
            self.decoder_layers.append(nn.Linear(
                in_features=out_, out_features=in_
            ))
            out_ = in_
            if in_ >= kwargs["input_shape"]:
                break
            
            
        
        self.decoder_layers = nn.ModuleList(self.decoder_layers)
        self.encoder_layers = nn.ModuleList(self.encoder_layers)
        

    def forward(self, features):
        for i in range(0,self.deep):
            features = self.encoder_layers[i](features)
            features = nn.ReLU()(features)
        
        features = self.middle_rapp(features)
        features = nn.ReLU()(features)
        
        for i in range(0,self.deep):
            features = self.decoder_layers[i](features)
            if i==self.deep:
                features = self.last(features)
            else:
                features = nn.ReLU()(features)
            
        return features
    

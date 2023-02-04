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
            if out_ <= 4:
                break
        
            
        self.middle_rapp = nn.Linear(
                in_features=in_, out_features=in_
            )
        
        out_ = in_*2
        for i in range(0,kwargs["deep"]):
            if out_ >= kwargs["input_shape"]:
                out_ = kwargs["input_shape"]
            if i==kwargs["deep"] -1:
                in_ = kwargs["input_shape"]
            
            self.decoder_layers.append(nn.Linear(
                in_features=in_, out_features=out_
            ))
            if out_ >= kwargs["input_shape"]:
                break
            in_ = out_
            out_ = in_*2
            
            
            
        
        self.decoder_layers = nn.ModuleList(self.decoder_layers)
        self.encoder_layers = nn.ModuleList(self.encoder_layers)
        

    def forward(self, features):
        for layer in self.encoder_layers:
            features = layer(features)
            features = nn.ReLU()(features)
        
        features = self.middle_rapp(features)
        features = nn.ReLU()(features)
        
        for off,layer in enumerate(self.decoder_layers):
            features = layer(features)
            if off==len(self.decoder_layers)-1:
                features = self.last(features)
            else:
                features = nn.ReLU()(features)
            
        return features
    

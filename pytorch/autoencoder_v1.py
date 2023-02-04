import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
            
        self.middle_rapp = nn.Linear(
                in_features=in_, out_features=in_
            )

        for i in range(0,kwargs["deep"]):
            if i==kwargs["deep"]:
                in_ = kwargs["input_shape"]
            
            self.decoder_layers.append(nn.Linear(
                in_features=out_, out_features=in_
            ))
            in_ = out_*2
            out_ = in_
        
        
        

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
    

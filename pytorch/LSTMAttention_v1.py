
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorflow import keras #just for downloading dataset
from torchvision import models #just for debugging

class LSTMWithAttention(nn.Module):
  def __init__(self, input_size, time_steps,cells=20,lay_2 = 30,features=4,**kwargs):
        super(LSTMWithAttention, self).__init__(**kwargs)
        self.lstm = nn.LSTM(input_size=input_size,hidden_size=cells,num_layers=time_steps,batch_first=True,bidirectional=True)
        self.attention = Attention(cells*2,time_steps)

        self.batch_norm1 = nn.BatchNorm1d(cells*2) 
        self.linear = nn.Linear(cells*2,lay_2)
        self.batch_norm2 = nn.BatchNorm1d(lay_2) 
        self.features = nn.Linear(lay_2,features)
  def forward(self, x):
    x,_ = self.lstm(x)
    x = self.attention(x)
    x = self.batch_norm1(x)
    x = self.linear(x)
    x = self.batch_norm2(x)
    x = self.features(x)
    return x

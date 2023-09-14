import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.signal import stft
from torch.utils.data import Dataset, DataLoader


# 自定义数据集类
class PNW_loader(Dataset):
    def __init__(self, 
                 csv, 
                 data,
                 left_index = 0,
                 right_index = 6000,
                 window_length = 100,
                 nfft = 100
                ):
        
        self.csv = csv
        self.data = data
        self.left_index = left_index
        self.right_index = right_index
        self.class_list = ['earthquake','explosion','surface event','sonic boom','thunder','plane crash']
        self.sample_rate = 100
        self.window_length = window_length
        self.nfft = nfft
        
    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        # 
        random_line = self.csv.iloc[idx]   
        classname = self.get_data_name(random_line,idx)
        label = np.array(self.class_list.index(classname))
        #normliazation the data before transfer to specturm
        norm_data = self.z_norm(self.data[idx,self.left_index:self.right_index,:])
        spec = self.cal_norm_spectrogram(norm_data)
        # select the specific volumn
        text = torch.tensor(label,dtype=torch.long)
        norm_data = torch.tensor(norm_data,dtype=torch.float)
        spec = torch.tensor(spec,dtype=torch.float)
        
        # 
        return norm_data,text,spec
    

    def z_norm(self,x):
        for i in range(3):
            x_std = x[:,i].std()+1e-3
            x[:,i] = (x[:,i] - x[:,i].mean())/x_std
        return x
    
    def get_data_name(self,random_line,idx):
        separators = [',', '$']
        # use different split symbol for getting key_correct
        key_correct = re.split('|'.join(map(re.escape, separators)), random_line['trace_name'])
        
        class_name = random_line['source_type']
        # print split result
        data_index = int(key_correct[1])
        key_name = 'data/' + key_correct[0]
        return class_name
    
    def cal_norm_spectrogram(self,x):
        spec = np.zeros([3,int(x.shape[0]/self.window_length * 2),int(self.nfft/2)])
        for i in range(3):
            _, _, spectrogram = stft(x[:,i], fs=self.sample_rate, window='hann', nperseg=self.window_length, noverlap=int(self.window_length/2), nfft=self.nfft,boundary='zeros')
            spectrogram = spectrogram[1:,1:]
            spec[i,:] = np.abs(spectrogram).transpose(1,0)
        return spec
    
# How to load the model from param
def load_model(model_name,model,device_type='cpu'):
    if device_type == 'cpu':
        param = torch.load(model_name,map_location=torch.device('cpu'))
    if device_type == 'gpu': 
        param = torch.load(model_name)
    model.load_state_dict(param)
    return model

# The simple two layer mlp of downstream network architectrue
class Linear_BN_relu(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Linear_BN_relu, self).__init__()   
        layers = [
            nn.Linear(in_ch,out_ch),
            nn.ReLU(),
        ]
        self.linear=nn.Sequential(*layers) 
        
    def forward(self,x):
        """
        :param x:
        """
        out = self.linear(x)
        return out

class downstream_class3(torch.nn.Module):
    def __init__(self):
        super(downstream_class3, self).__init__()
        
        self.lin1 = Linear_BN_relu(384, 100)    
        self.lin2 = torch.nn.Linear(100, 3)
        # self.act1 = torch.nn.Softmax(dim = -1)
  
    def forward(self, x):
        
        x = self.lin1(x)
        x = self.lin2(x)
        # x = self.act1(x)
        return x
    
# define a function for plot spectrum data
def plot_spec(data,i):
    plt.figure(figsize=(10,6))
    plt.imshow(data[i,0,:].T,aspect='auto',vmax= 0.3)
    plt.ylim(0,49)
    plt.yticks([])
    plt.show()
    
    
if __name__ == "__main__":
    model = downstream_class3()
    input_data = torch.zeros([64,384])
    result = model(input_data)
    print(result.shape)
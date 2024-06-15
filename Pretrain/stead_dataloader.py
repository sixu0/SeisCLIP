import re
import torch
import numpy as np
import pandas as pd
from scipy.signal import stft
from torch.utils.data import Dataset, DataLoader

import h5py


# 自定义数据集类
class stead_loader(Dataset):
    def __init__(self, csv, sample_rate=100, window_length=100, nfft=100, hdf5_path='./data/chunk2/chunk2.hdf5'):
        self.hdf5_path = hdf5_path
        self.csv = csv
        self.selected_columns = ['p_arrival_sample', 'p_weight', 'p_travel_sec', 's_arrival_sample', 's_weight', 'source_distance_km', 'back_azimuth_deg', 'coda_end_sample']
        self.sample_rate = sample_rate
        self.window_length = window_length
        self.nfft = nfft
        
    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        # 根据索引获取样本
        random_line = self.csv.iloc[idx]   
        dt_stead = h5py.File(self.hdf5_path, mode = "r")
        dataset = dt_stead.get('data/'+random_line.iloc[-1])
        data = np.array(dataset)
        data = self.z_norm(data)
        spec = self.cal_norm_spectrogram(data)
        
        # 选取指定的列
        selected_signal = random_line[self.selected_columns]
        label = self.norm_text(selected_signal)
        spec = torch.tensor(spec)
        text = torch.tensor(label)
        data = torch.tensor(data).T
        
        # 返回样本
        return data,text,spec
    

    def z_norm(self,x):
        for i in range(3):
            x_std = x[:,i].std()+1e-3
            x[:,i] = (x[:,i] - x[:,i].mean())/x_std
        return x
    
    
    def cal_norm_spectrogram(self,x):
        spec = np.zeros([3,int(x.shape[0]/self.window_length * 2),int(self.nfft/2)])
        for i in range(3):
            _, _, spectrogram = stft(x[:,i], fs=self.sample_rate, window='hann', nperseg=self.window_length, noverlap=int(self.window_length/2), nfft=self.nfft,boundary='zeros')
            spectrogram = spectrogram[1:,1:]
            # spectrogram = (spectrogram - spectrogram.mean())/spectrogram.std()+1e-3
            spec[i,:] = np.abs(spectrogram).transpose(1,0)
        return spec
    
    def norm_text(self,selected_signal):

        string = selected_signal['coda_end_sample']
        # 定义多个分隔符
        separators = ['[[',"."]

        # 使用多个分隔符对字符串进行分割并保留所有子字符串
        pattern = "|".join(map(re.escape, separators))
        result = re.split(pattern, string)

        selected_signal['coda_end_sample'] = int(result[1])
        y = np.array(selected_signal.values,dtype='float')
        # ['p_arrival_sample', 'p_weight', 'p_travel_sec', 's_arrival_sample', 's_weight', 'source_distance_km', 'back_azimuth_deg', 'coda_end_sample']
        # normalize P_sample,p_travel,s_sample,source_distance,azimuth, coda and sample
        y[0] = y[0]/6000
        y[2] = y[2]/60
        y[3] = y[3]/6000
        y[5] = y[5]/300
        y[6] = y[6]/360
        y[7] = y[7]/6000
        y = self.replace_nan_with_zero(y)
        return y
        
    def replace_nan_with_zero(self,arr):
        mask = np.isnan(arr)  # 创建一个布尔掩码，标识出数组中的 NaN 值
        arr[mask] = 0         # 将掩码中对应位置的元素替换为 0
        return arr


def read_stead_data(csv_path='./data/chunk2/chunk2.csv'):
    
    csv_stead = pd.read_csv(csv_path) 
    
    print(f'total events in csv file: {len(csv_stead)}')
    csv_stead = csv_stead[(csv_stead.trace_category == 'earthquake_local')]
    print(f'total events selected: {len(csv_stead)}')
    return csv_stead
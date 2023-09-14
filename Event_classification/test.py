# @Time    : 14/9/23
# @Author  : Xu Si
# @Affiliation  : University of Science and Technolog of China
# @Email   : xusi@mail.ustc.edu.cn
# @File    : test.py

# import lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import confusion_matrix, classification_report

# import model file and utils file
from model.model_seismic_clip import *
from utils.utils import *


def read_data(waveform_path,metadata_path):
    #import a little dataset from PNW, there are 32 waveform with their correponding metadata.
    val_csv_test = pd.read_csv(metadata_path)
    #metadata shape
    print(val_csv_test.shape)
    data = np.load(waveform_path)
    loaded_tensor = data['waveform']
    # Define the dataloader
    train_dataset_600_50 = PNW_loader(val_csv_test,loaded_tensor,window_length = 20,nfft=100)
    train_dataloader = DataLoader(train_dataset_600_50, batch_size=32, shuffle = False)
    return train_dataloader

def load_SeisCLIP(main_model_path,sub_model_path):
    ### 
    device = torch.device("cpu")
    finetune_model = AUDIO_CLIP(
    embed_dim = 384, text_input = 8,text_width = 512,text_layers=2,spec_tdim=600, spec_fdim = 50,spec_fstr = 10,spec_model_size = 'small224',device_name = device).to(device)
    # load model
    finetune_model = load_model(main_model_path,finetune_model)
    finetune_down_model = downstream_class3().to(device)
    finetune_down_model = load_model(sub_model_path,finetune_down_model)
    return finetune_model,finetune_down_model

def go_test_evaluate(dataloader,finetune_model,finetune_down_model):
    for batch in dataloader:
        
        print('Waveform Size:',batch[0].shape)
        print('Label Size:',batch[1].shape)    
        print('Spectrum Size:',batch[2].shape)        
    # since the pytorch cross_entropy has softmax during calculation process. Therefore, we need add a softmax actication function when using the network for prediction.
    act = torch.nn.Softmax(dim=-1)
    temp_output = act(finetune_down_model(finetune_model.encode_audio(batch[2])))
    finetune_pred_ve = temp_output.detach().numpy()
    label = batch[1].detach().numpy()

    print('Pred:',finetune_pred_ve.argmax(axis=1))
    print('Label:',label)
    # calculate confusion_matrix
    cm = confusion_matrix(label, finetune_pred_ve.argmax(axis=1))
    report = classification_report(label, finetune_pred_ve.argmax(axis=1))
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)

def main():
    #Before run the code, you must firstly put the data and model in this path.
    wave_path       = './data/waveform_test.npz'
    meta_path       = './data/metadata_test.csv'
    main_model_path = './pretrained_models/finetune_model_class_main.pt'
    sub_model_path  = './pretrained_models/finetune_model_class_sub.pt'
    
    dataloader = read_data(wave_path,meta_path)
    finetune_model,finetune_down_model = load_SeisCLIP(main_model_path,sub_model_path)
    go_test_evaluate(dataloader,finetune_model,finetune_down_model)
    
    
if __name__ == "__main__":
    main()

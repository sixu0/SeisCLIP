import torch
import pandas as pd
import numpy as np




def process_csv(DiTing_330km_csv_pre,ifnormalized=False):
    # deal with mag_type_index
    DiTing_330km_csv_pre.insert(7, 'mag_type_index', DiTing_330km_csv_pre['mag_type'])
    DiTing_330km_csv_pre.loc[DiTing_330km_csv_pre['mag_type']=='Ml','mag_type_index'] = int(0)
    DiTing_330km_csv_pre.loc[DiTing_330km_csv_pre['mag_type']=='ML','mag_type_index'] = int(0)
    DiTing_330km_csv_pre.loc[DiTing_330km_csv_pre['mag_type']=='Ms','mag_type_index'] = int(1)
    DiTing_330km_csv_pre.loc[DiTing_330km_csv_pre['mag_type']=='MS','mag_type_index'] = int(1)
    DiTing_330km_csv_pre.loc[DiTing_330km_csv_pre['mag_type']=='mB','mag_type_index'] = int(2)
    DiTing_330km_csv_pre.loc[DiTing_330km_csv_pre['mag_type']=='mb','mag_type_index'] = int(2)
    
    DiTing_330km_csv_pre.loc[DiTing_330km_csv_pre['mag_type']=='Mb','mag_type_index'] = int(2)
    
    # deal with p_clarity_index
    DiTing_330km_csv_pre.insert(10, 'p_clarity_index', DiTing_330km_csv_pre['p_clarity'])
    DiTing_330km_csv_pre.loc[DiTing_330km_csv_pre['p_clarity']==' ','p_clarity_index'] = int(0)
    DiTing_330km_csv_pre.loc[DiTing_330km_csv_pre['p_clarity']=='E','p_clarity_index'] = int(1)
    DiTing_330km_csv_pre.loc[DiTing_330km_csv_pre['p_clarity']=='I','p_clarity_index'] = int(2)
    DiTing_330km_csv_pre.loc[DiTing_330km_csv_pre['p_clarity']=='(','p_clarity_index'] = int(3)
    DiTing_330km_csv_pre.loc[DiTing_330km_csv_pre['p_clarity']=='n','p_clarity_index'] = int(4)
    
    # deal with p_motion_index
    DiTing_330km_csv_pre.insert(12, 'p_motion_index', DiTing_330km_csv_pre['p_motion'])
    DiTing_330km_csv_pre.loc[DiTing_330km_csv_pre['p_motion']==' ','p_motion_index'] = int(0)
    DiTing_330km_csv_pre.loc[DiTing_330km_csv_pre['p_motion']=='U','p_motion_index'] = int(1)
    DiTing_330km_csv_pre.loc[DiTing_330km_csv_pre['p_motion']=='R','p_motion_index'] = int(2)
    DiTing_330km_csv_pre.loc[DiTing_330km_csv_pre['p_motion']=='D','p_motion_index'] = int(3)
    DiTing_330km_csv_pre.loc[DiTing_330km_csv_pre['p_motion']=='C','p_motion_index'] = int(4)
    DiTing_330km_csv_pre.loc[DiTing_330km_csv_pre['p_motion']=='n','p_motion_index'] = int(5)
    
    DiTing_330km_csv_pre['baz'] = pd.to_numeric(DiTing_330km_csv_pre['baz'], errors='coerce')
    DiTing_330km_csv_pre.loc[np.isnan(DiTing_330km_csv_pre['baz']), 'baz'] = 0

    DiTing_330km_csv_pre['P_residual'] = pd.to_numeric(DiTing_330km_csv_pre['P_residual'], errors='coerce')
    DiTing_330km_csv_pre.loc[np.isnan(DiTing_330km_csv_pre['P_residual']), 'P_residual'] = 0

    DiTing_330km_csv_pre['S_residual'] = pd.to_numeric(DiTing_330km_csv_pre['S_residual'], errors='coerce')
    DiTing_330km_csv_pre.loc[np.isnan(DiTing_330km_csv_pre['S_residual']), 'S_residual'] = 0

    DiTing_330km_csv_pre['evmag'] = pd.to_numeric(DiTing_330km_csv_pre['evmag'], errors='coerce')
    DiTing_330km_csv_pre.loc[np.isnan(DiTing_330km_csv_pre['evmag']), 'evmag'] = 1.6
    
    DiTing_330km_csv_pre['st_mag'] = pd.to_numeric(DiTing_330km_csv_pre['st_mag'], errors='coerce')
    DiTing_330km_csv_pre.loc[np.isnan(DiTing_330km_csv_pre['st_mag']), 'st_mag'] = 1.6
    DiTing_330km_csv_pre.loc[DiTing_330km_csv_pre['st_mag']>8.0, 'st_mag'] = 8.0
    
    selected_columns = ['evmag', 'p_pick', 's_pick', 'dis', 'st_mag', 'baz', 'Z_P_amplitude_snr', 'Z_P_power_snr',
                    'Z_S_amplitude_snr', 'Z_S_power_snr']
    
    if ifnormalized:
        DiTing_330km_csv_pre['evmag'] = DiTing_330km_csv_pre['evmag']/7.7
        DiTing_330km_csv_pre['st_mag'] = DiTing_330km_csv_pre['st_mag']/8.0
        DiTing_330km_csv_pre['dis'] = DiTing_330km_csv_pre['dis']/330.0
        
        DiTing_330km_csv_pre.loc[DiTing_330km_csv_pre['baz']>360, 'baz'] -= 360
        DiTing_330km_csv_pre['baz'] = DiTing_330km_csv_pre['baz']/360.0
        
        
        DiTing_330km_csv_pre.loc[DiTing_330km_csv_pre['p_pick']<0, 'p_pick'] = 0
        DiTing_330km_csv_pre.loc[DiTing_330km_csv_pre['p_pick']>4000, 'p_pick'] = 4000
        DiTing_330km_csv_pre['p_pick'] = DiTing_330km_csv_pre['p_pick']/4000
        
        DiTing_330km_csv_pre.loc[DiTing_330km_csv_pre['s_pick']<0, 's_pick'] = 0
        DiTing_330km_csv_pre.loc[DiTing_330km_csv_pre['s_pick']>6000, 's_pick'] = 6000
        DiTing_330km_csv_pre['s_pick'] = DiTing_330km_csv_pre['s_pick']/6000
        
        DiTing_330km_csv_pre.loc[DiTing_330km_csv_pre['Z_P_amplitude_snr']>50,'Z_P_amplitude_snr'] = 50
        DiTing_330km_csv_pre['Z_P_amplitude_snr'] = DiTing_330km_csv_pre['Z_P_amplitude_snr']/50.0
        
        DiTing_330km_csv_pre.loc[DiTing_330km_csv_pre['Z_S_amplitude_snr']>5, 'Z_S_amplitude_snr'] = 5
        DiTing_330km_csv_pre['Z_S_amplitude_snr'] = DiTing_330km_csv_pre['Z_S_amplitude_snr']/5.0
        
        DiTing_330km_csv_pre.loc[DiTing_330km_csv_pre['Z_P_power_snr']<-10, 'Z_P_power_snr'] = -10
        DiTing_330km_csv_pre.loc[DiTing_330km_csv_pre['Z_P_power_snr']>30, 'Z_P_power_snr'] = 30
        DiTing_330km_csv_pre['Z_P_power_snr'] = (DiTing_330km_csv_pre['Z_P_power_snr']+10)/40
        
        DiTing_330km_csv_pre.loc[DiTing_330km_csv_pre['Z_S_power_snr']<-10, 'Z_S_power_snr'] = -10
        DiTing_330km_csv_pre.loc[DiTing_330km_csv_pre['Z_S_power_snr']>15, 'Z_S_power_snr'] = 15
        DiTing_330km_csv_pre['Z_S_power_snr'] = (DiTing_330km_csv_pre['Z_S_power_snr']+10)/25
        
# 0.003 57668.855 300
# -44.963 89.86 -30 30
# 0.0 833.487 20
# -129.66 50.964 -30 30
    
    return DiTing_330km_csv_pre


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
    
    
def cal_acc_location(label,pred,returnlat=False):
    label = label.cpu().detach().numpy()
    pred  =  pred.cpu().detach().numpy()
    
    pred = unscale(pred)  
    label = unscale(label)   
    latlon_km = 40075 / 360.

    err_dist = np.sqrt((pred[:, 0] - label[:, 0])**2 + (pred[:, 1] - label[:, 1])**2) * latlon_km
    err_lat = np.abs(pred[:, 0] - label[:, 0])
    err_lon = np.abs(pred[:, 1] - label[:, 1])
    err_depth = np.abs(pred[:, 2] - label[:, 2])
    err_mag = np.abs(pred[:, 3] - label[:, 3])
    return err_dist,err_depth,err_mag


def cal_acc_dis(label,pred):
    label = label.cpu().detach().numpy()
    pred  =  pred.cpu().detach().numpy()
    
    pred = unscale(pred)  
    label = unscale(label)   
    latlon_km = 40075 / 360.

    err_dist = np.sqrt((pred[:, 0] - label[:, 0])**2 + (pred[:, 1] - label[:, 1])**2) * latlon_km
    err_lat = np.abs(pred[:, 0] - label[:, 0])
    err_lon = np.abs(pred[:, 1] - label[:, 1])
    return err_dist

def cal_acc_dep_mag(label,pred,f_type = 'dep'):
    label = label.cpu().detach().numpy()
    pred  =  pred.cpu().detach().numpy()
    
    if f_type == 'dep':
        pred = unscale(pred,dep_type = True)  
        label = unscale(label,dep_type = True) 
    if f_type == 'mag':
        pred = unscale(pred,dep_type = False)  
        label = unscale(label,dep_type = False)  
        
    err_fea = np.abs(pred[:, 0] - label[:, 0])
    return err_fea


def cal_acc_degree(label,pred):
    # label = label.cpu().detach().numpy()
    # pred  =  pred.cpu().detach().numpy()
    abs_diff = torch.abs(label - pred)
    strike = torch.mean(torch.minimum(abs_diff[:,0], 1 - abs_diff[:,0]))*360
    dip = torch.mean(abs_diff[:,1])*90
    slip = torch.mean(torch.minimum(abs_diff[:,2], 1 - abs_diff[:,2]))*360
    return strike.cpu().detach().numpy(),dip.cpu().detach().numpy(),slip.cpu().detach().numpy()


def scale(x):
    minlatitude = 32
    maxlatitude = 36
    minlongitude = -120
    maxlongitude = -116
    maxdepth = 30e3
    minmag = 3
    maxmag = 6
    """ Function to scale the data in the range +/- 1 """
    if len(x.shape) == 2:
        x[:, 0] = (x[:, 0] - minlatitude) / (maxlatitude - minlatitude)
        x[:, 1] = (x[:, 1] - minlongitude) / (maxlongitude - minlongitude)
        x[:, 2] = x[:, 2] / maxdepth
        x[:, 3] = (x[:, 3] - minmag) / (maxmag - minmag)
    elif len(x.shape) == 3:
        x[:, :, 0] = (x[:, :, 0] - minlatitude) / (maxlatitude - minlatitude)
        x[:, :, 1] = (x[:, :, 1] - minlongitude) / (maxlongitude - minlongitude)
        x[:, :, 2] = x[:, :, 2] / maxdepth
        x[:, :, 3] = (x[:, :, 3] - minmag) / (maxmag - minmag)
    x = (x - 0.5) * 2
    return x


def unscale(x,dep_type = True):    
    minlatitude = 30
    maxlatitude = 45
    minlongitude = 130
    maxlongitude = 150
    maxdepth = 1000
    minmag = 3
    maxmag = 8
    
    """ Function to unscale the data """
    # x = x / 2 + 0.5
    if len(x.shape) == 2:
        if x.shape[1] == 1:
            if dep_type:
                x[:, 0] = x[:, 0] * maxdepth
            elif dep_type == False:
                x[:, 0] = x[:, 0] * maxmag
        if x.shape[1] == 2:
            x[:, 1] = x[:, 1] * (maxlatitude - minlatitude) + minlatitude
            x[:, 0] = x[:, 0] * (maxlongitude - minlongitude) + minlongitude
        if x.shape[1] == 4:
            x[:, 1] = x[:, 1] * (maxlatitude - minlatitude) + minlatitude
            x[:, 0] = x[:, 0] * (maxlongitude - minlongitude) + minlongitude
            x[:, 2] = x[:, 2] * maxdepth
            x[:, 3] = x[:, 3] * maxmag
        
    # elif len(x.shape) == 3:
    #     x[:, :, 0] = x[:, :, 0] * (maxlatitude - minlatitude) + minlatitude
    #     x[:, :, 1] = x[:, :, 1] * (maxlongitude - minlongitude) + minlongitude
    #     x[:, :, 2] = x[:, :, 2] * maxdepth * 1e-3
    #     x[:, :, 3] = x[:, :, 3] * (maxmag - minmag) + minmag
    return x


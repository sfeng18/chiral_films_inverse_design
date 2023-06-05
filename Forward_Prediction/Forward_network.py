import torch
import torch.nn as nn
from torch.nn import functional as F
import spectrum as sp
import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt
import matplotlib
import math
import csv 
# from paras import tag_wave, ideal_Gvalue
matplotlib.use('TkAgg')
task_sfx = 'dye_test_abs_250'
from paras import db_abs, col_max, col_min, thick_max, thick_min, str_max, str_min, gra_max, gra_min
# from paras import bit_str, bit_center, bit_width, tag_wave ,oth_wave,target_locat,oth_locat

#导入数据
Color_Dict = {
    '蓝2': 'blue2',
    '蓝6': 'blue6',
    '蓝71': 'blue71',
    '刚果红': 'congo red',
    '绿': 'green',
    '橘红': 'orange',
    '紫': 'purple',
    '红': 'red',
    '台盼蓝': 'tai',
    '黄': 'yellow',
}

Color_Values = ['yellow', 'orange', 'congo red', 'red', 'purple', 'tai', 'blue2', 'blue6', 'blue71', 'green']
G_in_dim = 20
NColors = 10
ngpu = 1  # The number of available GPUs, use 0 to run in CPU mode.
# Decide which device we run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
def getData1(path):
    datalist=[]
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            row = list(map(float, row))
            datalist.append(row)
    return datalist

def minmaxscaler(data, max, min):  #MinMaxScaler normalization method
    a = (data - min) / (max - min)
    return a

def inverse_minmaxscaler(data, max, min):  #denormalization function
    d = data * (max - min) + min
    return d

def colorData(dye_property_file):
    dye_property = sp.read_file(dye_property_file)
    color_dict = {}  #Store 121 parameters corresponding to the color
    j = 1
    for cl in Color_Values:
        color_dict[j] = []  #Store 121 pigment characteristic absorption spectrum data
        color_last = len(dye_property[cl]['abs'])  #Pick out 121 data
        color_step = int(color_last / 120)
        for i in range(0, color_last, color_step):  #Different colors have different amounts of data
            color_dict[j].append(dye_property[cl]['abs'][i])
        j += 1
    return color_dict

color_dict = colorData('./Data/dye_ABS.msgz')#Data file path

#Color data changed from 1 to 121
def color_transform(data):
    l = 121
    lis = [0.0 for _ in range(l)]
    colornum = torch.round(data).item()
    if colornum in range(1, NColors + 1):
        return color_dict[colornum]
    else:
        return lis

#The total data changed from 4 to 124, and normalized at the same time
def data_transform(data):
    clist = torch.squeeze(torch.tensor(color_transform(data[0])))
    data1 = torch.flatten(torch.cat((clist, data[1:]), 0))
    return data1

#Normalize the data, where the colors are globally normalized
def norm(data):
    color_x = data[:-3]  #One-dimensional color data
    color_x = minmaxscaler(color_x, col_max, col_min)
    color_x = torch.unsqueeze(color_x, 0)
    #Column-wise normalization for the other three parameters
    thick_x = minmaxscaler(data[-3], thick_max, thick_min)
    thick_x = torch.unsqueeze(torch.unsqueeze(thick_x, 0), 0)
    #Convert one-dimensional array to two-dimensional array
    # thick_x=torch.from_numpy(np.expand_dims(np.array(thick_x),axis=1))
    str_x = minmaxscaler(data[-2], str_max, str_min)
    str_x = torch.unsqueeze(torch.unsqueeze(str_x, 0), 0)
    gra_x = minmaxscaler(data[-1], gra_max, gra_min)
    gra_x = torch.unsqueeze(torch.unsqueeze(gra_x, 0), 0)
    x_un = torch.cat([color_x, thick_x, str_x, gra_x], dim=1)
    return x_un

class Batch_Net(nn.Module):
    """
    On the basis of Activation_Net above, a method to speed up the convergence speed is added - batch normalization
    """

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, n_hidden_5, out_dim):
        super(Batch_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3), nn.BatchNorm1d(n_hidden_3), nn.Dropout(p=0.5))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4), nn.BatchNorm1d(n_hidden_4), nn.ReLU(True))
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, n_hidden_5), nn.BatchNorm1d(n_hidden_5), nn.ReLU(True))
        self.layer6 = nn.Sequential(nn.Linear(n_hidden_5, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x

#Extract the entire forward prediction network
Forward_net =torch.load('./Forward_Prediction/net_predict_ABS_all_121_dye_test_250.pkl',map_location=torch.device('cpu') ).to(device)

def data_in_range(d):
    return (NColors >= d[0] >= 1 and thick_max >= d[1] >= thick_min and str_max >= d[2] >= str_min and gra_max >= d[3] >= gra_min)

def merge (datalist):
    a=datalist.tolist()
    b=str(a[0])+'-'+str(a[1])+'-'+str(a[2])+'-'+str(a[3])
    return b



#read data
Forward_net.eval()
# Get a batch of real images from the dataloader
result_data=getData1('./Forward_Prediction/random_20gabs.csv')
result_data=torch.tensor(result_data)
tag_wave=0
result_size = result_data.size(0)
right = []
net_G_data = []
loss_G_data = []
gan_G_data = []
hist_data = {}
wave = np.arange(200,805,5)
with open('./Forward_Prediction/20_gabs_outfile.csv','a',newline="",encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    onerow=[]
    onerow.append('feature：Color - thickness - stretch - gray')
    for i in wave:#Append the array content one by one
            onerow.append(i)
    writer.writerow(onerow)
    for j in range(0, result_size):
        complete_data=data_transform(result_data[j])#The total data changed from 4 to 124, and normalized at the same time
        complete_data_norm=norm(complete_data)#Normalize the data, where the colors are globally normalized
        net_G=Forward_net(complete_data_norm)
        feature=merge(result_data[j])
        newrow=[]
        newrow.append(feature)
    # r,g,b=rgb(net_G)
    # h,s,v=rgb2hsv(r,g,b)
        myarray=net_G.squeeze(0).detach().numpy()
        for i in myarray:#Append the array content one by one
            newrow.append(i)
        # writer.writerow(net_G.squeeze(0).detach().numpy().tolist())
        writer.writerow(newrow)

import torch
import torch.nn as nn
from torch.nn import functional as F
import spectrum as sp
import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt
import matplotlib
import math
import os
from paras import tag_wave, ideal_Gvalue, db_abs
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d import art3d
import matplotlib as mpl
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import art3d
import matplotlib.font_manager as fm
matplotlib.use('TkAgg')

#Import Data

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

if db_abs == 'ABS':
    col_max, col_min = 1.806, 0.04
elif db_abs == 'PROP':
    col_max, col_min = 0.671413806451613, 0.07797111363636361
else:
    raise RuntimeError('ERROR: Unkown db_abs value: %s' % db_abs)
thick_max, thick_min = 80.0, 17.0
str_max, str_min = 267.0, 20.0
gra_max, gra_min = 8.0, 1.0

# Decide which device we run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


def minmaxscaler(data, max, min):  #MinMaxScaler normalization method
    a = (data - min) / (max - min)
    return a


def inverse_minmaxscaler(data, max, min):  #denormalization function
    d = data * (max - min) + min
    return d

if db_abs == 'ABS':
    def colorMatrix(dye_property_file):
        dye_property = sp.read_file(dye_property_file)
        color_matrix = []  #Store 121 parameters corresponding to the color
        j = 1
        for cl in Color_Values:
            color_info = []  #Store 121 pigment characteristic absorption spectrum data
            color_last = len(dye_property[cl]['abs'])  #Pick out 121 data
            color_step = int(color_last / 120)
            for i in range(0, color_last, color_step):  #Different colors have different amounts of data
                color_info.append(dye_property[cl]['abs'][i])
            color_matrix.append(color_info)
            j += 1
        return torch.tensor(color_matrix, dtype=torch.float32)

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

    color_dict = colorData('./Data/dye_ABS.msgz')
    color_matrix = colorMatrix('./Data/dye_ABS.msgz').to(device)
else:
    def colorMatrix(dye_property_file):
        dye_property = sp.read_file(dye_property_file)
        color_matrix = []  #Store 121 parameters corresponding to the color
        j = 1
        for cl in Color_Values:
            color_info = []  #Store 121 pigment characteristic absorption spectrum data
            color_last = len(dye_property[cl])  #Pick out 121 data
            color_step = int(color_last / 120)
            for i in range(0, color_last, color_step):  #Different colors have different amounts of data
                color_info.append(dye_property[cl][i][1])
            color_matrix.append(color_info)
            j += 1
        return torch.tensor(color_matrix, dtype=torch.float32)

    def colorData(dye_property_file):
        dye_property = sp.read_file(dye_property_file)
        color_dict = {}  #Store 121 parameters corresponding to the color
        j = 1
        for cl in Color_Values:
            color_dict[j] = []  #Store 121 pigment characteristic absorption spectrum data
            color_last = len(dye_property[cl])  #Pick out 121 data
            color_step = int(color_last / 120)
            for i in range(0, color_last, color_step):  #Different colors have different amounts of data
                color_dict[j].append(dye_property[cl][i][1])
            j += 1
        return color_dict

    color_dict = colorData('./Data/dye_prop.msgz')
    color_matrix = colorMatrix('./Data/dye_prop.msgz').to(device)


#Color data changed from 1 to 121
def color_transform2(color):
    color_labels = torch.from_numpy(np.arange(1.0, NColors + 1.0, dtype=np.float32)).to(device)
    color_distance = F.relu(1.0 - torch.abs(color_labels - color))
    color_onehot = F.softmax(color_distance * 20, 0)
    return torch.matmul(color_onehot.reshape(1, -1), color_matrix)


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
    clist = torch.squeeze(color_transform(data[0]))
    data1 = torch.flatten(torch.cat((clist, data[1:]), 0))
    return data1


#The total data changed from 4 to 124, and normalized at the same time
def data_transform_dict(data):
    clist = torch.squeeze(torch.tensor(color_transform(data[0])))
    data1 = torch.flatten(torch.cat((clist, data[1:]), 0))
    return data1


def norm(data):
    #Globally normalize colors
    color_x = data[:-3]  
    color_x = minmaxscaler(color_x, col_max, col_min)
    color_x = torch.unsqueeze(color_x, 0)
    #Column-wise normalization for the other three parameters
    thick_x = minmaxscaler(data[-3], thick_max, thick_min)
    thick_x = torch.unsqueeze(torch.unsqueeze(thick_x, 0), 0)
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
        
#build generator
class Generator(nn.Module):

    def __init__(self, ngpu):
        self.ngpu = ngpu
        super(Generator, self).__init__()
        self.data = nn.Sequential(
            nn.Linear(G_in_dim, 40),
            nn.LazyBatchNorm1d(),
            nn.ReLU(True),
            nn.Linear(40, 80),
            nn.LazyBatchNorm1d(),
            nn.ReLU(True),
            nn.Linear(80, 20),
            nn.LazyBatchNorm1d(),
            nn.ReLU(True),
            nn.Linear(20, 5),
        )

    def forward(self, x):
        x = self.data(x)
        return x

#Extract the entire forward prediction network
Forward_net = torch.load('./Inverse_Design_2_Fluorescence/%g/net_predict_%s_%g.pkl' % (tag_wave,db_abs, tag_wave)).to(device)
Generator_net = torch.load('./Inverse_Design_2_Fluorescence/%g/netG_%s_%g_%g.pkl' % (tag_wave, db_abs, tag_wave, ideal_Gvalue)).to(device)


def data_in_range(d):
    return (NColors >= d[0] >= 1 and thick_max >= d[1] >= thick_min and str_max >= d[2] >= str_min and gra_max >= d[3] >= gra_min)


def data_real(d):
    if abs(d[0] - round(float(d[0]))) > 0.3:
        return False
    for thick in (17, 30, 48, 60, 80):
        if abs(d[1] - thick) < 3:
            return True
    return False


#read data
Generator_net.eval()
Forward_net.eval()
# Get a batch of real images from the dataloader
noise1 = torch.randn(2000, G_in_dim, device=device)
result_data = Generator_net(noise1)
# print(result_data)
result_size = result_data.size(0)
right = []
net_G_data = []
loss_G_data = []
gan_G_data = []
hist_data = {}
for j in range(0, result_size):
    result_data[j][0] = inverse_minmaxscaler(result_data[j][0], NColors, 1)
    result_data[j][1] = inverse_minmaxscaler(result_data[j][1], thick_max, thick_min)
    result_data[j][2] = inverse_minmaxscaler(result_data[j][2], str_max, str_min)
    result_data[j][3] = inverse_minmaxscaler(result_data[j][3], gra_max, gra_min)
    # if result_data[j][4].to(torch.float64) >= -0.6:
    if data_in_range(result_data[j]) and data_real(result_data[j]):
        # print(result_data[j])
        gan_G_data.append(result_data[j][4].item())
        data_to_exp = result_data[j]
        data_to_exp[0] = data_to_exp[0].round()
        cl = Color_Values[int(data_to_exp[0]) - 1]
        if cl not in hist_data:
            hist_data[cl] = [data_to_exp[1:].tolist()]
        else:
            hist_data[cl].append(data_to_exp[1:].tolist())
        complete_data = data_transform_dict(data_to_exp[:-1])
        complete_data_norm = norm(complete_data)
        net_G = Forward_net(complete_data_norm)
        net_G_data.append(net_G.item())
        loss_G = net_G - result_data[j][4]
        loss_G_data.append(loss_G.item())
        OutFile = './Inverse_Design_2_Fluorescence/%g/cgan_test_result_%s_%g.csv' % (tag_wave,db_abs, tag_wave)
        if not os.path.isfile(OutFile):
            Head = 'Color,Thick,Strain,Gray,G_GAN,G_Predict,dG\n'
            open(OutFile, 'wt').writelines(Head)
        with open(OutFile, 'a') as f:
            f.write(','.join(['%g'%_ for _ in result_data[j].tolist()[:4] + [result_data[j][4].item(), net_G.item(), loss_G.item()]]) + '\n')


plt.plot(net_G_data, 'b')
plt.plot(gan_G_data, 'g')
plt.plot(loss_G_data, 'r')
plt.show()


def factor_pair(num):
    '''Factoring one number into two closest factors'''
    a = int(math.sqrt(num))
    while a > 1:
        if num % a == 0:
            return (a, num // a)
        a -= 1
    return (1, num)


NPlot = len(hist_data)
Ncol, Nrow = factor_pair(NPlot)
if Ncol == 1 and Nrow > 3 * Ncol:
    Ncol, Nrow = factor_pair(NPlot + 1)

TickSize = 12
LabelSize = 14
TitleSize = 16
plt.figure(figsize=(14.4, 7.2))
Out_Colors = [_ for _ in Color_Values if _ in hist_data]
for i, cl in enumerate(Out_Colors):
    plt.subplot(Ncol, Nrow, i + 1)
    XYC = np.array(hist_data[cl])[:, :3]
    plt.scatter(
        XYC[:, 0],
        XYC[:, 1],
        c=XYC[:, 2],
        vmin=gra_min,
        vmax=gra_max,
        cmap='jet',
    )
    if (i + 1) % Nrow == 0:
        cbar = plt.colorbar()
        cbar.set_label('Grayscale', fontsize=LabelSize)
    plt.title("Color: " + cl, fontsize=TitleSize)
    if i // Nrow == Nrow - 1:
        plt.xlabel(r"Thickness ($\mu$m)", fontsize=LabelSize)
    if i % Nrow == 0:
        plt.ylabel("Strain (%)", fontsize=LabelSize)
    plt.xlim(thick_min, thick_max)
    plt.ylim(str_min, str_max)
plt.suptitle('Target: %g nm, G= %g' % (tag_wave, ideal_Gvalue), fontsize=TitleSize)
plt.savefig('./Inverse_Design_2_Fluorescence/%g/G_%s_%g_%g.png' % ( tag_wave,db_abs, ideal_Gvalue, tag_wave), dpi=300)


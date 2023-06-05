# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 10:40:15 2022

@author: DELL
"""

import torch
from torch import nn, optim
from torch.autograd import Variable
import new_spectrum as sp
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import metrics
from os import path

Color_Dict = {'蓝2': 'blue2', '蓝6': 'blue6', '蓝71': 'blue71','刚果红':'congo red','绿':'green','橘红':'orange','紫':'purple','红':'red','台盼蓝':'tai','黄':'yellow'}
Color_Values=['yellow','orange','congo red','red','purple','tai','blue2','blue6','blue71','green']


def draw_scatter(x,y):
    fig = plt.figure(dpi=100,figsize=(5,5))
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.rcParams['lines.linewidth'] = 3
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_xlabel('Experiment',fontsize=35)
    ax1.set_ylabel('Prediction',fontsize=35)
    ax1.scatter(x, y, s=10, c='k', marker='.')
    plt.xticks([-2.0,0.0,2.0],fontsize=25)
    plt.yticks([-2.0,0.0,2.0],fontsize=25)
    a=np.arange(-2.0,2.0,0.1)
    b=a
    plt.plot(a,b)
    bwith = 3
    TK = plt.gca()
    TK.spines['bottom'].set_linewidth(bwith)
    TK.spines['left'].set_linewidth(bwith)
    TK.spines['top'].set_linewidth(bwith)
    TK.spines['right'].set_linewidth(bwith)
    plt.show()

def draw_plot(y1,y2):
    x=[]
    for i in range(800,199,-5):
        x.append(i)
    plt.figure(dpi=100,figsize=(10,10))
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.rcParams['lines.linewidth'] = 3
    plt.xlabel('Wavelenth/nm',fontsize=30)
    plt.ylabel('G_value',fontsize=30)
    plt.plot(x,y1,c='r',label='Experiment')
    plt.plot(x,y2,c='b',label='Prediction')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(loc=0,fontsize=25)
    bwith = 3
    TK = plt.gca()
    TK.spines['bottom'].set_linewidth(bwith)
    TK.spines['left'].set_linewidth(bwith)
    TK.spines['top'].set_linewidth(bwith)
    TK.spines['right'].set_linewidth(bwith)
    plt.show()

def rand_curves(len, num, Nsmooth=10, scale=None):
    y = np.random.normal(scale=1, size=num)
    curves = []
    for _ in range(len + Nsmooth - 1):
        curves.append(y)
        y = y + np.random.normal(scale=1, size=num)
    curves = np.array(curves)
    curves -= np.mean(curves, axis=0).reshape((1, -1))
    if scale is not None:
        factors = abs(scale) / np.max(np.abs(curves), axis=0)
        curves = curves * factors.reshape((1, -1))
    ones = np.ones((Nsmooth,)) / Nsmooth
    final_curves = []
    for cv in curves.T:
        final_curves.append(np.convolve(cv, ones, mode='valid'))
    return np.array(final_curves)

def extend_dataset(train, target, multifact=9, scale=0.001):
    train = np.array(train)
    target = np.array(target)
    Ntrain = len(train)
    Ncurves = multifact * Ntrain
    noise_curves = rand_curves(121, 2 * Ncurves, Nsmooth=10, scale=scale)
    noise_curves2 = noise_curves[Ncurves:, :]
    noise_paras = np.random.normal(scale=scale / 3, size=(Ncurves, 4))
    new_train, new_target = [train], [target]
    for i in range(multifact):
        N_st = i * Ntrain
        N_ed = (i + 1) * Ntrain
        my_train = np.hstack([train[:, :121] + noise_curves[N_st:N_ed, :], train[:, -4:] + noise_paras[N_st:N_ed, :]])
        my_target = target + noise_curves2[N_st:N_ed, :]
        new_train.append(my_train)
        new_target.append(my_target)
    return np.vstack(new_train), np.vstack(new_target)

class Batch_Net(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, n_hidden_5, out_dim):
        super(Batch_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3), nn.BatchNorm1d(n_hidden_3), nn.ReLU(True))
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


# define some hyperparameters
batch_size = 16
lr = 0.0001
Noise_scale = 0.001
my_net = Batch_Net(125, 200, 250, 250, 250, 200, 121)
optimizer = torch.optim.Adam(my_net.parameters(),lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5)
criterion = nn.SmoothL1Loss()

dye_property = sp.read_file('./Data/dye_ABS.msgz')
sdb_statistics = sp.read_file('./Data/sdb.msgz')

###########################################################
x = []
for value in Color_Dict.values():
    x1 = []
    for i in range(0,len(dye_property[value]['abs']),int(len(dye_property[value]['abs'])/120)):
        x1.append(dye_property[value]['abs'][i])
    x2=x1[:]
    for key in sdb_statistics.keys():
        sdb_statistics_key = key.split('-')
        if sdb_statistics_key[0] == value:
            for j in range(2,6):
                x1.append(float(sdb_statistics_key[j]))
            x.append(x1)
            x1 = x2[:]
###################################################################
m=[]
for value in Color_Dict.values():
    for key in sdb_statistics.keys():
        sdb_statistics_key = key.split('-')
        if sdb_statistics_key[0] == value:
            m0 = []
            for index,value0 in enumerate(sdb_statistics[key]['G']):
                if index%int(len(sdb_statistics[key]['G'])/120) == 0:
                    m0.append(value0)
            m.append(m0)
###################################################################

TRAIN_PERCENT = 0.8
train_size = int(len(x) * TRAIN_PERCENT)

if path.exists('./Data/train_idx.npy'):
    train_idx = np.load('./Data/train_idx.npy')
else:
    train_idx = np.random.choice(len(x), train_size, replace=False)
    np.save('./Data/train_idx.npy', train_idx)
test_idx = np.setdiff1d(np.arange(len(x)), train_idx)

train = []
target = []
test = []
test_true = []
for train_num in train_idx:
    train.append(x[train_num])
    target.append(m[train_num])
for test_num in test_idx:
    test.append(x[test_num])
    test_true.append(m[test_num])

train, target = extend_dataset(train, target, scale=Noise_scale)
train = torch.tensor(train, dtype=torch.float32)
target = torch.tensor(target, dtype=torch.float32)
test = torch.tensor(test,dtype=torch.float32)

#################################################################

epoch = 0
for i in range(10000):
    img = Variable(train)
    label = Variable(target)
    out = my_net(img)
    loss = criterion(out, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch += 1
    if epoch % 1000 == 0:
        print('epoch: {}, loss: {:.4}'.format(epoch, loss))

# torch.save(my_net, 'my_net.pkl')

my_net.eval()

predict = my_net(test)
predict = torch.squeeze(predict).detach().numpy().tolist()
predict = sum(predict,[])
test_true = sum(test_true,[])

predict0 = my_net(train)
predict0 = torch.squeeze(predict0).detach().numpy().tolist()
predict0 = sum(predict0,[])
target = target.detach().numpy().tolist()

######################################################################

print('test picture')
draw_scatter(test_true,predict)
test_true = np.array(test_true)
predict = np.array(predict)
correlation = np.corrcoef(test_true,predict)
R2 = correlation**2
print('test R2:',R2[0][1])

print('train picture')
draw_scatter(target, predict0)
target = np.array(target).reshape(-1)
predict0 = np.array(predict0).reshape(-1)
correlation = np.corrcoef(target,predict0)
R3 = correlation**2
print('train R2:',R3[0][1])

###################################################################

print('Test random sample')
target = np.array(target).reshape(-1,121)
predict0 = np.array(predict0).reshape(-1,121)
random_sample = random.sample(range(0,len(target)-1),1)
for i in random_sample:
    draw_plot(target[i],predict0[i])

print('Train random sample')
test_true = np.array(test_true).reshape(-1,121)
predict = np.array(predict).reshape(-1,121)
random_sample = random.sample(range(0,len(test_true)-1),1)
for i in random_sample:
    draw_plot(test_true[i],predict[i])

#####################

#some evaluation indexes of training set
y_true=np.array(target).reshape(-1)
y_pred=np.array(predict0).reshape(-1)
# MSE
MSEdata0=metrics.mean_squared_error(y_true, y_pred)
# RMSE
RMSEdata0=np.sqrt(metrics.mean_squared_error(y_true, y_pred)) 
# MAE
MAEdata0=metrics.mean_absolute_error(y_true, y_pred)
print('train\n'+'MSE:'+str(MSEdata0)+'\n'+'RMSE:'+str(RMSEdata0)+'\n'+'MAE:'+str(MAEdata0)+'\n')

#some evaluation indexes of testing set
y_true1=np.array(test_true)
y_pred1=np.array(predict)
# MSE
MSEdata1=metrics.mean_squared_error(y_true1, y_pred1)
# RMSE
RMSEdata1=np.sqrt(metrics.mean_squared_error(y_true1, y_pred1))
# MAE
MAEdata1=metrics.mean_absolute_error(y_true1, y_pred1)
print('test\n'+'MSE:'+str(MSEdata1)+'\n'+'RMSE:'+str(RMSEdata1)+'\n'+'MAE:'+str(MAEdata1)+'\n')

# filename='model evaluation.txt'
# with open(filename, 'w') as f:
#     f.write('train\n'+'R2:'+str(R3[0][1])+'\n'+'MSE:'+str(MSEdata0)+'\n'+'RMSE:'+str(RMSEdata0)+'\n'+'MAE:'+str(MAEdata0)+'\n')
#     f.write('test\n'+'R2:'+str(R2[0][1])+'\n'+'MSE:'+str(MSEdata1)+'\n'+'RMSE:'+str(RMSEdata1)+'\n'+'MAE:'+str(MAEdata1)+'\n')

import torch
from torch import nn, optim
from torch.autograd import Variable
import spectrum as sp
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn import metrics


Color_Dict = {'蓝2': 'blue2', '蓝6': 'blue6', '蓝71': 'blue71','刚果红':'congo red','绿':'green','橘红':'orange','紫':'purple','红':'red','台盼蓝':'tai','黄':'yellow'}
Color_Values=['yellow','orange','congo red','red','purple','tai','blue2','blue6','blue71','green']
# Color_Values.sort()
def minmaxscaler(data):   #MinMaxScaler normalization method
    data_list=data.tolist()
    Min = min(data_list)
    Max = max(data_list)    
    unifydata=(data - Min)/(Max-Min)
    return unifydata,Max,Min



#MAPE and SMAPE need to be implemented by themselves
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

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


# define some hyperparameters
batch_size = 16
lr = 0.0001
my_net = Batch_Net(124, 200, 250, 250, 250, 200, 3)

optimizer = torch.optim.Adam(my_net.parameters(),lr)
#optimizer = optim.SGD(my_net.parameters(), lr)
#optimizer = optim.SGD(my_net.parameters(), lr = 0.01, momentum=0.9,weight_decay=1e-5) 
#optimizer = torch.optim.RMSprop(my_net.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5)

criterion = nn.SmoothL1Loss()
#criterion = nn.MSELoss(reduction='mean')

dye_property = sp.read_file('./Data/dye_ABS.msgz')
sdb_statistics = sp.read_file('./Data/sdb.msgz')


x = []#1493*125 database
a=[]
for value in Color_Values:
    x1 = []
    for i in range(0,len(dye_property[value]['abs']),int(len(dye_property[value]['abs'])/120)):
        x1.append(dye_property[value]['abs'][i])
    x2=x1[:]
    for key in sdb_statistics.keys():
        sdb_statistics_key = key.split('-')
        if sdb_statistics_key[0] == value:
            for j in range(3,6):
                x1.append(float(sdb_statistics_key[j]))
            a.extend(x1)
            x.append(x1)
            x1 = x2[:]
a= torch.from_numpy(np.array(a).reshape(1493,124))
color_x=a[:,:-3].flatten()#Convert two-dimensional array to one-dimensional
color_x,col_max,col_min=minmaxscaler(color_x)
color_x= torch.from_numpy(np.array(color_x).reshape(1493,121))
#Column-wise normalization for the other three parameters
thick_x,thick_max,thick_min=minmaxscaler(a[:,-3])
thick_x=torch.from_numpy(np.expand_dims(np.array(thick_x),axis=1))#Convert one-dimensional array to two-dimensional array
str_x,str_max,str_min=minmaxscaler(a[:,-2])
str_x=torch.from_numpy(np.expand_dims(np.array(str_x),axis=1))
gra_x,gra_max,gra_min=minmaxscaler(a[:,-1])
gra_x=torch.from_numpy(np.expand_dims(np.array(gra_x),axis=1))
x_un=torch.cat([color_x,thick_x,str_x,gra_x],dim=1)
max_min_list=[col_max,col_min,thick_max,thick_min,str_max,str_min,gra_max,gra_min]

x = x_un.tolist()

m=[]#Pick out 121 G values
for value in Color_Values:
    for key in sdb_statistics.keys():
        sdb_statistics_key = key.split('-')
        if sdb_statistics_key[0] == value:
            m0 = []
            for index1,value1 in enumerate(sdb_statistics[key]['X']):
                if value1 == 445.0:
                    k = index1
                if value1 == 520.0:
                    y=index1
                if value1 == 634.0:
                    e=index1
            m0.append(sdb_statistics[key]['G'][k])
            m0.append(sdb_statistics[key]['G'][y])
            m0.append(sdb_statistics[key]['G'][e])
            m.append(m0)



TRAIN_PERCENT = 0.95
train_size = int(len(x) * TRAIN_PERCENT)
train_idx = np.random.choice(len(x), train_size, replace=False)
#Find the difference of collection elements in 2 arrays. Return value: sorted unique values that are in ar1 but not in ar2
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

train_data = torch.tensor(train, dtype=torch.float32)
target_train_data = torch.tensor(target, dtype=torch.float32)
test_data = torch.tensor(test,dtype=torch.float32)
target_test_data=torch.tensor(test_true,dtype=torch.float32).view(75,3)

epoch = 0
for i in range(13000):
    img = Variable(train_data)
    label = Variable(target_train_data)
    out = my_net(img)
    loss = criterion(out, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch += 1
    if epoch % 100 == 0:
        print('epoch: {}, loss: {:.4}'.format(epoch, loss))

# torch.save(my_net, 'net_newall.pkl')

plt.switch_backend('TKAgg')
def draw_scatter(x,y):
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_title('Result Analysis')
    ax1.set_xlabel('True')
    ax1.set_ylabel('Predict')
    ax1.scatter(x, y, s=10, c='k', marker='.')
    a = np.arange(-2, 2, 0.1)
    b = a
    plt.plot(a,b)
    plt.show()

def draw_plot(y1,y2):
    x=[]
    for i in range(800,199,-5):
        x.append(i)
    plt.figure()
    plt.title('Compare of Predict and True')
    plt.xlabel('λ/nm')
    plt.ylabel('G')
    plt.plot(x,y1,c='k')
    plt.plot(x,y2,c='y')
    plt.show()


my_net.eval()
predict = my_net(test_data)
predict = torch.squeeze(predict).detach().numpy().tolist()
b = np.array(target_test_data)
target_test_data=b.tolist()
plt.figure()
plt.plot(target_test_data,'b')
plt.plot(predict,'r')
plt.show()
predict0 = my_net(train_data)
predict0 = torch.squeeze(predict0).detach().numpy().tolist()
target_train_data =target_train_data.detach().numpy().tolist()
d = np.array(target_train_data)
target_train_data=d.tolist()
plt.plot(target_train_data,'b')
plt.plot(predict0,'r')
plt.show()

print('test picture')
draw_scatter(target_test_data,predict)
target_test_data= np.array(target_test_data)
predict = np.array(predict)
correlation = np.corrcoef(target_test_data,predict)
R2 = correlation**2
print(R2)
print('train picture')
draw_scatter(target_train_data, predict0)

####################
#training set metrics
# MSE
y_true=np.array(target_train_data)
y_pred=np.array(predict0)
MSEdata0=metrics.mean_squared_error(y_true, y_pred)# 8.107142857142858
# RMSE
RMSEdata0=np.sqrt(metrics.mean_squared_error(y_true, y_pred)) # 2.847304489713536
# MAE
MAEdata0=metrics.mean_absolute_error(y_true, y_pred) # 1.9285714285714286
# MAPE
MAPEdata0=mape(y_true, y_pred)# 76.07142857142858
# SMAPE
SMAPEdata0=smape(y_true, y_pred) # 57.76942355889724
print('MSE:'+str(MSEdata0)+'\n'+'RMSE:'+str(RMSEdata0)+'\n'+'MAE:'+str(MAEdata0)+'\n'+'MAPE:'+str(MAPEdata0)+'\n'+'SMAPE:'+str(SMAPEdata0)+'\n')

#Test set metrics
y_true1=np.array(target_test_data)
y_pred1=np.array(predict)
MSEdata1=metrics.mean_squared_error(y_true1, y_pred1)# 8.107142857142858
# RMSE
RMSEdata1=np.sqrt(metrics.mean_squared_error(y_true1, y_pred1)) # 2.847304489713536
# MAE
MAEdata1=metrics.mean_absolute_error(y_true1, y_pred1) # 1.9285714285714286
# MAPE
MAPEdata1=mape(y_true1, y_pred1)# 76.07142857142858
# SMAPE
SMAPEdata1=smape(y_true1, y_pred1) # 57.76942355889724
print('MSE:'+str(MSEdata1)+'\n'+'RMSE:'+str(RMSEdata1)+'\n'+'MAE:'+str(MAEdata1)+'\n'+'MAPE:'+str(MAPEdata1)+'\n'+'SMAPE:'+str(SMAPEdata1)+'\n')

torch.save(my_net, './Inverse_Design_1_Laser/net_newall_445+520+634_200_5.27.pkl')



#%% Import Libraries
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as Function
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import make_grid , save_image
from scipy import stats, integrate
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
sns.set(color_codes=True)

import time


# Load data set
movies=pd.read_csv('DataMovie/ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
users=pd.read_csv('DataMovie/ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings=pd.read_csv('DataMovie/ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

from sklearn.model_selection import train_test_split

trainingset, testset= train_test_split(ratings, train_size=0.7)

len(trainingset)

# convert to array as it is quicker 
trainingset=np.array(trainingset, dtype='int')
testset=np.array(testset, dtype='int')

# get total no. of movies and users in order to then make a matrix of the data
nb_users= int(max(max(trainingset[:,0]), max(testset[:,0])))
nb_movies=int(max(max(trainingset[:,1]), max(testset[:,1])))
print(nb_users, nb_movies)

# make matrix of users in lines and movies in columns
def convert(data):
    new_data=[] #initialise list
    for id_users in range(1, nb_users+1):
        id_movies=data[:,1][data[:,0]==id_users]
        id_ratings=data[:,2][data[:,0]==id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

trainingset=convert(trainingset)
testset=convert(testset)



# convert data into torch sensors
training_set = torch.FloatTensor(trainingset)
test_set=torch.FloatTensor(testset)

new=[]
for i in range(1, len(training_set)):
    x=training_set[i:i+1]
    if len(x[x>=0])>0:
        new.append(x)
    else:
        continue

len(new)

# convert ratings (1-5) into binary ratings 1 (liked) and 0 (not liked)

training_set[training_set==0] = -1 # not rated
training_set[training_set==1] = 0
training_set[training_set==2] = 0
training_set[training_set>=3] = 1

test_set[test_set==0] = -1 # not rated
test_set[test_set==1] = 0
test_set[test_set==2] = 0
test_set[test_set>=3] = 1


#%% Build RBM Class
class RBM(nn.Module):
    def __init__(self,
                 n_vis=794,
                 n_hid=625,
                 k=10):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hid,n_vis)*1e-2)
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hid))
        self.k = k
    
    def sample_p(self,p):
        return Function.relu(torch.sign(p - Variable(torch.rand(p.size()))))
    
    def v_to_h(self,v):
        p_h = Function.sigmoid(Function.linear(v,self.W,self.h_bias))
        sample_h = self.sample_p(p_h)
        return p_h,sample_h
    
    def h_to_v(self,h):
        p_v = Function.sigmoid(Function.linear(h,self.W.t(),self.v_bias))
        sample_v = self.sample_p(p_v)
        return p_v,sample_v
        
    def forward(self,v):
        pre_h1,h1 = self.v_to_h(v)
        
        h_ = h1
        
        for _ in range(self.k):
            pre_v_,v_ = self.h_to_v(h_)
            pre_h_,h_ = self.v_to_h(v_)
        
        return v,v_, h_
    
    def free_energy_cost(self,v):
        vbias_term = v.mv(self.v_bias)
        wx_b = Function.linear(v,self.W,self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()
    
    

        

rbm = RBM(k=1)
train_op = optim.SGD(rbm.parameters(),0.1)


batch_size_=500
reconerr=[]
feerror=[]

for epoch in range(10):
    loss_ = []
    reconstruction_error=0
    s=0
    for n in range(0, len(training_set)- batch_size_, batch_size_):
        vk=training_set[n:n+batch_size_]
        vk=Variable(vk)
        
        v0=training_set[n:n+batch_size_]
        v0=Variable(v0)
              
        ph0,_=rbm.v_to_h(v0)
        for k in range(1):
            _,hk =rbm.v_to_h(vk)
            _,vk=rbm.h_to_v(hk)
            
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.v_to_h(vk)
        
        loss = rbm.free_energy(v0) - rbm.free_energy(vk)
        loss_.append(loss.data[0])
        train_op.zero_grad()
        
        loss.backward()
        train_op.step()
    
        reconstruction_error+=torch.mean(torch.abs(v0-vk))
        s+=1
        
    reconerr.append(reconstruction_error.data.numpy()[0]/s)
    feerror.append(np.mean(loss_))
    
    print ('Reconstructions error:  ' + str(reconstruction_error.data.numpy()[0]/s)+ 
          ' - free energy loss: ' + str(np.mean(loss_)))  

for i in feerror:
    print(i)

# Testing

import time
start_time = time.time()

test_loss=0                                       # need to measure error. loss function could use RMSE (done in autoencoders)
s=0 
vis1=[]
for id_user in range(0, nb_users): #batch learning
    v=test_set[id_user:id_user+ 1]  # training set inputs are used to activate neurons of our RBM
    vt=test_set[id_user:id_user + 1] #target
    
    v=Variable(v)
    vt=Variable(vt)
    if len(vt[vt>=0])>0:
        _,h =rbm.v_to_h(v)
        _,v=rbm.h_to_v(h)
        
        test_loss += torch.mean(torch.abs(vt[vt>=0]-v[vt>=0]))   #update train loss
        s+=1
    
    v=v.data.numpy()
    vis1.append(v)


print ('Reconstructions error:  ' +  str(test_loss/s))
print("--- training time is %s seconds ---" % (time.time() - start_time))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 14:41:50 2017

@author: IzzIbrahim
"""

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

#%% Load Data
BatchSize = 64
train_loading = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()])),batch_size=BatchSize)
test_loading = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor()])),batch_size=BatchSize)

#%% Arrange training data properly
input_data=[]
target_data=[]
for _, (data,target) in enumerate(train_loader):
        data =data.view(-1,784)
        data=data.numpy()
        input_data.append(data)
        target=target.view(-1)
        target=target.numpy()
        target_data.append(target)
        
n=np.array(target_data[0])
x=torch.from_numpy(n)
for i in range(1,938):
    y=np.array(target_data[i])
    y=torch.from_numpy(y)
    x=torch.cat((x, y),0)
    
full_input=np.array(input_data[0])
full_input=torch.Tensor(full_input)

for i in range(1,938):
    intermed=np.array(input_data[i])
    intermed=torch.Tensor(intermed)
    full_input=torch.cat((full_input, intermed),0)
x=x.numpy()
target_array=np.zeros((60000,10))

count=0
for n in x:
    target_array[count][n]=1
    count+=1

target_array=torch.Tensor(target_array)
final_training_data=torch.cat((full_input, target_array), 1)

#%% Arrange testing data properly
input_data=[]

for _, (data,target) in enumerate(test_loader):
        data =data.view(-1,784)
        data=data.numpy()
        input_data.append(data)  
full_input=np.array(input_data[0])
full_input=torch.Tensor(full_input)

for i in range(1,157):
    intermed=np.array(input_data[i])
    intermed=torch.Tensor(intermed)
    
    full_input=torch.cat((full_input, intermed),0)     

test_dummy=np.zeros((len(full_input), 10))
test_dummy=torch.Tensor(test_dummy)
test_set=torch.cat((full_input, test_dummy), 1)

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

#%% Train RBM 
rbm = RBM(k=10, n_vis=784, n_hid=625)
training_optimiser = optim.SGD(rbm.parameters(),0.1)

for epoch in range(5):
    free_energy_loss = []   # initialise values
    reconstruction_error=0
    s=0
    for n in range(0, len(full_input)- BatchSize, BatchSize): 
        training_data=full_input[n:n+BatchSize]
        training_data= Variable(training_data)
        training_data = training_data.bernoulli()   # set values to bernoulli RBM
    
        v,v1,h1 = rbm(training_data)
        
        loss = rbm.free_energy_cost(v) - rbm.free_energy_cost(v1)
        free_energy_loss.append(loss.data[0])
        
        training_optimiser.zero_grad()
        loss.backward()
        training_optimiser.step()
        
        reconstruction_error+=torch.mean(torch.abs(v-v1))
        s+=1
        
    print ('Epoch: ' + str(epoch+1) +  
           ' - reconstructions error:  ' + str(reconstruction_error.data.numpy()[0]/s) + 
           ' - free energy loss: ' + str(np.mean(free_energy_loss)))  
    
#%% Test RBM
start_time = time.time()
output=[]
target_output=[]

test_loss = 0
s=0
for n in range(0,len(test_set)):
    testing_data=Variable(test_set)
    testing_data = testing_data[n:n+1]

    v,v1,h1 = rbm(testing_data)
    reconstruction_error+=torch.mean(torch.abs(v-v1))
    s+=1
    
    loss = rbm.free_energy_cost(v) - rbm.free_energy_cost(v1)
    free_energy_loss.append(loss.data[0])
    
    output.append(v1)
    target_output.append(v)

print ('Reconstructions error:  ' + str(reconstruction_error.data.numpy()[0]/s)+ 
          ' - free energy loss: ' + str(np.mean(free_energy_loss)))  
print("--- testing time is %s seconds ---" % (time.time() - start_time))
    

#%% Save Model
# save entire model
f='rbm.pkl'
torch.save(rbm, f)

rbm=torch.load('rbm.pkl')

#%% Arrange Output
target_data=[]
for _, (data,target) in enumerate(test_loader):
    target=target.view(-1)
    target=target.numpy()
    target_data.append(target)

n=np.array(target_data[0])
target_1=torch.from_numpy(n)

for i in range(1,157):
    y=np.array(target_data[i])
    y=torch.from_numpy(y)
    target_1=torch.cat((target_1, y),0)
target_1=target_1.numpy()
output_array=[]

for n in range(0, 10000):
    output_n=output[n][0].data.numpy()
    output_array.append(output_n)
    
output_array=np.array(output_array)
digits=[]

for n in range(0, 10000):
    digit=output_array[n][784:]
    digits.append(digit)

#%% Classify

def getdigit(outputarray):
    results=[]
    index=[]
    for i in range(0,len(outputarray)):
        for n in range(0, 10):
            if outputarray[i][n]==1:
                results.append(int(n))
                index.append(int(i))
        if all(outputarray[i][0:10]==0):
            results.append(int(0))
            index.append(int(i))
    total=list(zip(index,results))
    return total
output_digits=getdigit(digits)
classification=[]
for i, x in enumerate(output_digits):
    classification.append(x)
classification=np.array(classification)
df_class=pd.DataFrame(classification)
final_class=df_class.drop_duplicates(subset=0, keep='first')
final_class=final_class.reset_index()
correct=0
for i in range(0, 10000):
    if target_1[i]==final_class[1][i]:
        correct+=1      
accuracy=correct/len(target_1)
print ( 'Classification accuracy is %s ' %accuracy)




#%% Save Weights
weights=rbm.W
weights=weights.data.numpy()
x=weights.reshape(490000)

#Visualise
fig=sns.set_palette('Set2')
fig=sns.set_style('white')
fig=sns.kdeplot(x,bw=0.1)

fig.set(yticks=[])
plt.xlabel('Weight Connections', fontsize=16)
plt.savefig('Weights Distribution.png')
plt.show()



#%% Prune

m=np.percentile(abs(x), 25)

i = []
for w in x:
    if w < m and w > -m:
        w=0      
    else:
        w=w
    i.append(w)
i=np.array(i).reshape(625, 784)
w=torch.Tensor(i)
o=torch.nn.Parameter(w)
rbm.W=o

indexes=np.where(i==0)[0]

#%% Binary Quantising
i=[]

for w in x:
    if w>=0:
        w=float(1)
    else:
        w=float(-1)
    i.append(w)
    
i=np.array(i).reshape(625, 784)
m=torch.Tensor(i)
o=torch.nn.Parameter(m)
rbm.W=o

#%% Ternary Quantising
delta=0.7/len(x) * sum(x)
q=[]
for w in x:
    if w > delta:
        w=float(1)
    elif w<-delta:
        w=float(-1)
    else:
        w=float(0)
    q.append(w)
    
q=np.array(q).reshape(625, 784)
w=torch.Tensor(q)
w=torch.nn.Parameter(w)
rbm.W=w
    
#%% Ternary Quantising

E=np.percentile(abs(x), 50)

q=[]

for w in x:
    if w > delta:
        w=float(E)
    elif w<-delta:
        w=float(-E)
    else:
        w=float(0)
    q.append(w)
    
q=np.array(q).reshape(625, 784)
w=torch.Tensor(q)
w=torch.nn.Parameter(w)
rbm.W=w

#%% K-means

weights=x.reshape(-1,1)

kmeans = KMeans(n_clusters=4, random_state=0).fit(weights)

labels=kmeans.labels_
centres=kmeans.cluster_centers_

kmeans_array=[]
for m in labels:
    m=centres[m]
    kmeans_array.append(m)
kmeans_array_np=np.array(kmeans_array).reshape(625,784)

w=torch.Tensor(kmeans_array_np)
w=torch.nn.Parameter(w)
rbm.W=w





















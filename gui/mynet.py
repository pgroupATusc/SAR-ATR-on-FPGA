import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv, GATConv, GraphConv, GravNetConv, SAGEConv  # noqa
import  torch_geometric

import torch
import torch.nn as nn
import torch.nn.functional as F



featurelength = 48
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.gconv1 = SAGEConv(1, featurelength)
        self.gconv2 = SAGEConv(featurelength, featurelength)
        self.gconv3 = SAGEConv(featurelength, featurelength)
        self.gconv4 = SAGEConv(featurelength, featurelength)
        
        self.fc_att_mean_1 = nn.Linear(featurelength, featurelength)
        self.fc_att_max_1 = nn.Linear(featurelength, featurelength)
        self.channelatten_1 = SAGEConv(featurelength, 1)
        self.fc_att_mean_2 = nn.Linear(featurelength, featurelength)
        self.fc_att_max_2 = nn.Linear(featurelength, featurelength)
        self.channelatten_2 = SAGEConv(featurelength, 1)
        self.fc_att_mean_3 = nn.Linear(featurelength, featurelength)
        self.fc_att_max_3 = nn.Linear(featurelength, featurelength)
        self.channelatten_3 = SAGEConv(featurelength, 1)
        self.fc_att_mean_4 = nn.Linear(featurelength, featurelength)
        self.fc_att_max_4 = nn.Linear(featurelength, featurelength)
        self.channelatten_4 = SAGEConv(featurelength, 1)
        
        self.pool = torch.nn.MaxPool2d(2, stride=2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(featurelength*16*16, 256)
        self.fc2 = nn.Linear(256, 10)
        

    def forward(self, x_image, edgearray, train):

        xi=x_image[None,:]
        
        xi = torch.reshape(xi, [1, 128 * 128])
             
        x = torch.transpose(xi, 0, 1)        
        x_0 = x

        x_1 = F.relu(self.gconv1(x_0, edgearray[0]))
        if train == True:
            x_1 = self.dropout(x_1)
        x_1 = torch.transpose(x_1, 0, 1)
        x_1 = torch.reshape(x_1, [1, featurelength, 128, 128])
        x_1 = self.pool(x_1)
        x_1 = torch.reshape(x_1, [featurelength, 64*64])
        x_1 = torch.transpose(x_1, 0, 1)
        
         # attention layer 1
        x_att_mean_1 = torch.mean(x_1, dim=0)
        x_att_mean_1 = x_att_mean_1[None,:]
        x_att_mean_1 = self.fc_att_mean_1(x_att_mean_1)
        x_att_max_1 = torch.max(x_1, dim=0)[0]
        x_att_max_1 = x_att_max_1[None,:]
        x_att_max_1 = self.fc_att_max_1(x_att_max_1)
        x_att_1 = F.sigmoid(x_att_max_1 + x_att_mean_1)
        x_1_1 = torch.mul(x_1, x_att_1)
        
        x_att_channel_1 = self.channelatten_1(x_1, edgearray[1])
        x_att_1 = F.sigmoid(x_att_channel_1)
        #x_att_1= x_att_1[:,None]
        x_1_2 = torch.mul(x_1, x_att_1)
        
        x_1 = x_1 + x_1_1 + x_1_2
        
        x_2 = F.relu(self.gconv2(x_1, edgearray[1]))
        if train == True:
            x_2 = self.dropout(x_2)
        x_2 = torch.transpose(x_2, 0, 1)
        x_2 = torch.reshape(x_2, [1, featurelength, 64, 64])
        x_2 = self.pool(x_2)
        x_2 = torch.reshape(x_2, [featurelength, 32*32])
        x_2 = torch.transpose(x_2, 0, 1)
        
        x_att_mean_2 = torch.mean(x_2, dim=0)
        x_att_mean_2 = x_att_mean_2[None,:]
        x_att_mean_2 = self.fc_att_mean_2(x_att_mean_2)
        x_att_max_2 = torch.max(x_2, dim=0)[0]
        x_att_max_2 = x_att_max_2[None,:]
        x_att_max_2 = self.fc_att_max_2(x_att_max_2)
        x_att_2 = F.sigmoid(x_att_max_2 + x_att_mean_2)
        x_2_1 = torch.mul(x_2, x_att_2)
        
        x_att_channel_2 = self.channelatten_2(x_2, edgearray[2])
        x_att_2 = F.sigmoid(x_att_channel_2)
        #x_att_2= x_att_2[:,None]
        x_2_2 = torch.mul(x_2, x_att_2)
        
        x_2 = x_2 + x_2_1 + x_2_2
        
        x_3 = F.relu(self.gconv3(x_2, edgearray[2]))
        if train == True:
            x_3 = self.dropout(x_3)
        x_3 = torch.transpose(x_3, 0, 1)
        x_3 = torch.reshape(x_3, [1, featurelength, 32, 32])
        x_3 = self.pool(x_3)
        x_3 = torch.reshape(x_3, [featurelength, 16*16])
        x_3 = torch.transpose(x_3, 0, 1)
        
        x_att_mean_3 = torch.mean(x_3, dim=0)
        x_att_mean_3 = x_att_mean_3[None,:]
        x_att_mean_3 = self.fc_att_mean_3(x_att_mean_3)
        x_att_max_3 = torch.max(x_3, dim=0)[0]
        x_att_max_3 = x_att_max_3[None,:]
        x_att_max_3 = self.fc_att_max_3(x_att_max_3)
        x_att_3 = F.sigmoid(x_att_max_3 + x_att_mean_3)
        x_3_1 = torch.mul(x_3, x_att_3)
        
        x_att_channel_3 = self.channelatten_3(x_3, edgearray[3])
        x_att_3 = F.sigmoid(x_att_channel_3)
        #x_att_3= x_att_3[:,None]
        x_3_2 = torch.mul(x_3, x_att_3)
        
        x_3 = x_3 + x_3_1 + x_3_2
        
        
        x_4 = F.relu(self.gconv4(x_3, edgearray[3]))
        if train == True:
            x_4 = self.dropout(x_4)
        x_4 = torch.transpose(x_4, 0, 1)
        x_4 = torch.reshape(x_4, [1, featurelength, 16, 16])
        #x_4 = self.pool(x_4)
        x_4 = torch.reshape(x_4, [1, featurelength* 16*16])


        x_5 = x_4
        
        x_5 = F.relu(self.fc1(x_5))
        #print(x_5)
        # if train == True:
        #     x_5 = self.dropout(x_5)
            
        x_5 = self.fc2(x_5)
        #print(x_5)
   
        return F.log_softmax(x_5, dim=1)



class Net2(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.gconv1 = SAGEConv(2, featurelength)
        self.gconv2 = SAGEConv(featurelength, featurelength)
        self.gconv3 = SAGEConv(featurelength, featurelength)
        self.gconv4 = SAGEConv(featurelength, featurelength)
        
        self.fc_att_mean_1 = nn.Linear(featurelength, featurelength)
        self.fc_att_max_1 = nn.Linear(featurelength, featurelength)
        self.channelatten_1 = SAGEConv(featurelength, 1)
        self.fc_att_mean_2 = nn.Linear(featurelength, featurelength)
        self.fc_att_max_2 = nn.Linear(featurelength, featurelength)
        self.channelatten_2 = SAGEConv(featurelength, 1)
        self.fc_att_mean_3 = nn.Linear(featurelength, featurelength)
        self.fc_att_max_3 = nn.Linear(featurelength, featurelength)
        self.channelatten_3 = SAGEConv(featurelength, 1)
        self.fc_att_mean_4 = nn.Linear(featurelength, featurelength)
        self.fc_att_max_4 = nn.Linear(featurelength, featurelength)
        self.channelatten_4 = SAGEConv(featurelength, 1)
        
        self.pool = torch.nn.MaxPool2d(2, stride=2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(featurelength*16*16, 256)
        self.fc2 = nn.Linear(256, 2)
        

    def forward(self, x_image, edgearray, train):

        xi=x_image[None,:]
        
        xi = torch.reshape(xi, [2, 128 * 128])
             
        x = torch.transpose(xi, 0, 1)        
        x_0 = x

        x_1 = F.relu(self.gconv1(x_0, edgearray[0]))
        if train == True:
            x_1 = self.dropout(x_1)
        x_1 = torch.transpose(x_1, 0, 1)
        x_1 = torch.reshape(x_1, [1, featurelength, 128, 128])
        x_1 = self.pool(x_1)
        x_1 = torch.reshape(x_1, [featurelength, 64*64])
        x_1 = torch.transpose(x_1, 0, 1)
        
         # attention layer 1
        x_att_mean_1 = torch.mean(x_1, dim=0)
        x_att_mean_1 = x_att_mean_1[None,:]
        x_att_mean_1 = self.fc_att_mean_1(x_att_mean_1)
        x_att_max_1 = torch.max(x_1, dim=0)[0]
        x_att_max_1 = x_att_max_1[None,:]
        x_att_max_1 = self.fc_att_max_1(x_att_max_1)
        x_att_1 = F.sigmoid(x_att_max_1 + x_att_mean_1)
        x_1_1 = torch.mul(x_1, x_att_1)
        
        x_att_channel_1 = self.channelatten_1(x_1, edgearray[1])
        x_att_1 = F.sigmoid(x_att_channel_1)
        #x_att_1= x_att_1[:,None]
        x_1_2 = torch.mul(x_1, x_att_1)
        
        x_1 = x_1 + x_1_1 + x_1_2
        
        x_2 = F.relu(self.gconv2(x_1, edgearray[1]))
        if train == True:
            x_2 = self.dropout(x_2)
        x_2 = torch.transpose(x_2, 0, 1)
        x_2 = torch.reshape(x_2, [1, featurelength, 64, 64])
        x_2 = self.pool(x_2)
        x_2 = torch.reshape(x_2, [featurelength, 32*32])
        x_2 = torch.transpose(x_2, 0, 1)
        
        x_att_mean_2 = torch.mean(x_2, dim=0)
        x_att_mean_2 = x_att_mean_2[None,:]
        x_att_mean_2 = self.fc_att_mean_2(x_att_mean_2)
        x_att_max_2 = torch.max(x_2, dim=0)[0]
        x_att_max_2 = x_att_max_2[None,:]
        x_att_max_2 = self.fc_att_max_2(x_att_max_2)
        x_att_2 = F.sigmoid(x_att_max_2 + x_att_mean_2)
        x_2_1 = torch.mul(x_2, x_att_2)
        
        x_att_channel_2 = self.channelatten_2(x_2, edgearray[2])
        x_att_2 = F.sigmoid(x_att_channel_2)
        #x_att_2= x_att_2[:,None]
        x_2_2 = torch.mul(x_2, x_att_2)
        
        x_2 = x_2 + x_2_1 + x_2_2
        
        x_3 = F.relu(self.gconv3(x_2, edgearray[2]))
        if train == True:
            x_3 = self.dropout(x_3)
        x_3 = torch.transpose(x_3, 0, 1)
        x_3 = torch.reshape(x_3, [1, featurelength, 32, 32])
        x_3 = self.pool(x_3)
        x_3 = torch.reshape(x_3, [featurelength, 16*16])
        x_3 = torch.transpose(x_3, 0, 1)
        
        x_att_mean_3 = torch.mean(x_3, dim=0)
        x_att_mean_3 = x_att_mean_3[None,:]
        x_att_mean_3 = self.fc_att_mean_3(x_att_mean_3)
        x_att_max_3 = torch.max(x_3, dim=0)[0]
        x_att_max_3 = x_att_max_3[None,:]
        x_att_max_3 = self.fc_att_max_3(x_att_max_3)
        x_att_3 = F.sigmoid(x_att_max_3 + x_att_mean_3)
        x_3_1 = torch.mul(x_3, x_att_3)
        
        x_att_channel_3 = self.channelatten_3(x_3, edgearray[3])
        x_att_3 = F.sigmoid(x_att_channel_3)
        #x_att_3= x_att_3[:,None]
        x_3_2 = torch.mul(x_3, x_att_3)
        
        x_3 = x_3 + x_3_1 + x_3_2
        
        
        x_4 = F.relu(self.gconv4(x_3, edgearray[3]))
        if train == True:
            x_4 = self.dropout(x_4)
        x_4 = torch.transpose(x_4, 0, 1)
        x_4 = torch.reshape(x_4, [1, featurelength, 16, 16])
        #x_4 = self.pool(x_4)
        x_4 = torch.reshape(x_4, [1, featurelength* 16*16])


        x_5 = x_4
        
        x_5 = F.relu(self.fc1(x_5))
        #print(x_5)
        if train == True:
            x_5 = self.dropout(x_5)
            
        x_5 = self.fc2(x_5)
        #print(x_5)
   
        return F.log_softmax(x_5, dim=1)
import dearpygui.dearpygui as dpg


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

import time

import scipy.io
train_data = scipy.io.loadmat('../Layer/hw18_v4/binversion/train.mat')
test_data = scipy.io.loadmat('../Layer/hw18_v4/binversion/test.mat')

import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import time





X_train, y_train = np.array(train_data['train_data']), np.array(train_data['train_label'])
X_test, y_test =  np.array(test_data['test_data']), np.array(test_data['test_label'])

X_train_image = X_train
X_test_image = X_test

y_train = y_train - 1
y_test = y_test - 1

device_cpu = torch.device("cpu")

X_train_image = torch.from_numpy(X_train_image).to(device_cpu)
X_train_image = X_train_image.type(torch.float32)
X_test_image = torch.from_numpy(X_test_image).to(device_cpu)
X_test_image = X_test_image.type(torch.float32)
train_label = torch.from_numpy(y_train).to(device_cpu) 
train_label = train_label.type(torch.int64)
test_label = torch.from_numpy(y_test).to(device_cpu)
test_label = test_label.type(torch.int64)

from mynet import Net

net_cpu = Net()
net_cpu.to(device_cpu)
net_cpu.load_state_dict(torch.load('./MSTAR.pt', map_location=torch.device('cpu') ))

edges = [torch_geometric.utils.grid(128,128)[0].to(device_cpu),
        torch_geometric.utils.grid(64,64)[0].to(device_cpu),
        torch_geometric.utils.grid(32,32)[0].to(device_cpu),
        torch_geometric.utils.grid(16,16)[0].to(device_cpu)
    ]


label_mstar = ['2S1', 'BRDM2', 'BTR70', 'T62', 'ZIL131', 'BMP2', 'BTR-60', 'D7',  'T72',  'ZSU234']

print(label_mstar[y_train[0].item()])

font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
org1 = (50, 50)
org2 = (50, 100)
org3 = (50, 150)

# fontScale
fontScale = 0.5
   
# Blue color in BGR
color1 = (255, 0, 0)
color2 = (255, 0, 0)

# Line thickness of 2 px
thickness = 2
   

def save_callback():
    print("Save Clicked")

dpg.create_context()
dpg.create_viewport(title='Custom Title', width=600, height=400)
dpg.setup_dearpygui()

with dpg.window(label="Example Window"):
    dpg.add_progress_bar(label = 'CPU', width = 500, height = 100)
    dpg.add_progress_bar(label = 'FPGA', width = 500, height = 100)

dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()


# for i in range(2700):
#     window_name = 'image'
#     iimage = cv2.resize(X_train[i], [640, 640])
#     iimage = cv2.putText(iimage, 'groundtruth label: ' + label_mstar[y_train[i].item()], org1, font, 
#                    fontScale, color1, thickness, cv2.LINE_AA)
    
#     start = time.time()
#     outputs = net_cpu(X_train_image[i], edges, False)
#     end = time.time()
#     totaltime = end - start
#     timestr = " %.0f " % (1.0/totaltime)

#     _, predicted = torch.max(outputs.data, 1)
#     predictedint = int(predicted[0])

#     iimage = cv2.putText(iimage, 'Predited label: ' + label_mstar[predictedint], org2, font, 
#                    fontScale, color2, thickness, cv2.LINE_AA)
#     iimage = cv2.putText(iimage, 'Frame Rate: ' + timestr + ' images/second', org3, font, 
#                    fontScale, color2, thickness, cv2.LINE_AA)
#     cv2.imshow(window_name, iimage)
#     cv2.waitKey(1)




# cv2.destroyAllWindows()
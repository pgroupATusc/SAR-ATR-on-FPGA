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


from subprocess import run
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sarfpga 
import re

X_train, y_train = np.array(train_data['train_data']), np.array(train_data['train_label'])
X_test, y_test =  np.array(test_data['test_data']), np.array(test_data['test_label'])

X_train_image = X_train
X_test_image = X_test

y_train = y_train - 1
y_test = y_test - 1


myacc = sarfpga.acceleratorArray("./combine_top.xclbin")
myacc.preparation()
myacc.loadweight()



label_mstar = ['2S1', 'BRDM2', 'BTR70', 'T62', 'ZIL131', 'BMP2', 'BTR-60', 'D7',  'T72',  'ZSU234']

print(label_mstar[y_train[0].item()])

font = cv2.FONT_HERSHEY_SIMPLEX

# org
org1 = (50, 50)  
org2 = (50, 100)
org3 = (50, 150)
org4 = (50, 200)

# fontScale
fontScale = 0.5
   
# Blue color in BGR
color1 = (255, 0, 0)
color2 = (255, 0, 0)

# Line thickness of 2 px
thickness = 2


data = run("sensors | grep POWER | head -n 1| grep -o -E '[0-9.]+' | head -n 1 ",capture_output=True,shell=True)
powerinfo = str(data.stdout)


orglabel = (20, 20)
scaletarget = [240, 160]

realimage = []

realimage.append(cv2.resize(cv2.imread("pic/1-2S1.png"), scaletarget))
realimage[0] = cv2.putText(realimage[0], '2S1', orglabel, font, 
                    fontScale, color2, thickness, cv2.LINE_AA)
realimage.append(cv2.resize(cv2.imread("pic/2-BRDM2.png"), scaletarget))
realimage[1] = cv2.putText(realimage[1], 'BRDM2', orglabel, font, 
                    fontScale, color2, thickness, cv2.LINE_AA)
realimage.append(cv2.resize(cv2.imread("pic/3-BTR70.png"), scaletarget))
realimage[2] = cv2.putText(realimage[2], 'BTR70', orglabel, font, 
                    fontScale, color2, thickness, cv2.LINE_AA)
realimage.append(cv2.resize(cv2.imread("pic/4-T62.png"), scaletarget))
realimage[3] = cv2.putText(realimage[3], 'T62', orglabel, font, 
                    fontScale, color2, thickness, cv2.LINE_AA)
realimage.append(cv2.resize(cv2.imread("pic/5-ZIL131.png"), scaletarget))
realimage[4] = cv2.putText(realimage[4], 'ZIL131', orglabel, font, 
                    fontScale, color2, thickness, cv2.LINE_AA)
realimage.append(cv2.resize(cv2.imread("pic/6-BMP2.png"), scaletarget))
realimage[5] = cv2.putText(realimage[5], 'BMP2', orglabel, font, 
                    fontScale, color2, thickness, cv2.LINE_AA)
realimage.append(cv2.resize(cv2.imread("pic/7-BTR-60.png"), scaletarget))
realimage[6] = cv2.putText(realimage[6], 'BTR-60', orglabel, font, 
                    fontScale, color2, thickness, cv2.LINE_AA)
realimage.append(cv2.resize(cv2.imread("pic/8-D7.png"), scaletarget))
realimage[7] = cv2.putText(realimage[7], 'D7', orglabel, font, 
                    fontScale, color2, thickness, cv2.LINE_AA)
realimage.append(cv2.resize(cv2.imread("pic/9-T72.png"), scaletarget))
realimage[8] = cv2.putText(realimage[8], 'T72', orglabel, font, 
                    fontScale, color2, thickness, cv2.LINE_AA)
realimage.append(cv2.resize(cv2.imread("pic/10-ZSU234.png"), scaletarget))
realimage[9] = cv2.putText(realimage[9], 'ZSU234', orglabel, font, 
                    fontScale, color2, thickness, cv2.LINE_AA)

for epoch in range(100000):
    timestr = 0
    sumtime = 0
    Latency = 0


    for i in range(0, 2700, 2):
        window_name = 'SAR ATR on FPGA'
        iimage = cv2.resize(X_train[i], [640, 640])
        iimage = cv2.putText(iimage, 'groundtruth label: ' + label_mstar[y_train[i].item()], org1, font, 
                    fontScale, color1, thickness, cv2.LINE_AA)
        imagefilename = "../Layer/data/input_images/image_%d.bin" % i
        labelfilename = "../Layer/data/input_images_label/label_%d.bin" % i
        myacc.loadinput(imagefilename)

        start = time.time()
        myacc.inference()
        end = time.time()
        totaltime = end - start
        sumtime += totaltime

        if i % 100 == 0:
            timestr = " %.0f " % (100.0/sumtime)
            Latency = sumtime/100.0 * 1000
            sumtime = 0
           

        predictlabel = myacc.checkresult(labelfilename)

        if i % 500 == 0:
            data = run("sensors | grep POWER | head -n 1| grep -o -E '[0-9.]+' | head -n 1",capture_output=True,shell=True)
            powerinfo = str(data.stdout)
            powerinfo = re.findall(r'\b\d+\b',powerinfo)
            powerinfo = powerinfo[0]
        
        iimage = cv2.putText(iimage, 'Epoch %d, image %d' % (epoch, i), (400, 50), font, 
                    fontScale, color2, thickness, cv2.LINE_AA)
        iimage = cv2.putText(iimage, 'Latency %.3f ms' % Latency, (400, 100), font, 
                    fontScale, color2, thickness, cv2.LINE_AA)

        iimage = cv2.putText(iimage, 'Predited label: ' + label_mstar[y_train[i].item()], org2, font, 
                    fontScale, color2, thickness, cv2.LINE_AA)
        iimage = cv2.putText(iimage, 'Frame Rate: ' + timestr + ' images/second', org3, font, 
                    fontScale, color2, thickness, cv2.LINE_AA)
        iimage = cv2.putText(iimage, 'Power: ' + powerinfo + ' W', org4, font, 
                    fontScale, color2, thickness, cv2.LINE_AA)
        
        cv2.imshow("Real object", realimage[y_train[i].item()])
        cv2.imshow(window_name, iimage)
        cv2.waitKey(1)




# cv2.destroyAllWindows()
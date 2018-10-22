import torch
import torch.nn as nn
import torch.nn.functional as Funct
import sys
import segData
from PIL import Image
from torchvision.transforms import ToPILImage
import torchvision.transforms as t

from Unet import Unet
from Unet import ConvBlock
from Unet import First_ConvBlock
from Unet import Get_Sample
from Unet import Up_Block

import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

def iou_r(outputs, label):
    outputs = (outputs > 0.5).cpu().type(torch.cuda.FloatTensor)
    label = label
    intersection = (outputs * label).type(torch.cuda.FloatTensor)
    ex = 0
    result = []
    for i in range(batch_size):
        k = torch.sum(label[i][0])
        ret = (torch.sum(intersection[i][0])/(torch.sum(outputs[i][0])+k-torch.sum(intersection[i][0]) + 1e-7))
        result.append(ret.data)
        
    ret = sum(result)
    return ret

def iou_b(outputs, label):
    outputs = (outputs < 0.5).cpu().type(torch.cuda.FloatTensor)
    label = (label == 0).type(torch.cuda.FloatTensor)
    intersection = (outputs * label).type(torch.cuda.FloatTensor)

    result = []
    for i in range(batch_size):
        ret = (torch.sum(intersection[i][0])/(torch.sum(outputs[i][0])+torch.sum(label[i][0])-torch.sum(intersection[i][0]) + 1e-7))
        result.append(ret.data)
    
    ret = sum(result)
    return ret

if __name__ == "__main__":
    torch.cuda.init()
    batch_size = 8
    test_dataset = segData.DataS("val")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)   
    model = torch.load('1_epoch_model.pt')
    #model = torch.nn.DataParallel(model)
    
    result = 0.0
    result2 = 0.0
    result3 = 0.0
    #result4 = 0.0
    ex_r = 0
    for img, label in test_loader:
        img = Variable(img.cuda())
        label = label.type(torch.cuda.FloatTensor)
        output = model(img)
        print(output)
        mi = iou_r(output, label)        
        mi2 = iou_b(output, label)
        
        result += mi
        result2 += mi2
        
        torch.cuda.empty_cache()
        
    result /= (len(test_dataset))
    result2 /= (len(test_dataset))
    print('%f'%((result.data*100)))
    print('%f'%((result2.data*100)))
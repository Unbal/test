# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

# Training settings
batch_size = 64

# MNIST Dataset
train_dataset = datasets.MNIST(root='./data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./data/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

class trans_block(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(trans_block, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size =1, stride=1)
        self.bn1 = nn.BatchNorm2d(inplanes)        
        self.conv2 = nn.ConvTranspose2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(inplanes)
        self.conv3 = nn.Conv2d(inplanes, outplanes, kernel_size =1, stride=1)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        return out
    
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) #여기서 말하는 channel은 걍 필터의 갯수정도라고 생각해두면 될것 같다.
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) #kernel_size == 필터의 크기를 말하는것 같다.
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10) #원래 구하는방법이 있긴하지만 그냥 아무숫자나 적어두고 워닝뜨면 그때 고쳐라!
        self.up = trans_block(20,64)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2 = nn.ConvTranspose2d(20,64,kernel_size = 4,stride =2 ,padding = 1)
    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        print(x.shape)
        y = self.up1(x)
        z = self.up2(x)
        x = self.up(x)
        print('\n\nTrans_block = ',x.shape)
        print('Upsample = ', y.shape)
        print('Conv2d = ', z.shape,'\n\n')
        x = x.view(in_size, -1)  # flatten the tensor
        print(x.shape)
        x = self.fc(x)
        return F.log_softmax(x)


model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        print(output.shape)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, 2):
    train(epoch)
    test()

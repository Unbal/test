import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from collections import OrderedDict

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="3"

class First_ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(First_ConvBlock, self).__init__()
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batch_norm(x)
        x = self.conv2(x)
        return x

class DeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeConvBlock, self).__init__()
        self.batch_norm_1 = nn.BatchNorm2d(in_channels)
        self.batch_norm_2 = nn.BatchNorm2d(out_channels)
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.batch_norm_1(x)
        x = self.deconv1(F.relu(x))
        x = self.batch_norm_2(x)
        x = self.deconv2(F.relu(x))
        return x
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.batch_norm_1 = nn.BatchNorm2d(in_channels)
        self.batch_norm_2 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.batch_norm_1(x))
        x = self.conv1(x)
        x = F.relu(self.batch_norm_2(x))
        x = self.conv2(x)
        return x
    
class Up_Block(nn.Module):
    def __init__(self, inplanes):
        super(Up_Block, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size =1, stride=1)
        self.deconv = nn.ConvTranspose2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn = nn.BatchNorm2d(inplanes)  
        
    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = F.relu(self.bn(self.deconv(out)))
        out = F.relu(self.bn(self.conv(out)))
        return out
    
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        
        self.encoder_1 = First_ConvBlock(3,64)
        self.upsample1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1),
            nn.BatchNorm2d(64))
        
        self.encoder_2 = ConvBlock(64,128)
        self.upsample2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128))
        self.encoder_3 = ConvBlock(128,256)
        self.upsample3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1),
            nn.BatchNorm2d(256))
        self.encoder_4 = ConvBlock(256,512)
        self.upsample4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1),
            nn.BatchNorm2d(512))
        self.encoder_5 = ConvBlock(512,1024)

        self.decoder_1 = DeConvBlock(1536,512)
        self.downsample = nn.Sequential(
            nn.Conv2d(1536, 512, kernel_size=1),
            nn.BatchNorm2d(512))
        self.decoder_2 = DeConvBlock(768,256)
        self.downsample2 = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=1),
            nn.BatchNorm2d(256))
        self.decoder_3 = DeConvBlock(384,128)
        self.downsample3 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=1),
            nn.BatchNorm2d(128))
        self.decoder_4 = DeConvBlock(192,64)
        self.downsample4 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1),
           nn.BatchNorm2d(64))
        self.final = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )  

        self.maxpool = nn.MaxPool2d((2, 2), stride=(2, 2), return_indices=True)       
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.ups = Up_Block(1024)
        self.up2 = Up_Block(512)
        self.up3 = Up_Block(256)
        self.up4 = Up_Block(128)
        
        self.unpool = nn.MaxUnpool2d(2, stride=2)  # get masks
        
        #self.classifier = nn.Softmax()

    def forward(self, x):       
        
        size_1 = x.size()
        enc1 = self.encoder_1(x)
        temp = self.upsample1(x)
        enc1+=temp
        x,indices1 = self.maxpool(enc1)

        size_2 = x.size()
        enc2 = self.encoder_2(x)
        temp = self.upsample2(x)
        enc2+=temp
        x,indices2 = self.maxpool(enc2)  
        
        size_3 = x.size()
        enc3 = self.encoder_3(x)
        temp = self.upsample3(x)
        enc3+=temp
        x,indices3 = self.maxpool(enc3)  
        
        size_4 = x.size()
        enc4 = self.encoder_4(x)
        temp = self.upsample4(x)
        enc4+=temp
        x,indices4 = self.maxpool(enc4)  
        
        center = self.encoder_5(x)
        
        temp = torch.cat([enc4, self.up1(center)], 1)
        dec1 = self.decoder_1(temp) #여기서 upsample안하고 maxunpool로하려니깐 center와 indices4의 채널갯수가 맞지않음
        dec1 += self.downsample(temp)
        temp = torch.cat([enc3, self.up1(dec1)], 1)
        dec2 = self.decoder_2(temp)
        dec2 += self.downsample2(temp)

        temp = torch.cat([enc2, self.up1(dec2)], 1)
        dec3 = self.decoder_3(temp)
        dec3 += self.downsample3(temp)
        
        temp = torch.cat([enc1, self.up1(dec3)], 1)
        dec4 = self.decoder_4(temp)
        dec4 += self.downsample4(temp)
        
        final = self.final(dec4)
        
        #x = self.classifier(x)
        return final

if __name__ == "__main__":
    from segData import DataS
    import torch.optim as optim
    from torch.autograd import Variable
    from torch.utils.data import DataLoader
    import torchvision.transforms as t
    
    train_dataset = DataS('train')
    train_loader= DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
    test_dataset = DataS('val')
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=8)
    
    model = Unet()
    if torch.cuda.is_available(): 
        model = model.cuda()
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    model = model.train()
    max_epochs = 5
    for i in range(max_epochs):
        running_loss = 0.0
        for j, data in enumerate(train_loader, 0):
            inputs, labels = data
            if torch.cuda.is_available(): 
                inputs = inputs.cuda()
                labels = labels.cuda()
            inputs = Variable(inputs)
            labels = Variable(labels)                       
            model.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data
            if j % 1000 == 999:
                print('[epoch: %d, j: %5d] average loss: %.8f' % (i + 1, j + 1, running_loss / 1000))
                running_loss = 0.0
    torch.save(model, 'model2_before.pt')

import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.ao.quantization import QuantStub, DeQuantStub

class CNN_simple(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, quantize=False):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1)  
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        
        self.pool = nn.AdaptiveAvgPool2d(1) 
        
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.quantize = quantize
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        if self.quantize:
            x = self.quant(x)
        x = F.relu(self.conv1(x))         
        x = F.relu(self.conv2(x))         
        x = self.pool(x)                  
        x = torch.flatten(x, 1)           
        x = F.relu(self.fc1(x))           
        x = self.fc2(x)                   
        if self.quantize:
            x = self.dequant(x)
        return F.log_softmax(x, dim=1)
    


    

class CNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, quantize=False):
        super(CNN, self).__init__()
        self.quantize = quantize
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(0.4)
        )

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        if self.quantize:
            x = self.quant(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if self.quantize:
            x = self.dequant(x)
        return F.log_softmax(x, dim=1)


    def fuse_model(self, is_qat=False):
        fuse_modules = torch.ao.quantization.fuse_modules_qat if is_qat else torch.ao.quantization.fuse_modules
        fuse_modules(self.features, ['0', '1', '2'], inplace=True)  
        fuse_modules(self.features, ['3', '4', '5'], inplace=True)  
        fuse_modules(self.features, ['8', '9', '10'], inplace=True)  
        fuse_modules(self.features, ['11', '12', '13'], inplace=True)  
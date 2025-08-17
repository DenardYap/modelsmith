import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.ao.quantization import QuantStub, DeQuantStub
import brevitas.nn as qnn
from brevitas.quant import Int32Bias
from torch.ao.quantization import QuantStub, DeQuantStub

import brevitas.nn as qnn
from brevitas.quant import Int32Bias

"""
TODO:
Quantize CNN and see how well they are doing 
- Model size
- Speed (both training and inference)
- Energy Consumptions 
- 8-bit, 4-bit, 2-bit, and 1-bit
"""

# Quantized CNN - 8 bit version 
class QCNN_simple_8bit(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(QCNN_simple_8bit, self).__init__()
        self.conv1 = qnn.QuantConv2d(input_channels, 32, 3, bias=True, weight_bit_width=8)
        self.conv2 = qnn.QuantConv2d(32, 64, 3, bias=True, weight_bit_width=8)
        self.relu1 = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)
        self.relu2 = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)
        self.relu3 = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)
        self.dropout1 = nn.Dropout(0.25)
        
        self.pool = nn.AdaptiveAvgPool2d(1) 
        
        self.fc1   = qnn.QuantLinear(64, 128, bias=True, weight_bit_width=8)
        self.fc2   = qnn.QuantLinear(128, num_classes, bias=True, weight_bit_width=8)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        print(torch.max(x), torch.min(x))
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)                  
        x = torch.flatten(x, 1)           
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)                      
        return F.log_softmax(x, dim=1)
    

# Quantized CNN - 4 bit version 
class QCNN_simple_4bit(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(QCNN_simple_4bit, self).__init__()
        self.quant_inp = qnn.QuantIdentity(bit_width=4, return_quant_tensor=True)
        self.conv1 = qnn.QuantConv2d(input_channels, 32, 3, bias=True, weight_bit_width=4, bias_quant=Int32Bias)
        self.conv2 = qnn.QuantConv2d(32, 64, 3, bias=True, weight_bit_width=4, bias_quant=Int32Bias)
        self.relu1 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
        self.relu2 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
        self.relu3 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
        self.dropout1 = nn.Dropout(0.25)
        self.pool = nn.AdaptiveAvgPool2d(1) 
        self.fc1   = qnn.QuantLinear(64, 128, bias=True, weight_bit_width=4, bias_quant=Int32Bias)
        self.fc2   = qnn.QuantLinear(128, num_classes, bias=True, weight_bit_width=4, bias_quant=Int32Bias)

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)                  
        x = torch.flatten(x, 1)           
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)                      
        return F.log_softmax(x, dim=1)
    
# Quantized CNN - 2 bit version 
class QCNN_simple_2bit(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(QCNN_simple_2bit, self).__init__()
        self.conv1 = qnn.QuantConv2d(input_channels, 32, 3, bias=True, weight_bit_width=2)
        self.conv2 = qnn.QuantConv2d(32, 64, 3, bias=True, weight_bit_width=2)
        self.relu1 = qnn.QuantReLU(bit_width=2, return_quant_tensor=True)
        self.relu2 = qnn.QuantReLU(bit_width=2, return_quant_tensor=True)
        self.relu3 = qnn.QuantReLU(bit_width=2, return_quant_tensor=True)
        self.dropout1 = nn.Dropout(0.25)
        
        self.pool = nn.AdaptiveAvgPool2d(1) 
        
        self.fc1   = qnn.QuantLinear(64, 128, bias=True, weight_bit_width=2)
        self.fc2   = qnn.QuantLinear(128, num_classes, bias=True, weight_bit_width=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)                  
        x = torch.flatten(x, 1)           
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)                      
        return F.log_softmax(x, dim=1)
    
# Quantized CNN - 1 bit version 
class QCNN_simple_1bit(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(QCNN_simple_1bit, self).__init__()
        self.conv1 = qnn.QuantConv2d(input_channels, 32, 3, bias=True, weight_bit_width=1)
        self.conv2 = qnn.QuantConv2d(32, 64, 3, bias=True, weight_bit_width=1)
        self.relu1 = qnn.QuantReLU(bit_width=1, return_quant_tensor=True)
        self.relu2 = qnn.QuantReLU(bit_width=1, return_quant_tensor=True)
        self.relu3 = qnn.QuantReLU(bit_width=1, return_quant_tensor=True)
        self.dropout1 = nn.Dropout(0.25)
        
        self.pool = nn.AdaptiveAvgPool2d(1) 
        
        self.fc1   = qnn.QuantLinear(64, 128, bias=True, weight_bit_width=1)
        self.fc2   = qnn.QuantLinear(128, num_classes, bias=True, weight_bit_width=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)                  
        x = torch.flatten(x, 1)           
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)                      
        return F.log_softmax(x, dim=1)

class CNN_simple(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, quantize=False):
        super(CNN_simple, self).__init__()
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
    

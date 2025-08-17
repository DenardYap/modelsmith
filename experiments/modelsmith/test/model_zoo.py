import os
import sys

# Prefer local src over any installed package named "modelsmith"
_current_dir = os.path.dirname(__file__)
_src_path = os.path.abspath(os.path.join(_current_dir, "..", "src"))
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

# Write a simple linear layer for classifying MNIST images

import torch
import torch.nn as nn
import torch.nn.functional as F
from modelsmith import BinLinear

# Standard PyTorch LeNet5
class StandardLeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Input: 32x32
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # 32x32 -> 32x32
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 16x16 -> 12x12
        # After max pooling: 16 channels of 6x6 feature maps = 16 * 6 * 6 = 576
        self.fc1 = nn.Linear(576, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # 32x32 -> 16x16
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 16x16 -> 6x6
        x = x.view(x.size(0), -1)  # Flatten: 16 * 6 * 6 = 576
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# Standard PyTorch ResNet18
class StandardBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class StandardResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * StandardBasicBlock.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(StandardBasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * StandardBasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)

# Standard PyTorch ViT
class StandardViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=1000, dim=768, depth=12, heads=12):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        self.transformer = nn.ModuleList([])
        for _ in range(depth):
            self.transformer.append(nn.ModuleList([
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),  # Self-attention projection
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 4),  # MLP
                nn.Linear(dim * 4, dim)  # MLP
            ]))

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # Split into patches
        p = self.patch_size
        x = img.unfold(2, p, p).unfold(3, p, p).transpose(1, 3).transpose(2, 3)
        x = x.reshape(x.size(0), -1, 3 * p * p)
        
        # Patch embedding
        x = self.patch_to_embedding(x)
        
        # Add cls token and position embedding
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding

        # Transformer
        for norm1, attn, norm2, ff1, ff2 in self.transformer:
            x = x + attn(norm1(x))
            x = x + ff2(F.relu(ff1(norm2(x))))

        # Classification head
        x = x[:, 0]
        return self.mlp_head(x)

class BinaryConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.alpha = nn.Parameter(self.weight.abs().mean())
        
    def forward(self, x):
        # Binarize weights
        weight_bin = torch.sign(self.weight)
        # Apply straight-through estimator
        weight_bin = weight_bin.detach() + self.weight - self.weight.detach()
        
        # Normalize and binarize input
        x = x - x.mean(dim=(2, 3), keepdim=True)
        beta = x.abs().mean()
        x_bin = torch.sign(x)
        x_bin = x_bin.detach() + x - x.detach()
        
        # For now, use standard convolution since we haven't implemented binary convolution
        # TODO: Implement binary convolution using binary_matmul
        out = F.conv2d(x_bin, weight_bin, None, self.stride, self.padding)
        return out * self.alpha * beta

class BinaryLeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Input: 32x32
        self.conv1 = BinaryConv2d(1, 6, kernel_size=5, padding=2)  # 32x32 -> 32x32
        self.conv2 = BinaryConv2d(6, 16, kernel_size=5)  # 16x16 -> 12x12
        # After max pooling: 16 channels of 6x6 feature maps = 16 * 6 * 6 = 576
        self.fc1 = BinLinear(576, 120)
        self.fc2 = BinLinear(120, 84)
        self.fc3 = BinLinear(84, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # 32x32 -> 16x16
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 16x16 -> 6x6
        x = x.view(x.size(0), -1)  # Flatten: 16 * 6 * 6 = 576
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class BinaryBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = BinaryConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BinaryConv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                BinaryConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BinaryResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64
        
        self.conv1 = BinaryConv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = BinLinear(512 * BinaryBasicBlock.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BinaryBasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BinaryBasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)

class BinaryViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=1000, dim=768, depth=12, heads=12):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = BinLinear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        self.transformer = nn.ModuleList([])
        for _ in range(depth):
            self.transformer.append(nn.ModuleList([
                nn.LayerNorm(dim),
                BinLinear(dim, dim),  # Self-attention projection
                nn.LayerNorm(dim),
                BinLinear(dim, dim * 4),  # MLP
                BinLinear(dim * 4, dim)  # MLP
            ]))

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            BinLinear(dim, num_classes)
        )

    def forward(self, img):
        # Split into patches
        p = self.patch_size
        x = img.unfold(2, p, p).unfold(3, p, p).transpose(1, 3).transpose(2, 3)
        x = x.reshape(x.size(0), -1, 3 * p * p)
        
        # Patch embedding
        x = self.patch_to_embedding(x)
        
        # Add cls token and position embedding
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding

        # Transformer
        for norm1, attn, norm2, ff1, ff2 in self.transformer:
            x = x + attn(norm1(x))
            x = x + ff2(F.relu(ff1(norm2(x))))

        # Classification head
        x = x[:, 0]
        return self.mlp_head(x)


class SimpleLinear(nn.Module):
    def __init__(self, input_dim=28*28, num_classes=10):
        super(SimpleLinear, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return F.log_softmax(out, dim=1)
    

class BinaryLinear(nn.Module):
    def __init__(self, input_dim=28*28, num_classes=10):
        super(BinaryLinear, self).__init__()
        # MNIST images are 28x28 = 784 pixels
        self.linear = BinLinear(input_dim, num_classes)  # 10 classes for digits 0-9

    def forward(self, x):
        # Flatten the input images: (batch_size, 1, 28, 28) -> (batch_size, 784)
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return F.log_softmax(out, dim=1)

import torch
import torch.nn as nn
import torch.nn.functional as F
from modelsmith import BinLinear

class BinaryMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[512, 256, 128], num_classes=10, quantize_first=False, quantize_last=False):
        super().__init__()
        self.input_dim = input_dim
        self.flatten = nn.Flatten()
        
        # Build layers dynamically
        layers = []
        prev_dim = input_dim

        if len(hidden_dims) > 0:
            first_hidden_dim = hidden_dims[0]
            # First layer: optionally keep full precision
            if quantize_first:
                layers.append(BinLinear(prev_dim, first_hidden_dim))
            else:
                layers.append(nn.Linear(prev_dim, first_hidden_dim))
            layers.append(nn.BatchNorm1d(first_hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = first_hidden_dim

            # Middle hidden layers: keep binarized
            for hidden_dim in hidden_dims[1:]:
                layers.append(BinLinear(prev_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim

        # Output layer: optionally keep full precision
        if quantize_last:
            layers.append(BinLinear(prev_dim, num_classes))
        else:
            layers.append(nn.Linear(prev_dim, num_classes))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # Ensure input is flattened to match input_dim
        x = self.flatten(x)
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {x.shape[1]}")
        x = self.layers(x)
        return F.log_softmax(x, dim=1)

class StandardMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[512, 256, 128], num_classes=10):
        super().__init__()
        self.input_dim = input_dim
        self.flatten = nn.Flatten()
        
        # Build layers dynamically
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # Ensure input is flattened to match input_dim
        x = self.flatten(x)
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {x.shape[1]}")
        x = self.layers(x)
        return F.log_softmax(x, dim=1)

# Different MLP configurations
def get_tiny_mlp(binary=True, num_classes=10, input_dim=784):
    """2-layer MLP with small hidden dimension"""
    ModelClass = BinaryMLP if binary else StandardMLP
    return ModelClass(input_dim=input_dim, hidden_dims=[128, 64], num_classes=num_classes)

def get_small_mlp(binary=True, num_classes=10, input_dim=784):
    """3-layer MLP with medium hidden dimensions"""
    ModelClass = BinaryMLP if binary else StandardMLP
    return ModelClass(input_dim=input_dim, hidden_dims=[256, 128, 64], num_classes=num_classes)

def get_medium_mlp(binary=True, num_classes=10, input_dim=784):
    """4-layer MLP with larger hidden dimensions"""
    ModelClass = BinaryMLP if binary else StandardMLP
    return ModelClass(input_dim=input_dim, hidden_dims=[512, 256, 128, 64], num_classes=num_classes)

def get_large_mlp(binary=True, num_classes=10, input_dim=784):
    """5-layer MLP with large hidden dimensions"""
    ModelClass = BinaryMLP if binary else StandardMLP
    return ModelClass(input_dim=input_dim, hidden_dims=[1024, 512, 256, 128, 64], num_classes=num_classes)

def get_wide_mlp(binary=True, num_classes=10, input_dim=784):
    """3-layer MLP with very wide hidden layers"""
    ModelClass = BinaryMLP if binary else StandardMLP
    return ModelClass(input_dim=input_dim, hidden_dims=[2048, 1024, 512], num_classes=num_classes)

def get_deep_mlp(binary=True, num_classes=10, input_dim=784):
    """8-layer MLP with medium-sized hidden layers"""
    ModelClass = BinaryMLP if binary else StandardMLP
    return ModelClass(input_dim=input_dim, hidden_dims=[512, 512, 256, 256, 128, 128, 64, 64], num_classes=num_classes)

if __name__ == "__main__":
    # Test all configurations
    x = torch.randn(32, 1, 28, 28)  # MNIST-sized input batch
    models = [
        ("Tiny MLP", get_tiny_mlp()),
        ("Small MLP", get_small_mlp()),
        ("Medium MLP", get_medium_mlp()),
        ("Large MLP", get_large_mlp()),
        ("Wide MLP", get_wide_mlp()),
        ("Deep MLP", get_deep_mlp()),
    ]
    
    for name, model in models:
        y = model(x)
        print(f"\n{name}:")
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {y.shape}")
        print(f"Parameter count: {sum(p.numel() for p in model.parameters()):,}")

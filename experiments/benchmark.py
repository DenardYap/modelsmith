from model_zoo import CNN
import torch.quantization as quant
from train import test, prepare_qat_model, convert_to_quantized_model
import torch 
import time
import os
from torchvision import datasets, transforms

# float32 
# QAT -> float32 -> int8 -> float32 (noised version)
# int4, int2 -> 1.58 bit (3 bits) 2^1.58 ~ 3 (Microsoft's paper)
# 0 - 255
print(torch.backends.quantized.supported_engines)

FMNIST_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))  
])

FMNIST_test_dataset = datasets.FashionMNIST(
    root='./data',
    train=False, 
    download=True,
    transform=FMNIST_transform
)

test_loader = torch.utils.data.DataLoader(
    FMNIST_test_dataset,
    batch_size=1000,
    shuffle=False
)

# def load_quantized_model(model_path, input_channels=1, num_classes=10):
#     model = CNN(input_channels=input_channels, num_classes=num_classes, quantize=True)
#     model.qconfig = quant.get_default_qat_qconfig('x86')  # Ensure this matches training
#     model = prepare_qat_model(model)   # attaches qconfig etc.
#     # model = torch.ao.quantization.convert(model, inplace=False)
#     torch.ao.quantization.convert(model, inplace=True)  # Converts for inference
#     model.eval()
#     model.load_state_dict(torch.load(model_path))
#     return model

torch.backends.quantized.engine = 'qnnpack'
def load_quantized_model(model_path, input_channels=1, num_classes=10):
    model = CNN(input_channels=input_channels, num_classes=num_classes, quantize=True)
    model.fuse_model(is_qat=True)
    model = prepare_qat_model(model)   
    model = torch.ao.quantization.convert(model, inplace=False)
    model.load_state_dict(torch.load(model_path))
    # torch.ao.quantization.convert(model, inplace=True)  # Converts for inference
    model.eval()
    return model

best_model_path = "model_weights/FASHION_MNIST/0_5.pth"
best_quantized_model_path =  "model_weights/FASHION_MNIST/quantized_converted/0_3.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

quantized_model = load_quantized_model(best_quantized_model_path)
quantized_model.to(device)


# load normal model
model = CNN(1, 10)
model.load_state_dict(torch.load(best_model_path))
model.to(device)
model.eval()
# Fuse to see if it gets faster?
# model.fuse_model()

def get_acc(model, data_loader):
    accuracy, loss = test(model, device, data_loader)
    print(f"Model Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")
    

def get_speed(model, data_loader):

    sample_input = next(iter(data_loader))[0][:1].to(device)  # One image
    N = 1000

    # Warmup
    for _ in range(100):
        _ = model(sample_input)

    # Timing
    start = time.time()
    for _ in range(N):
        _ = model(sample_input)
    end = time.time()

    avg_latency = (end - start) / N * 1000  # in ms
    print(f"Avg Inference Latency: {avg_latency:.2f} ms")


def get_size(model_path):
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / 1e6
    print(f"Model Size: {size_mb:.2f} MB")


get_acc(model, test_loader)
get_acc(quantized_model, test_loader)
get_speed(model, test_loader)
get_speed(quantized_model, test_loader)
get_size(best_model_path)
get_size(best_quantized_model_path)

"""

Non-quantized model: 

Without fuse: 
Avg Inference Latency: 0.70 ms
Avg Inference Latency: 0.70 ms
Avg Inference Latency: 0.82 ms

With fuse:
Avg Inference Latency: 0.63 ms
Avg Inference Latency: 0.61 ms
Avg Inference Latency: 0.66 ms
"""
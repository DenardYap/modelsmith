from model_zoo import *
import torch.quantization as quant
from train import test, prepare_qat_model, convert_to_quantized_model
import torch 
import time
import os
from torchvision import datasets, transforms
import argparse

def get_channels_and_classes(dataset_name):

    if dataset_name == "MNIST":
        return 1, 10
    elif dataset_name == "FMNIST":
        return 1, 10
    elif dataset_name == "CIFAR10":
        return 3, 10
    else:
        raise Exception("Dataset not recognized")

def load_model(name, dataset_name):
    input_channels, num_classes = get_channels_and_classes(dataset_name)

    if name == "CNN_simple":
        return CNN_simple(input_channels=input_channels, num_classes=num_classes)
    elif name == "QCNN_simple_4bit":
        return QCNN_simple_4bit(input_channels=input_channels, num_classes=num_classes)
    else:
        raise Exception("Model name not recognized")
    
def get_dataset(name):
    if name == "MNIST":

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST(
                    root='./data',
                    train=False,
                    download=True, 
                    transform=transform
                    )

    elif name == "FMNIST":
                
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))  
        ])

        dataset = datasets.FashionMNIST(
            root='./data',
            train=False, 
            download=True,
            transform=transform
        )

    elif name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 mean and std
        ])
        dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        raise Exception("Dataset name not recognized")
    
    return dataset, transform

def main():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Process some command line arguments.')

    # Add command line arguments
    parser.add_argument('--dataset_name', type=str, required=True, help='dataset name')
    parser.add_argument('--model_name', type=str, required=True, help='model name')

    # Parse arguments
    args = parser.parse_args()

    dataset_name = args.dataset_name
    model_name = args.model_name

    test_loader = torch.utils.data.DataLoader(
        dataset_name,
        batch_size=1000,
        shuffle=False
    )
    model = load_model(model_name)

if __name__ == '__main__':
    main()



best_model_path = "model_weights/FASHION_MNIST/0_5.pth"
best_quantized_model_path =  "model_weights/FASHION_MNIST/quantized_converted/0_3.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
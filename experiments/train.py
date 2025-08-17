import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from model_zoo import CNN
import os
import torch.quantization as quant

kf = KFold(n_splits=2, shuffle=True, random_state=42)

torch.backends.quantized.engine = 'qnnpack'
def prepare_qat_model(model):
    model.train()
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
    torch.ao.quantization.prepare_qat(model, inplace=True)
    # quant.prepare_qat(model, inplace=True)
    
    return model

def convert_to_quantized_model(model):
    model.eval()  
    quantized_model = quant.convert(model)
    return quantized_model


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Test function
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    model.to(device)
    # with torch.no_grad():
    with torch.inference_mode():  # Better than torch.no_grad() for inference
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy, test_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(dataset, model_weight_path, use_quantization=False, input_channels=1, num_classes=10, num_epochs=5):
    if use_quantization:
        print("Using Quantization-Aware-Training")
        model_weight_path += "/quantized"
        model_weight_path_converted = model_weight_path + "_converted" 

        os.makedirs(model_weight_path_converted, exist_ok=True)
    os.makedirs(model_weight_path, exist_ok=True)

    best_accuracy = 0.0
    best_model = None
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\nTraining fold {fold + 1}/{kf.get_n_splits()}...")
        
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=1000, shuffle=False)
        
        model =  CNN(input_channels=input_channels, num_classes=num_classes, quantize=use_quantization).to(device)
        if use_quantization:
            print("Fusing module for quantization...")
            model.fuse_model(is_qat= use_quantization)
            model = prepare_qat_model(model)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(1, num_epochs + 1):
            train(model, device, train_loader, optimizer, epoch)
            accuracy, val_loss = test(model, device, val_loader)
            print(f"Validation Accuracy for fold {fold + 1} Epoch {epoch}: {accuracy:.2f}%")
            if use_quantization:
                quantized_model = torch.ao.quantization.convert(model.eval(), inplace=False)
                accuracy, val_loss = test(quantized_model, device, val_loader)
                print(f"Validation Accuracy for quantized model fold {fold + 1} Epoch {epoch}: {accuracy:.2f}%")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                
                if use_quantization:
                    best_quantized_model = quantized_model.state_dict()
                    save_path = f"{model_weight_path_converted}/{fold}_{epoch}.pth"
                    torch.save(best_quantized_model, save_path)
                # else:
                best_model = model.state_dict()
                save_path = f"{model_weight_path}/{fold}_{epoch}.pth"
                torch.save(best_model, save_path)
                model.train()
                print(f"Best model saved with accuracy at fold {fold + 1} at EPOCH {epoch}: {best_accuracy:.2f}%\n")

if __name__ == "__main__":
    num_epochs = 5

    CIFAR10_model_weight_path = "model_weights/CIFAR10"
    CIFAR10_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 mean and std
    ])
    CIFAR10_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=CIFAR10_transform)


    MNIST_model_weight_path = "model_weights/MNIST"
    MNIST_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    MNIST_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=MNIST_transform)


    FMNIST_model_weight_path = "model_weights/FASHION_MNIST"
    FMNIST_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))  # Fashion-MNIST mean and std
    ])
    FMNIST_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=FMNIST_transform)
    # main(MNIST_dataset, MNIST_model_weight_path, use_quantization, 1, 10)
    use_quantization = True
    main(FMNIST_dataset, FMNIST_model_weight_path, use_quantization, 1, 10)

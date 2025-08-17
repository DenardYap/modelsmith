# Train linear layers defined in model_zoo.py 

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from model_zoo import SimpleLinear, BinaryLinear
import os
from torch.utils.data import DataLoader, TensorDataset, random_split
import logging 
import matplotlib.pyplot as plt

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)


        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    train_loss = total_train_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    return train_accuracy, train_loss

# Test/validate function
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

def main(dataset, LinearModel, model_name, model_weight_path, lr = 0.001, input_dim=28*28, num_classes=10, num_epochs=5):
    print(f"Creating {model_weight_path}/{model_name}...")
    os.makedirs(f"{model_weight_path}/{model_name}", exist_ok=True)
    logging.basicConfig(
        filename=f"{model_weight_path}/{model_name}/training.log", 
        level=logging.INFO,       
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    best_loss = float("infinity")
    best_accuracy = 0.0
    best_model = None


    # Metrics history
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []


    train_size = int(0.8 * len(dataset))  
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
    model = LinearModel(input_dim=input_dim, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr) # Might need to train higher Learning rate for quantized model
    
    for epoch in range(1, num_epochs + 1):
        train_acc, train_loss = train(model, device, train_loader, optimizer, epoch)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        accuracy, val_loss = test(model, device, val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(accuracy)
        print(f"Validation Accuracy at Epoch {epoch}: {accuracy:.2f}%")
        if val_loss < best_loss:
            best_loss = val_loss
            best_accuracy = accuracy
            
            best_model = model.state_dict()
            save_path = f"{model_weight_path}/{model_name}/{epoch}.pth"
            torch.save(best_model, save_path)
            print("================================================================================")
            print(f"EPOCH: {epoch} | ACC: {best_accuracy:.2f}% | LOSS: {best_loss:.2f}")
            logging.info(f"EPOCH: {epoch} | ACC: {best_accuracy:.2f}% | LOSS: {best_loss:.2f}")
            print("================================================================================")

        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(val_accuracies, label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        plot_path = os.path.join(model_weight_path, model_name, f"training_plot.png")
        plt.savefig(plot_path)

if __name__ == "__main__":

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

    # MNIST experiments 
    # model_name = "SimpleLinear"
    # main(MNIST_dataset, SimpleLinear, model_name, MNIST_model_weight_path, 28*28, 10)
    num_epochs = 200
    lrs = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    for lr in lrs:
        model_name = f"BinaryLinearTest_lr_{lr}"
        main(MNIST_dataset, BinaryLinear, model_name, MNIST_model_weight_path, lr, 28*28, 10, num_epochs)
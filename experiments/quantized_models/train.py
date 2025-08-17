import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from model_zoo import QCNN_simple_8bit, QCNN_simple_4bit, QCNN_simple_2bit, QCNN_simple_1bit, CNN_simple
import os
from torch.utils.data import DataLoader, TensorDataset, random_split
import logging 


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

def main(dataset, CNN, model_name, model_weight_path, input_channels=1, num_classes=10, num_epochs=5):
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

    train_size = int(0.8 * len(dataset))  
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
    print("input_channelsinput_channelsinput_channels", input_channels, num_classes)
    model = CNN(input_channels=input_channels, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Might need to train higher Learning rate for quantized model
    
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        accuracy, val_loss = test(model, device, val_loader)
        print(f"Validation Accuracy at Epoch {epoch}: {accuracy:.2f}%")
        if val_loss < best_loss:
            best_loss = val_loss
            best_accuracy = accuracy
            
            # else:
            best_model = model.state_dict()
            save_path = f"{model_weight_path}/{model_name}/{epoch}.pth"
            torch.save(best_model, save_path)
            model.train()
            print("================================================================================")
            print(f"EPOCH: {epoch} | ACC: {best_accuracy:.2f}% | LOSS: {best_loss:.2f}%")
            logging.info(f"EPOCH: {epoch} | ACC: {best_accuracy:.2f}% | LOSS: {best_loss:.2f}%")
            print("================================================================================")



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

    # MNIST experiments 
    # model_name = "CNN_simple"
    # main(MNIST_dataset, CNN_simple, model_name, MNIST_model_weight_path, 1, 10)
    # model_name = "QCNN_simple_8bit"
    # main(MNIST_dataset, QCNN_simple_8bit, model_name, MNIST_model_weight_path, 1, 10)
    model_name = "QCNN_simple_4bit"
    main(MNIST_dataset, QCNN_simple_4bit, model_name, MNIST_model_weight_path, 1, 10)
    # model_name = "QCNN_simple_2bit"
    # main(MNIST_dataset, QCNN_simple_2bit, model_name, MNIST_model_weight_path, 1, 10)
    # model_name = "QCNN_simple_1bit"
    # main(MNIST_dataset, QCNN_simple_1bit, model_name, MNIST_model_weight_path, 1, 10)

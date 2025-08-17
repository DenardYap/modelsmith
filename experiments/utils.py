import time
import torch 

def benchmark_model(model, dataloader, device):
    model.eval()
    model.to(device)
    start = time.time()

    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            _ = model(data)

    end = time.time()
    return end - start

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return 100. * correct / total




# float_time = benchmark_model(float_model, test_loader, torch.device('cpu'))
# quant_time = benchmark_model(quantized_model, test_loader, torch.device('cpu'))

# print(f"Float32 inference time: {float_time:.4f} s")
# print(f"Quantized inference time: {quant_time:.4f} s")

# import os

# float_size = os.path.getsize("float_model.pth") / 1024
# quant_size = os.path.getsize("quant_model.pth") / 1024

# print(f"Float32 model size: {float_size:.2f} KB")
# print(f"Quantized model size: {quant_size:.2f} KB")

# float_acc = evaluate(float_model, test_loader, torch.device('cpu'))
# quant_acc = evaluate(quantized_model, test_loader, torch.device('cpu'))

# print(f"Float32 accuracy: {float_acc:.2f}%")
# print(f"Quantized accuracy: {quant_acc:.2f}%")
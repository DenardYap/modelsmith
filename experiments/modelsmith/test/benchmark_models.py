import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import numpy as np
from model_zoo import (
    get_tiny_mlp,
    get_small_mlp,
    get_medium_mlp,
    get_large_mlp,
    get_wide_mlp,
    get_deep_mlp,
)
from torch.cuda.amp import autocast, GradScaler
import psutil
import os
from tqdm import tqdm
from tabulate import tabulate
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.optim.lr_scheduler import SequentialLR

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, value, model):
        if self.best_value is None:
            self.best_value = value
            self.best_model_state = model.state_dict()
            return False

        if self.mode == 'max':
            if value > self.best_value + self.min_delta:
                self.best_value = value
                self.counter = 0
                self.best_model_state = model.state_dict()
            else:
                self.counter += 1
        else:  # mode == 'min'
            if value < self.best_value - self.min_delta:
                self.best_value = value
                self.counter = 0
                self.best_model_state = model.state_dict()
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False

    def restore_best_model(self, model):
        model.load_state_dict(self.best_model_state)

def get_model_size(model):
    """Calculate model size in MB and number of parameters"""
    param_size = 0
    param_count = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_count += param.nelement()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb, param_count

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2

def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None, desc="Training", scheduler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_times = []

    pbar = tqdm(train_loader, desc=desc, leave=False)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        start_time = time.time()
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        batch_time = time.time() - start_time
        batch_times.append(batch_time)
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Update progress bar with running statistics
        accuracy = 100. * correct / total
        avg_loss = running_loss / (batch_idx + 1)
        pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'Acc': f'{accuracy:.2f}%',
            'LR': f'{optimizer.param_groups[0]["lr"]:.2e}',
            'Time/batch': f'{np.mean(batch_times):.4f}s'
        })

    accuracy = 100. * correct / total
    avg_loss = running_loss / len(train_loader)
    avg_batch_time = np.mean(batch_times)
    return accuracy, avg_loss, avg_batch_time

def evaluate(model, test_loader, criterion, device, desc="Testing"):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_times = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc=desc, leave=False)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            start_time = time.time()
            
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            
            batch_time = time.time() - start_time
            batch_times.append(batch_time)
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update progress bar with running statistics
            accuracy = 100. * correct / total
            avg_loss = test_loss / (batch_idx + 1)
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{accuracy:.2f}%',
                'Time/batch': f'{np.mean(batch_times):.4f}s'
            })

    accuracy = 100. * correct / total
    avg_loss = test_loss / len(test_loader)
    avg_batch_time = np.mean(batch_times)
    return accuracy, avg_loss, avg_batch_time

def get_training_config(is_binary):
    """Get training configuration based on model type. Can be overridden via env vars."""
    max_epochs = int(os.getenv('MS_MAX_EPOCHS', '300' if is_binary else '100'))
    patience = int(os.getenv('MS_PATIENCE', '100' if is_binary else '10'))
    warmup_epochs = int(os.getenv('MS_WARMUP_EPOCHS', '10' if is_binary else '3'))
    initial_lr = float(os.getenv('MS_LR', '0.01' if is_binary else '0.001'))
    min_lr = float(os.getenv('MS_MIN_LR', '1e-6'))
    return {
        'max_epochs': max_epochs,
        'patience': patience,
        'min_delta': 0.001,
        'warmup_epochs': warmup_epochs,
        'initial_lr': initial_lr,
        'min_lr': min_lr,
    }

def benchmark_model(model_class, dataset_name, batch_size=128, device='cuda', desc="", is_binary=False):
    # Get training configuration
    config = get_training_config(is_binary)
    
    # Set up data loaders
    if dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        num_classes = 10
        input_dim = 28 * 28  # MNIST is 28x28
    elif dataset_name == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_test)
        num_classes = 10
        input_dim = 32 * 32 * 3  # CIFAR10 is 32x32x3

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize model with correct input dimensions
    if isinstance(model_class, type):
        model = model_class(num_classes=num_classes)
    else:
        # For lambda functions that take input_dim
        model = model_class(num_classes=num_classes, input_dim=input_dim)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['initial_lr'], weight_decay=0.01)
    scaler = GradScaler() if device == 'cuda' else None
    
    # Learning rate schedulers
    warmup_scheduler = LinearLR(optimizer, 
                              start_factor=0.1, 
                              end_factor=1.0,
                              total_iters=config['warmup_epochs'] * len(train_loader))
    
    cosine_scheduler = CosineAnnealingLR(optimizer,
                                        T_max=(config['max_epochs'] - config['warmup_epochs']) * len(train_loader),
                                        eta_min=config['min_lr'])
    
    scheduler = SequentialLR(optimizer,
                            schedulers=[warmup_scheduler, cosine_scheduler],
                            milestones=[config['warmup_epochs'] * len(train_loader)])

    # Early stopping
    early_stopping = EarlyStopping(patience=config['patience'], 
                                 min_delta=config['min_delta'],
                                 mode='max')

    # Record metrics
    model_size_mb, param_count = get_model_size(model)
    initial_memory = get_memory_usage()
    
    metrics = {
        'train_accuracy': [],
        'test_accuracy': [],
        'train_loss': [],
        'test_loss': [],
        'train_batch_time': [],
        'test_batch_time': [],
        'memory_usage': [],
        'model_size_mb': model_size_mb,
        'param_count': param_count,
        'epochs_trained': 0
    }

    print(f"\n{desc}")
    print(f"Model Size: {model_size_mb:.2f}MB")
    print(f"Parameter Count: {param_count:,}")
    print(f"Initial Memory Usage: {initial_memory:.2f}MB")
    print(f"Training Config: {config}")

    # Training loop
    pbar = tqdm(range(config['max_epochs']), desc=f"{desc} Epochs")
    best_test_acc = 0
    for epoch in pbar:
        train_acc, train_loss, train_time = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            desc=f"{desc} Train",
            scheduler=scheduler
        )
        test_acc, test_loss, test_time = evaluate(
            model, test_loader, criterion, device,
            desc=f"{desc} Test"
        )
        
        current_memory = get_memory_usage() - initial_memory
        best_test_acc = max(best_test_acc, test_acc)
        
        metrics['train_accuracy'].append(train_acc)
        metrics['test_accuracy'].append(test_acc)
        metrics['train_loss'].append(train_loss)
        metrics['test_loss'].append(test_loss)
        metrics['train_batch_time'].append(train_time)
        metrics['test_batch_time'].append(test_time)
        metrics['memory_usage'].append(current_memory)
        metrics['epochs_trained'] = epoch + 1

        # Update progress bar with metrics
        pbar.set_postfix({
            'Train Acc': f'{train_acc:.2f}%',
            'Test Acc': f'{test_acc:.2f}%',
            'Best Test': f'{best_test_acc:.2f}%',
            'Train Loss': f'{train_loss:.4f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.2e}',
        })

        # Early stopping check
        if early_stopping(test_acc, model):
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            print(f"Best test accuracy: {best_test_acc:.2f}%")
            # Restore best model
            early_stopping.restore_best_model(model)
            break

    return metrics

def print_comparison(binary_metrics, standard_metrics, model_name):
    """Print side-by-side comparison of binary and standard models"""
    print(f"\n{model_name} Comparison:")
    print("=" * 80)
    
    comparison = [
        ["Metric", "Binary Model", "Standard Model", "Difference"],
        ["Best Test Accuracy", 
         f"{max(binary_metrics['test_accuracy']):.2f}%",
         f"{max(standard_metrics['test_accuracy']):.2f}%",
         f"{max(binary_metrics['test_accuracy']) - max(standard_metrics['test_accuracy']):.2f}%"],
        ["Final Test Accuracy",
         f"{binary_metrics['test_accuracy'][-1]:.2f}%",
         f"{standard_metrics['test_accuracy'][-1]:.2f}%",
         f"{binary_metrics['test_accuracy'][-1] - standard_metrics['test_accuracy'][-1]:.2f}%"],
        ["Epochs Trained",
         f"{binary_metrics['epochs_trained']}",
         f"{standard_metrics['epochs_trained']}",
         f"{binary_metrics['epochs_trained'] - standard_metrics['epochs_trained']}"],
        ["Avg Train Time/Batch",
         f"{np.mean(binary_metrics['train_batch_time']):.4f}s",
         f"{np.mean(standard_metrics['train_batch_time']):.4f}s",
         f"{np.mean(binary_metrics['train_batch_time']) - np.mean(standard_metrics['train_batch_time']):.4f}s"],
        ["Model Size",
         f"{binary_metrics['model_size_mb']:.2f}MB",
         f"{standard_metrics['model_size_mb']:.2f}MB",
         f"{binary_metrics['model_size_mb'] - standard_metrics['model_size_mb']:.2f}MB"],
        ["Parameter Count",
         f"{binary_metrics['param_count']:,}",
         f"{standard_metrics['param_count']:,}",
         f"{binary_metrics['param_count'] - standard_metrics['param_count']:,}"],
        ["Peak Memory Usage",
         f"{max(binary_metrics['memory_usage']):.2f}MB",
         f"{max(standard_metrics['memory_usage']):.2f}MB",
         f"{max(binary_metrics['memory_usage']) - max(standard_metrics['memory_usage']):.2f}MB"]
    ]
    
    print(tabulate(comparison, headers="firstrow", tablefmt="grid"))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    models_to_test = {
        # 'Tiny MLP': {
        #     'standard': lambda num_classes, input_dim: get_tiny_mlp(binary=False, num_classes=num_classes, input_dim=input_dim),
        #     'binary': lambda num_classes, input_dim: get_tiny_mlp(binary=True, num_classes=num_classes, input_dim=input_dim),
        # },
        # 'Small MLP': {
        #     'standard': lambda num_classes, input_dim: get_small_mlp(binary=False, num_classes=num_classes, input_dim=input_dim),
        #     'binary': lambda num_classes, input_dim: get_small_mlp(binary=True, num_classes=num_classes, input_dim=input_dim),
        # },
        # 'Medium MLP': {
        #     'standard': lambda num_classes, input_dim: get_medium_mlp(binary=False, num_classes=num_classes, input_dim=input_dim),
        #     'binary': lambda num_classes, input_dim: get_medium_mlp(binary=True, num_classes=num_classes, input_dim=input_dim),
        # },
        # 'Large MLP': {
        #     'standard': lambda num_classes, input_dim: get_large_mlp(binary=False, num_classes=num_classes, input_dim=input_dim),
        #     'binary': lambda num_classes, input_dim: get_large_mlp(binary=True, num_classes=num_classes, input_dim=input_dim),
        # },
        # 'Wide MLP': {
        #     'standard': lambda num_classes, input_dim: get_wide_mlp(binary=False, num_classes=num_classes, input_dim=input_dim),
        #     'binary': lambda num_classes, input_dim: get_wide_mlp(binary=True, num_classes=num_classes, input_dim=input_dim),
        # },
        'Deep MLP': {
            'standard': lambda num_classes, input_dim: get_deep_mlp(binary=False, num_classes=num_classes, input_dim=input_dim),
            'binary': lambda num_classes, input_dim: get_deep_mlp(binary=True, num_classes=num_classes, input_dim=input_dim),
        }
    }

    datasets = ['MNIST', 'CIFAR10']
    results = {}

    for dataset_name in datasets:
        results[dataset_name] = {}
        print(f"\nBenchmarking on {dataset_name}")
        print("=" * 80)
        
        for model_name, model_variants in models_to_test.items():
            print(f"\nTesting {model_name}")
            print("-" * 40)
            
            
            standard_metrics = benchmark_model(
                model_variants['standard'],
                dataset_name,
                batch_size=128,
                device=device,
                desc=f"{model_name} (Standard)",
                is_binary=False
            )
            
            # Run binary and standard models
            binary_metrics = benchmark_model(
                model_variants['binary'],
                dataset_name,
                batch_size=128,
                device=device,
                desc=f"{model_name} (Binary)",
                is_binary=True
            )
            # Store results
            results[dataset_name][f"{model_name}_binary"] = binary_metrics
            results[dataset_name][f"{model_name}_standard"] = standard_metrics
            
            # Print comparison
            print_comparison(binary_metrics, standard_metrics, f"{dataset_name} - {model_name}")

if __name__ == '__main__':
    main() 
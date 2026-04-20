import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Seed for reproducibility
torch.manual_seed(42)

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        # Initializing gates slightly negative to give the sparsity penalty a head start
        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), -1.0))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

class SelfPruningNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    os.makedirs('experiments', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)
    
    lambdas = [1e-4, 1e-3, 1e-2]
    epochs = 10
    results = []

    print(f"{'Lambda':<10} | {'Test Accuracy (%)':<20} | {'Sparsity Level (%)':<20}")
    print("-" * 55)

    for lam in lambdas:
        model = SelfPruningNetwork().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # Train
        for epoch in range(epochs):
            model.train()
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                
                outputs = model(inputs)
                ce_loss = criterion(outputs, targets)
                
                # Sparsity loss
                sparsity_loss = 0
                for m in model.modules():
                    if isinstance(m, PrunableLinear):
                        sparsity_loss += torch.sigmoid(m.gate_scores).sum()
                
                loss = ce_loss + lam * sparsity_loss
                loss.backward()
                optimizer.step()
                
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_acc = 100. * correct / total
        
        # Calculate Sparsity
        total_gates = 0
        pruned_gates = 0
        all_gates = []
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, PrunableLinear):
                    gates = torch.sigmoid(m.gate_scores)
                    total_gates += gates.numel()
                    pruned_gates += (gates < 1e-2).sum().item()
                    if lam == 1e-3:
                        all_gates.extend(gates.cpu().numpy().flatten())
                        
        sparsity_lvl = 100. * pruned_gates / total_gates
        results.append([lam, test_acc, sparsity_lvl])
        
        print(f"{lam:<10} | {test_acc:<20.2f} | {sparsity_lvl:<20.2f}")
        
        if lam == 1e-3:
            plt.figure(figsize=(8, 5))
            plt.hist(all_gates, bins=50, color='royalblue', edgecolor='black')
            plt.axvline(x=0.01, color='red', linestyle='--', label='Pruning Threshold (0.01)')
            plt.title('Gate Value Distribution (λ=1e-3)')
            plt.xlabel('Gate Value (σ(g))')
            plt.ylabel('Count')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig('plots/gate_histogram.png', dpi=150)
            plt.close()

    with open('experiments/results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Lambda', 'Test_Accuracy', 'Sparsity_Level'])
        for r in results:
            writer.writerow(r)

if __name__ == '__main__':
    main()

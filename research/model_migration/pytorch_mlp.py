"""
Simple Multi-Layer Perceptron (MLP) implementation in PyTorch.
This model can be used for basic classification tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SimpleMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) with two hidden layers.
    
    Architecture:
    - Input Layer: input_size neurons
    - Hidden Layer 1: 128 neurons with ReLU activation
    - Hidden Layer 2: 64 neurons with ReLU activation
    - Output Layer: output_size neurons with log softmax activation for classification
    """
    def __init__(self, input_size=784, hidden_size1=128, hidden_size2=64, output_size=10):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        
    def forward(self, x):
        # Flatten the input if it's not already flat
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


def train_model(model, train_loader, epochs=5, lr=0.01):
    """
    Train the PyTorch model on the provided data loader.
    
    Args:
        model: The PyTorch model to train
        train_loader: DataLoader containing the training data
        epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        Trained model
    """
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f'Epoch: {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')
    
    return model


def evaluate_model(model, test_loader):
    """
    Evaluate the PyTorch model on test data.
    
    Args:
        model: The PyTorch model to evaluate
        test_loader: DataLoader containing the test data
    
    Returns:
        Accuracy of the model on the test data
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy


# Example usage:
if __name__ == "__main__":
    # For demonstration purposes only
    # In a real scenario, you would load actual data
    
    # Create a simple MLP model
    model = SimpleMLP(input_size=784, output_size=10)
    print("PyTorch MLP Model Structure:")
    print(model)
    
    # Print model size (number of parameters)
    pytorch_params = sum(p.numel() for p in model.parameters())
    print(f"Number of model parameters: {pytorch_params}")
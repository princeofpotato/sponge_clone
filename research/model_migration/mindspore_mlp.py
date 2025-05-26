"""
Simple Multi-Layer Perceptron (MLP) implementation in MindSpore.
This is a migration of the PyTorch MLP model to demonstrate framework transition.
"""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context
from mindspore.train.model import Model
from mindspore.train.callback import LossMonitor
from mindspore.nn.metrics import Accuracy


# Set execution mode
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class SimpleMLP(nn.Cell):
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
        self.fc1 = nn.Dense(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Dense(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Dense(hidden_size2, output_size)
        self.log_softmax = nn.LogSoftmax(axis=1)
        
    def construct(self, x):
        # Flatten the input if it's not already flat
        if len(x.shape) > 2:
            x = ops.reshape(x, (x.shape[0], -1))
            
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.log_softmax(self.fc3(x))
        return x


def train_model(model, train_dataset, epochs=5, lr=0.01):
    """
    Train the MindSpore model on the provided dataset.
    
    Args:
        model: The MindSpore model to train
        train_dataset: MindSpore dataset containing the training data
        epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        Trained model
    """
    # Define loss function and optimizer
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = nn.SGD(model.trainable_params(), learning_rate=lr)
    
    # Define the model for training
    train_model = Model(model, net_loss, optimizer, metrics={"Accuracy": Accuracy()})
    
    # Train the model
    train_model.train(epochs, train_dataset, callbacks=[LossMonitor(per_print_times=100)])
    
    return model


def evaluate_model(model, test_dataset):
    """
    Evaluate the MindSpore model on test data.
    
    Args:
        model: The MindSpore model to evaluate
        test_dataset: MindSpore dataset containing the test data
    
    Returns:
        Accuracy of the model on the test data
    """
    # Define loss function for evaluation
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    
    # Define the model for evaluation
    eval_model = Model(model, net_loss, metrics={"Accuracy": Accuracy()})
    
    # Evaluate the model
    result = eval_model.eval(test_dataset)
    
    print(f"Test Accuracy: {result['Accuracy']*100:.2f}%")
    return result['Accuracy']


# Example usage:
if __name__ == "__main__":
    # For demonstration purposes only
    # In a real scenario, you would load actual data
    
    # Create a simple MLP model
    model = SimpleMLP(input_size=784, output_size=10)
    print("MindSpore MLP Model Structure:")
    print(model)
    
    # Print model size (number of parameters)
    mindspore_params = sum([p.size for p in model.trainable_params()])
    print(f"Number of model parameters: {mindspore_params}")
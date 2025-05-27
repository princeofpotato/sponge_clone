"""
Basic usage example for the Evo2 genome modeling framework
"""

import os
import sys
import numpy as np

# Add the parent directory to the path to import the Evo2 package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import mindspore as ms
    from mindspore import context
    MINDSPORE_AVAILABLE = True
except ImportError:
    MINDSPORE_AVAILABLE = False
    print("Warning: MindSpore not available. Install with 'pip install mindspore'")
    print("This example requires MindSpore to run.")
    sys.exit(1)

from src.model import GenomeModel
from src.utils import preprocess_sequence, convert_to_mindspore_tensor

def run_example():
    """
    Run a basic example of genome modeling with Evo2
    """
    print("Setting up MindSpore context...")
    # Set up MindSpore context
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    
    print("Creating sample data...")
    # Create some sample DNA sequences
    sample_sequences = [
        "ATGCGTACGATCGATCGATCGTAGCTAGCTAGCTACGATCG",
        "GCTAGCTAGCTAGCTAGCATCGATCGATGCTAGCTAGCTAG",
        "CTAGCTAGCTAGCATCGACTACGATCGATCGATCGATCGAT"
    ]
    
    # Preprocess sequences
    processed_sequences = []
    for seq in sample_sequences:
        one_hot = preprocess_sequence(seq, seq_type='dna', max_length=100)
        processed_sequences.append(one_hot)
    
    # Convert to batch
    batch_data = np.stack(processed_sequences, axis=0)
    inputs = convert_to_mindspore_tensor(batch_data)
    
    print(f"Input shape: {inputs.shape}")
    
    # Create and initialize the model
    print("Initializing GenomeModel...")
    model = GenomeModel(
        sequence_length=100,
        num_features=4,  # A, C, G, T
        embedding_dim=32,
        num_encoder_layers=2,
        num_heads=4,
        dropout_rate=0.1,
        num_classes=3  # Example classification task
    )
    
    # Set the model to training mode
    model.set_train(True)
    
    # Forward pass
    print("Running forward pass...")
    outputs = model(inputs)
    
    print(f"Output shape: {outputs.shape}")
    print(f"Model output: {outputs}")
    
    print("Example completed successfully!")
    return outputs

if __name__ == "__main__":
    if MINDSPORE_AVAILABLE:
        run_example()
    else:
        print("This example requires MindSpore to run.")
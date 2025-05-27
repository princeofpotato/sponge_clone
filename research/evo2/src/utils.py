"""
Utility functions for Evo2 genome modeling
"""

import numpy as np
from typing import Dict, List, Tuple, Union

try:
    import mindspore as ms
    import mindspore.nn as nn
    from mindspore import Tensor, context
    MINDSPORE_AVAILABLE = True
except ImportError:
    MINDSPORE_AVAILABLE = False
    print("Warning: MindSpore not available. Install with 'pip install mindspore'")


def one_hot_encode_dna(sequence: str) -> np.ndarray:
    """
    Convert a DNA sequence string to one-hot encoding
    
    Args:
        sequence: DNA sequence string containing A, C, G, T
        
    Returns:
        One-hot encoded array of shape (4, seq_length)
    """
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    seq_length = len(sequence)
    encoding = np.zeros((5, seq_length), dtype=np.float32)
    
    for i, char in enumerate(sequence.upper()):
        if char in mapping:
            encoding[mapping[char], i] = 1.0
    
    # Remove the N channel to get standard ACGT encoding
    return encoding[:4, :]


def one_hot_encode_protein(sequence: str) -> np.ndarray:
    """
    Convert a protein sequence string to one-hot encoding
    
    Args:
        sequence: Protein sequence string containing amino acids
        
    Returns:
        One-hot encoded array of shape (20, seq_length)
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    mapping = {aa: i for i, aa in enumerate(amino_acids)}
    seq_length = len(sequence)
    encoding = np.zeros((20, seq_length), dtype=np.float32)
    
    for i, char in enumerate(sequence.upper()):
        if char in mapping:
            encoding[mapping[char], i] = 1.0
    
    return encoding


def preprocess_sequence(sequence: str, seq_type: str = 'dna', max_length: int = 1000) -> np.ndarray:
    """
    Preprocess a biological sequence for model input
    
    Args:
        sequence: Biological sequence string
        seq_type: Type of sequence ('dna' or 'protein')
        max_length: Maximum sequence length
        
    Returns:
        Preprocessed sequence as numpy array
    """
    # Truncate or pad the sequence
    if len(sequence) > max_length:
        sequence = sequence[:max_length]
    else:
        sequence = sequence + 'N' * (max_length - len(sequence))
    
    # Convert to one-hot encoding
    if seq_type.lower() == 'dna':
        one_hot = one_hot_encode_dna(sequence)
    elif seq_type.lower() == 'protein':
        one_hot = one_hot_encode_protein(sequence)
    else:
        raise ValueError(f"Unknown sequence type: {seq_type}. Use 'dna' or 'protein'")
    
    return one_hot


def convert_to_mindspore_tensor(data: np.ndarray) -> Union[np.ndarray, 'Tensor']:
    """
    Convert numpy array to MindSpore tensor if MindSpore is available
    
    Args:
        data: Input numpy array
        
    Returns:
        MindSpore tensor or numpy array if MindSpore is not available
    """
    if MINDSPORE_AVAILABLE:
        return Tensor(data)
    return data


def evaluate_model(model: 'nn.Cell', 
                  sequences: List[str], 
                  labels: List[int], 
                  batch_size: int = 32, 
                  seq_type: str = 'dna') -> Dict[str, float]:
    """
    Evaluate the model on a dataset
    
    Args:
        model: The genome model to evaluate
        sequences: List of biological sequences
        labels: List of labels/targets
        batch_size: Batch size for evaluation
        seq_type: Type of sequence ('dna' or 'protein')
        
    Returns:
        Dictionary with evaluation metrics
    """
    if not MINDSPORE_AVAILABLE:
        raise ImportError("MindSpore is required for model evaluation")
    
    num_samples = len(sequences)
    model.set_train(False)
    
    all_preds = []
    all_labels = []
    
    # Process in batches
    for i in range(0, num_samples, batch_size):
        batch_seqs = sequences[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        
        # Preprocess sequences
        processed_seqs = [preprocess_sequence(seq, seq_type) for seq in batch_seqs]
        batch_inputs = np.stack(processed_seqs, axis=0)
        batch_inputs = convert_to_mindspore_tensor(batch_inputs)
        
        # Convert labels
        batch_labels = np.array(batch_labels, dtype=np.int32)
        batch_labels = convert_to_mindspore_tensor(batch_labels)
        
        # Get predictions
        predictions = model(batch_inputs)
        
        # Store for metrics calculation
        all_preds.append(predictions.asnumpy())
        all_labels.append(batch_labels.asnumpy())
    
    # Concatenate all predictions and labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Calculate metrics
    accuracy = np.mean((all_preds.argmax(axis=1) == all_labels).astype(np.float32))
    
    return {
        'accuracy': float(accuracy),
        'num_samples': num_samples
    }
"""
Evo2: Genome modeling and design across all domains of life
MindSpore implementation
"""

from .model import GenomeModel
from .utils import preprocess_sequence, evaluate_model

__version__ = '0.1.0'
__all__ = ['GenomeModel', 'preprocess_sequence', 'evaluate_model']
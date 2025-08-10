"""
Data generation module for creating training and testing datasets.
Contains scripts for generating both correct and incorrect arithmetic examples.
"""

from .generate_datasets import generate_dataset
from .generate_incorrect_datasets import generate_incorrect_dataset

__all__ = ['generate_dataset', 'generate_incorrect_dataset'] 
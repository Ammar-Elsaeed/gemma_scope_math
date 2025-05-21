import os
from itertools import product
import numpy as np

def generate_arithmetic_dataset(operation, output_file):
    """Generate dataset for arithmetic operations with numbers 0-99.
    
    Args:
        operation (str): The operation to use ('+' or '-')
        output_file (str): Path to the output file
    """
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Generate all combinations of numbers from 0 to 99
    numbers = range(100)
    combinations = product(numbers, numbers)
    
    # Write to file
    with open(os.path.join("data", output_file), 'w') as f:
        for a, b in combinations:
            line = f"calc: {a} {operation} {b} =\n"
            f.write(line)
    
    # Count lines in generated file
    with open(os.path.join("data", output_file), 'r') as f:
        line_count = sum(1 for _ in f)
    
    print(f"Generated {output_file} with {line_count} examples")

def generate_paired_random_datasets(num_examples=10000):
    """Generate paired addition and subtraction datasets with same random operands.
    
    Args:
        num_examples (int): Number of examples to generate
    """
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Generate random operands between 1000 and 10000
    operands = np.random.randint(1000, 10001, size=(num_examples, 2))
    
    # Generate addition dataset
    with open(os.path.join("data", "random_addition.txt"), 'w') as f:
        for a, b in operands:
            line = f"calc: {a} + {b} =\n"
            f.write(line)
    
    # Generate subtraction dataset with same operands
    with open(os.path.join("data", "random_subtraction.txt"), 'w') as f:
        for a, b in operands:
            line = f"calc: {a} - {b} =\n"
            f.write(line)
    
    print(f"Generated random_addition.txt and random_subtraction.txt with {num_examples} examples each")
    print("The datasets use the same random operands in range [1000, 10000]")

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate simple datasets (0-99)
    generate_arithmetic_dataset('+', 'addition.txt')
    generate_arithmetic_dataset('-', 'subtraction.txt')
    
    # Generate paired random datasets (1000-10000)
    generate_paired_random_datasets(10000)

if __name__ == "__main__":
    main() 
import numpy as np
import random
from typing import List, Tuple
import os

def generate_operands(num_examples: int, max_val: int = 10000) -> List[Tuple[int, int, str]]:
    """Generate random operand pairs with operation."""
    examples = []
    # Generate addition examples (first half)
    for _ in range(num_examples // 2):
        a = random.randint(0, max_val)
        b = random.randint(0, max_val)
        examples.append((a, b, '+'))
    
    # Generate subtraction examples (second half)
    for _ in range(num_examples // 2):
        # Generate two random numbers and sort them to ensure positive result
        a = random.randint(0, max_val)
        b = random.randint(0, max_val)
        a, b = max(a, b), min(a, b)  # Ensure a >= b
        examples.append((a, b, '-'))
    
    # Shuffle all examples
    random.shuffle(examples)
    return examples

def generate_datasets(size: int = 20000, max_val: int = 10000) -> Tuple[List[str], List[str]]:
    """Generate both correct and incorrect datasets using the same operands."""
    correct_examples = []
    incorrect_examples = []
    
    # Generate operands and operations
    examples = generate_operands(size, max_val)
    
    for a, b, op in examples:
        if op == '+':
            correct = a + b
            # Generate wrong answer that's different from correct answer
            while True:
                wrong = random.randint(0, 2 * max_val)
                if wrong != correct:
                    break
            correct_examples.append(f"{a} + {b} = {correct}")
            incorrect_examples.append(f"{a} + {b} = {wrong}")
        else:  # op == '-'
            correct = a - b
            # Generate wrong answer that's different from correct answer
            while True:
                wrong = random.randint(0, max_val)  # Max possible difference is max_val
                if wrong != correct:
                    break
            correct_examples.append(f"{a} - {b} = {correct}")
            incorrect_examples.append(f"{a} - {b} = {wrong}")
    
    return correct_examples, incorrect_examples

def save_dataset(examples: List[str], filename: str):
    """Save dataset to a file."""
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", filename), "w") as f:
        for example in examples:
            f.write(example + "\n")

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Generate both datasets using the same operands
    print("Generating datasets...")
    correct_examples, incorrect_examples = generate_datasets()
    
    # Save datasets
    save_dataset(correct_examples, "correct_arithmetic.txt")
    save_dataset(incorrect_examples, "incorrect_arithmetic.txt")
    print(f"Saved {len(correct_examples)} examples in each dataset")
    
    # Print some example pairs
    print("\nExample pairs (Correct vs Incorrect):")
    for i in range(5):
        print(f"Correct:   {correct_examples[i]}")
        print(f"Incorrect: {incorrect_examples[i]}")
        print()

if __name__ == "__main__":
    main() 
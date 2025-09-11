# Utility functions for loading and working with ablation results
# Add these functions to your causality_experiment.ipynb notebook
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sae_lens import SAE, HookedSAETransformer
import torch.nn.functional as F
import os
import re
import pickle
import json
from typing import Dict, List, Tuple, Optional

def calc_correct_answer(problem):
    """
    Calculate the correct answer for a given problem.
    
    Args:
        problem: Problem string (e.g., "2 + 2")
    
    Returns:
        str: Correct answer as a string
    """
    # Remove any non-numeric characters except for +, -, *, /
    clean_problem = re.sub(r'[^\d\s\+\-\*/]', '', problem)
    
    try:
        # Evaluate the expression safely
        answer = eval(clean_problem)
        return str(answer)
    except Exception as e:
        print(f"Error evaluating problem '{problem}': {e}")
        return "ERROR"
        
def benchmark_correct_answers(dataset_path, batch_size=64, start_batch=0, end_batch=None,):
    """
    Load the same questions as batch_quiz_model and calculate correct answers.
    
    Args:
        dataset_path: Path to the dataset file
        batch_size: Number of problems to process simultaneously (same as used in batch_quiz_model)
        start_batch: Starting batch index (0-based, same as used in batch_quiz_model)
        end_batch: Ending batch index (exclusive, same as used in batch_quiz_model)
        prefix: Prefix that was added to each problem in batch_quiz_model
        postfix: Postfix that was added to each problem in batch_quiz_model
    
    Returns:
        List of tuples (full_prompt, correct_answer)
    """
    # Load the dataset
    with open(dataset_path, 'r') as f:
        problems = [line.strip() for line in f.readlines() if line.strip()]
    
    # Calculate total number of batches
    total_batches = (len(problems) + batch_size - 1) // batch_size
    
    # Set end_batch if not specified
    if end_batch is None:
        end_batch = total_batches
    
    # Validate batch indices (same validation as batch_quiz_model)
    if start_batch < 0 or start_batch >= total_batches:
        raise ValueError(f"start_batch {start_batch} is out of range [0, {total_batches})")
    if end_batch < start_batch or end_batch > total_batches:
        raise ValueError(f"end_batch {end_batch} is out of range [{start_batch}, {total_batches}]")
    
    # Calculate problem indices for the specified batch range (same as batch_quiz_model)
    start_idx = start_batch * batch_size
    end_idx = min(end_batch * batch_size, len(problems))
    
    # Get subset of problems for this run
    subset_problems = problems[start_idx:end_idx]
    
    results = []
    for problem in subset_problems:
        # Create the same full prompt as batch_quiz_model does
        full_prompt =problem 
        
        # Calculate correct answer for the original problem (without prefix/postfix)
        correct_answer = calc_correct_answer(problem)
        
        results.append((full_prompt, correct_answer))
    
    print(f"Calculated correct answers for batches {start_batch} to {end_batch-1} ({len(subset_problems)} problems)")
    
    return results

def calculate_accuracy(model_results, correct_answers):
    """
    Calculate accuracy by comparing extracted numbers from model answers with correct answers.
    Handles negative numbers and step-by-step responses.
    Skipped examples (no valid numbers found) are counted as incorrect in accuracy calculation.
    
    Args:
        model_results: List of tuples (prompt, model_answer) from batch_quiz_model
        correct_answers: List of tuples (prompt, correct_answer) from benchmark_correct_answers
    
    Returns:
        tuple: (accuracy, correct_count, total_count, detailed_results, skipped_count)
            - accuracy: Accuracy as a float between 0 and 1 (skipped examples count as incorrect)
            - correct_count: Number of correct answers
            - total_count: Total number of questions (including skipped)
            - detailed_results: List of tuples (prompt, correct_answer, model_answer, extracted_number, is_correct)
            - skipped_count: Number of prompts skipped due to no valid numbers found
    """
    import re
    
    if len(model_results) != len(correct_answers):
        raise ValueError(f"Mismatch in result lengths: {len(model_results)} vs {len(correct_answers)}")
    
    correct_count = 0
    skipped_count = 0
    detailed_results = []
    
    for i, ((model_prompt, model_answer), (correct_prompt, correct_answer)) in enumerate(zip(model_results, correct_answers)):
        # Remove question pattern if it exists
        question_pattern = r'\d+[\+\-\*/]\d+\s*=?\s*'
        cleaned_answer = re.sub(question_pattern, '', model_answer)
        
        # Remove step numbering patterns (e.g., "1.", "2.", etc.)
        step_pattern = r'\b\d+\.\s*'
        cleaned_answer = re.sub(step_pattern, '', cleaned_answer)
        
        # Extract the first number (with optional minus sign) from the cleaned model answer
        number_pattern = r'-?\d+'
        numbers = re.findall(number_pattern, cleaned_answer)
        
        # Handle skipped cases (no numbers found) - count as incorrect
        if not numbers:
            skipped_count += 1
            extracted_number = "NO_NUMBER_FOUND"
            is_correct = False
        else:
            # Get the first extracted number
            extracted_number = numbers[0]
            # Check if extracted number matches the correct answer
            is_correct = extracted_number == correct_answer.strip()
            if is_correct:
                correct_count += 1
        
        detailed_results.append((
            model_prompt,
            correct_answer,
            model_answer,
            extracted_number,
            is_correct
        ))
    
    total_count = len(detailed_results)  # Count all results including skipped
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    return accuracy, correct_count, total_count, detailed_results, skipped_count

def print_accuracy_report(accuracy, correct_count, total_count, detailed_results, skipped_count=0, show_errors_only=False, max_examples=10):
    """
    Print a detailed accuracy report.
    
    Args:
        accuracy: Accuracy as a float
        correct_count: Number of correct answers
        total_count: Total number of questions
        detailed_results: Detailed results from calculate_accuracy
        skipped_count: Number of prompts skipped due to insufficient digits
        show_errors_only: If True, only show incorrect answers
        max_examples: Maximum number of examples to show
    """
    incorrect_count = total_count - correct_count
    print(f"Accuracy: {accuracy:.3f} ({correct_count}/{total_count})")
    print(f"Correct: {correct_count}")
    print(f"Incorrect: {incorrect_count} (including {skipped_count} skipped)")
    if skipped_count > 0:
        print(f"  - Actually incorrect: {incorrect_count - skipped_count}")
        print(f"  - Skipped (no valid numbers): {skipped_count}")
    print("-" * 80)
    
    examples_shown = 0
    for prompt, correct, model_answer, extracted, is_correct in detailed_results:
        if show_errors_only and is_correct:
            continue
        
        if examples_shown >= max_examples:
            break
        
        status = "✓" if is_correct else "✗"
        skipped_indicator = " (SKIPPED)" if extracted == "NO_NUMBER_FOUND" else ""
        print(f"{status} Prompt: {prompt}")
        print(f"  Correct: {correct}")
        print(f"  Model output: '{model_answer}'")
        print(f"  Extracted: '{extracted}'{skipped_indicator}")
        print()
        
        examples_shown += 1
    
    if examples_shown < len(detailed_results):
        remaining = len(detailed_results) - examples_shown
        print(f"... and {remaining} more examples")

def load_ablation_results(json_path: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Load ablation results from a JSON file.

    Args:
        json_path: Path to the JSON file containing ablation results

    Returns:
        Dictionary with condition names as keys and list of (prompt, answer) tuples as values
        Example: {
            'no_ablation': [('5+7= ', '12'), ('3+4= ', '7'), ...],
            'operation_ablation': [('5+7= ', '11'), ('3+4= ', '6'), ...],
            'topk_ablation': [('5+7= ', '10'), ('3+4= ', '5'), ...]
        }
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Ensure tuples, not lists, for each (prompt, answer) pair
    return {k: [tuple(pair) for pair in v] for k, v in data.items()}

def load_correct_answers(pickle_path: str) -> List[Tuple[str, str]]:
    """
    Load correct answers from pickle file.
    
    Args:
        pickle_path: Path to the pickle file containing correct answers
        
    Returns:
        List of (prompt, correct_answer) tuples
    """
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

def compare_ablation_conditions(ablation_results: Dict[str, List[Tuple[str, str]]], 
                               correct_answers: List[Tuple[str, str]],
                               conditions_to_compare: Optional[List[str]] = None) -> Dict[str, Dict]:
    """
    Compare multiple ablation conditions against correct answers using your existing functions.
    
    Args:
        ablation_results: Results from load_ablation_results()
        correct_answers: Results from load_correct_answers() or benchmark_correct_answers()
        conditions_to_compare: List of condition names to compare. If None, compares all conditions.
        
    Returns:
        Dictionary with condition names as keys and accuracy metrics as values
    """
    if conditions_to_compare is None:
        conditions_to_compare = list(ablation_results.keys())
    
    comparison_results = {}
    
    for condition in conditions_to_compare:
        if condition not in ablation_results:
            print(f"Warning: Condition '{condition}' not found in ablation results")
            continue
            
        print(f"\\n{'='*50}")
        print(f"Analyzing condition: {condition.upper()}")
        print(f"{'='*50}")
        
        # Use your existing functions
        model_results = ablation_results[condition]
        accuracy, correct_count, total_count, detailed_results, skipped_count = calculate_accuracy(
            model_results, correct_answers
        )
        
        # Store results
        comparison_results[condition] = {
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_count': total_count,
            'incorrect_count': total_count - correct_count,
            'skipped_count': skipped_count,
            'detailed_results': detailed_results
        }
        
        # Print report
        print_accuracy_report(accuracy, correct_count, total_count, detailed_results, 
                            skipped_count, show_errors_only=True, max_examples=5)
    
    return comparison_results

def analyze_ablation_experiment(results_dir: str, 
                               layer: int, 
                               dataset_name: str, 
                               num_feats: int,
                               dataset_path: str,
                               batch_size: int = 1024,
                               start_batch: int = 0,
                               end_batch: Optional[int] = None) -> Dict:
    """
    Complete analysis of an ablation experiment.
    
    Args:
        results_dir: Directory containing the ablation results
        layer: Layer number used in the experiment
        dataset_name: Name of the dataset (e.g., 'addition', 'subtraction')
        num_feats: Number of features that were ablated
        dataset_path: Path to the original dataset file
        batch_size: Batch size used (for generating correct answers)
        start_batch: Start batch index (for generating correct answers)
        end_batch: End batch index (for generating correct answers)
        
    Returns:
        Dictionary containing all analysis results
    """
    
    # Find the most recent results file for this configuration
    import glob
    pattern = f"{results_dir}/layer_{layer}_{dataset_name}_{num_feats}_feats_*.pkl"
    result_files = glob.glob(pattern)
    
    if not result_files:
        raise FileNotFoundError(f"No ablation results found matching pattern: {pattern}")
    
    # Get the most recent file
    latest_file = max(result_files, key=os.path.getctime)
    print(f"Loading ablation results from: {latest_file}")
    
    # Load ablation results
    ablation_results = load_ablation_results(latest_file)
    print(f"Found conditions: {list(ablation_results.keys())}")
    
    # Generate or load correct answers
    correct_answers_pattern = f"{results_dir}/correct_answers_{dataset_name}_{num_feats}_feats.pkl"
    correct_answers_files = glob.glob(correct_answers_pattern)
    
    if correct_answers_files:
        print(f"Loading correct answers from: {correct_answers_files[0]}")
        correct_answers = load_correct_answers(correct_answers_files[0])
    else:
        print("Generating correct answers...")
        correct_answers = benchmark_correct_answers(
            dataset_path, batch_size=batch_size, 
            start_batch=start_batch, end_batch=end_batch
        )
    
    # Compare all conditions
    print(f"\\n{'='*60}")
    print("ABLATION EXPERIMENT ANALYSIS")
    print(f"Layer: {layer}, Dataset: {dataset_name}, Features: {num_feats}")
    print(f"{'='*60}")
    
    comparison_results = compare_ablation_conditions(ablation_results, correct_answers)
    
    # Summary comparison
    print(f"\\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Condition':<20} {'Accuracy':<10} {'Correct':<8} {'Total':<8} {'Skipped':<8}")
    print("-" * 60)
    
    for condition, metrics in comparison_results.items():
        print(f"{condition:<20} {metrics['accuracy']:<10.3f} "
              f"{metrics['correct_count']:<8} {metrics['total_count']:<8} "
              f"{metrics['skipped_count']:<8}")
    
    return {
        'ablation_results': ablation_results,
        'correct_answers': correct_answers,
        'comparison_results': comparison_results,
        'results_file': latest_file
    }

def quick_ablation_analysis(results_pickle_path: str, 
                           correct_answers_pickle_path: str) -> Dict:
    """
    Quick analysis when you already have the specific file paths.
    
    Args:
        results_pickle_path: Path to ablation results pickle file
        correct_answers_pickle_path: Path to correct answers pickle file
        
    Returns:
        Dictionary containing analysis results
    """
    print(f"Loading results from: {results_pickle_path}")
    ablation_results = load_ablation_results(results_pickle_path)
    
    print(f"Loading correct answers from: {correct_answers_pickle_path}")
    correct_answers = load_correct_answers(correct_answers_pickle_path)
    
    comparison_results = compare_ablation_conditions(ablation_results, correct_answers)
    
    return {
        'ablation_results': ablation_results,
        'correct_answers': correct_answers, 
        'comparison_results': comparison_results
    }

# Example usage:
"""
# After running your modified parallel_ablation.py script:

# Method 1: Automatic analysis (finds the latest files)
results = analyze_ablation_experiment(
    results_dir="./ablation_results",
    layer=12,
    dataset_name="addition", 
    num_feats=1,
    dataset_path="./data/addition.txt",
    batch_size=1024*4,
    start_batch=0,
    end_batch=3
)

# Method 2: Manual analysis with specific file paths
results = quick_ablation_analysis(
    results_pickle_path="./ablation_results/layer_12_addition_1_feats_20250811_143022.pkl",
    correct_answers_pickle_path="./ablation_results/correct_answers_addition_1_feats.pkl"
)

# Access specific condition results:
no_ablation_accuracy = results['comparison_results']['no_ablation']['accuracy']
operation_ablation_accuracy = results['comparison_results']['operation_ablation']['accuracy']

print(f"No ablation accuracy: {no_ablation_accuracy:.3f}")
print(f"Operation ablation accuracy: {operation_ablation_accuracy:.3f}")
print(f"Difference: {no_ablation_accuracy - operation_ablation_accuracy:.3f}")
"""

# Complete Example: How to use the compatible ablation script with your notebook functions

"""
This example shows how to:
1. Run the modified parallel_ablation_compatible.py script
2. Load the results in your notebook
3. Use your existing analysis functions

STEP 1: Run the modified ablation script
========================================
"""

# Example command to run the compatible ablation script:
"""
python src/ablation/parallel_ablation_compatible.py \
    --layer 12 \
    --dataset "./data/addition.txt" \
    --ablation_features "./feature_metrics/layer_12_subtraction_features.csv" \
    --batch_size 1024 \
    --max_new_tokens 12 \
    --num_feats 3 \
    --run_no_ablation \
    --ablate_topk_features \
    --save_correct_answers \
    --output_dir "./ablation_results"
"""

"""
This will create files like:
- ablation_results/layer_12_addition_1_feats_20250811_143022.pkl
- ablation_results/layer_12_addition_1_feats_20250811_143022.json  
- ablation_results/layer_12_addition_1_feats_20250811_143022.txt
- ablation_results/correct_answers_addition_1_feats.pkl

STEP 2: Load and analyze results in your notebook
=================================================
"""

# Add these cells to your causality_experiment.ipynb notebook:

"""
# Cell 1: Load the utility functions (copy ablation_analysis_utils.py content or import it)
import pickle
import json
import os
import glob
from typing import Dict, List, Tuple, Optional

def load_ablation_results(pickle_path: str) -> Dict[str, List[Tuple[str, str]]]:
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

def load_correct_answers(pickle_path: str) -> List[Tuple[str, str]]:
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

# Cell 2: Load your ablation results
results_dir = "./ablation_results"
layer = 12
dataset_name = "addition"
num_feats = 1

# Find the most recent results file
pattern = f"{results_dir}/layer_{layer}_{dataset_name}_{num_feats}_feats_*.pkl"
result_files = glob.glob(pattern)
latest_file = max(result_files, key=os.path.getctime)

print(f"Loading ablation results from: {latest_file}")
ablation_results = load_ablation_results(latest_file)
print(f"Available conditions: {list(ablation_results.keys())}")

# Cell 3: Load correct answers
correct_answers_file = f"{results_dir}/correct_answers_{dataset_name}_{num_feats}_feats.pkl"
if os.path.exists(correct_answers_file):
    correct_answers = load_correct_answers(correct_answers_file)
    print(f"Loaded {len(correct_answers)} correct answers")
else:
    # Generate them using your existing function
    correct_answers = benchmark_correct_answers(
        f"./data/{dataset_name}.txt", 
        batch_size=1024*4, 
        start_batch=0, 
        end_batch=3
    )

# Cell 4: Analyze each condition using your existing functions
conditions = ['no_ablation', 'operation_ablation', 'topk_ablation']

for condition in conditions:
    if condition in ablation_results:
        print(f"\\n{'='*50}")
        print(f"Analyzing: {condition.upper()}")
        print(f"{'='*50}")
        
        model_results = ablation_results[condition]
        accuracy, correct_count, total_count, detailed_results, skipped_count = calculate_accuracy(
            model_results, correct_answers
        )
        
        print_accuracy_report(accuracy, correct_count, total_count, detailed_results, 
                            skipped_count, show_errors_only=True, max_examples=5)

# Cell 5: Compare conditions
print(f"\\n{'='*60}")
print("CONDITION COMPARISON")
print(f"{'='*60}")

comparison_data = {}
for condition in conditions:
    if condition in ablation_results:
        model_results = ablation_results[condition]
        accuracy, correct_count, total_count, detailed_results, skipped_count = calculate_accuracy(
            model_results, correct_answers
        )
        comparison_data[condition] = {
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_count': total_count,
            'skipped_count': skipped_count
        }

# Print comparison table
print(f"{'Condition':<20} {'Accuracy':<10} {'Correct':<8} {'Total':<8} {'Skipped':<8}")
print("-" * 60)
for condition, metrics in comparison_data.items():
    print(f"{condition:<20} {metrics['accuracy']:<10.3f} "
          f"{metrics['correct_count']:<8} {metrics['total_count']:<8} "
          f"{metrics['skipped_count']:<8}")

# Calculate differences
if 'no_ablation' in comparison_data and 'operation_ablation' in comparison_data:
    accuracy_diff = comparison_data['no_ablation']['accuracy'] - comparison_data['operation_ablation']['accuracy']
    print(f"\\nAccuracy drop from ablation: {accuracy_diff:.3f}")
"""

"""
STEP 3: Expected Data Formats
=============================

Your functions expect:

1. model_results: List of tuples (prompt, model_answer)
   Example: [("5+7= ", "12"), ("3+4= ", "7")]

2. correct_answers: List of tuples (prompt, correct_answer)  
   Example: [("5+7= ", "12"), ("3+4= ", "7")]

The modified script saves results in exactly this format in the pickle files.

STEP 4: Key Improvements
========================

1. Compatible Format: Results are saved as lists of (prompt, answer) tuples
2. Multiple Formats: Pickle (fast), JSON (readable), text (backwards compatible)
3. Correct Answers: Automatically generated and saved
4. Easy Loading: Simple functions to load and analyze results
5. Batch Processing: Handles large datasets efficiently

STEP 5: File Structure After Running
====================================

ablation_results/
├── layer_12_addition_1_feats_20250811_143022.pkl    # Main results (use this)
├── layer_12_addition_1_feats_20250811_143022.json   # Human readable
├── layer_12_addition_1_feats_20250811_143022.txt    # Original format
├── correct_answers_addition_1_feats.pkl             # Correct answers
├── layer_12_addition_2_feats_20250811_143055.pkl    # Results for 2 features
└── ...

STEP 6: Integration with Existing Workflow
==========================================

Your existing notebook functions work unchanged:
- calculate_accuracy()
- print_accuracy_report()
- benchmark_correct_answers()

Just load the data from pickle files instead of generating it fresh each time.
"""

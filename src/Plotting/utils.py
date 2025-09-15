import json


def load_baseline_results():
    # loads baseline performance for the 4 datasets into a dictionary with dataset names as keys, and performance metrics as values
    # Use gemma_scope_math/answers/baseline_metrics.json -> metrics -> dataset -> answer_directly -> {accuracy, correct_count, incorrect_count, skipped_count}
    with open("gemma_scope_math/answers/baseline_metrics.json") as f:
        data = json.load(f)
    return {
        dataset: metrics["answer_directly"]
        for dataset, metrics in data["metrics"].items()
    }

def load_dataset_ablation_results(dataset_name):
    # loads ablation results for a given dataset into a dictionary. keys are [layer][number of ablated features][ablation_type], where layer is 0-25, number of ablated features is odd numbers from [1,11], and ablation_type is one of [operation_ablation, topk_ablation]
    # -> values are lists of length two, where the first element is the prompt (e.g. "Answer Directly: X+Y= "), and the second element is the generated text from the model
    with open(f"gemma_scope_math/aggregated_results/{dataset_name}.json") as f:
        dataset_results = json.load(f)
    return dataset_results 

def calculate_and_format_ablation_results(dataset_results):
    
import json
import ollama
# from human_eval.data import HUMAN_EVAL
# from human_eval.execution import evaluate_function
import os
from datasets import load_dataset
from human_eval.evaluation import evaluate_functional_correctness

dataset = load_dataset("openai_humaneval")["test"]

# Set the local model name (must be available in Ollama)
MODEL_NAME = "deepseek-r1:8b"  # Change to your preferred model (e.g., "mistral")

# Prompt template for generating function implementations
PROMPT_TEMPLATE = """### Instruction:
Please complete the python function below. The final complete version of your function must be returned within a code block. Here is the unfinished function:\n
### Function:
{prompt}

### Response:
"""

def generate_code(prompt: str) -> str:
    """Generates Python code using a local model in Ollama."""
    response = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(prompt=prompt)}])
    return response['message']['content']

# JSONL file path
sample_file = "generated_samples.jsonl"


for i in range(len(dataset)):  
        example = dataset.select([i])[0]
        if (example['task_id'] == "HumanEval/108"):
            print(i)
            exit(1)

with open(sample_file, "w") as f:
    pass

# write generated code

with open(sample_file, "a") as f:
    for i in range(len(dataset)):  
        example = dataset.select([i])[0]
        prompt = example["prompt"]
        generated_code = generate_code(prompt)

        # make sure `generated_code` not empty
        if not generated_code.strip():
            print(f" Warning: Empty generated code for sample {i+1}")
            continue

        
        json.dump({"task_id": example["task_id"], "completion": generated_code}, f)
        f.write("\n")  
        print(f"task_id {example['task_id']} completed")

# check if JSONL file written 
if os.path.exists(sample_file):
    with open(sample_file, "r") as f:
        lines = f.readlines()
        if len(lines) == 0:
            raise ValueError(f" Error: {sample_file} is empty! No generated samples found.")
else:
    raise FileNotFoundError(f" Error: {sample_file} does not exist!")

# run eval
results = evaluate_functional_correctness(sample_file=sample_file)

# pass@1
pass_at_1 = results["pass@1"]
print(f"\n Final Pass@1 Score: {pass_at_1:.2%}")

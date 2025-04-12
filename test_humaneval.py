import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from human_eval.human_eval.evaluation import evaluate_functional_correctness


os.environ["TOKENIZERS_PARALLELISM"] = "false"
scratch = os.getenv("SCRATCH")

# load model
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

print("Model successfully loaded!")

# load HumanEval 
dataset = load_dataset("openai_humaneval")["test"]

# code generation function
def generate_code(prompt, max_length=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)  
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_length, do_sample=True, top_p=0.9, temperature=0.7
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# JSONL file path

sample_file = "/scratch/user/lsc206573/generated_samples.jsonl"


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

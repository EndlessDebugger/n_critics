import json
import os
from datasets import load_dataset


# load code samples 
print(">>> loading code samples...")
sample_file="/scratch/user/lsc206573/nlp/mbpp/generated_samples.jsonl"

if not os.path.exists(sample_file):
    raise FileNotFoundError(f"Sample file '{sample_file}' does not exist")

sample_dict={}
with open(sample_file,"r") as f:
    for line in f:
        try:
            sample=json.loads(line.strip())
            task_id=sample["task_id"]
            code=sample["completion"]
            sample_dict[task_id]=code
        except json.JSONDecodeError:
            continue

# load mbpp dataset
print(">>> loading mbpp dataset...")
mbpp_file="/scratch/user/lsc206573/nlp/mbpp/mbpp.jsonl"
dataset=load_dataset("json",data_files=mbpp_file)
dataset=dataset["train"]
print(f">>> Dataset loaded. Total tasks: {len(dataset)}")

# fuse dataset and samples
print(">>> fusing samples and original dataset")
prompts=[]
for i in range(len(dataset)):
    datapoint=dataset[i]
    task_id=datapoint["task_id"]
    task=datapoint["text"]
    code=sample_dict[task_id]
    prompt=f"the following function \n{code}\n is written for solving this problem: \"{task}\"  there might be some errors in the code, such as: using undefined variables; unexpected indent; unclosed parenthesis. Try to refine the code to solve the problem"
    prompts.append(prompt)

dataset_with_prompt=dataset.add_column("prompt",prompts)

out_file = "mbpp_with_prompt.jsonl"

dataset_with_prompt.to_json(out_file, orient="records", lines=True)
print(f">>> Saved augmented dataset to {out_file}")


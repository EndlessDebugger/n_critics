import os
import json
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

#os.environ["TOKENIZERS_PARALLELISM"] = "false"

print(">>> Loading model...")
model_path = "/scratch/user/lsc206573/nlp/Deepseek-Coder-V2-Lite-Instruct/"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", torch_dtype=torch.float16, local_files_only=True, trust_remote_code=True
)
print(">>> Model loaded.")

# load dataset
print(">>> Loading dataset...")
dataset = load_dataset("json", data_files="/scratch/user/lsc206573/nlp/mbpp/mbpp.jsonl")
dataset = dataset["train"]
print(f">>> Dataset loaded. Total tasks: {len(dataset)}")

# output path
sample_file = "/scratch/user/lsc206573/nlp/mbpp/generated_samples.jsonl"

# if output file does not exist, create one
if not os.path.exists(sample_file):
    with open(sample_file, "w") as f:
        pass
    print(">>> Created empty sample file.")

# start from break point
completed_task_ids = set()
with open(sample_file, "r") as f:
    for line in f:
        try:
            data = json.loads(line.strip())
            if "task_id" in data:
                completed_task_ids.add(data["task_id"])
        except json.JSONDecodeError:
            continue
print(f">>> Found {len(completed_task_ids)} completed samples.")

import re

def generate_code(prompt, max_length=512):
    full_prompt = f"{prompt}\n\n# Python function:\n"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    # extract generated part
    generated_ids = output[0][ inputs["input_ids"].shape[-1] : ]

    # 2) decode generation
    raw = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    # 3) split by lines，dump comment or empty lines
    lines = raw.splitlines()
    code_start = 0
    for i, line in enumerate(lines):
        # when a line starts with import、def、class ，or looks like x = ... , regard it as the beginning
        if re.match(r'^(import\s+\w+|class\s+\w+|def\s+\w+\s*\(.*\)\s*:|\w+\s*=)', line):
            code_start = i
            break

    # 4) concatenate the rest
    code_only = "\n".join(lines[code_start:]).rstrip()

    # free cache
    del inputs, output
    torch.cuda.empty_cache()
    gc.collect()

    return code_only

# main loop
print(">>> Starting task generation loop...")
for i in range(len(dataset)):
    example = dataset[i]
    task_id = example["task_id"]

    if task_id in completed_task_ids:
        print(f">>> Skipping task {task_id} (already completed)")
        continue

    prompt = example["text"]
    print(f">>> Generating for task {task_id} ({i+1}/{len(dataset)})")

    try:
        generated_code = generate_code(prompt)

        if not generated_code.strip():
            print(f">>> Empty result for task {task_id}, skipping.")
            continue

        # write to JSONL 
        with open(sample_file, "a") as f:
            json.dump({"task_id": task_id, "completion": generated_code}, f)
            f.write("\n")

        print(f">>> Saved completion for task {task_id}")

    except Exception as e:
        print(f">>> Error processing task {task_id}: {e}")
        continue

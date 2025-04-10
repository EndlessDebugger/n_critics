import os
import json
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from human_eval.evaluation import evaluate_functional_correctness

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 模型加载
model_path = "/scratch/user/lsc206573/nlp/Deepseek-Coder-V2-Lite-Instruct/"
tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", torch_dtype=torch.float16,local_files_only=True,trust_remote_code=True
)

print("Model loaded.")
model.generation_config.max_length = 600


# HumanEval 数据集
dataset = load_dataset("openai_humaneval")["test"]

# 文件路径
sample_file = "/scratch/user/lsc206573/generated_samples.jsonl"

# 加载已完成 task_id（断点续跑）
completed_task_ids = set()
if os.path.exists(sample_file):
    with open(sample_file, "r") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if "task_id" in data:
                    completed_task_ids.add(data["task_id"])
            except json.JSONDecodeError:
                continue
    print(f"Found {len(completed_task_ids)} completed samples.")

# 代码生成函数
def generate_code(prompt, max_length=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            use_cache=False,
        )
    code = tokenizer.decode(output[0], skip_special_tokens=True)

    # 释放显存和 Python 内存
    del inputs, output
    torch.cuda.empty_cache()
    gc.collect()

    return code

# 生成并写入 JSONL 文件
for i in range(len(dataset)):
    example = dataset[i]
    task_id = example["task_id"]

    if task_id in completed_task_ids:
        print(f"Skipping {task_id} (already completed)")
        continue

    prompt = example["prompt"]
    try:
        generated_code = generate_code(prompt)

        if not generated_code.strip():
            print(f"Empty result for {task_id}, skipping.")
            continue

        # 立即写入
        with open(sample_file, "a") as f:
            json.dump({"task_id": task_id, "completion": generated_code}, f)
            f.write("\n")

        print(f"Saved completion for {task_id} ({i+1}/{len(dataset)})")

    except Exception as e:
        print(f"Error processing {task_id}: {e}")
        continue

# 统一评测
print("Evaluating all completions...")
results = evaluate_functional_correctness(sample_file=sample_file)

# 输出 pass@1
pass_at_1 = results.get("pass@1", 0.0)
print(f"\nFinal Pass@1 Score: {pass_at_1:.2%}")

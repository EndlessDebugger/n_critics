import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
# from tqdm import tqdm
import pickle


import os

os.environ["TRANSFORMERS_CACHE"] = "/scratch/user/dortizchaves/n_critics/daniel"
os.environ["HF_HUB_CACHE"] = "/scratch/user/dortizchaves/n_critics/daniel"
os.environ["HF_HOME"] = "/scratch/user/dortizchaves/n_critics/daniel"


PROMPT_TEMPLATE = """### Instruction:
Please implement the python function as described below. Strictly return only the code. The final complete version of your function must not within a code block (without backticks) and should be immedately runnable. Do not explain any part of the solution\n
### Function:
    {prompt}

    ### Response:
    """


# from tqdm import tqdm

# Load model and tokenizer
model_name = "deepseek-ai/Deepseek-Coder-V2-Lite-Instruct"  # You can change to any decoder-only model
#dir_name = ".cache/huggingface/hub/models--Deepseek-Coder-V2-Lite-Instruct/snapshots/e434a23f91ba5b4923cf6c9d9a238eb4a08e3a11/"
#tokem_name = "./daniel/.cache/huggingface/hub/temp"
dir_name = "$SCRATCH/n_critics/.cache/huggingface/hub/models--Deepseek-Coder-V2-Lite-Instruct/"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./.cache", local_files_only=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./.cache", local_files_only=True, trust_remote_code=True, torch_dtype=torch.bfloat16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model = torch.compile(model)
model.eval()

print("Model loaded")
dataset = load_dataset("codeparrot/apps", split="test[:700]", cache_dir="./.cache")
print("dataset loaded")

#1)
# Function to generate and evaluate output
def generate_solution(batch, max_new_tokens=256):
   # prompt = PROMPT_TEMPLATE.format(prompt=problem_description["question"])
    
    prompts = [PROMPT_TEMPLATE.format(prompt=ex["question"]) for ex in batch]
     #   prompt+= f"\n{problem_description['input_ouput']}"
    encoded = tokenizer(prompts, return_tensors="pt", truncation=True, max_length=512, padding=True, return_attention_mask=True)
    input_ids = encoded["input_ids"].to(device)
    attn_mask = encoded["attention_mask"].to(device)
    #input_ids = tokenizer(problem_description, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.9,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
    generated_code = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return generated_code



generations = []
# counter = 0
checkpoint_file = "correct_justdeepseek_apps.pkl"
total_iterations = 700

batch_size = 4

try:
    with open(checkpoint_file, "rb") as f:
        generations = pickle.load(f)
    start = len(generations)
    print(f"resumed from checkpoint {start}")
except:
    start = 0
    print("no checkpoint")

# with tqdm(total=total_iterations, desc="Processing", unit="step") as pbar:
for i in range(start, total_iterations, batch_size):
    try:
        # counter += 1
        #first = dataset[i:i+batch_size]
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        batch = [dict(row) for row in batch]  # Make sure it's list-of-dict
        #generations.append([generate_solution(first)])
        results = generate_solution(batch)
        generations.extend(results)

        if len(generations) % batch_size*10 == 0:
            print(f"checkpoint saved at {len(generations)}")
            with open(checkpoint_file, "wb") as f:
                pickle.dump(generations, f)
            
        if len(generations) % batch_size*100 == 0:
            print(len(generations))
            # break  

    except StopIteration:
        break  # Stop when iteration is exhauste

with open(checkpoint_file, "wb") as f:
    pickle.dump(generations, f)

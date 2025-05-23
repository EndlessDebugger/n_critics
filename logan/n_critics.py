import time 
start_time = time.time()
print("Beginning imports...")
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
from contextlib import contextmanager
from tqdm import tqdm 
from itertools import islice
import pickle

from human_eval.evaluation import evaluate_functional_correctness
from human_eval.data import read_problems, write_jsonl
from ncriticstask import NCriticsTask
delta_t = time.time() - start_time
print(f"Completed imports in {delta_t} seconds.") 

def load_tasks(init_prompt = "DO NOT include ANY comments, test cases, or any description. ONLY RUNNABLE CODE.\n", num_samples=1) -> list[NCriticsTask]:
    """Initializes a list of NCriticsTask objects from the problem set."""
    task_list = []
    problems = read_problems()
    # problems = dict(islice(problems.items(), 8))  # DEBUG
    for task_id, task in problems.items():
        for _ in range(num_samples):
            task_obj = NCriticsTask(id=task_id, problem_statement=init_prompt + task["prompt"])
            task_list.append(task_obj)
    task_list *= num_samples  # Duplicate tasks for the number of samples
    return task_list 


@contextmanager
def load_model(model_name: str):
    """Load model temporarily on GPU with FP16, unload after use."""
    cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    trc = False
    if "deepseek" in model_name.lower():
        trc = True
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=trc)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, cache_dir=cache_dir, trust_remote_code=trc)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    tokenizer.padding_side = "left"
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    try:
        yield model, tokenizer
    finally:
        model.cpu()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def extract_between_triple_quotes(text: str):
    start = text.find("```") + 3  # Find start pos after first set of triple quotes + python\n. Model includes this to indicate type of code ig.
    end = text.find("```", start)  # Find end pos at next set of triple quotes
    new_text = text[start:end].strip()  # Remove whitespace
    if new_text.startswith("python"):
        new_text = new_text[6:]
    return new_text.strip()
    

def generate_response(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompts: list, max_length=2048):

    chat_prompts = [[{"role": "user", "content": p}] for p in prompts]
    
    inputs = tokenizer.apply_chat_template(
        chat_prompts,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=8192  # TODO: change depending on model type instead of hard-coding
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            inputs,
            max_new_tokens=max_length,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            use_cache=True  # speed boost
        )

    input_lens = inputs.shape[1]  # inputs is of shape [batch_size, sequence_len], sequence_len same for all bc of padding. So this is an integer
    responses = [
        tokenizer.decode(output[input_lens:], skip_special_tokens=True)  # get all model outputs with inputs removed 
        for output in output_ids
    ]
    responses = [extract_between_triple_quotes(r) for r in responses]
    return responses


def safe_generate_primary_responses(model_name: str, tasks: list[NCriticsTask], max_length: int = 1500, batch_size: int = 4):
    """Generate responses with dynamic batch adjustment on OOM."""
    with load_model(model_name) as (model, tokenizer):
        total_tasks = len(tasks)
        i = 0
        
        # Initialize tqdm progress bar
        with tqdm(total=total_tasks, desc="Generating responses") as pbar:
            all_prompts = [] # DEBUG
            while i < total_tasks:
                try:
                    end = min(i + batch_size, total_tasks)
                    batch = tasks[i:end]
                    prompts = [task.prompt for task in batch]
                    all_prompts.extend(prompts) # DEBUG
                    responses = generate_response(model, tokenizer, prompts=prompts, max_length=max_length)
                    
                    # Assign responses to tasks and update progress
                    for task, response in zip(batch, responses):
                        task.model_response = response
                    i += batch_size
                    
                    # Update tqdm progress bar
                    pbar.update(batch_size)
                    
                    # Clean up after successful batch
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                except torch.cuda.OutOfMemoryError:
                    if batch_size <= 1: 
                        with open('example_NCriticsTask.pkl', 'wb') as f:
                            pickle.dump(tasks[i], f)
                        raise torch.cuda.OutOfMemoryError(f"safe_generate_primary_response failed even with batch size = {batch_size}. Prompt lengths were {[len(prompt) for prompt in all_prompts]}.")
                    else: 
                        print(f"OOM at batch starting index {i}. Reducing batch size from {batch_size} to {batch_size // 2}")
                        torch.cuda.empty_cache()
                        gc.collect()
                        batch_size = max(1, batch_size // 2)


def get_critiques(critic_models: list[str], tasks: list[NCriticsTask], batch_size: int):
    """Engage multiple LLMs to obtain an ensemble of critiques."""
     # clear previous critic responses 
    for task in tasks: 
        task.critic_responses = []
    for critic_name in critic_models:
        with load_model(critic_name) as (model, tokenizer):
            for i in tqdm(range(0, len(tasks), batch_size), desc="Getting critiques"):
                batch = tasks[i:i + batch_size]
                critique_prompts = []
                for task in batch:
                    # generate critique prompt
                    critique_prompt = f"Given the following problem statement:\n\n{task.problem_statement}\n\n"
                    critique_prompt += f"Please critique the following response:\n\n{task.model_response}\n\n"
                    critique_prompt += "In 5 sentences or less, address the following:\n"
                    critique_prompt += "1. Does the response correctly and fully address the problem given in the prompt?\n"
                    critique_prompt += "2. Is the response time-optimal?\n"
                    critique_prompt += "3. Is the response memory-optimal?\n"
                    critique_prompt += "4. Does the output have any syntax errors?\n"
                    critique_prompt += "If the response is fully satisfactory, print nothing but the following line: 'The response is fully satisfactory.'\n\n"
                    critique_prompts.append(critique_prompt)
                # generate critic responses 
                critique = generate_response(model, tokenizer, prompts=critique_prompts, max_length=256)
                # add to task object 
                for batch, response in zip(batch, critique):
                    task.critic_responses.append(response)


def refine_prompts(tasks: list[NCriticsTask]):
    """Refine the primary model's response based on critiques."""
    for task in tqdm(tasks, desc=f"Refining prompts"): 
        refined_prompt = f"Given the problem: \n\n{task.problem_statement}\n\n"
        refined_prompt += f"Your answer was: \n\n## YOUR CODE RESPONSE ##\n{task.model_response}\n\n"
        refined_prompt += f"Critiques: \n\n## CRITIQUES ##\n"
        for critique in task.critic_responses:
            refined_prompt += critique + '\n'
        refined_prompt += '\n'
        refined_prompt += "Update your response. DO NOT INCLUDE comments, test cases, explanations. Provide ONLY runnable code!"
        task.prompt = refined_prompt


def evaluate_n_critics(output_filename: str):
    """Evaluate the final responses of the primary model."""
    # Prevent transformers' parallelization from messing up human-eval's parallelization
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # evaluate the model inferences using the HumanEval evaluation script
    results = evaluate_functional_correctness(sample_file=output_filename)
    pass_at_1 = results["pass@1"]
    return pass_at_1
    

def write_to_jsonl(tasks: list[NCriticsTask], iter_num: int):
    # write model inferences to a jsonl file
    folder = "results"
    if not os.path.exists(folder):
        os.mkdir(folder)
    fname = "iter_" + str(iter_num) + ".jsonl"
    out_path = os.path.join(folder, fname) 
    results_as_dict_list = [task.as_dict() for task in tasks]
    write_jsonl(out_path, results_as_dict_list)
    return out_path


def n_critics_algorithm(primary_model: str,  
                        critic_models: list, 
                        initial_prompt: str = "", 
                        max_iterations: int = 1, 
                        num_samples: int = 1,
                        batch_size: int = 8) -> float:
    """
    N-Critics algorithm implementation.

    :param primary_model (str): name of the primary model to use for generating responses. Must be valid HuggingFace model name.
    :param critic_models (list): names of the critic models to use for critiques. Must be valid HuggingFace model names.
    :param initial_prompt (str): initial prompt to use for generating responses. 
    :param max_iterations (int): maximum number of refinement iterations. 
    :param num_samples (int): number of samples to generate for each programming problem statement.
    :param batch_size (int): batch size for increased processing speed. (Higher GPU memory usage.)

    :return (float): final pass@1 score of the primary model after all iterations.
    """
    # load NCriticsTask objects from the problem set 
    tasks = load_tasks(initial_prompt, num_samples)

    # Generate initial responses for each task
    safe_generate_primary_responses(primary_model, tasks, batch_size=batch_size)
    
    # evaluation files 
    eval_filenames = []

    # include initial responses, with no critiques 
    filename = write_to_jsonl(tasks, 0)
    eval_filenames.append(filename) 

    for i in range(max_iterations):
        print('#'*50) 
        print(f"Iteration {i+1} of {max_iterations}")
        print('#'*50)
        get_critiques(critic_models, tasks, batch_size=batch_size)
        refine_prompts(tasks)
        safe_generate_primary_responses(primary_model, tasks, batch_size=batch_size)
        filename = write_to_jsonl(tasks, i+1)
        eval_filenames.append(filename) 

    scores = [evaluate_n_critics(fname) for fname in eval_filenames]
    return scores


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the N-Critics algorithm.")
    parser.add_argument("--max_iterations", type=int, default=1, help="Number of refinement iterations. Enter 1-8.")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples per problem. Enter 1-10.")
    args = parser.parse_args()
    
    if args.max_iterations < 1 or args.max_iterations > 8: 
      raise ValueError("Value of max_iterations must be between 1 and 8.")
    
    if args.num_samples < 1 or args.num_samples > 10: 
      raise ValueError("Value of num_samples must be between 1 and 8.")
    
    primary_model = "deepseek-ai/Deepseek-Coder-V2-Lite-Instruct"
    critic_models = ["google/gemma-3-12b-it", "meta-llama/Llama-3.2-3B-Instruct"]
    initial_prompt = "Complete the following programming problem: \n"

    scores = n_critics_algorithm(primary_model, 
                                critic_models, 
                                initial_prompt=initial_prompt, 
                                max_iterations=args.max_iterations,
                                num_samples=args.num_samples)
    scores_filename = os.path.join("results", "pass1scores.txt")
    with open(scores_filename, "w") as file:
        print("\n Final Pass@1 Scores: ")
        for i, score in enumerate(scores):
            print(f"iteration {i}: {score:.2f}")
            file.write(f"iteration {i}: {score:.2f}")

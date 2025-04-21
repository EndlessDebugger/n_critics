import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
from contextlib import contextmanager
from tqdm import tqdm 

from human_eval.evaluation import evaluate_functional_correctness
from human_eval.data import read_problems, write_jsonl
from ncriticstask import NCriticsTask

def load_tasks(init_prompt = "", num_samples=1) -> list[NCriticsTask]:
    """Initializes a list of NCriticsTask objects from the problem set."""
    task_list = []
    problems = read_problems()
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
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    try:
        yield tokenizer, model
    finally:
        model.cpu()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    

def generate_response(model: AutoModelForCausalLM, tokenizer:AutoTokenizer, prompt: str | list = "", max_length=256):
    """Generate a response using the model and tokenizer."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)  
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_length, do_sample=True, top_p=0.9, temperature=0.7
        )
    if type(prompt) == str:
        return tokenizer.decode(output[0], skip_special_tokens=True)
    elif type(prompt) == list: 
        return tokenizer.batch_decode(output, skip_special_tokens=True)
    else:
        raise ValueError("Prompt must be a string or a list of strings.")


def generate_primary_responses(model_name: str, tasks: list[NCriticsTask], batch_size: int, max_length=256):
    """Generate responses for a list of tasks."""
    with load_model(model_name) as (tokenizer, model):
        for i in tqdm(range(0, len(tasks), batch_size), desc=f"Generating with {model_name}"):
            batch = tasks[i:i + batch_size]
            prompts = [task.prompt for task in batch]
            responses = generate_response(model, tokenizer, prompt=prompts, max_length=max_length)
            for task, response in zip(batch, responses):
                task.model_response = response


def get_critiques(critic_models: list[str], tasks: list[NCriticsTask], batch_size: int):
    """Engage multiple LLMs to obtain an ensemble of critiques."""
     # clear previous critic responses 
    for task in tasks: 
        task.critic_responses = []
    for critic_name in critic_models:
        with load_model(critic_name) as (tokenizer, critic):
            for i in tqdm(range(0, len(tasks), batch_size), desc="Getting critiques"):
                batch = tasks[i:i + batch_size]
                critique_prompts = []
                for task in batch:
                    # generate critique prompt
                    critique_prompt = f"Given the following problem statement:\n\n{task.problem_statement}\n\n"
                    critique_prompt += f"Please critique the following response:\n\n{task.model_response}\n\n"
                    critique_prompt += "In five sentences or less, address the following:\n"
                    critique_prompt += "1. Does the response correctly and fully address the problem given in the prompt?\n"
                    critique_prompt += "2. Is the response time-optimal?\n"
                    critique_prompt += "3. Is the response memory-optimal?\n"
                    critique_prompt += "If the response is fully satisfactory, print nothing but the following line: 'The response is fully satisfactory.'\n\n"
                    critique_prompts.append(critique_prompt)
                # generate critic responses 
                critique = generate_response(critic, tokenizer, prompt=critique_prompts)
                # add to task object 
                for batch, response in zip(batch, critique):
                    task.critic_responses.append(response)


def refine_prompts(tasks: list[NCriticsTask]):
    """Refine the primary model's response based on critiques."""
    for task in tqdm(tasks, desc=f"Refining prompts"): 
        refined_prompt = f"You were originally tasked with writing code to solve the following programming problem: \n\n{task.problem_statement}\n\n"
        refined_prompt += f"You generated the following code: \n\n## YOUR CODE RESPONSE ##\n{task.model_response}\n\n"
        refined_prompt += f"Your answer received the following critiques: \n\n## CRITIQUES ##\n"
        for critique in task.critic_responses:
            refined_prompt += critique + '\n'
        refined_prompt += '\n'
        refined_prompt += "Update your response based on the critiques."
        task.prompt = refined_prompt


def evaluate_n_critics(tasks: list[NCriticsTask], output_filename: str):
    """Evaluate the final responses of the primary model."""
    # write model inferences to a jsonl file
    if not output_filename.endswith(".jsonl"):
        output_filename += ".jsonl"
    results_as_dict_list = [task.as_dict() for task in tasks]
    write_jsonl(output_filename, results_as_dict_list)
    # evaluate the model inferences using the HumanEval evaluation script
    results = evaluate_functional_correctness(sample_file=output_filename)
    pass_at_1 = results["pass@1"]
    return pass_at_1



def n_critics_algorithm(primary_model: str,  
                        critic_models: list, 
                        initial_prompt: str = "", 
                        max_iterations: int = 1, 
                        num_samples: int = 1,
                        out_filename: str = "n_critics_results.jsonl") -> float:
    """
    N-Critics algorithm implementation.

    :param primary_model (str): name of the primary model to use for generating responses. Must be valid HuggingFace model name.
    :param critic_models (list): names of the critic models to use for critiques. Must be valid HuggingFace model names.
    :param initial_prompt (str): initial prompt to use for generating responses. 
    :param max_iterations (int): maximum number of refinement iterations. 
    :param num_samples (int): number of samples to generate for each programming problem statement.
    :param out_filename (str): name of the output file to save the results. Must be a valid JSONL filename, 
    ending in .jsonl or with no file extension. 

    :return (float): final pass@1 score of the primary model after all iterations.
    """
    # load NCriticsTask objects from the problem set 
    tasks = load_tasks(initial_prompt, num_samples)

    # Generate initial responses for each task
    generate_primary_responses(primary_model, tasks)

    for i in range(max_iterations):
        print('#'*50) 
        print(f"Iteration {i+1} of {max_iterations}")
        print('#'*50)
        get_critiques(critic_models, tasks)
        refine_prompts(tasks)
        generate_primary_responses(primary_model, tasks)

    score = evaluate_n_critics(tasks, out_filename)
    return score


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the N-Critics algorithm.")
    parser.add_argument("--max_iterations", type=int, default=1, help="Number of refinement iterations. Enter 1-8.")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples per problem. Enter 1-10.")
    parser.add_argument("--out_filename", type=str, default="n_critics_results.jsonl", help="Output filename. Must have .jsonl extension or no extension.")
    args = parser.parse_args()
    
    if args.max_iterations < 1 or args.max_iterations > 8: 
      raise ValueError("Value of max_iterations must be between 1 and 8.")
    
    if args.num_samples < 1 or args.num_samples > 10: 
      raise ValueError("Value of num_samples must be between 1 and 8.")
    
    primary_model = "deepseek-ai/Deepseek-Coder-V2-Lite-Instruct"
    critic_models = ["google/gemma-3-12b-it", "meta-llama/Llama-3.2-3B-Instruct"]
    initial_prompt = "Complete the following programming problem: \n"
    problems = read_problems()

    score = n_critics_algorithm(primary_model, 
                                critic_models, 
                                initial_prompt=initial_prompt, 
                                max_iterations=args.max_iterations,
                                num_samples=args.num_samples,
                                out_filename=args.out_filename)
    print(f"\n Final Pass@1 Score: {score:.2%}")

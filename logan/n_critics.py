import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
from contextlib import contextmanager

from human_eval.human_eval.evaluation import evaluate_functional_correctness
from human_eval.human_eval.data import read_problems, write_jsonl
from task import Task

def load_tasks(init_prompt = "", num_samples=1) -> list[Task]:
    """Initializes a list of Task objects from the problem set."""
    task_list = []
    problems = read_problems()
    for task_id, task in problems.items():
        for _ in range(num_samples):
            task_obj = Task(id=task_id, problem_statement=init_prompt + task["prompt"])
            task_list.append(task_obj)
    task_list *= num_samples  # Duplicate tasks for the number of samples
    return task_list 


@contextmanager
def load_model(model_name: str):
    """Load model temporarily on GPU with FP16, unload after use."""
    cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True, cache_dir=cache_dir)
    )
    try:
        yield tokenizer, model
    finally:
        model.cpu()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def generate_response(model: AutoModelForCausalLM, tokenizer:AutoTokenizer, prompt="", max_length=256):
    """Generate a response using the model and tokenizer."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)  
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_length, do_sample=True, top_p=0.9, temperature=0.7
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def generate_primary_responses(model_name: str, tasks: list[Task], max_length=256):
    """Generate responses for a list of tasks."""
    with load_model(model_name) as (tokenizer, model):
        for task in tasks:
            task.model_response = generate_response(model, tokenizer, prompt=task.prompt, max_length=max_length)


def get_critiques(critic_models: list[str], tasks: list[Task]):
    """Engage multiple LLMs to obtain an ensemble of critiques."""
     # clear previous critic responses 
    for task in tasks: 
        task.critic_responses = []
    for critic_name in critic_models:
        with load_model(critic_name) as (tokenizer, critic):
            for task in tasks:
            # generate critique prompt
                critique_prompt = f"Given the following problem statement:\n\n{task.problem_statement}\n\n"
                critique_prompt += f"Please critique the following response:\n\n{task.model_response}\n\n"
                critique_prompt += "In five sentences or less, address the following:\n"
                critique_prompt += "1. Does the response correctly and fully address the problem given in the prompt?\n"
                critique_prompt += "2. Is the response time-optimal?\n"
                critique_prompt += "3. Is the response memory-optimal?\n"
                critique_prompt += "If the response is fully satisfactory, print nothing but the following line: 'The response is fully satisfactory.'\n\n"
                # generate critic responses 
                critique = generate_response(critic, tokenizer, prompt=critique_prompt)
                task.critic_responses.append(critique)


def refine_prompts(tasks: list[Task]):
    """Refine the primary model's response based on critiques."""
    for task in tasks: 
        refined_prompt = f"You were originally tasked with writing code to solve the following programming problem: \n\n{task.problem_statement}\n\n"
        refined_prompt += f"You generated the following code: \n\n## YOUR CODE RESPONSE ##\n{task.model_response}\n\n"
        refined_prompt += f"Your answer received the following critiques: \n\n## CRITIQUES ##\n"
        for critique in task.critic_responses:
            refined_prompt += critique + '\n'
        refined_prompt += '\n'
        refined_prompt += "Update your response based on the critiques."
        task.prompt = refined_prompt


def evaluate_n_critics(tasks: list[Task], output_filename: str = "n_critics_results.jsonl"):
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



def n_critics_algorithm(primary_model, critic_models: list, initial_prompt: str = "", max_iterations=4, num_samples=1):
    """N-Critics algorithm implementation."""
    # load Task objects from the problem set 
    tasks = load_tasks(initial_prompt, num_samples)

    # Generate initial responses for each task
    generate_primary_responses(primary_model, tasks)

    for i in range(len(max_iterations)):
        get_critiques(critic_models, tasks)
        refine_prompts(tasks)
        generate_primary_responses(primary_model, tasks)

    score = evaluate_n_critics(tasks)
    return score


# Example usage
if __name__ == "__main__":
    primary_model = "deepseek-ai/Deepseek-Coder-V2-Lite-Instruct"
    critic_models = ["gemma", "llama3"]
    initial_prompt = "Complete the following programming problem: \n"
    problems = read_problems()

    score = n_critics_algorithm(primary_model, 
                                critic_models, 
                                initial_prompt=initial_prompt, 
                                max_iterations=4,
                                num_samples=1)
    print(f"\n Final Pass@1 Score: {score:.2%}")

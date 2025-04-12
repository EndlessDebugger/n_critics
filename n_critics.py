# import os
# import json
# from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc
from contextlib import contextmanager

@contextmanager
def load_model(model_name):
    """Load model temporarily on GPU with FP16, unload after use."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.float16
    )
    try:
        yield tokenizer, model
    finally:
        model.cpu()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def generate_response(model_name, prompt, max_length=256):
    with load_model(model_name) as (tokenizer, model):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)  
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=max_length, do_sample=True, top_p=0.9, temperature=0.7
            )
        return tokenizer.decode(output[0], skip_special_tokens=True)

def get_critiques(models, prompt):
    """Engage multiple LLMs to obtain an ensemble of critiques."""
    critiques = []
    for model in models:
        #Example prompt, will probably change with list later
        critique_prompt = f"Please critique the following response:\n\n{prompt}"
        critique = generate_response(model, critique_prompt)
        critiques.append(critique)
    return critiques

def refine_prompt(orig_response, orig_prompt, critiques):
    """Refine the primary model's response based on critiques."""
    refinement_prompt = f"You were originally tasked with writing code to solve the following programming problem: \n\n{orig_prompt}\n\n"
    refinement_prompt += f"You generated the following code: \n\n## YOUR CODE RESPONSE ##\n{orig_response}\n\n"
    refinement_prompt += f"Your answer received the following critiques: \n\n"
    for critique in critiques:
      refinement_prompt += critique + '\n'
    refinement_prompt += '\n'
    refinement_prompt += "Please update your response based on the critiques your answer received."
    return refinement_prompt

def n_critics_algorithm(primary_model, critic_models, initial_prompt, max_iterations=4):
    """N-Critics algorithm implementation."""
    output = generate_response(primary_model, initial_prompt)
    i = 0

    while i < max_iterations:
        critiques = get_critiques(critic_models + [primary_model], output)

        # Check if critiques suggest the output is satisfactory
        if "The response is fully satisfactory" in " ".join(critiques).lower():
            return output
        
        refined_prompt = refine_prompt(output, critiques)
        output = generate_response(primary_model, refined_prompt)

        i += 1

    return output  # Return the final refined output


# Example usage
if __name__ == "__main__":
    primary_model = "deepseek-ai/Deepseek-Coder-V2-Lite-Instruct"
    critic_models = ["gemma", "llama3"]
    initial_prompt = "Explain quantum mechanics in simple terms."

    refined_output = n_critics_algorithm(primary_model, critic_models, initial_prompt)
    print(refined_output)

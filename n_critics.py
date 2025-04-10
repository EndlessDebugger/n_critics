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

def refine_prompt(prompt, critiques):
    """Refine the prompt based on critiques."""
    refinement_prompt = f"Refine the following response based on these critiques:\n\nCritiques:\n{critiques}\n\nOriginal response:\n{prompt}"
    return refinement_prompt

def n_critics_algorithm(primary_model, critic_models, initial_prompt, max_iterations=4):
    """N-Critics algorithm implementation."""
    output = generate_response(primary_model, initial_prompt)
    i = 0

    while i < max_iterations:
        critiques = get_critiques(critic_models + [primary_model], output)

        # Check if critiques suggest the output is satisfactory
        if "satisfactory" in " ".join(critiques).lower():
            return output
        
        refined_prompt = refine_prompt(output, critiques)
        output = generate_response(primary_model, refined_prompt)

        i += 1

    return output  # Return the final refined output

# Example usage
primary_model = "deepseek-ai/Deepseek-Coder-V2-Lite-Instruct"
critic_models = ["gemma", "llama3"]
initial_prompt = "Explain quantum mechanics in simple terms."

refined_output = n_critics_algorithm(primary_model, critic_models, initial_prompt)
print(refined_output)

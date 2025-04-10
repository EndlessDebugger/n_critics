import ollama

def generate_response(model, prompt):
    """Generate a response from a specified LLM model using Ollama."""
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

def get_critiques(models, prompt):
    """Engage multiple LLMs to obtain an ensemble of critiques."""
    critiques = []
    for model in models:
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
primary_model = "mistral"
critic_models = ["gemma", "llama3"]
initial_prompt = "Explain quantum mechanics in simple terms."

refined_output = n_critics_algorithm(primary_model, critic_models, initial_prompt)
print(refined_output)

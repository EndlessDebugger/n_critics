from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "deepseek-ai/Deepseek-Coder-V2-Lite-Instruct"
save_path = "./Deepseek-Coder-V2-Lite-Instruct"

# 下载并保存 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
tokenizer.save_pretrained(save_path)

# 下载并保存模型
model = AutoModelForCausalLM.from_pretrained(model_name)
model.save_pretrained(save_path)

print(f"Model and tokenizer saved to: {save_path}")


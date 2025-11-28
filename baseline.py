from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1",use_fast=False)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

prompt = "<s>[INST] What's your name? [/INST]"
input_ids = tok(prompt, return_tensors="pt").input_ids
out = model.generate(input_ids, max_new_tokens=100)

print(tok.decode(out[0], skip_special_tokens=False))

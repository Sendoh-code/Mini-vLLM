from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1",use_fast=False)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

prompt = "<s>[INST] What's your name? [/INST]"
data = ["where are you from?","Hi, model, My name is Jack the junior, what's your name?","I come from China the eastern country, where are you from?","what's your name?"]
input_ids = [tok("<s>[INST]"+ i + "[/INST]", return_tensors="pt").input_ids for i in data]
for i in input_ids:
    out = model.generate(i)
    print(tok.decode(out[0], skip_special_tokens=False))

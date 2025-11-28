from model_runner import ModelRunner
from utils import slice_kv, gather_kv,build_prompt
import torch
import time

class MiniVLLMEngine:
    def __init__(self, model_name:str):
        self.runner = ModelRunner(model_name)
        self.tokenizer = self.runner.tokenizer
        self.next_request_id = 0

        # 全局状态
        self.KVManager = {}          # req_id -> past_kv
        self.TokenManager = {}       # req_id -> token_ids
        self.RequestTable = {}       # req_id -> metadata (prompt, finished flag, etc.)

    def register_request(self, prompt: str):
        req_id = self.next_request_id
        self.next_request_id += 1

        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.runner.device)

        # 初始化全局状态
        self.KVManager[req_id] = None           # prefill 后才会有
        self.TokenManager[req_id] = prompt_ids[0].tolist()  # prompt token_ids
        self.RequestTable[req_id] = {
            "prompt": prompt,
            "finished": False
        }

        return req_id

    def prefill(self, req_ids:list):
        input_text = [self.RequestTable[req_id]['prompt'] for req_id in req_ids]
        next_tokens, kvcaches = self.runner.forward_prefill(input_batch=input_text)

        for i,req_id in enumerate(req_ids):
            self.TokenManager[req_id].append(int(next_tokens[i]))
            self.KVManager[req_id] = slice_kv(kvcaches,i)

    def decode_step(self, req_ids:list):
        input_nextids = torch.tensor(
            [self.TokenManager[req_id][-1] for req_id in req_ids],
            device=self.runner.device
        )
        inputcache = gather_kv(req_ids,self.KVManager)

        next_tokens, kvcaches = self.runner.forward_decode(input_nextids,inputcache)

        for i,req_id in enumerate(req_ids):
            self.TokenManager[req_id].append(int(next_tokens[i]))
            self.KVManager[req_id] = slice_kv(kvcaches,i)


if __name__=='__main__':
    test = MiniVLLMEngine("mistralai/Mistral-7B-Instruct-v0.1")
    prompts = ["Hi, Where are you from?","what's your name?"]
    prompts = [build_prompt(i) for i in prompts]
    req_ids = [test.register_request(i) for i in prompts]
    print(f'req_ids{req_ids}')
    test.prefill(req_ids)
    for i in range(100):
        test.decode_step(req_ids)
    token_list = test.TokenManager[0]
    text = test.tokenizer.decode(token_list, skip_special_tokens=False)
    print(f"raw tokens:{token_list}")
    print(f"text:{text}")
    #print(test.runner.tokenizer(build_prompt("what's your name?"), return_tensors="pt").attention_mask)
    #print(test.runner.tokenizer(build_prompt("Where are you from?"), return_tensors="pt").attention_mask)
    token_list = test.TokenManager[1]
    text = test.tokenizer.decode(token_list, skip_special_tokens=False)
    print(f"raw tokens:{token_list}")
    print(f"text:{text}")


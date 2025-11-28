from collections import deque
from engine import MiniVLLMEngine
import random


class MiniScheduler:
    def __init__(self,model_name = "mistralai/Mistral-7B-Instruct-v0.1"):
        self.prefill_queue = deque()
        self.decode_queue = deque()
        self.waiting_queue = deque()
        self.finished = []
        self.engine = MiniVLLMEngine(model_name=model_name)

    def _add_new(self, prompt:str):
        self.waiting_queue.append(prompt)

    def _add_toprefill(self, prompt:str):
        """为请求分配内存，记录KVcache之后才能进入prefillqueue"""
        req_id = self.engine.register_request(prompt)
        self.prefill_queue.append(req_id)

    def _add_todecode(self, req_id:str):
        self.decode_queue.append(req_id)

    def prepare(self,prompts:list[str]):
        for req in prompts:
            self._add_new(req)

    def to_prefill(self):
        batch_size = random.randint(4,6)
        batch_size = min(batch_size,len(self.waiting_queue))
        for _ in range(batch_size):
            self._add_toprefill(self.waiting_queue.popleft())

    def to_decode(self,req_batch:list[str]):
        for req in req_batch:
            self._add_todecode(req)

    def batch_prefill(self):
        batch_size = random.randint(4,6) #实际上的每次batch prefill的batch size取决于可用的KVcache块，这里由于我们无法操作内存，用random模拟动态batch
        batch_size = min(batch_size,len(self.prefill_queue))
        batch = []
        for _ in range(batch_size):
            batch.append(self.prefill_queue.popleft())
        if len(batch)!=0:
            self.engine.prefill(batch)
            self.to_decode(batch)
             
    
    def batch_decode_step(self,req_batch):
        self.engine.decode_step(req_batch)

    def next_batch(self,req_batch:list[str]):
        """在一步decode之后检查已经结束的推理，在下一batch中移除这些并补上新的"""
        finish_count = 0
        new_batch = []
        for req_id in req_batch:
            if self.engine.TokenManager[req_id][-1]==2:
                # 输出结果，移出batch
                self.finished.append(req_id)
                finish_count+=1
            else:
                new_batch.append(req_id)
        finish_count = min(finish_count,len(self.decode_queue))
        for _ in range(finish_count):
            new_batch.append(self.decode_queue.popleft())
        return new_batch

if __name__=='__main__':
    test = MiniScheduler()
    raw_prompt_list = ["what's your name?","where are you from?","what's your name?","where are you from?"]
    
    test.prepare(raw_prompt_list)
    if test.waiting_queue:
        test.to_prefill()
    test.batch_prefill()
    
    new_batch = [test.decode_queue.popleft() for _ in range(4)]
    

    while len(test.finished)!=len(raw_prompt_list):
        test.batch_decode_step(new_batch)
        if test.waiting_queue:
            test.to_prefill()
        test.batch_prefill()
        
        new_batch = test.next_batch(new_batch)
        
    count = 0
    for i in test.finished:
        token_list = test.engine.TokenManager[i]
        text = test.engine.tokenizer.decode(token_list, skip_special_tokens=False)
        print(f"{count} task----------------------------")
        print(f"raw tokens:{token_list}")
        print(f"text:{text}")
        count += 1
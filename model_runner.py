import torch # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM # type: ignore
from kv_cache import KVCacheManager

class ModelRunner:
    """使用model()重写transformer的generate方法"""
    # mistralai/Mistral-7B-v0.1
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.1"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            )
        self.model.to(self.device)
        self.model.eval()
    

    def forward_prefill(self, input_batch:list[str]):
        """接受prompt，计算并返回所有KVcache->next_ids, past_kv"""
        # tokenize
        encoded = self.tokenizer(
            input_batch,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(self.device)
        input_ids=encoded.input_ids
        attention_mask=encoded.attention_mask

        # inference
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True
            )
        logits = outputs.logits
        past_kv = outputs.past_key_values

        next_ids = torch.argmax(logits[:,-1,:],dim=-1)
        return next_ids, past_kv


        

    def forward_decode(self, token_id, past_kv):
        """借助prefill时得到的KVcache，对prompt后续的token进行推理"""
        input_ids = token_id.unsqueeze(1)
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_kv,
            use_cache=True
        )
        logits = outputs.logits
        past_kv = outputs.past_key_values

        next_ids = torch.argmax(logits[:,-1,:],dim=-1)
        return next_ids, past_kv
        


if __name__=='__main__':
    test = ModelRunner()
    prompts = ["what's your name?","where are you from?"]
    a,b = test.forward_prefill(prompts)
    c,d = test.forward_decode(a,b)
    print("Here's test results")
    past_kv=b
    print("KV TYPE:", type(past_kv))
    print("LAYER TYPE:", type(past_kv.layers[0]))
    print("ATTRIBUTES:", dir(past_kv.layers[0]))

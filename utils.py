import torch
def build_prompt(user_text):
    return f"<s>[INST] {user_text} [/INST]"

def slice_kv(past_kv, i):
    """
    从 batched past_key_values 中切出第 i 条请求的 KV。

    参数:
        past_kv: tuple of (k, v)
            例如: ((k0, v0), (k1, v1), ..., (kL, vL))
        i: int
            要提取的 batch index

    返回:
        single_kv: tuple of (k_i, v_i)  # batch=1
    """
    sliced = []
    for (k, v) in past_kv:
        # k,v shape = (batch, heads, seq_len, head_dim)
        k_i = k[i:i+1].clone()  # 保留 batch 维度为 1
        v_i = v[i:i+1].clone()
        sliced.append((k_i, v_i))
    return tuple(sliced)

def gather_kv(req_ids, KVManager):
    """
    将多个请求的单独 KV cache 合并成一个 batch 的 KV。

    参数:
        req_ids: list[int]
            组合成 batch 的 request id 顺序
        KVManager: dict
            req_id -> past_kv（tuple of (k,v)）

    返回:
        batched_kv: tuple of (k_cat, v_cat) for each layer
    """
    # 获取层数
    first_kv = KVManager[req_ids[0]]
    num_layers = len(first_kv)

    batched = []

    for layer_idx in range(num_layers):
        k_list = []
        v_list = []

        for req_id in req_ids:
            k, v = KVManager[req_id][layer_idx]
            k_list.append(k)  # 每个 k shape: (1, heads, seq_len, dim)
            v_list.append(v)

        # 拼 batch: (N requests, heads, seq_len, dim)
        k_cat = torch.cat(k_list, dim=0)
        v_cat = torch.cat(v_list, dim=0)

        batched.append((k_cat, v_cat))

    return tuple(batched)

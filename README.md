# 🚀 mini-vLLM — A Minimal yet Functional vLLM-Style Inference Engine

**mini-vLLM** is a lightweight educational re-implementation of the core ideas behind the vLLM inference engine.
It focuses on **understanding**, not performance, and faithfully reproduces the essential mechanisms of modern high-performance LLM inference:

* **Prefill + incremental decode** execution
* **KV-cache construction, slicing, and gathering**
* **Batch inference over multiple requests**
* **ChatML-formatted prompts for Instruct models**
* **Global token/KV/request state management**
* **A scheduler-like control flow (request lifecycle management)**

The codebase avoids CUDA kernels, PagedAttention, memory paging, and other optimizations so that the core logic remains transparent and easy to learn.

---

# ✨ Features

### ✔ Prefill (full-sequence forward pass)

Runs a full forward pass to initialize:

* per-request KV caches
* the next token for each request

One prefill per request, just like vLLM.

---

### ✔ Incremental decoding (one token at a time)

Each decode step receives:

* the last token for each active request
* the gathered KV cache for the batch

and returns:

* one new token per request
* updated KV cache tensors (32 layers for Mistral-7B)

---

### ✔ KV-cache management (slice + gather)

Fully supports the Transformers 4.36 **tuple-based KV format**:

* `slice_kv()` extracts the KV tensors for a single request
* `gather_kv()` stacks multiple KV caches into a batch layout

This mirrors vLLM’s logical KV-management behavior (without paging).

---

### ✔ ChatML prompt formatting (Instruct-model-friendly)

All user prompts are wrapped as:

```
<s>[INST] user_message [/INST]
```

This is necessary to make Mistral Instruct behave like a chat assistant instead of generating Q/A lists or essays.

---

### ✔ Global state tracking

Three centralized state stores:

* **TokenManager:** `req_id → token_id list`
* **KVManager:** `req_id → tuple-of-KV tensors`
* **RequestTable:** metadata (prompt, finished flag, etc.)

This design matches real LLM serving architectures.

---

### ✔ Scheduler-style control flow

The engine implements a simplified scheduler that:

* tracks active vs. finished requests
* checks EOS tokens
* updates token/KV states each iteration
* reconstructs batch KV with `gather_kv()`

This architecture is intentionally compatible with full vLLM-style dynamic batching.

---

# 📁 Project Structure

```
mini-vllm/
│
├── model_runner.py      # Pure inference: prefill() + decode_step()
├── utils.py             # KV slicing/gathering utilities
└── engine.py            # MiniVLLMEngine (state + scheduling logic)
```

---

# 🧠 Inference Workflow

### 1. Register a request

Applies ChatML formatting and tokenizes the prompt.

```python
req_id = engine.register_request("What’s your name?")
```

---

### 2. Prefill

Builds the initial KV cache.

```python
engine.prefill([req_id])
```

---

### 3. Decode loop

Decodes step-by-step, updating per-request KV and token streams.

```python
for _ in range(30):
    engine.decode_step([req_id])
```

---

### 4. Retrieve final text

```python
print(engine.get_text(req_id))
```

Example output:

```
<s><s>[INST] What's your name? [/INST] 
My name is Mistral 7B v0.1. But you can call me Mistral.</s>
```

---

# 🛠 Requirements

* Python 3.10+
* PyTorch 2.x
* transformers == 4.36.x
* CUDA GPU (recommended but not required)

---

# 📌 Notes

* This project focuses on **logic clarity**, not speed.
* It intentionally omits:

  * PagedAttention
  * GPU memory paging
  * Multi-GPU execution
  * CUDA kernel fusion

The goal is to make the inference architecture fully understandable.

---

# 🧭 Roadmap

* [ ] Complete Scheduler (dynamic batching + queueing)
* [ ] EOS / stop-token handling
* [ ] Output post-processing (remove ChatML markers)
* [ ] True multi-request decode demonstration
* [ ] Prefix-sharing experiments
* [ ] KV-paging simulator
* [ ] Throughput comparison (HF generate vs. mini-vLLM)

---

# 🤝 Acknowledgements

Inspired by **vLLM**, **HuggingFace Transformers**, and recent literature on efficient LLM inference.
The implementation mirrors real production systems while remaining compact and easy to study.

---

# 📜 License

MIT License.

---

If you'd like, I can also prepare:

### ✔ A shorter README

### ✔ A more academic IEEE/NeurIPS-style README

### ✔ A README with diagrams

### ✔ A README including installation commands and examples

### ✔ A Chinese version

Just tell me what style you prefer.

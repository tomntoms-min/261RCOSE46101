## Environment Setup

The experiments were conducted in Google Colab with CUDA-enabled GPUs.

### 1. Install Dependencies

```bash
pip uninstall -y triton transformers accelerate bitsandbytes datasets

pip install -U triton transformers accelerate bitsandbytes datasets

pip install -U google-generativeai

pip install -q transformers accelerate bitsandbytes pandas matplotlib seaborn tqdm datasets

pip install -q git+https://github.com/dsbowen/strong_reject.git@main

pip install -q -U torchao
```

After installation, restart the runtime:

```text
Runtime → Restart Runtime
```

---

### 2. Authentication Setup

Store the following secrets in Google Colab:

* HF_TOKEN
* GEMINI_API_KEY

Then initialize authentication:

```python
from google.colab import userdata
from huggingface_hub import login
import google.generativeai as genai

hf_token = userdata.get("HF_TOKEN")
login(hf_token)

gemini_api_key = userdata.get("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)

print("Authentication initialized.")
```

---

### 3. Gemini Judge Initialization

```python
import google.generativeai as genai

judge_model = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash",
    safety_settings=[
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
)

print("Gemini judge initialized.")
```

---

### 4. Utility Functions

```python
import gc
import torch

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
```

---

### 5. Target Model Loading

The experiments use:

```text
meta-llama/Meta-Llama-3-8B-Instruct
```

with 8-bit quantization.

```python
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16,
    bnb_8bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=hf_token
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    token=hf_token
)

embed_layer = model.get_input_embeddings()
tok_emb = embed_layer.weight.data

tokenizer.pad_token = tokenizer.eos_token

print("Target model loaded.")
```

---

### Experiment Workflow

1. Install dependencies
2. Restart runtime
3. Authenticate Hugging Face and Gemini APIs
4. Initialize Gemini judge
5. Load target model
6. Run:

   * `omgc_robustness_evaluation.py`
   * `omgc_magic_comparison.py`
   * `omgc_multi_judge_assessment.py`
7. Collect StrongREJECT and Gemini evaluation scores
8. Export CSV/PDF/PNG results

```
```

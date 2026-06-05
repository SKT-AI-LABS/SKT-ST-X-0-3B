# SKT-ST-X-0-3B 🚀
**The Most Efficient 3B Mixture of Experts (MoE) Model for Bilingual Reasoning.**

SKT-ST-X-0-3B is a sovereign Small Language Model (SLM) engineered by **SKT AI LABS, India**. Designed to deliver high-level reasoning and coding capabilities while maintaining an incredibly small footprint, this model is built for developers who demand speed and efficiency.

## 🌟 Why SKT-ST-X-0-3B?
 * **Sovereign AI Innovation:** Built by SKT AI LABS, India, for robust local performance.

 * **High-Speed MoE:** Uses a 2-expert-per-token architecture, ensuring lightning-fast inference on consumer-grade hardware.

 * **Bilingual Mastery:** Expertly trained in both **English and Hindi**.
 
 * **Edge-Ready:** 3B total parameters (~1.1B active), making it perfect for custom AI agents and local deployment.


## 📊 Quick Performance Overview
| Capability | Performance |
|---|---|
| **Logic & Reasoning** | Strong |
| **Coding (Python)** | Efficient |
| **VRAM Usage** | Optimized (Low) |
| **Context Length** | 8K Tokens |

## ⚡ Quick Start
Get started in seconds:
```bash
pip install transformers accelerate torch bitsandbytes

```
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "sKT-Ai-Labs/SKT-ST-X-0-3B"

# Load model
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Inference
prompt = "What is the future of sovereign AI?"
formatted = f"<|user|>\n{prompt}\n<|assistant|>\n"
inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=150)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

```
## 🛠 Advanced Features

 * **LoRA Fine-tuning:** Easily adapt to custom tasks with minimal compute.
 * **4-bit Quantization:** Run efficiently on devices with limited VRAM.


## 🤝 Community & Support
Love this project? **Please click the ⭐ button** at the top to support Indian AI development!

 * **Issues:** Found a bug? Open an Issue.
 * **Contact:** support@sktailabs.in

 * **Hugging Face:** View on Hugging Face


## 📜 License
Released under the **Apache-2.0 License**.
## 📝 Citation
```bibtex
@misc{SKT-ST-X-0-3B,
  author = {SKT AI LABS, India},
  title = {SKT-ST-X-0-3B: A Compact Mixture of Experts Model},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/sKT-Ai-Labs/SKT-ST-X-0-3B}
}

```

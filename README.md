# SKT-ST-X-0-3B: Efficient Small Language Model

<div align="center">

## A Practical Language Model for Edge and Resource-Constrained Environments

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.30%2B-yellow.svg)](https://huggingface.co/docs/transformers)
[![HuggingFace](https://img.shields.io/badge/🤗%20Model-Hub-green.svg)](https://huggingface.co/SKT-Ai-Labs/SKT-ST-X-0-3B)

</div>

---

## Overview

SKT-ST-X-0-3B is a 3-billion parameter language model optimized for efficiency and practical deployment. With only 1.1B active parameters through Mixture of Experts (MoE) architecture, it achieves reasonable performance on resource-constrained hardware while maintaining quality across English and Hindi tasks.

### Key Specifications

| Aspect | Details |
|--------|---------|
| **Total Parameters** | 3B |
| **Active Parameters** | 1.1B (per token) |
| **Architecture** | 2-Expert Mixture of Experts |
| **Context Length** | 8K tokens |
| **Languages** | English, Hindi |
| **Training Data** | Diverse multilingual corpus |
| **License** | Apache 2.0 |

---

## Quick Start

### Installation

```bash
pip install transformers torch
```

### Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "SKT-Ai-Labs/SKT-ST-X-0-3B",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("SKT-Ai-Labs/SKT-ST-X-0-3B")

inputs = tokenizer("Explain quantum computing:", return_tensors="pt")
outputs = model.generate(inputs.input_ids, max_length=100)
print(tokenizer.decode(outputs[0]))
```

### With Quantization (Reduced Memory)

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "SKT-Ai-Labs/SKT-ST-X-0-3B",
    quantization_config=bnb_config,
    device_map="auto"
)
```

---

## Hardware Requirements

### Recommended Specifications

| Use Case | GPU Memory | System RAM | Storage |
|----------|-----------|-----------|---------|
| Inference | 4-6GB | 8GB | 6-8GB |
| Fine-tuning (LoRA) | 6-8GB | 16GB | 6-8GB |
| Full Training | 16GB+ | 32GB+ | 10GB |

### Supported Hardware

- NVIDIA GPUs (A100, H100, RTX 3090, RTX 3060, etc.)
- AMD GPUs (with ROCm)
- Apple Silicon (M1/M2 Pro and above)
- CPU inference (slower but functional)
- Quantized versions for mobile/edge devices

---

## Performance Benchmarks

### Inference Speed (Tokens per Second)

Measured on consumer hardware with default precision:

```
Hardware               | Speed (tok/s) | Notes
-----------------------|---------------|------------------
NVIDIA RTX 3090        | ~80-100       | High-end consumer GPU
NVIDIA RTX 3060        | ~40-50        | Mid-range consumer GPU
NVIDIA RTX 2060        | ~20-30        | Budget GPU
Apple M1 Pro (CPU)     | ~8-12         | Laptop (FP32)
Quantized (4-bit)      | +30-50% boost | Reduced accuracy
```

### Model Quality

The model is evaluated on standard benchmarks:

- **MMLU**: Comparable to similarly-sized models
- **HellaSwag**: Standard performance metrics
- **Bilingual Tasks**: Good performance on English and Hindi
- **Code Generation**: Basic to intermediate capabilities

*Note: This is a 3B model. Expect performance limitations compared to larger models (7B+, 13B+).*

---

## Features

### Technical Capabilities

| Feature | Details |
|---------|---------|
| **Mixture of Experts** | 2 experts per token for conditional compute |
| **Flash Attention** | Optimized attention for faster inference |
| **LoRA Support** | Efficient fine-tuning with <5% parameters |
| **Quantization** | 4-bit, 8-bit quantization support |
| **Bilingual** | Native support for English and Hindi |
| **Context Window** | 8K token context for document handling |

### Supported Tasks

- ✓ Text generation and completion
- ✓ Instruction following (with appropriate fine-tuning)
- ✓ Question answering
- ✓ Summarization (with prompting)
- ✓ Code generation (basic)
- ✓ Translation between English/Hindi
- ✗ Image/Vision tasks (not supported)
- ✗ Complex reasoning (limited by model size)

---

## Use Cases

### Production-Ready Applications

1. **Edge Deployment**
   - On-device inference on servers with limited resources
   - Privacy-focused applications (data stays local)
   - Reduced latency compared to cloud APIs

2. **Mobile & IoT**
   - Quantized models for mobile apps
   - Smart device applications
   - Low-power edge computing

3. **Cost-Effective Inference**
   - Self-hosted inference alternative to cloud APIs
   - Reduced operational costs for high-volume inference
   - No per-token pricing

4. **Multilingual Support**
   - English-Hindi translation
   - Bilingual chatbots
   - Regional language support

### Limitations to Consider

- **Size Limitations**: 3B parameters mean reduced reasoning and world knowledge
- **Accuracy Trade-offs**: Smaller than state-of-the-art models
- **Context**: 8K tokens may be insufficient for very long documents
- **Specialization**: May require fine-tuning for domain-specific tasks

---

## Installation & Setup

### From Source

```bash
git clone https://github.com/SKT-AI-LABS/SKT-ST-X-0-3B.git
cd SKT-ST-X-0-3B
pip install -r requirements.txt
```

### Docker Setup

```bash
docker build -t skt-st-x .
docker run --gpus all -it skt-st-x
```

---

## Fine-Tuning

### LoRA Fine-Tuning Example

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

### Training Script

See `examples/fine_tune.py` for a complete training example with proper evaluation and checkpointing.

---

## Deployment

### Inference Server (vLLM)

```bash
python -m vllm.entrypoints.openai_api_server \
    --model SKT-Ai-Labs/SKT-ST-X-0-3B \
    --tensor-parallel-size 1
```

### Hugging Face Inference Endpoints

Available through Hugging Face's commercial inference service for production deployments.

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas where help is needed:
- Performance optimizations
- Additional language support
- Evaluation benchmarks
- Documentation improvements
- Bug reports and fixes

---

## Model Card

For detailed model information, training data, and evaluation results, see [MODEL_CARD.md](MODEL_CARD.md).

---

## Citation

If you use SKT-ST-X-0-3B in research, please cite:

```bibtex
@model{skt2024,
  author = {SKT AI Labs},
  title = {SKT-ST-X-0-3B: An Efficient Bilingual Language Model},
  year = {2024},
  organization = {SKT AI Labs}
}
```

---

## License

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) file for details.

---

## Community & Support

- **Issues**: [GitHub Issues](https://github.com/SKT-AI-LABS/SKT-ST-X-0-3B/issues)
- **Discussions**: [GitHub Discussions](https://github.com/SKT-AI-LABS/SKT-ST-X-0-3B/discussions)
- **Email**: support@sktailabs.in
- **Documentation**: [docs.sktailabs.in](https://docs.sktailabs.in)

---

## Roadmap

### Planned Improvements

| Feature | Timeline | Status |
|---------|----------|--------|
| Additional language support | Q3 2026 | Planned |
| Vision capabilities | Q4 2026 | Research phase |
| Improved reasoning | Q2 2026 | In development |
| Function calling | Q2 2026 | Planned |
| Structured outputs (JSON) | Q2 2026 | Planned |

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.

---

## Acknowledgments

- Built on Hugging Face Transformers
- Community contributors and testers
- Research foundations from the open-source AI community

---

Made with care by **SKT AI Labs**  
Open source. Privacy-first. Practical AI.

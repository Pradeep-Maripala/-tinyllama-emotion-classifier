# TinyLlama Fine-Tuning for Emotion Classification

Fine-tuning a 1.1B parameter language model for emotion classification using Parameter-Efficient Fine-Tuning (PEFT) with LoRA — all on a standard laptop.

---

## Overview

Large language models are powerful but expensive to fine-tune. This project demonstrates that **LoRA (Low-Rank Adaptation)** enables effective fine-tuning of a 1.1B parameter model on consumer hardware — no GPU cluster required.

Using TinyLlama as the base model and a custom emotion dataset, the fine-tuned adapter learns to detect emotions in text while modifying less than 1% of the original model's parameters. An interactive CLI allows direct side-by-side comparison of the base model vs. the fine-tuned model.

---

## Key Highlights

- Fine-tuned on **Apple Silicon (MPS)** using Metal Performance Shaders acceleration
- LoRA targets `q_proj` and `v_proj` attention layers — base model weights stay frozen
- Chat-style prompt formatting using TinyLlama's `<|im_start|>` / `<|im_end|>` tokens
- Interactive CLI to compare base model vs. fine-tuned model in real time
- Training tracked with **Weights & Biases (wandb)**

---

## Methods

### Why LoRA?

Full fine-tuning of a 1.1B parameter model requires significant GPU memory and compute. LoRA instead:

1. Freezes all original model weights
2. Injects small trainable rank-decomposition matrices into attention layers
3. Learns task-specific behavior through these low-rank updates only
4. At inference, merges the adapter back — zero additional latency

### Architecture

```
TinyLlama-1.1B-Chat (frozen)
        +
LoRA Adapter
  - r = 8
  - lora_alpha = 16
  - lora_dropout = 0.1
  - target_modules = [q_proj, v_proj]
        ↓
Fine-tuned Emotion Classifier
```

### Prompt Format

```
<|im_start|>user
Classify the emotion of the following text:
"I am feeling very happy today!"
<|im_end|>
<|im_start|>assistant
joy
<|im_end|>
```

### Training Configuration

| Parameter | Value |
|---|---|
| Base model | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| Epochs | 3 |
| Batch size | 3 |
| Learning rate | 2e-4 |
| Warmup steps | 100 |
| Weight decay | 0.01 |
| Max sequence length | 512 |
| Hardware | Apple Silicon (MPS) |
| Experiment tracking | Weights & Biases |

---

## Repository Structure

```
├── finetune_emotion.py          # LoRA fine-tuning script
├── comparemodel_emotion.py      # Interactive CLI: base vs fine-tuned comparison
├── emotion_data.txt             # Training dataset (format: text;label)
├── emotion_finetuned_model/
│   ├── adapter_config.json      # LoRA configuration
│   ├── adapter_model.safetensors # Trained LoRA weights
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
├── .env.example                 # API key template
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites
- Python 3.9+
- Apple Silicon Mac (MPS) or CUDA GPU
- HuggingFace account
- Weights & Biases account

### Clone the repo
```bash
git clone https://github.com/pradeep-maripala/tinyllama-emotion-classifier.git
cd tinyllama-emotion-classifier
```

### Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your HF_TOKEN and WANDB_API_KEY
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Fine-tune the model
```bash
python finetune_emotion.py
```

### Compare base vs fine-tuned model (interactive CLI)
```bash
python comparemodel_emotion.py
```
```
=== Emotion Classification Comparison ===
Enter a sentence to classify its emotion.
For example: 'I am feeling very happy today!'

Enter your sentence: I can't believe this happened to me
--------------------------------------------------
Base Model:
  Emotion: 'surprise'
  Inference Time: 1.2341 seconds

Fine-Tuned Model:
  Emotion: 'anger'
  Inference Time: 1.1823 seconds
--------------------------------------------------
```

---

## Data Format

`emotion_data.txt` uses a simple semicolon-separated format:

```
I am so happy today;joy
This makes me really angry;anger
I feel so sad and alone;sadness
```

---

## Requirements

```
torch
transformers
peft
datasets
huggingface_hub
accelerate
wandb
python-dotenv
```

---

## Security Note

API keys are loaded from environment variables. Never hardcode credentials in your scripts. Use the provided `.env.example` as a template.

---

## Background & Motivation

This project explores how far PEFT techniques can stretch consumer hardware for NLP tasks. With the rise of small but capable models like TinyLlama, Phi-2, and Mistral-7B, efficient fine-tuning is increasingly accessible — this project documents that process end to end.

---

## Author

**Pradeep Maripala**
MS Data Science, University of Iowa
[LinkedIn](https://www.linkedin.com/in/maripala-pradeep-13425b211/) | [Email](mailto:maripalapradeep27@gmail.com)

import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
from peft import LoraConfig, get_peft_model

os.environ["TOKENIZERS_PARALLELISM"]="false"

from huggingface_hub import login
login("<your_key>")

import wandb
wandb.login(key="<your_key>")


# =============================================
# Step 1: Load the TXT Data for Emotion Classification
# =============================================
# The text file should have one example per line in the format:
# <text>;<label>
data = []
with open("emotion_data.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if ";" in line:
            parts = line.split(";", 1)
            text_part = parts[0].strip()
            label = parts[1].strip()
            # Only add the example if both text and label are non-empty.
            if text_part and label:
                data.append({"text": text_part, "label": label})
        else:
            # Optionally, log or print lines that do not meet the expected format.
            pass

# Create a Hugging Face Dataset from the list of dictionaries.
dataset = Dataset.from_dict({
    "text": [d["text"] for d in data],
    "label": [d["label"] for d in data]
})

# =============================================
# Step 2: Load Model, Tokenizer and Setup LoRA
# =============================================
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure there is a pad token (set to eos_token if not defined)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)



# Configure LoRA for efficient fine-tuning.
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, lora_config)

# =============================================
# Step 3: Tokenize the Dataset with an Updated Prompt for Emotion Classification
# =============================================
def tokenize_function(examples):
    texts = []
    for text, label in zip(examples["text"], examples["label"]):
        prompt = (
            f"<|im_start|>user\nClassify the emotion of the following text:\n\"{text}\"\n<|im_end|>\n"
            f"<|im_start|>assistant\n{label}\n<|im_end|>"
        )
        texts.append(prompt)
    tokenized = tokenizer(texts, truncation=True, max_length=512, padding="max_length")
    # For a causal LM, we use input_ids as labels.
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text", "label"]
)

# =============================================
# Step 4: Set Training Arguments and Initialize Trainer
# =============================================
training_args = TrainingArguments(
    output_dir="./emotion_finetuned_model",
    per_device_train_batch_size=3,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=50,
    save_steps=500,
    fp16=False,
    warmup_steps=100,
    weight_decay=0.01,
    use_mps_device=True
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# =============================================
# Step 5: Fine-Tune the Model
# =============================================
trainer.train()

model.save_pretrained("emotion_finetuned_model")
tokenizer.save_pretrained("emotion_finetuned_model")

print("Fine-tuning complete. Model saved to 'emotion_finetuned_model'.")

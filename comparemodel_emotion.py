import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Define model name and paths
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
finetuned_model_path = "emotion_finetuned_model"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the base model
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)
base_model.resize_token_embeddings(len(tokenizer))
base_model.config.use_cache = True  # Enable cache for faster inference

print("Loading fine-tuned model...")
# Load the base model again for fine-tuning
finetuned_base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)
# Load the LoRA fine-tuned model as a PEFT model
finetuned_model = PeftModel.from_pretrained(finetuned_base_model, "emotion_finetuned_model")

# Merge LoRA weights into the base model
finetuned_model = finetuned_model.merge_and_unload()

# Set the eos_token_id to the id of "<|im_end|>"
eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

# Function to run inference and measure time
def run_inference(model, input_ids):
    start_time = time.time()
    with torch.no_grad():  # Disable gradient calculation for inference
        output_ids = model.generate(
            input_ids,
            max_new_tokens=2,  # Generate only one token/word (i.e. one emotion)
            do_sample=True,
            temperature=0.7,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    elapsed_time = time.time() - start_time
    return output_ids, elapsed_time

print("\n=== Emotion Classification Comparison ===")
print("Enter a sentence to classify its emotion.")
print("For example: 'I am feeling very happy today!'")
print("Type 'quit' to exit.\n")

while True:
    sentence_input = input("Enter your sentence: ").strip()
    if sentence_input.lower() == "quit":
        break

    # Construct the prompt for emotion classification using chat-style tokens
    prompt = (
        f"<|im_start|>user\nClassify the emotion of the following text:\n\"{sentence_input}\"\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(base_model.device)
    
    # Run inference on the base model
    base_output_ids, base_time = run_inference(base_model, input_ids)
    # Run inference on the fine-tuned model
    finetuned_output_ids, finetuned_time = run_inference(finetuned_model, input_ids)
    
    # Process the base model's output
    base_text = tokenizer.decode(base_output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    base_emotion = base_text.split()[0] if base_text else ""
    
    # Process the fine-tuned model's output
    finetuned_text = tokenizer.decode(finetuned_output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    finetuned_emotion = finetuned_text.split()[0] if finetuned_text else ""
    
    # Print the results and performance comparison
    print("\n" + "="*50)
    print("Base Model:")
    print(f"Emotion: '{base_emotion}'")
    print(f"Inference Time: {base_time:.4f} seconds")
    
    print("\nFine-Tuned Model:")
    print(f"Emotion: '{finetuned_emotion}'")
    print(f"Inference Time: {finetuned_time:.4f} seconds")
    
    print("\nPerformance Comparison:")
    time_diff = base_time - finetuned_time
    if time_diff > 0:
        print(f"Fine-tuned model was {abs(time_diff):.4f} seconds faster")
    else:
        print(f"Base model was {abs(time_diff):.4f} seconds faster")
    print("="*50 + "\n")

print("Exiting interactive query loop.")

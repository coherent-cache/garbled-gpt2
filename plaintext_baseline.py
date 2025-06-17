#!/usr/bin/env python3
import time
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def main():
    # Load tiny GPT-2 model and tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()  # Set to evaluation mode

    # Prepare input text
    input_text = "The quick brown fox jumps"
    inputs = tokenizer(input_text, return_tensors="pt")

    # Run inference and measure time
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds

    # Get the predicted next token (argmax of logits)
    logits = outputs.logits
    next_token_logits = logits[0, -1, :]
    predicted_token_id = torch.argmax(next_token_logits).item()
    predicted_token = tokenizer.decode([predicted_token_id])

    # Print results
    print(f"Input: {input_text}")
    print(f"Inference time: {inference_time:.2f} ms")
    print(f"Output logits shape: {logits.shape}")
    print(f"Predicted next token: {predicted_token}")


if __name__ == "__main__":
    main()

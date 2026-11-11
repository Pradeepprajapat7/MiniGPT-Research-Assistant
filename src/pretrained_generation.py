
import torch
from src.pretrained_model import load_model


def generate_text(
    prompt,
    tokenizer,
    model,
    device,
    max_new_tokens=100
):
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    return generated_text


if __name__ == "__main__":
    tokenizer, model, device = load_model()

    prompts = [
    "Artificial intelligence is",
    "Deep learning works by",
    "The Transformer architecture",
    "A language model learns",
    "PyTorch is useful for"
]

for prompt in prompts:
    print("\nPrompt:", prompt)

    result = generate_text(
        prompt,
        tokenizer,
        model,
        device,
        max_new_tokens=50
    )

    print("Generated:", result)
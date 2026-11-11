
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

MODEL_NAME = "distilbert/distilgpt2"


def load_model():
    device_name = (
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    torch_device = torch.device(device_name)

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Loading pretrained model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    )

    model.to(torch_device)
    model.eval()

    print("Device:", device_name)
    print("Parameters:", sum(
        p.numel() for p in model.parameters()
    ))

    return tokenizer, model, device_name


if __name__ == "__main__":
    tokenizer, model, device_name = load_model()

    print("Model loaded successfully!")

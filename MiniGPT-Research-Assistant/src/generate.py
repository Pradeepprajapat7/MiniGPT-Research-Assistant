
import torch


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    device="cpu"
):
    model.eval()

    token_ids = tokenizer.encode(prompt)
    x = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device
    )

    for _ in range(max_new_tokens):
        # Keep only the context the model supports.
        x_cond = x[:, -model.block_size:]

        logits = model(x_cond)

        # Next-token prediction.
        logits = logits[:, -1, :]

        logits = logits / temperature

        probabilities = torch.softmax(logits, dim=-1)

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        x = torch.cat((x, next_token), dim=1)

    model.train()
    return tokenizer.decode(x[0].tolist())

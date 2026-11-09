
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    batches = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1)
            )

            total_loss += loss.item()
            batches += 1

    model.train()
    return total_loss / max(batches, 1)


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=10,
    learning_rate=3e-4,
    device="cpu"
):
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}"
        )

        for x, y in progress:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1)
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        train_loss = total_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch + 1}: "
            f"Train Loss={train_loss:.4f}, "
            f"Validation Loss={val_loss:.4f}"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            "checkpoints/minigpt.pt"
        )

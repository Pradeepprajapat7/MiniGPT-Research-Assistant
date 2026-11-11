
import math
import torch
import torch.nn as nn


def scaled_dot_product_attention(q, k, v, causal=False):

    # Step 1: Calculate attention scores
    scores = torch.matmul(q, k.transpose(-2, -1))

    # Step 2: Scale scores
    scores = scores / math.sqrt(q.size(-1))

    # Step 3: Apply causal mask if required
    if causal:
        query_length = q.size(-2)
        key_length = k.size(-2)

        mask = torch.tril(
            torch.ones(
                query_length,
                key_length,
                device=q.device,
                dtype=torch.bool
            )
        )

        scores = scores.masked_fill(~mask, float("-inf"))

    # Step 4: Normalize scores
    attention_weights = torch.softmax(scores, dim=-1)

    # Step 5: Weighted sum of values
    output = torch.matmul(attention_weights, v)

    return output, attention_weights


class SelfAttention(nn.Module):

    def __init__(self, embed_dim):

        super().__init__()

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        output, weights = scaled_dot_product_attention(
            q, k, v
        )

        return output, weights


if __name__ == "__main__":

    torch.manual_seed(42)

    # Batch = 1, Sequence length = 4, Embedding dimension = 8
    x = torch.randn(1, 4, 8)

    model = SelfAttention(embed_dim=8)

    output, weights = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Attention weights shape:", weights.shape)

    print("\nAttention weights:")
    print(weights)

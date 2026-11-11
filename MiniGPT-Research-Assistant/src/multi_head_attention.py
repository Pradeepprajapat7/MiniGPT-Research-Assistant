
import math
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):

        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # One projection for Q, K and V
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)

        # Final output projection
        self.output_projection = nn.Linear(
            embed_dim, embed_dim
        )

    def forward(self, x, causal=True):

        batch_size, seq_length, embed_dim = x.shape

        # Generate Query, Key and Value
        qkv = self.qkv(x)

        # Split into three tensors and separate attention heads
        qkv = qkv.view(
            batch_size,
            seq_length,
            3,
            self.num_heads,
            self.head_dim
        )

        q, k, v = qkv.unbind(dim=2)

        # Move heads before sequence dimension
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Calculate attention scores
        scores = torch.matmul(
            q, k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        # Causal masking
        if causal:

            mask = torch.tril(
                torch.ones(
                    seq_length,
                    seq_length,
                    device=x.device,
                    dtype=torch.bool
                )
            )

            scores = scores.masked_fill(
                ~mask, float("-inf")
            )

        # Attention weights
        weights = torch.softmax(scores, dim=-1)

        # Weighted sum
        output = torch.matmul(weights, v)

        # Combine all heads
        output = output.transpose(1, 2).contiguous()

        output = output.view(
            batch_size,
            seq_length,
            self.embed_dim
        )

        # Final projection
        output = self.output_projection(output)

        return output, weights


if __name__ == "__main__":

    torch.manual_seed(42)

    x = torch.randn(2, 5, 64)

    model = MultiHeadSelfAttention(
        embed_dim=64,
        num_heads=4
    )

    output, weights = model(x)

    print("Input:", x.shape)
    print("Output:", output.shape)
    print("Attention weights:", weights.shape)

    loss = output.mean()
    loss.backward()

    print("Backward pass successful!")

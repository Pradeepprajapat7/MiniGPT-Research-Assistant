
import torch

from src.attention import (
    scaled_dot_product_attention,
    SelfAttention
)

from src.multi_head_attention import (
    MultiHeadSelfAttention
)


def test_scaled_attention():

    q = torch.randn(2, 4, 8)
    k = torch.randn(2, 4, 8)
    v = torch.randn(2, 4, 8)

    output, weights = scaled_dot_product_attention(
        q, k, v, causal=True
    )

    assert output.shape == (2, 4, 8)
    assert weights.shape == (2, 4, 4)

    # Each row must sum to 1
    assert torch.allclose(
        weights.sum(dim=-1),
        torch.ones(2, 4)
    )

    # Future tokens must not receive attention
    assert torch.allclose(
        torch.triu(weights, diagonal=1),
        torch.zeros_like(weights)
    )

    print("Scaled attention: PASSED")


def test_self_attention():

    x = torch.randn(2, 4, 8)

    model = SelfAttention(embed_dim=8)

    output, weights = model(x)

    assert output.shape == (2, 4, 8)

    print("Self attention: PASSED")


def test_multi_head_attention():

    x = torch.randn(2, 5, 64)

    model = MultiHeadSelfAttention(
        embed_dim=64,
        num_heads=4
    )

    output, weights = model(x)

    assert output.shape == (2, 5, 64)
    assert weights.shape == (2, 4, 5, 5)

    loss = output.square().mean()
    loss.backward()

    assert model.qkv.weight.grad is not None

    print("Multi-head attention: PASSED")
    print("Backpropagation: PASSED")


if __name__ == "__main__":

    test_scaled_attention()
    test_self_attention()
    test_multi_head_attention()

    print("\nAll tests completed successfully!")

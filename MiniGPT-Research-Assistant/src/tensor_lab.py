
import torch

# 1. Scalar
scalar = torch.tensor(10)

# 2. Vector
vector = torch.tensor([1, 2, 3, 4])

# 3. Matrix
matrix = torch.tensor([
    [1, 2],
    [3, 4]
])

# 4. 3D Tensor
tensor_3d = torch.randn(2, 3, 4)

print("Scalar:", scalar)
print("Vector:", vector)
print("Matrix:\n", matrix)

print("3D Tensor Shape:", tensor_3d.shape)
print("Dimensions:", tensor_3d.ndim)

# Matrix multiplication
a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([[5., 6.], [7., 8.]])

result = torch.matmul(a, b)

print("Matrix Multiplication:\n", result)




x = torch.tensor(3.0, requires_grad=True)

# Forward pass
y = x**2 + 3*x + 2

print("Output:", y.item())

# Backward pass
y.backward()

# Gradient
print("Gradient:", x.grad.item())

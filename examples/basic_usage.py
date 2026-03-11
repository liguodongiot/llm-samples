import torch
from matmul_ops import MatMul, get_op


def main():
    A = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
    B = torch.randn(4, 5, dtype=torch.float32, requires_grad=True)
    
    result = MatMul.apply(A, B)
    print(f"Result shape: {result.shape}")
    print(f"Result:\n{result}")
    
    loss = result.sum()
    loss.backward()
    print(f"A.grad shape: {A.grad.shape}")
    print(f"B.grad shape: {B.grad.shape}")
    
    op = get_op("matmul")
    result2 = op.apply(A, B)
    print(f"\nVia registry: {torch.allclose(result, result2)}")


if __name__ == "__main__":
    main()
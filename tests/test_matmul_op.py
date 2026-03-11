import pytest
import torch
from matmul_ops import MatMul


class TestMatMul:
    def test_forward_correctness(self):
        A = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
        B = torch.randn(4, 5, dtype=torch.float32, requires_grad=True)
        
        result = MatMul.apply(A, B)
        expected = torch.matmul(A, B)
        
        assert torch.allclose(result, expected)
    
    def test_backward_gradient(self):
        A = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
        B = torch.randn(4, 5, dtype=torch.float32, requires_grad=True)
        
        result = MatMul.apply(A, B)
        loss = result.sum()
        loss.backward()
        
        assert A.grad is not None
        assert B.grad is not None
    
    def test_gradcheck(self):
        A = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        B = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
        
        def matmul_fn(a, b):
            return MatMul.apply(a, b)
        
        assert torch.autograd.gradcheck(matmul_fn, (A, B), eps=1e-6, atol=1e-4)
    
    def test_invalid_dimensions(self):
        A = torch.randn(3, 4, dtype=torch.float32)
        B = torch.randn(5, 6, dtype=torch.float32)
        
        with pytest.raises(ValueError):
            MatMul.apply(A, B)
    
    def test_invalid_dtype(self):
        A = torch.randint(0, 10, (3, 4))
        B = torch.randn(4, 5, dtype=torch.float32)
        
        with pytest.raises(TypeError):
            MatMul.apply(A, B)
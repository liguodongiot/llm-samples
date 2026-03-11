import torch
from torch.autograd import Function

from ..utils import validate_matrix_dimensions, validate_tensor_type


class MatMul(Function):
    """Custom matrix multiplication operator with autograd support.
    
    Implements forward and backward passes for matrix multiplication
    as a custom PyTorch autograd function.
    """
    
    @staticmethod
    def forward(ctx, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Forward pass: compute A @ B.
        
        Args:
            A: First matrix of shape (m, n)
            B: Second matrix of shape (n, k)
        
        Returns:
            Result matrix of shape (m, k)
        """
        validate_matrix_dimensions(A, B)
        validate_tensor_type(A)
        validate_tensor_type(B)
        
        ctx.save_for_backward(A, B)
        
        return torch.matmul(A, B)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass: compute gradients.
        
        Args:
            grad_output: Gradient of loss w.r.t. output
        
        Returns:
            Tuple of (grad_A, grad_B)
        """
        A, B = ctx.saved_tensors
        
        grad_A = torch.matmul(grad_output, B.t())
        grad_B = torch.matmul(A.t(), grad_output)
        
        return grad_A, grad_B
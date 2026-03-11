import torch


def validate_matrix_dimensions(A: torch.Tensor, B: torch.Tensor) -> None:
    """Validate matrix dimensions for multiplication.
    
    Args:
        A: First matrix tensor of shape (m, n)
        B: Second matrix tensor of shape (n, k)
    
    Raises:
        ValueError: If dimensions are incompatible
    """
    if A.dim() != 2:
        raise ValueError(f"A must be 2D tensor, got {A.dim()}D")
    if B.dim() != 2:
        raise ValueError(f"B must be 2D tensor, got {B.dim()}D")
    if A.shape[1] != B.shape[0]:
        raise ValueError(
            f"Matrix dimensions incompatible: A.shape={A.shape}, B.shape={B.shape}"
        )


def validate_tensor_type(tensor: torch.Tensor, allowed_dtypes=None) -> None:
    """Validate tensor is float32 or float64.
    
    Args:
        tensor: Tensor to validate
        allowed_dtypes: List of allowed dtypes. Defaults to [float32, float64]
    
    Raises:
        TypeError: If tensor dtype is not allowed
    """
    if allowed_dtypes is None:
        allowed_dtypes = [torch.float32, torch.float64]
    if tensor.dtype not in allowed_dtypes:
        raise TypeError(f"Expected tensor dtype in {allowed_dtypes}, got {tensor.dtype}")
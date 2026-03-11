import pytest
import torch
from matmul_ops.utils import validate_matrix_dimensions, validate_tensor_type


class TestValidateMatrixDimensions:
    def test_valid_dimensions(self):
        A = torch.randn(3, 4)
        B = torch.randn(4, 5)
        validate_matrix_dimensions(A, B)
    
    def test_invalid_inner_dimensions(self):
        A = torch.randn(3, 4)
        B = torch.randn(5, 6)
        with pytest.raises(ValueError):
            validate_matrix_dimensions(A, B)
    
    def test_A_not_2d(self):
        A = torch.randn(3, 4, 5)
        B = torch.randn(4, 6)
        with pytest.raises(ValueError):
            validate_matrix_dimensions(A, B)
    
    def test_B_not_2d(self):
        A = torch.randn(3, 4)
        B = torch.randn(4, 5, 6)
        with pytest.raises(ValueError):
            validate_matrix_dimensions(A, B)


class TestValidateTensorType:
    def test_valid_float32(self):
        A = torch.randn(3, 4, dtype=torch.float32)
        validate_tensor_type(A)
    
    def test_valid_float64(self):
        A = torch.randn(3, 4, dtype=torch.float64)
        validate_tensor_type(A)
    
    def test_invalid_int(self):
        A = torch.randint(0, 10, (3, 4))
        with pytest.raises(TypeError):
            validate_tensor_type(A)
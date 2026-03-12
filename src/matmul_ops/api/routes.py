import torch
from fastapi import APIRouter, HTTPException

from ..ops.matmul import MatMul
from ..utils.validators import validate_matrix_dimensions, validate_tensor_type
from .models import MatrixInput, MatrixOutput


router = APIRouter()


@router.post("/matmul", response_model=MatrixOutput)
def matmul(input_data: MatrixInput) -> MatrixOutput:
    """Perform matrix multiplication A @ B."""
    try:
        A = torch.tensor(input_data.A, dtype=torch.float32)
        B = torch.tensor(input_data.B, dtype=torch.float32)
        
        validate_matrix_dimensions(A, B)
        validate_tensor_type(A)
        validate_tensor_type(B)
        
        result = MatMul.apply(A, B)
        
        return MatrixOutput(result=result.tolist())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TypeError as e:
        raise HTTPException(status_code=422, detail=str(e))

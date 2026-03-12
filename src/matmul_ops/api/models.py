from typing import List
from pydantic import BaseModel, Field


class MatrixInput(BaseModel):
    """Input model for matrix multiplication."""
    A: List[List[float]] = Field(..., description="First matrix as nested list")
    B: List[List[float]] = Field(..., description="Second matrix as nested list")


class MatrixOutput(BaseModel):
    """Output model for matrix multiplication."""
    result: List[List[float]] = Field(..., description="Result matrix as nested list")

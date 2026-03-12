from typing import List
from pydantic import BaseModel, Field, model_validator


class MatrixInput(BaseModel):
    """Input model for matrix multiplication."""
    model_config = {"strict": True}
    
    A: List[List[float]] = Field(..., description="First matrix as nested list")
    B: List[List[float]] = Field(..., description="Second matrix as nested list")

    @model_validator(mode='before')
    @classmethod
    def validate_types(cls, data):
        if isinstance(data, dict):
            for key in ['A', 'B']:
                if key in data:
                    for row in data[key]:
                        for val in row:
                            if isinstance(val, bool) or not isinstance(val, (int, float)):
                                raise ValueError(f"{key} must contain only numbers")
                            if isinstance(val, int) and not isinstance(val, bool):
                                raise ValueError(f"{key} must contain floats, got int")
        return data


class MatrixOutput(BaseModel):
    """Output model for matrix multiplication."""
    result: List[List[float]] = Field(..., description="Result matrix as nested list")

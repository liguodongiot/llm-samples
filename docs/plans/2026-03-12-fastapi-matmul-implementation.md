# FastAPI MatMul Service Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add FastAPI HTTP service to expose the MatMul operator via REST API.

**Architecture:** FastAPI as HTTP layer, Pydantic for validation, reuse existing MatMul operator.

**Tech Stack:** FastAPI, Uvicorn, Pydantic

---

### Task 1: Add FastAPI to dependencies

**Files:**
- Modify: `pyproject.toml:16-20`

**Step 1: Update pyproject.toml**

Add fastapi and uvicorn to dev dependencies:

```python
[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
]
```

**Step 2: Install dependencies**

```bash
pip install -e ".[dev]"
```

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add fastapi and uvicorn dependencies"
```

---

### Task 2: Create Pydantic models

**Files:**
- Create: `src/matmul_ops/api/models.py`

**Step 1: Write the test**

```python
# tests/test_api_models.py
import pytest
from matmul_ops.api.models import MatrixInput, MatrixOutput

def test_matrix_input_valid():
    data = {"A": [[1.0, 2.0], [3.0, 4.0]], "B": [[5.0, 6.0], [7.0, 8.0]]}
    model = MatrixInput(**data)
    assert model.A == [[1.0, 2.0], [3.0, 4.0]]

def test_matrix_output_valid():
    data = {"result": [[19.0, 22.0], [43.0, 50.0]]}
    model = MatrixOutput(**data)
    assert model.result == [[19.0, 22.0], [43.0, 50.0]]
```

**Step 2: Run test**

```bash
pytest tests/test_api_models.py -v
Expected: FAIL - ModuleNotFoundError: No module named 'matmul_ops.api'
```

**Step 3: Create models.py**

```python
from typing import List
from pydantic import BaseModel, Field


class MatrixInput(BaseModel):
    """Input model for matrix multiplication."""
    A: List[List[float]] = Field(..., description="First matrix as nested list")
    B: List[List[float]] = Field(..., description="Second matrix as nested list")


class MatrixOutput(BaseModel):
    """Output model for matrix multiplication."""
    result: List[List[float]] = Field(..., description="Result matrix as nested list")
```

**Step 4: Run test**

```bash
pytest tests/test_api_models.py -v
Expected: PASS
```

**Step 5: Commit**

```bash
git add src/matmul_ops/api/models.py tests/test_api_models.py
git commit -m "feat: add Pydantic models for matrix API"
```

---

### Task 3: Create API endpoint

**Files:**
- Create: `src/matmul_ops/api/routes.py`
- Create: `src/matmul_ops/api/__init__.py`

**Step 1: Write the test**

```python
# tests/test_api_routes.py
import pytest
from fastapi.testclient import TestClient
from matmul_ops.api.main import app

client = TestClient(app)

def test_matmul_success():
    response = client.post("/matmul", json={
        "A": [[1.0, 2.0], [3.0, 4.0]],
        "B": [[5.0, 6.0], [7.0, 8.0]]
    })
    assert response.status_code == 200
    assert response.json() == {"result": [[19.0, 22.0], [43.0, 50.0]]}

def test_matmul_invalid_dimensions():
    response = client.post("/matmul", json={
        "A": [[1.0, 2.0]],
        "B": [[1.0], [2.0], [3.0]]
    })
    assert response.status_code == 400

def test_matmul_invalid_dtype():
    response = client.post("/matmul", json={
        "A": [[1, 2]],
        "B": [[3, 4]]
    })
    assert response.status_code == 422
```

**Step 2: Run test**

```bash
pytest tests/test_api_routes.py -v
Expected: FAIL - ModuleNotFoundError
```

**Step 3: Create routes.py**

```python
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
```

**Step 4: Create __init__.py**

```python
from .routes import router

__all__ = ["router"]
```

**Step 5: Run test**

```bash
pytest tests/test_api_routes.py -v
Expected: PASS
```

**Step 6: Commit**

```bash
git add src/matmul_ops/api/ tests/test_api_routes.py
git commit -m "feat: add matmul API endpoint"
```

---

### Task 4: Create FastAPI app

**Files:**
- Create: `src/matmul_ops/api/main.py`

**Step 1: Create main.py**

```python
from fastapi import FastAPI
from .routes import router


app = FastAPI(
    title="MatMul API",
    description="Matrix multiplication operator API",
    version="0.1.0",
)

app.include_router(router)


@app.get("/")
def root():
    return {"message": "MatMul API", "docs": "/docs"}
```

**Step 2: Update __init__.py**

```python
from .routes import router

__all__ = ["router"]
```

**Step 3: Test the app**

```bash
uvicorn matmul_ops.api.main:app --host localhost --port 8000
# In another terminal:
curl -X POST http://localhost:8000/matmul \
  -H "Content-Type: application/json" \
  -d '{"A": [[1, 2], [3, 4]], "B": [[5, 6], [7, 8]]}'
# Expected: {"result": [[19.0, 22.0], [43.0, 50.0]]}
```

**Step 4: Commit**

```bash
git add src/matmul_ops/api/main.py src/matmul_ops/api/__init__.py
git commit -m "feat: create FastAPI application"
```

---

### Task 5: Run all tests and verify

**Step 1: Run all tests**

```bash
pytest tests/ -v
```

**Step 2: Commit**

```bash
git add .
git commit -m "feat: add FastAPI service for matrix multiplication"
```

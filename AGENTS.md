# AGENTS.md - Agentic Coding Guidelines

Guidelines for agentic coding agents operating in this repository.

## Project Overview

`matmul_ops` is a PyTorch custom matrix multiplication operator library for learning purposes.

## Project Structure

```
matmul_ops/
├── src/matmul_ops/
│   ├── ops/
│   │   ├── base.py          # Abstract base class
│   │   ├── matmul.py        # MatMul operator
│   │   └── registry.py     # Operator registry
│   └── utils/
│       └── validators.py    # Input validation
├── tests/
│   ├── test_matmul_op.py
│   ├── test_registry.py
│   └── test_validators.py
└── examples/
    └── basic_usage.py
```

## Build, Test, and Lint Commands

### Installation

```bash
pip install -e .           # Basic
pip install -e ".[dev]"    # With dev dependencies
```

### Running Tests

```bash
pytest tests/                                    # All tests
pytest tests/test_matmul_op.py                  # Single file
pytest tests/test_matmul_op.py::TestMatMul      # Single class
pytest tests/test_matmul_op.py::TestMatMul::test_forward_correctness  # Single test
pytest tests/ --cov=matmul_ops --cov-report=term-missing  # Coverage
pytest tests/ -v                                # Verbose
```

### Docker

```bash
docker build -t matmul_ops .
docker run matmul_ops pytest tests/ -v
```

## Code Style Guidelines

### Imports

Organize in order: standard library, third-party, local application.

```python
import torch
from torch.autograd import Function

from ..utils import validate_matrix_dimensions, validate_tensor_type
```

### Type Hints

Use type hints for all function arguments and return types.

```python
def forward(ctx, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
def validate_matrix_dimensions(A: torch.Tensor, B: torch.Tensor) -> None:
```

### Naming Conventions

- **Classes**: PascalCase (`MatMul`, `BaseOp`, `OperatorRegistry`)
- **Functions/methods**: snake_case (`validate_matrix_dimensions`)
- **Private methods**: Leading underscore (`_operators`)
- **File names**: snake_case (`test_matmul_op.py`)

### Docstrings

Use docstrings for all public classes and functions:

```python
class MatMul(Function):
    """Custom matrix multiplication operator with autograd support."""
    
    @staticmethod
    def forward(ctx, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Forward pass: compute A @ B.
        
        Args:
            A: First matrix of shape (m, n)
            B: Second matrix of shape (n, k)
        
        Returns:
            Result matrix of shape (m, k)
        """
```

### Error Handling

- Use specific exception types (`ValueError`, `TypeError`, `KeyError`)
- Include descriptive error messages
- Validate inputs early and fail fast

```python
if A.dim() != 2:
    raise ValueError(f"A must be 2D tensor, got {A.dim()}D")
```

### Operator Implementation Pattern

1. Inherit from `torch.autograd.Function`
2. Implement `forward()` and `backward()` as static methods
3. Use `ctx.save_for_backward()` to store tensors for backward
4. Use `ctx.saved_tensors` to retrieve saved tensors in backward

```python
class MatMul(Function):
    @staticmethod
    def forward(ctx, A, B):
        validate_matrix_dimensions(A, B)
        ctx.save_for_backward(A, B)
        return torch.matmul(A, B)
    
    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        return torch.matmul(grad_output, B.t()), torch.matmul(A.t(), grad_output)
```

### Registry Pattern

```python
register_op("matmul", MatMul)
op = get_op("matmul")
result = op.apply(A, B)
```

### Testing Guidelines

- Group tests in classes with descriptive names
- Use `pytest.raises` for exception testing
- Include gradient tests with `torch.autograd.gradcheck`

```python
class TestMatMul:
    def test_forward_correctness(self):
        assert torch.allclose(result, expected)
    
    def test_invalid_dimensions(self):
        with pytest.raises(ValueError):
            MatMul.apply(A, B)
```

### Module Exports

Define `__all__` in each module:

```python
__all__ = ["MatMul", "get_op", "validate_matrix_dimensions", "validate_tensor_type"]
```

## Key Dependencies

- `torch>=2.0.0` - PyTorch
- `pytest>=7.0.0` - Testing framework
- `pytest-cov>=4.0.0` - Coverage plugin

## Notes

- Learning-focused project for PyTorch autograd functions
- Requires float32 or float64 tensors
- Only 2D tensors supported
- All operators must implement gradient computation

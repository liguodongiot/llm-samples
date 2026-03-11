# Matrix Multiplication Operator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a PyTorch custom matrix multiplication operator with autograd support, modular structure, and comprehensive tests.

**Architecture:** Implement using `torch.autograd.Function` with a layered module design (base abstract class, concrete operators, registry, validators).

**Tech Stack:** Python 3.8+, PyTorch 2.0+, pytest

---

### Task 1: Create pyproject.toml

**Files:**
- Create: `pyproject.toml`

**Step 1: Write the project configuration**

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "matmul_ops"
version = "0.1.0"
description = "A PyTorch custom matrix multiplication operator library"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.8"
dependencies = [
    "torch>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
]

[tool.setuptools.packages.find]
where = ["src"]
```

**Step 2: Verify file created**

Run: `ls pyproject.toml`
Expected: File exists

---

### Task 2: Create utils/validators.py

**Files:**
- Create: `src/matmul_ops/utils/__init__.py`
- Create: `src/matmul_ops/utils/validators.py`

**Step 1: Write validators module**

`src/matmul_ops/utils/__init__.py`:
```python
from .validators import validate_matrix_dimensions, validate_tensor_type

__all__ = ["validate_matrix_dimensions", "validate_tensor_type"]
```

`src/matmul_ops/utils/validators.py`:
```python
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


def validate_tensor_type(tensor: torch.Tensor) -> None:
    """Validate tensor is float32.
    
    Args:
        tensor: Tensor to validate
    
    Raises:
        TypeError: If tensor is not float32
    """
    if tensor.dtype != torch.float32:
        raise TypeError(f"Expected float32 tensor, got {tensor.dtype}")
```

**Step 2: Verify files created**

Run: `ls src/matmul_ops/utils/`
Expected: `__init__.py validators.py`

---

### Task 3: Create ops/base.py

**Files:**
- Create: `src/matmul_ops/ops/base.py`

**Step 1: Write base abstract class**

```python
from abc import ABC, abstractmethod
import torch


class BaseOp(ABC):
    """Abstract base class for custom operators.
    
    All operators should inherit from this class and implement
    forward and backward methods.
    """
    
    @staticmethod
    @abstractmethod
    def forward(ctx, *args, **kwargs):
        """Forward pass of the operator.
        
        Args:
            ctx: Context object for storing variables for backward
            *args: Variable length argument list
            **kwargs: Keyword arguments
        
        Returns:
            Output tensor(s)
        """
        pass
    
    @staticmethod
    @abstractmethod
    def backward(ctx, *grad_outputs):
        """Backward pass of the operator.
        
        Args:
            ctx: Context object with stored variables
            *grad_outputs: Gradients from upstream
        
        Returns:
            Gradients with respect to inputs
        """
        pass
```

**Step 2: Verify file created**

Run: `ls src/matmul_ops/ops/base.py`
Expected: File exists

---

### Task 4: Create tests for validators

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_validators.py`

**Step 1: Write failing tests**

```python
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
    
    def test_invalid_float64(self):
        A = torch.randn(3, 4, dtype=torch.float64)
        with pytest.raises(TypeError):
            validate_tensor_type(A)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_validators.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'matmul_ops')

**Step 3: Create minimal __init__.py**

`src/matmul_ops/__init__.py`:
```python
__version__ = "0.1.0"
```

**Step 4: Run tests again**

Run: `pip install -e . && pytest tests/test_validators.py -v`
Expected: Tests run but fail (no validate_matrix_dimensions exported yet)

**Step 5: Commit**

```bash
git add pyproject.toml src/matmul_ops/__init__.py src/matmul_ops/utils/ tests/test_validators.py
git commit -m "feat: add project config and validators module"
```

---

### Task 5: Create ops/matmul.py

**Files:**
- Create: `src/matmul_ops/ops/matmul.py`

**Step 1: Write the matrix multiplication operator**

```python
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
```

**Step 2: Verify file created**

Run: `ls src/matmul_ops/ops/matmul.py`
Expected: File exists

---

### Task 6: Create ops/registry.py

**Files:**
- Create: `src/matmul_ops/ops/registry.py`

**Step 1: Write the operator registry**

```python
class OperatorRegistry:
    """Registry for custom operators.
    
    Provides a simple mechanism to register and retrieve operators by name.
    """
    
    def __init__(self):
        self._operators = {}
    
    def register(self, name: str, op_class) -> None:
        """Register an operator.
        
        Args:
            name: Name of the operator
            op_class: Operator class to register
        """
        if name in self._operators:
            raise ValueError(f"Operator '{name}' already registered")
        self._operators[name] = op_class
    
    def get(self, name: str):
        """Get an operator by name.
        
        Args:
            name: Name of the operator
        
        Returns:
            Registered operator class
        
        Raises:
            KeyError: If operator not found
        """
        if name not in self._operators:
            raise KeyError(f"Operator '{name}' not found in registry")
        return self._operators[name]
    
    def list_operators(self):
        """List all registered operator names.
        
        Returns:
            List of operator names
        """
        return list(self._operators.keys())


_global_registry = OperatorRegistry()


def register_op(name: str, op_class):
    """Register an operator in the global registry."""
    _global_registry.register(name, op_class)


def get_op(name: str):
    """Get an operator from the global registry."""
    return _global_registry.get(name)
```

**Step 2: Verify file created**

Run: `ls src/matmul_ops/ops/registry.py`
Expected: File exists

---

### Task 7: Update ops/__init__.py and test matmul

**Files:**
- Create: `src/matmul_ops/ops/__init__.py`

**Step 1: Write ops/__init__.py**

```python
from .base import BaseOp
from .matmul import MatMul
from .registry import register_op, get_op, OperatorRegistry

register_op("matmul", MatMul)

__all__ = ["BaseOp", "MatMul", "register_op", "get_op", "OperatorRegistry"]
```

**Step 2: Update main __init__.py**

`src/matmul_ops/__init__.py`:
```python
from .ops import MatMul, get_op
from .utils import validate_matrix_dimensions, validate_tensor_type

__version__ = "0.1.0"

__all__ = ["MatMul", "get_op", "validate_matrix_dimensions", "validate_tensor_type"]
```

**Step 3: Write test for matmul operator**

`tests/test_matmul_op.py`:
```python
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
        A = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
        B = torch.randn(4, 5, dtype=torch.float32, requires_grad=True)
        
        def matmul_fn(a, b):
            return MatMul.apply(a, b)
        
        assert torch.autograd.gradcheck(matmul_fn, (A, B), eps=1e-6, atol=1e-4)
    
    def test_invalid_dimensions(self):
        A = torch.randn(3, 4, dtype=torch.float32)
        B = torch.randn(5, 6, dtype=torch.float32)
        
        with pytest.raises(ValueError):
            MatMul.apply(A, B)
    
    def test_invalid_dtype(self):
        A = torch.randn(3, 4, dtype=torch.float64)
        B = torch.randn(4, 5, dtype=torch.float32)
        
        with pytest.raises(TypeError):
            MatMul.apply(A, B)
```

**Step 4: Run tests**

Run: `pip install -e . && pytest tests/test_matmul_op.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/matmul_ops/ops/ tests/test_matmul_op.py
git commit -m "feat: add MatMul operator with autograd support"
```

---

### Task 8: Test registry

**Files:**
- Create: `tests/test_registry.py`

**Step 1: Write registry tests**

```python
import pytest
from matmul_ops import get_op, MatMul


class TestRegistry:
    def test_get_matmul(self):
        op = get_op("matmul")
        assert op is MatMul
    
    def test_get_nonexistent(self):
        with pytest.raises(KeyError):
            get_op("nonexistent")
    
    def test_register_duplicate(self):
    from matmul_ops.ops.registry import OperatorRegistry
    
    registry = OperatorRegistry()
    registry.register("test", MatMul)
    
    with pytest.raises(ValueError):
        registry.register("test", MatMul)
```

**Step 2: Run tests**

Run: `pytest tests/test_registry.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_registry.py
git commit -m "test: add registry tests"
```

---

### Task 9: Create example and final docs

**Files:**
- Create: `examples/basic_usage.py`
- Create: `README.md`
- Create: `docs/api.md`

**Step 1: Write example**

```python
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
```

**Step 2: Create README.md**

```markdown
# matmul_ops

A PyTorch custom matrix multiplication operator library for learning purposes.

## Installation

```bash
pip install -e .
```

## Usage

```python
import torch
from matmul_ops import MatMul

A = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
B = torch.randn(4, 5, dtype=torch.float32, requires_grad=True)

result = MatMul.apply(A, B)
```

## Testing

```bash
pytest tests/
```

## License

MIT
```

**Step 3: Create docs/api.md**

```markdown
# API Reference

## MatMul

Custom matrix multiplication operator with autograd support.

### Methods

#### MatMul.apply(A, B)

Matrix multiplication A @ B.

**Parameters:**
- A (torch.Tensor): First matrix of shape (m, n), float32
- B (torch.Tensor): Second matrix of shape (n, k), float32

**Returns:**
- torch.Tensor: Result of shape (m, k)

**Raises:**
- ValueError: If dimensions are incompatible
- TypeError: If tensors are not float32

## Registry Functions

### get_op(name)

Get operator from global registry.

**Parameters:**
- name (str): Operator name

**Returns:**
- Operator class

## Validators

### validate_matrix_dimensions(A, B)

Validate matrix dimensions for multiplication.

### validate_tensor_type(tensor)

Validate tensor is float32.
```

**Step 4: Run example**

Run: `python examples/basic_usage.py`
Expected: Output shows result shape, values, and gradients

**Step 5: Commit**

```bash
git add examples/ docs/ README.md
git commit -m "docs: add example and documentation"
```

---

### Task 10: Final verification

**Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 2: Check project structure**

Run: `find . -type f -name "*.py" | grep -v __pycache__ | sort`
Expected: All planned files present

---

**Plan complete!**
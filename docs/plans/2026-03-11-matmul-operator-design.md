# Matrix Multiplication Operator Design

## Project Overview

- **Project Name**: matmul_ops
- **Type**: PyTorch Custom Operator Library
- **Core Functionality**: Implement matrix multiplication as a custom autograd function for learning purposes
- **Target Users**: Developers learning PyTorch custom operators

## Requirements

- Python 3.8+
- PyTorch 2.0+
- No batched matrix operations
- No GPU support
- float32 precision only

## Architecture

### Directory Structure

```
matmul_ops/
├── src/
│   └── matmul_ops/
│       ├── __init__.py
│       ├── ops/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── matmul.py
│       │   └── registry.py
│       └── utils/
│           ├── __init__.py
│           └── validators.py
├── tests/
│   ├── __init__.py
│   ├── test_matmul_op.py
│   ├── test_registry.py
│   └── test_validators.py
├── examples/
│   └── basic_usage.py
├── docs/
│   └── api.md
├── pyproject.toml
├── README.md
└── LICENSE
```

### Core Modules

#### 1. ops/base.py
- `BaseOp` abstract base class
- Abstract methods: `forward()`, `backward()`
- Defines interface contract for all operators

#### 2. ops/matmul.py
- `MatMul` class extending `torch.autograd.Function`
- Implements `forward()`: matrix multiplication A @ B
- Implements `backward()`: gradient computation
- Input validation for dimension compatibility

#### 3. ops/registry.py
- `OperatorRegistry` class
- Methods: `register(name, op_class)`, `get(name)`
- Dictionary-based storage

#### 4. utils/validators.py
- `validate_matrix_dimensions(A, B)`: check A.shape[1] == B.shape[0]
- `validate_tensor_type(tensor)`: ensure tensor is float32

## API Design

### Usage

```python
from matmul_ops import MatMul
import torch

A = torch.randn(3, 4, requires_grad=True)
B = torch.randn(4, 5, requires_grad=True)

# Direct usage
result = MatMul.apply(A, B)

# Via registry
from matmul_ops import get_op
op = get_op("matmul")
result = op.apply(A, B)
```

## Testing Strategy

- pytest framework
- Test matrix multiplication correctness (compare with torch.matmul)
- Test gradient correctness (autograd.gradcheck)
- Test error handling for dimension mismatches
- Test registry functionality

## Acceptance Criteria

1. Matrix multiplication produces correct results matching torch.matmul
2. Backward pass computes correct gradients
3. Invalid inputs raise appropriate errors
4. Registry correctly registers and retrieves operators
5. All tests pass
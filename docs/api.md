# API Reference

## MatMul

Custom matrix multiplication operator with autograd support.

### Methods

#### MatMul.apply(A, B)

Matrix multiplication A @ B.

**Parameters:**
- A (torch.Tensor): First matrix of shape (m, n), float32/float64
- B (torch.Tensor): Second matrix of shape (n, k), float32/float64

**Returns:**
- torch.Tensor: Result of shape (m, k)

**Raises:**
- ValueError: If dimensions are incompatible
- TypeError: If tensors are not float32 or float64

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

### validate_tensor_type(tensor, allowed_dtypes=None)

Validate tensor is float32 or float64.
# FastAPI Matrix Multiplication Service Design

## Overview

Add FastAPI HTTP service to expose the MatMul operator via REST API.

## Architecture

```
HTTP Request (JSON)
    ↓
FastAPI Endpoint
    ↓
Pydantic Validation
    ↓
MatMul Operator
    ↓
HTTP Response (JSON)
```

## API Specification

### POST /matmul

**Request Body:**
```json
{
  "A": [[1.0, 2.0], [3.0, 4.0]],
  "B": [[5.0, 6.0], [7.0, 8.0]]
}
```

**Success Response (200):**
```json
{
  "result": [[19.0, 22.0], [43.0, 50.0]]
}
```

**Error Responses:**
- `400`: Matrix dimensions incompatible
- `422`: Invalid tensor dtype (not float32/float64)
- `422`: JSON parse error

## Implementation Details

### Dependencies
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation (included with FastAPI)

### File Structure
```
src/matmul_ops/
├── api/
│   ├── __init__.py
│   ├── main.py           # FastAPI app
│   ├── models.py         # Pydantic models
│   └── routes.py        # API endpoints
```

### Pydantic Models

```python
class MatrixInput(BaseModel):
    A: List[List[float]]
    B: List[List[float]]

class MatrixOutput(BaseModel):
    result: List[List[float]]
```

### Error Handling

Map exceptions to HTTP status codes:
- `ValueError` (dimension mismatch) → 400
- `TypeError` (invalid dtype) → 422

## Deployment

### Running the Server

```bash
# Development
uvicorn matmul_ops.api.main:app --reload --host localhost --port 8000

# Production
uvicorn matmul_ops.api.main:app --host 0.0.0.0 --port 8000
```

### API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Testing

- Unit tests for Pydantic models
- Integration tests for API endpoints
- Error case tests

## Acceptance Criteria

1. POST /matmul returns correct matrix product
2. Invalid dimensions return 400 with descriptive error
3. Invalid dtype returns 422 with descriptive error
4. API documentation accessible at /docs
5. Unit tests pass

# matmul_ops

A PyTorch custom matrix multiplication operator library for learning purposes.

## 功能特性

- 基于 `torch.autograd.Function` 实现自定义算子
- 支持前向传播和反向传播（自动求导）
- 算子注册表模式，便于扩展
- 输入验证（维度检查、类型检查）
- FastAPI HTTP API 支持
- 完整的单元测试覆盖

## 安装

### 本地安装

```bash
pip install -e .
```

### Docker 安装

```bash
# 构建镜像
docker build -t matmul_ops .

# 运行示例
docker run matmul_ops
```

## 快速开始

### 基本用法

```python
import torch
from matmul_ops import MatMul

# 创建输入矩阵 (float32)
A = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
B = torch.randn(4, 5, dtype=torch.float32, requires_grad=True)

# 矩阵乘法
result = MatMul.apply(A, B)

print(f"Result shape: {result.shape}")  # torch.Size([3, 5])

# 反向传播计算梯度
loss = result.sum()
loss.backward()

print(f"A.grad shape: {A.grad.shape}")  # torch.Size([3, 4])
print(f"B.grad shape: {B.grad.shape}")  # torch.Size([4, 5])
```

### 使用注册表

```python
import torch
from matmul_ops import get_op

# 通过注册表获取算子
MatMul = get_op("matmul")

A = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
B = torch.randn(4, 5, dtype=torch.float32, requires_grad=True)

result = MatMul.apply(A, B)
```

### FastAPI 服务

#### 本地启动

```bash
# 安装 dev 依赖（包含 fastapi）
pip install -e ".[dev]"

# 启动服务
uvicorn matmul_ops.api.main:app --reload --host localhost --port 8000
```

#### Docker 启动

```bash
# 构建镜像
docker build -t matmul_ops .

# 运行服务
docker run -p 8000:8000 matmul_ops uvicorn matmul_ops.api.main:app --host 0.0.0.0 --port 8000
```

调用 API：

```bash
# 矩阵乘法
curl -X POST http://localhost:8000/matmul \
  -H "Content-Type: application/json" \
  -d '{"A": [[1, 2], [3, 4]], "B": [[5, 6], [7, 8]]}'

# 返回结果: {"result": [[19.0, 22.0], [43.0, 50.0]]}

# 查看 API 文档
curl http://localhost:8000/
# 返回: {"message":"MatMul API","docs":"/docs"}
```

访问 `http://localhost:8000/docs` 可查看 Swagger API 文档。

## 运行测试

### 本地运行

```bash
pytest tests/
```

### Docker 中运行

```bash
# 构建镜像
docker build -t matmul_ops .

# 运行测试
docker run matmul_ops pytest tests/ -v
```

### 查看测试覆盖率

```bash
pytest tests/ --cov=matmul_ops --cov-report=term-missing
```

## 项目结构

```
matmul_ops/
├── src/matmul_ops/          # 源代码
│   ├── ops/
│   │   ├── base.py          # 基础抽象类
│   │   ├── matmul.py        # 矩阵乘算子实现
│   │   └── registry.py      # 算子注册表
│   ├── api/
│   │   ├── main.py         # FastAPI 应用
│   │   ├── routes.py       # API 路由
│   │   └── models.py       # Pydantic 模型
│   └── utils/
│       └── validators.py    # 输入验证工具
├── tests/                   # 单元测试
│   ├── test_matmul_op.py
│   ├── test_registry.py
│   ├── test_validators.py
│   ├── test_api_models.py
│   └── test_api_routes.py
├── examples/
│   └── basic_usage.py       # 使用示例
└── docs/
    └── api.md               # API 文档
```

## 学习要点

### 1. torch.autograd.Function

自定义算子需要继承 `torch.autograd.Function` 并实现：

- `forward()`: 前向计算
- `backward()`: 反向传播梯度计算

### 2. 梯度计算原理

对于矩阵乘法 `C = A @ B`：

- `∂C/∂A = grad_output @ B^T`
- `∂C/∂B = A^T @ grad_output`

### 3. ctx.save_for_backward

在 `forward()` 中保存需要用于反向传播的张量，避免重复计算。

## 许可证

MIT

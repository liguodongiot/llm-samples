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
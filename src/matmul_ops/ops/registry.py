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
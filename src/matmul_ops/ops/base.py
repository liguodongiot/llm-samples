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
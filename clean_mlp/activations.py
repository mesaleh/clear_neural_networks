import numpy as np
from typing import Union, List
import logging

class Activation:
    """Base class for all activation functions."""
    
    def forward(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the activation function value.

        Args:
            x: Input data (scalar or numpy array).

        Returns:
            Activated output.
        """
        raise NotImplementedError

    def backward(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the derivative of the activation function with respect to its input 'x'.
           Note: 'x' here is typically the *input* to the activation function (often denoted 'z').

        Args:
            x: Input data where the derivative is evaluated (scalar or numpy array).

        Returns:
            Derivative of the activation function evaluated at x.
        """
        raise NotImplementedError


class ReLU(Activation):
    """Rectified Linear Unit activation function.

    Mathematical form:
        forward: f(x) = max(0, x)
        backward: f'(x) = 1 if x > 0 else 0
    """

    def forward(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute ReLU activation: max(0, x)"""
        logging.debug(f"ReLU forward - input shape: {x.shape}")
        result = np.maximum(0, x)
        logging.debug(f"ReLU forward - output shape: {result.shape}")
        return result

    def backward(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute ReLU derivative: 1 if x > 0 else 0"""
        logging.debug(f"ReLU backward - input shape: {x.shape}")
        result = np.where(x > 0, 1.0, 0.0)
        logging.debug(f"ReLU backward - output shape: {result.shape}")
        return result


class Tanh(Activation):
    """Hyperbolic tangent activation function.

    Mathematical form:
        forward: f(x) = tanh(x) = (e^x - e^-x)/(e^x + e^-x)
        backward: f'(x) = 1 - tanh^2(x)
    """

    def forward(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute tanh activation"""
        logging.debug(f"Tanh forward - input shape: {x.shape}")
        result = np.tanh(x)
        logging.debug(f"Tanh forward - output shape: {result.shape}")
        return result

    def backward(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute tanh derivative: 1 - tanh^2(x)"""
        logging.debug(f"Tanh backward - input shape: {x.shape}")
        result = 1.0 - np.tanh(x) ** 2
        logging.debug(f"Tanh backward - output shape: {result.shape}")
        return result


class Sigmoid(Activation):
    """Sigmoid activation function.

    Mathematical form:
        forward: f(x) = 1 / (1 + e^-x)
        backward: f'(x) = f(x) * (1 - f(x))
    """

    def forward(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute sigmoid activation with clipping for numerical stability."""
        logging.debug(f"Sigmoid forward - input shape: {x.shape}")
        # Clip input to avoid overflow in exp(-x) for large negative x
        clipped_x = np.clip(x, -500, 500)
        result = 1.0 / (1.0 + np.exp(-clipped_x))
        logging.debug(f"Sigmoid forward - output shape: {result.shape}")
        return result

    def backward(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute sigmoid derivative using the activation's output."""
        logging.debug(f"Sigmoid backward - input shape: {x.shape}")
        sig = self.forward(x) # Recompute sigmoid output based on input 'x'
        result = sig * (1.0 - sig)
        logging.debug(f"Sigmoid backward - output shape: {result.shape}")
        return result


class Softmax(Activation):
    """Softmax activation function.

    Normalizes outputs to a probability distribution.
    Mathematical form:
        forward: f(x_i) = e^x_i / Σ(e^x_j) for each sample in a batch.

    Backward pass:
        The derivative calculation (d(softmax)/dx) results in a Jacobian matrix.
        However, when Softmax is used as the output layer combined with
        Cross-Entropy loss, the gradient calculation for the backpropagation
        simplifies significantly (dLoss/dx = output_probabilities - target_labels).
        This simplification is typically handled directly when computing the initial
        gradient in the network's backward pass or within the Cross-Entropy loss
        gradient calculation itself.

        Therefore, the `backward` method here returns 1.0 as a placeholder,
        assuming the simplified gradient is computed elsewhere (e.g., in the
        network's loss computation or backward initiation).
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax activation safely using max subtraction trick.

        Args:
            x: Input data (numpy array, typically 2D: batch_size x features).

        Returns:
            Softmax probabilities (same shape as input).
        """
        if x.ndim == 1:
            # Handle 1D array (single sample)
            x = x.reshape(1, -1)

        # Check for and handle NaN or inf values first
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            logging.warning(f"Softmax received NaN or inf inputs: min={np.nanmin(x)}, max={np.nanmax(x)}")
            # Replace NaNs and infs with finite values
            x = np.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)

        # Clip extremely large values to prevent overflow before exp
        x = np.clip(x, -700, 700) # exp(±700) is near the limit of float64

        # Apply max subtraction trick for numerical stability along the feature axis
        x_max = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - x_max)
        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)

        # Handle potential division by zero if all inputs were extremely negative
        sum_exp_x = np.where(sum_exp_x == 0, 1e-15, sum_exp_x)

        result = exp_x / sum_exp_x

        # If original input was 1D, return 1D result
        if result.shape[0] == 1 and x.ndim == 1:
             return result.flatten()
        return result

    def backward(self, x: np.ndarray) -> np.ndarray:
        """Placeholder backward pass for Softmax.

        Returns 1.0, assuming the gradient calculation is handled externally,
        typically simplified when used with Cross-Entropy loss. See class docstring.

        Args:
            x: Input data where the derivative would be evaluated.

        Returns:
            An array of ones with the same shape as x.
        """
        logging.warning("Softmax.backward returns 1.0; assumes simplified gradient calculation with Cross-Entropy Loss elsewhere.")
        return np.ones_like(x)


class Power(Activation):
    """Power activation function.

    Mathematical form:
        forward: f(x) = x^n
        backward: f'(x) = nx^(n-1)
    """

    def __init__(self, n: float = 2.0):
        """Initialize with power value.

        Args:
            n: Power to raise input to (default: 2 for square)
        """
        self.n = n

    def forward(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute power activation"""
        logging.debug(f"Power forward - input shape: {x.shape if isinstance(x, np.ndarray) else 'scalar'}, power: {self.n}")
        result = np.power(x, self.n)
        logging.debug(f"Power forward - output shape: {result.shape}")
        return result

    def backward(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute power derivative"""
        logging.debug(f"Power backward - input shape: {x.shape}")
        # Handle potential issues with negative bases for non-integer powers if needed
        result = self.n * np.power(x, self.n - 1)
        logging.debug(f"Power backward - output shape: {result.shape}")
        return result


class Linear(Activation):
    """Linear activation function (identity).

    Mathematical form:
        forward: f(x) = x
        backward: f'(x) = 1
    """

    def forward(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute linear activation (identity)"""
        return x

    def backward(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute linear derivative (always 1)"""
        # Returns an array of ones with the same shape as the input x
        return np.ones_like(x)


# Dictionary mapping activation function names to their classes
ACTIVATION_FUNCTIONS = {
    'relu': ReLU,
    'tanh': Tanh,
    'sigmoid': Sigmoid,
    'softmax': Softmax,
    'power': Power,
    'linear': Linear
}

def get_activation(name: str, **kwargs) -> Activation:
    """Factory function to get an activation function instance by name.

    Args:
        name: Name of the activation function (case-insensitive).
        **kwargs: Additional arguments to pass to the activation function's constructor
                  (e.g., 'n' for the Power activation).

    Returns:
        An instance of the requested Activation class.

    Raises:
        ValueError: If the activation function name is not recognized.
    """
    name_lower = name.lower()
    if name_lower not in ACTIVATION_FUNCTIONS:
        raise ValueError(
            f"Unknown activation function '{name}'. "
            f"Available functions: {list(ACTIVATION_FUNCTIONS.keys())}"
        )
    return ACTIVATION_FUNCTIONS[name_lower](**kwargs)
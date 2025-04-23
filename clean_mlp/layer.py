import numpy as np
from typing import Optional, Tuple, Union
import logging

from activations import Activation, get_activation

class Layer:
    """
    Represents a single layer in a neural network, performing vectorized operations.

    Conceptually, a layer consists of multiple 'neurons', where each neuron
    computes a weighted sum of its inputs, adds a bias, and applies an activation
    function. This implementation achieves this efficiently using matrix operations
    rather than simulating individual neurons.

    Key Attributes:
        weights (np.ndarray): Weight matrix of shape (output_size, input_size). Each row
                               corresponds to the weights of a conceptual neuron connecting
                               all inputs to its output.
        biases (np.ndarray): Bias vector of shape (output_size,). Each element is the bias
                              for a conceptual neuron in this layer.
        activation_fn (Activation): The activation function applied element-wise to the
                                    weighted sum plus bias.
        z_values (np.ndarray): Stores the linear combination (weighted sum + bias) before
                                activation, computed during the forward pass. Shape: (batch_size, output_size).
        activations (np.ndarray): Stores the output of the layer after applying the activation
                                  function. Shape: (batch_size, output_size).
        inputs (np.ndarray): Stores the input received by the layer during the forward pass.
                             Shape: (batch_size, input_size).
        gradients (np.ndarray): Accumulated gradients of the loss with respect to the weights.
                                Shape: (output_size, input_size).
        bias_gradients (np.ndarray): Accumulated gradients of the loss with respect to the biases.
                                     Shape: (output_size,).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: Union[str, Activation, None] = 'linear',
        weight_init: str = 'xavier',
        use_bias: bool = True,
        initial_weights: Optional[np.ndarray] = None,  # Expected shape (output_size, input_size)
        initial_biases: Optional[np.ndarray] = None,   # Expected shape (output_size,)
        id: int = 0,
    ):
        """
        Initializes the layer.

        Args:
            input_size: Number of input features (size of the previous layer).
            output_size: Number of output features (number of conceptual neurons in this layer).
            activation: Activation function identifier (e.g., 'relu', 'sigmoid') or an Activation instance.
                        Defaults to 'linear'.
            weight_init: Weight initialization strategy ('xavier' or 'random').
            use_bias: Whether to include a bias term for the neurons.
            initial_weights: Optional pre-defined weight matrix. If provided, overrides `weight_init`.
                             Must have shape (output_size, input_size).
            initial_biases: Optional pre-defined bias vector. If provided, overrides default bias initialization.
                            Must have shape (output_size,). Ignored if `use_bias` is False.
            id: An identifier for the layer (optional, for logging/debugging).
        """
        self.input_size = input_size
        self.output_size = output_size
        self.id = id
        self.use_bias = use_bias

        # Build the activation function
        if isinstance(activation, str):
            self.activation_fn = get_activation(activation)
        elif isinstance(activation, Activation):
            self.activation_fn = activation
        elif activation is None:
            self.activation_fn = get_activation('linear')
        else:
            # Fallback to linear if type is unexpected
            logging.warning(f"Invalid activation type '{type(activation)}', using linear.")
            self.activation_fn = get_activation('linear')


        # Initialize weights directly
        if initial_weights is not None:
            if initial_weights.shape != (output_size, input_size):
                 raise ValueError(
                     f"Layer {id}: Initial weights shape {initial_weights.shape} "
                     f"does not match expected shape ({output_size}, {input_size})"
                 )
            self.weights = np.array(initial_weights, dtype=float)
            logging.debug(f"Layer #{self.id}: Using provided initial weights.")
        else:
            if weight_init == 'xavier':
                # Xavier/Glorot initialization: variance = 2 / (fan_in + fan_out)
                # Uniform distribution limits: sqrt(6 / (fan_in + fan_out))
                limit = np.sqrt(6.0 / (input_size + output_size))
                self.weights = np.random.uniform(-limit, limit, (output_size, input_size)).astype(float)
                logging.debug(f"Layer #{self.id}: Initializing weights with Xavier uniform ({limit:.4f}).")
            elif weight_init == 'random':
                # Simple random normal initialization with small scale
                self.weights = np.random.randn(output_size, input_size).astype(float) * 0.01
                logging.debug(f"Layer #{self.id}: Initializing weights with small random normal.")
            else:
                logging.warning(f"Layer #{self.id}: Unknown weight_init '{weight_init}', using small random normal.")
                self.weights = np.random.randn(output_size, input_size).astype(float) * 0.01


        # Initialize biases directly
        if use_bias:
            if initial_biases is not None:
                if initial_biases.shape != (output_size,):
                     raise ValueError(
                         f"Layer {id}: Initial biases shape {initial_biases.shape} "
                         f"does not match expected shape ({output_size},)"
                     )
                self.biases = np.array(initial_biases, dtype=float)
                logging.debug(f"Layer #{self.id}: Using provided initial biases.")
            else:
                # Common practice to initialize biases to zero
                self.biases = np.zeros(output_size, dtype=float)
                logging.debug(f"Layer #{self.id}: Initializing biases to zero.")
        else:
            # If bias is not used, create a zero vector for broadcasting compatibility.
            # It won't be updated during backpropagation if use_bias is False.
            self.biases = np.zeros(output_size, dtype=float)
            logging.debug(f"Layer #{self.id}: Bias not used, setting to zero vector.")

        # Placeholders for values computed during forward pass
        self.z_values    = None  # Shape: (batch_size, output_size) - Linear combination output
        self.activations = None  # Shape: (batch_size, output_size) - Activation function output
        self.inputs      = None  # Shape: (batch_size, input_size) - Inputs to this layer

        # Gradient accumulators (initialized to zero)
        self.gradients      = np.zeros_like(self.weights) # Shape: (output_size, input_size)
        self.bias_gradients = np.zeros_like(self.biases)  # Shape: (output_size,)

        logging.debug(
            f"Layer #{self.id} created: input_size={input_size}, "
            f"output_size={output_size}, activation={self.activation_fn.__class__.__name__}, "
            f"use_bias={use_bias}, weight_shape={self.weights.shape}, bias_shape={self.biases.shape}"
        )

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass through the layer using vectorized operations.

        Computes Z = X @ W.T + b, followed by A = activation_fn(Z).

        Args:
            inputs: Input data matrix of shape (batch_size, input_size).

        Returns:
            Output activations matrix of shape (batch_size, output_size).

        Raises:
            ValueError: If input shape is incorrect.
        """
        # Ensure input is a numpy array and handle 1D input (single sample)
        inputs = np.asarray(inputs, dtype=float)
        if inputs.ndim == 1:
            # If 1D, check if it matches input_size and reshape to (1, input_size)
            if inputs.shape[0] != self.input_size:
                raise ValueError(f"Layer {self.id}: Expected {self.input_size} inputs for a single sample, got {inputs.shape[0]}")
            inputs = inputs.reshape(1, -1) # Reshape to 2D array (1 sample)
        elif inputs.ndim == 2:
            # If 2D, check if the second dimension matches input_size
            if inputs.shape[1] != self.input_size:
                raise ValueError(f"Layer {self.id}: Expected input features {self.input_size}, got {inputs.shape[1]}")
        else:
            raise ValueError(f"Layer {self.id}: Unexpected input dimensions: {inputs.shape}. Expected 1D or 2D array.")

        self.inputs = inputs  # Store inputs for backward pass, shape (batch_size, input_size)

        # --- Vectorized Computation ---
        # 1. Weighted sum: Z = X @ W.T + b
        #    - inputs shape: (batch_size, input_size)
        #    - self.weights.T shape: (input_size, output_size)
        #    - Result shape: (batch_size, output_size)
        #    - self.biases shape: (output_size,) -> Broadcasts to (batch_size, output_size)
        z = np.dot(inputs, self.weights.T) + self.biases
        self.z_values = z # Store pre-activation values, shape (batch_size, output_size)

        # 2. Activation: A = activation_fn(Z)
        #    - Applies the activation function element-wise
        a = self.activation_fn.forward(z)
        self.activations = a # Store activations, shape (batch_size, output_size)

        return a

    def backward(self, gradients: np.ndarray, l2_lambda: float = 0.0) -> np.ndarray:
        """
        Performs the backward pass through the layer using vectorized operations.

        Computes the gradients of the loss with respect to the layer's weights (dW),
        biases (db), and the input to the layer (dX_prev), which is passed to the
        previous layer.

        Args:
            gradients: Gradient of the loss with respect to the output activations
                       of this layer (dL/dA). Shape: (batch_size, output_size).
            l2_lambda: L2 regularization strength (lambda).

        Returns:
            Gradient of the loss with respect to the *inputs* of this layer (dL/dX_prev).
            Shape: (batch_size, input_size).

        Raises:
            ValueError: If the incoming gradient shape is incorrect.
            RuntimeError: If forward pass hasn't been called (needed values are None).
        """
        if self.z_values is None or self.inputs is None:
             raise RuntimeError(f"Layer {self.id}: Must call forward() before backward().")

        # Ensure gradient shape matches layer output
        if gradients.ndim == 1:
            gradients = gradients.reshape(1, -1) # Handle single sample case
        if gradients.shape[1] != self.output_size:
            raise ValueError(f"Layer {self.id}: Expected {self.output_size} incoming gradients, got {gradients.shape[1]}. Shape: {gradients.shape}")
        if gradients.shape[0] != self.inputs.shape[0]:
             raise ValueError(f"Layer {self.id}: Gradient batch size ({gradients.shape[0]}) "
                              f"doesn't match input batch size ({self.inputs.shape[0]}).")

        batch_size = gradients.shape[0]

        # --- Vectorized Backpropagation ---

        # 1. Compute dL/dZ = dL/dA * dA/dZ
        #    - gradients (dL/dA) shape: (batch_size, output_size)
        #    - dA/dZ = self.activation_fn.backward(self.z_values) shape: (batch_size, output_size)
        #    - delta (dL/dZ) shape: (batch_size, output_size)
        dadz = self.activation_fn.backward(self.z_values)

        # Check for NaNs/Infs introduced by activation backward pass
        if np.any(np.isnan(dadz)) or np.any(np.isinf(dadz)):
            logging.warning(f"NaN or Inf detected in activation derivative (dA/dZ) in layer {self.id}")
            dadz = np.nan_to_num(dadz, nan=0.0, posinf=1e6, neginf=-1e6) # Replace with finite numbers

        delta = gradients * dadz # Element-wise multiplication

        # Check for NaNs/Infs after computing delta
        if np.any(np.isnan(delta)) or np.any(np.isinf(delta)):
            logging.warning(f"NaN or Inf detected in delta (dL/dZ) in layer {self.id}")
            delta = np.nan_to_num(delta, nan=0.0, posinf=1e6, neginf=-1e6) # Replace with finite numbers


        # 2. Compute gradient w.r.t. weights: dL/dW = dL/dZ * dZ/dW = delta.T @ X
        #    - delta.T shape: (output_size, batch_size)
        #    - self.inputs (X) shape: (batch_size, input_size)
        #    - dW (self.gradients) shape: (output_size, input_size)
        # Note: This computes the *total* gradient over the batch. Averaging (dividing by batch_size)
        # is often done in the optimizer or update step, but here we compute the sum.
        # Or, more commonly, the learning rate incorporates the 1/batch_size factor.
        # Let's compute the sum here, and let the Network/Optimizer handle scaling.
        dW = np.dot(delta.T, self.inputs)
        self.gradients = dW # Store/accumulate gradient

        # 3. Add L2 regularization gradient term (penalty derivative)
        #    - d(0.5 * lambda * W^2) / dW = lambda * W
        #    - We scale by batch_size to match common loss scaling conventions
        #      (where loss is averaged over batch).
        #      The corresponding L2 loss term added in Network.compute_loss should also
        #      be scaled appropriately (e.g., 0.5 * lambda / batch_size * sum(W^2)).
        if l2_lambda > 0.0:
            l2_grad = l2_lambda * self.weights
            self.gradients += l2_grad
            # Note: Regularization is typically *not* applied to biases.

        # 4. Compute gradient w.r.t. bias: dL/db = dL/dZ * dZ/db = delta * 1
        #    - Sum delta over the batch dimension.
        #    - db (self.bias_gradients) shape: (output_size,)
        if self.use_bias:
            db = np.sum(delta, axis=0)
            self.bias_gradients = db # Store/accumulate gradient
        # else: self.bias_gradients remains zero as initialized

        # 5. Compute gradient to pass to the previous layer: dL/dX_prev = dL/dZ * dZ/dX = delta @ W
        #    - delta shape: (batch_size, output_size)
        #    - self.weights shape: (output_size, input_size)
        #    - dX_prev shape: (batch_size, input_size)
        prev_layer_grad = np.dot(delta, self.weights)

        # Check for NaNs/Infs in the gradient being passed back
        if np.any(np.isnan(prev_layer_grad)) or np.any(np.isinf(prev_layer_grad)):
            logging.warning(f"NaN or Inf detected in gradient passed back from layer {self.id}")
            prev_layer_grad = np.nan_to_num(prev_layer_grad, nan=0.0, posinf=1e6, neginf=-1e6)

        return prev_layer_grad

    def update(self, learning_rate: float, batch_size: int):
        """
        Updates the layer's weights and biases using the accumulated gradients.

        Applies the update rule: W = W - learning_rate * (dL/dW / batch_size)
                                b = b - learning_rate * (dL/db / batch_size)

        Args:
            learning_rate: The learning rate hyperparameter.
            batch_size: The number of samples in the batch used to compute gradients.
                        Used for averaging the accumulated gradients.
        """
        if batch_size <= 0:
            raise ValueError("Batch size must be positive for gradient averaging.")

        # Average gradients over the batch
        avg_gradients = self.gradients / batch_size
        avg_bias_gradients = self.bias_gradients / batch_size

        # Check for very large gradients before update (optional: gradient clipping could be done here)
        grad_norm = np.linalg.norm(avg_gradients)
        if grad_norm > 1e6: # Arbitrary threshold for large gradients
            logging.warning(f"Layer {self.id}: Large gradient norm detected ({grad_norm:.2e}) before update.")
            # Optional: Clip gradients if they explode
            # clip_threshold = 10.0
            # avg_gradients = np.clip(avg_gradients, -clip_threshold, clip_threshold)
            # avg_bias_gradients = np.clip(avg_bias_gradients, -clip_threshold, clip_threshold)


        # Update weights
        self.weights -= learning_rate * avg_gradients

        # Update biases (only if used)
        if self.use_bias:
            self.biases -= learning_rate * avg_bias_gradients

        # No need to sync neurons anymore as they don't exist separately.

    def zero_grad(self):
        """Resets the accumulated gradients for weights and biases to zero."""
        self.gradients.fill(0.0)
        if self.use_bias: # Technically not needed as bias_gradients is zero if not use_bias
            self.bias_gradients.fill(0.0)

    def get_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the current weights and biases of the layer."""
        return self.weights.copy(), self.biases.copy()

    def get_gradients(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the current accumulated gradients for weights and biases."""
        return self.gradients.copy(), self.bias_gradients.copy()

    def summary(self) -> str:
        """Returns a string summary of the layer's configuration."""
        params = self.weights.size + (self.biases.size if self.use_bias else 0)
        return (
            f"Layer Summary (id={self.id}):\n"
            f"  Type: Fully Connected\n"
            f"  Input size: {self.input_size}\n"
            f"  Output size: {self.output_size}\n"
            f"  Activation: {self.activation_fn.__class__.__name__}\n"
            f"  Use Bias: {self.use_bias}\n"
            f"  Weights shape: {self.weights.shape}\n"
            f"  Biases shape: {self.biases.shape if self.use_bias else 'N/A'}\n"
            f"  Parameters: {params:,} parameters\n"
        )

    def __repr__(self):
        """Returns a concise string representation of the layer object."""
        return (f"Layer(id={self.id}, input_size={self.input_size}, "
                f"output_size={self.output_size}, "
                f"activation={self.activation_fn.__class__.__name__}, "
                f"use_bias={self.use_bias})")
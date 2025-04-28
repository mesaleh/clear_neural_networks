import numpy as np
from typing import List, Tuple, Union, Dict, Optional, Callable
import logging
import time
from datetime import datetime

from layer import Layer
from activations import get_activation, Activation, Softmax # Import Softmax to check instance type

# --- Loss Functions ---

LossFunctionType = Callable[[np.ndarray, np.ndarray], Tuple[float, np.ndarray]]

def mse_loss(outputs: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Computes Mean Squared Error loss and its gradient w.r.t. outputs.

    Loss = (1/N) * Σ(output_i - target_i)^2
    Gradient (dL/dOutput) = (2/N) * (output - target)

    Args:
        outputs: Predicted values (batch_size, output_dim).
        targets: True values (batch_size, output_dim).

    Returns:
        Tuple containing:
            - mse_value (float): The mean squared error.
            - gradient (np.ndarray): Gradient of MSE w.r.t. outputs.
    """
    if outputs.shape != targets.shape:
        raise ValueError(f"MSE Loss: Output shape {outputs.shape} must match target shape {targets.shape}")
    num_samples = outputs.shape[0]
    if num_samples == 0:
        return 0.0, np.zeros_like(outputs)

    error = outputs - targets
    loss = np.mean(error ** 2) # np.mean already handles the (1/N) factor

    # Gradient w.r.t. outputs
    gradient = 2 * error / num_samples
    return float(loss), gradient


def cross_entropy_loss(outputs: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Computes Cross-Entropy loss for multi-class classification (with Softmax output)
    and its gradient w.r.t. the *inputs* to the Softmax (often denoted 'z').

    Assumes 'outputs' are probabilities from Softmax and 'targets' are one-hot encoded.
    Loss = - (1/N) * Σ_samples Σ_classes [ target * log(output) ]

    Gradient (dL/dz) = (1/N) * (output - target)
        This is the simplified gradient w.r.t. the *pre-activation* values (z)
        of the final Softmax layer. It combines dL/da and da/dz.

    Args:
        outputs: Predicted probabilities from Softmax (batch_size, num_classes).
        targets: True labels, one-hot encoded (batch_size, num_classes).

    Returns:
        Tuple containing:
            - ce_value (float): The cross-entropy loss.
            - gradient (np.ndarray): Simplified gradient (dL/dz) w.r.t. Softmax inputs.
    """
    if outputs.shape != targets.shape:
        raise ValueError(f"CE Loss: Output shape {outputs.shape} must match target shape {targets.shape}")
    num_samples = outputs.shape[0]
    if num_samples == 0:
        return 0.0, np.zeros_like(outputs)

    # Clip outputs to avoid log(0)
    epsilon = 1e-15
    outputs_clipped = np.clip(outputs, epsilon, 1.0 - epsilon)

    # Compute cross-entropy loss
    # Element-wise multiplication and sum over classes, then mean over samples
    loss = -np.sum(targets * np.log(outputs_clipped)) / num_samples

    # Compute the simplified gradient (dL/dz = output - target) / N
    # This gradient is intended to be passed directly *into* the final layer's
    # backward pass, bypassing its activation.backward() step IF it's Softmax.
    gradient = (outputs - targets) / num_samples # Note: using original outputs, not clipped

    return float(loss), gradient


def binary_cross_entropy_loss(outputs: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Computes Binary Cross-Entropy loss (for Sigmoid output) and its gradient w.r.t. outputs.

    Typically used for binary classification or multi-label classification with Sigmoid outputs.
    Loss = - (1/N) * Σ_samples [ target * log(output) + (1 - target) * log(1 - output) ]
    Gradient (dL/dOutput) = (1/N) * [ -target/output + (1 - target)/(1 - output) ]

    Args:
        outputs: Predicted probabilities from Sigmoid (batch_size, num_outputs).
        targets: True labels (0 or 1) (batch_size, num_outputs).

    Returns:
        Tuple containing:
            - bce_value (float): The binary cross-entropy loss.
            - gradient (np.ndarray): Gradient of BCE w.r.t. outputs.
    """
    if outputs.shape != targets.shape:
        raise ValueError(f"BCE Loss: Output shape {outputs.shape} must match target shape {targets.shape}")
    num_samples = outputs.shape[0]
    if num_samples == 0:
        return 0.0, np.zeros_like(outputs)

    # Clip outputs to avoid log(0) or division by zero in gradient
    epsilon = 1e-15
    outputs_clipped = np.clip(outputs, epsilon, 1.0 - epsilon)

    # Compute BCE loss
    term1 = targets * np.log(outputs_clipped)
    term2 = (1 - targets) * np.log(1 - outputs_clipped)
    loss = -np.sum(term1 + term2) / num_samples

    # Compute gradient w.r.t. outputs
    grad_term1 = -targets / outputs_clipped
    grad_term2 = (1 - targets) / (1 - outputs_clipped)
    gradient = (grad_term1 + grad_term2) / num_samples

    return float(loss), gradient


# Dictionary mapping loss_type strings to loss functions
LOSS_FUNCTIONS: Dict[str, LossFunctionType] = {
    "mse": mse_loss,
    "cross_entropy": cross_entropy_loss,
    "binary_cross_entropy": binary_cross_entropy_loss
}

class Network:
    """
    A simple Feedforward Neural Network (Multilayer Perceptron).

    Manages a sequence of layers, the forward pass, backward pass (backpropagation),
    parameter updates, training loop, prediction, and evaluation.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activations: Optional[Union[List[str], List[Activation]]] = None,
        weight_init: str = 'xavier',
        use_bias: bool = True,
        initial_layer_weights: Optional[List[np.ndarray]] = None, # List of (output_size, input_size) arrays
        initial_layer_biases: Optional[List[np.ndarray]] = None,  # List of (output_size,) arrays
    ):
        """
        Initializes the neural network.

        Args:
            layer_sizes: A list of integers specifying the number of neurons in each layer,
                         starting with the input dimension and ending with the output dimension.
                         Example: [784, 128, 64, 10] for MNIST.
            activations: A list of activation function names (strings like 'relu', 'sigmoid')
                         or Activation instances, one for each layer *except* the input layer.
                         If None, defaults to 'linear' for all layers. Must have length
                         len(layer_sizes) - 1.
            weight_init: Weight initialization strategy ('xavier' or 'random') passed to layers.
            use_bias: Whether layers should use bias terms. Passed to layers.
            initial_layer_weights: Optional list of pre-defined weight matrices for each layer.
                                   The list should have length len(layer_sizes) - 1.
            initial_layer_biases: Optional list of pre-defined bias vectors for each layer.
                                  The list should have length len(layer_sizes) - 1.
        """
        if len(layer_sizes) < 2:
            raise ValueError("Network must have at least an input and an output layer size.")

        num_layers = len(layer_sizes) - 1

        # Setup activations
        if activations is None:
            self.activations = ['linear'] * num_layers
        elif len(activations) != num_layers:
            raise ValueError(f"Number of activation functions ({len(activations)}) must match "
                             f"number of layers ({num_layers}).")
        else:
            self.activations = activations

        # Validate initial weights/biases lengths if provided
        if initial_layer_weights is not None and len(initial_layer_weights) != num_layers:
            raise ValueError(f"Length of initial_layer_weights ({len(initial_layer_weights)}) "
                             f"must match number of layers ({num_layers}).")
        if initial_layer_biases is not None and len(initial_layer_biases) != num_layers:
            raise ValueError(f"Length of initial_layer_biases ({len(initial_layer_biases)}) "
                             f"must match number of layers ({num_layers}).")

        self.layers: List[Layer] = []
        for i in range(num_layers):
            # Get activation for this layer
            layer_activation = self.activations[i]

            # Get initial weights/biases if provided
            w_init = initial_layer_weights[i] if initial_layer_weights else None
            b_init = initial_layer_biases[i] if initial_layer_biases else None

            layer = Layer(
                input_size      = layer_sizes[i],
                output_size     = layer_sizes[i+1],
                activation      = layer_activation,
                weight_init     = weight_init,
                use_bias        = use_bias,
                initial_weights = w_init,   # Shape (output_size, input_size)
                initial_biases  = b_init,    # Shape (output_size,)
                id              = i # Layer index (0 to num_layers-1)
            )
            self.layers.append(layer)

        # Training history tracking
        self.training_history: Dict[str, List] = {
            'epoch': [],
            'loss': [],
            'val_loss': [],
            'learning_rate': [],
            'batch_size': [],
            'time_per_epoch': []
        }

        logging.info(f"Created neural network with architecture: {layer_sizes}")
        logging.info(f"Layer activations: {[l.activation_fn.__class__.__name__ for l in self.layers]}")

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass through all layers of the network.

        Args:
            inputs: Input data matrix of shape (batch_size, input_dim).

        Returns:
            Network output matrix of shape (batch_size, output_dim).
        """
        current_output = inputs
        for i, layer in enumerate(self.layers):
            logging.debug(f"Forward pass - Layer {i} input shape: {current_output.shape}")
            current_output = layer.forward(current_output)
            logging.debug(f"Forward pass - Layer {i} output shape: {current_output.shape}")
        return current_output

    def backward(self, initial_gradient: np.ndarray, loss_type: str, l2_lambda: float = 0.0):
        """
        Performs a backward pass (backpropagation) through all layers.

        Calculates gradients for weights and biases in each layer.

        Args:
            initial_gradient: The gradient of the loss function with respect to the
                              network's output (dL/dA_final) or, in the special case of
                              Softmax+CrossEntropy, the simplified gradient (dL/dZ_final).
            loss_type: The string identifier of the loss function used (e.g., "cross_entropy").
                       Used to handle the Softmax+CE optimization.
            l2_lambda: L2 regularization strength.
        """
        current_gradient = initial_gradient
        last_layer = self.layers[-1]

        # --- Special Handling for Softmax + Cross-Entropy ---
        # If the last layer is Softmax and the loss is Cross-Entropy,
        # the 'initial_gradient' received from cross_entropy_loss IS ALREADY dL/dZ_final.
        # In this specific case, we bypass the last layer's activation backward calculation.
        # The gradient is passed directly to the Layer.backward method, which computes
        # dW, db, and dX_prev based on dL/dZ (delta).

        is_softmax_ce_case = (isinstance(last_layer.activation_fn, Softmax) and
                              loss_type == "cross_entropy")

        if is_softmax_ce_case:
            logging.debug("Applying Softmax+CE gradient simplification.")
            # The initial_gradient IS dL/dZ for the last layer.
            # We call the last layer's backward directly with this gradient.
            # Note: Layer.backward internally calculates delta = received_gradient * activation_backward().
            #       Since our received_gradient IS dL/dZ, we *don't* want the activation_backward() multiply.
            #       However, Softmax.backward() is designed to return 1.0, so the multiplication
            #       delta = dLdZ * 1.0 = dLdZ has no effect, achieving the desired outcome.
            pass # Gradient is already dL/dZ, proceed as normal with the adjusted Softmax.backward

        logging.debug(f"Backward pass starting with gradient shape: {current_gradient.shape}")

        # Propagate gradient backwards through layers
        for i, layer in enumerate(reversed(self.layers)):
            layer_index = len(self.layers) - 1 - i
            logging.debug(f"Backward pass - Layer {layer_index} receiving gradient shape: {current_gradient.shape}")
            current_gradient = layer.backward(current_gradient, l2_lambda=l2_lambda)
            logging.debug(f"Backward pass - Layer {layer_index} passing gradient shape: {current_gradient.shape}")
            # Note: The returned gradient is dL/dX for *this* layer, which is dL/dA for the *previous* layer.

    def compute_loss(self, outputs: np.ndarray, targets: np.ndarray, loss_type: str = "mse", l2_lambda: float = 0.0) -> Tuple[float, np.ndarray]:
        """
        Computes the specified loss and the initial gradient for backpropagation.

        Handles the special case for Softmax + Cross-Entropy where the returned
        gradient is dL/dZ (output - target) instead of dL/dA.

        Args:
            outputs: Network predictions (batch_size, output_dim).
            targets: True labels (batch_size, output_dim).
            loss_type: Identifier for the loss function ('mse', 'cross_entropy', 'binary_cross_entropy').
            l2_lambda: L2 regularization strength.

        Returns:
            Tuple containing:
                - total_loss (float): Loss value including L2 regularization.
                - initial_gradient (np.ndarray): Gradient to start backpropagation (dL/dA or dL/dZ).
        """
        # Step 1: Select the appropriate loss function
        if loss_type not in LOSS_FUNCTIONS:
            raise ValueError(f"Unsupported loss_type '{loss_type}'. "
                             f"Valid options: {list(LOSS_FUNCTIONS.keys())}")
        loss_fn = LOSS_FUNCTIONS[loss_type]

        # Step 2: Compute the base loss and its gradient
        # For Cross-Entropy, loss_fn returns the simplified dL/dZ gradient.
        # For others (MSE, BCE), it returns dL/dA.
        base_loss, initial_gradient = loss_fn(outputs, targets)

        # Check for NaNs in base loss or gradient
        if np.isnan(base_loss):
             logging.error(f"NaN detected in base loss calculation (loss_type: {loss_type})")
             base_loss = np.nan_to_num(base_loss, nan=1e6) # Replace NaN loss
        if np.any(np.isnan(initial_gradient)):
            logging.error(f"NaN detected in initial gradient from loss function (loss_type: {loss_type})")
            initial_gradient = np.nan_to_num(initial_gradient, nan=0.0) # Replace NaN gradient elements

        # Step 3: Compute and add L2 regularization term to the loss
        reg_term = 0.0
        if l2_lambda > 0.0:
            batch_size = outputs.shape[0]
            if batch_size > 0:
                sum_sq_weights = 0.0
                for layer in self.layers:
                    sum_sq_weights += np.sum(layer.weights**2)
                # The factor 0.5 makes the derivative simply lambda * W
                # Divide by batch_size to keep loss scale consistent if base loss is averaged.
                reg_term = (0.5 * l2_lambda / batch_size) * sum_sq_weights
            else:
                 reg_term = 0.0 # Avoid division by zero if batch_size is 0

        total_loss = base_loss + reg_term
        return total_loss, initial_gradient

    def update(self, learning_rate: float, batch_size: int):
        """
        Updates the parameters (weights and biases) of all layers using their accumulated gradients.

        Args:
            learning_rate: The learning rate for the update step.
            batch_size: The number of samples in the batch, used for averaging gradients.
        """
        if batch_size <= 0:
             logging.warning(f"Attempting update with non-positive batch_size ({batch_size}). Skipping update.")
             return
        for i, layer in enumerate(self.layers):
            logging.debug(f"Updating layer {i}")
            layer.update(learning_rate, batch_size)

    def zero_grad(self):
        """Resets the gradients of all layers to zero before processing a new batch."""
        for layer in self.layers:
            layer.zero_grad()

    def train_batch(self, X_batch: np.ndarray, y_batch: np.ndarray, learning_rate: float, loss_type: str = "mse", l2_lambda: float = 0.0) -> float:
        """
        Trains the network on a single batch of data.

        Performs forward pass, loss calculation, backward pass, and parameter update.

        Args:
            X_batch: Input data for the batch (batch_size, input_dim).
            y_batch: Target labels for the batch (batch_size, output_dim).
            learning_rate: Learning rate for this update.
            loss_type: Loss function identifier.
            l2_lambda: L2 regularization strength.

        Returns:
            The computed loss for this batch (including regularization).
        """
        # 0. Reset gradients from previous batch
        self.zero_grad()

        # Check for NaN/Inf in input data
        if np.any(np.isnan(X_batch)) or np.any(np.isinf(X_batch)):
            logging.warning("NaN or Inf detected in input batch X_batch. Attempting to clean.")
            X_batch = np.nan_to_num(X_batch, nan=0.0, posinf=1e6, neginf=-1e6)

        # 1. Forward pass
        outputs = self.forward(X_batch)

        # Check for NaN/Inf in network output
        if np.any(np.isnan(outputs)) or np.any(np.isinf(outputs)):
            logging.warning("NaN or Inf detected in network output during training. Check weights/activations.")
            # Attempt to recover if possible, though this often indicates deeper issues
            outputs = np.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=0.0) # Clip for probability outputs

        # 2. Compute loss and initial gradient
        # Note: outputs might be clipped inside loss function (e.g., for log)
        loss, initial_gradient = self.compute_loss(outputs, y_batch, loss_type=loss_type, l2_lambda=l2_lambda)

        # Check for NaN/Inf in initial gradient
        if np.any(np.isnan(initial_gradient)) or np.any(np.isinf(initial_gradient)):
            logging.error("NaN or Inf detected in initial gradient after loss computation. Stopping batch.")
            # Returning a high loss might be better than proceeding with bad gradients
            return float(np.nan_to_num(loss, nan=1e6)) # Return high loss value

        # 3. Backward pass (backpropagation)
        self.backward(initial_gradient, loss_type, l2_lambda=l2_lambda)

        # 4. Update parameters
        batch_size = X_batch.shape[0]
        self.update(learning_rate, batch_size)

        return loss # Return the total loss for this batch

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        validation_split: float = 0.0,
        shuffle: bool = True,
        verbose: bool = True,
        log_every: int = 10,
        loss_type: str = "mse",
        l2_lambda: float = 0.0
    ) -> Dict[str, List]:
        """
        Trains the network for a specified number of epochs using batch gradient descent.

        Args:
            X: Training input data (num_samples, input_dim).
            y: Training target data (num_samples, output_dim).
            epochs: Number of training epochs.
            batch_size: Size of mini-batches for training.
            learning_rate: Learning rate for parameter updates.
            validation_split: Fraction of training data to use for validation (0.0 to 1.0).
            shuffle: Whether to shuffle the training data before each epoch.
            verbose: Whether to print training progress.
            log_every: Print progress every `log_every` epochs.
            loss_type: Loss function identifier ('mse', 'cross_entropy', 'binary_cross_entropy').
            l2_lambda: L2 regularization strength.

        Returns:
            A dictionary containing the training history (loss, val_loss, etc.).
        """
        X = np.asarray(X)
        y = np.asarray(y)
        num_samples = X.shape[0]

        if y.shape[0] != num_samples:
             raise ValueError("Number of samples in X and y must match.")

        # Prepare training and validation sets
        if validation_split > 0.0 and validation_split < 1.0:
            split_idx = int(num_samples * (1.0 - validation_split))
            # Shuffle data before splitting to ensure random validation set
            indices = np.random.permutation(num_samples)
            X_shuffled, y_shuffled = X[indices], y[indices]
            X_train, X_val = X_shuffled[:split_idx], X_shuffled[split_idx:]
            y_train, y_val = y_shuffled[:split_idx], y_shuffled[split_idx:]
            logging.info(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples.")
        elif validation_split == 0.0:
            X_train, y_train = X, y
            X_val, y_val = None, None
            logging.info(f"Training on {len(X_train)} samples, no validation split.")
        else:
            raise ValueError("validation_split must be between 0.0 and 1.0")

        num_train_samples = X_train.shape[0]

        # Adjust batch size if it's larger than the training set
        if batch_size > num_train_samples:
            logging.warning(f"Batch size ({batch_size}) is larger than training set size ({num_train_samples}). Setting batch size to {num_train_samples}.")
            batch_size = num_train_samples
        if batch_size <= 0:
            logging.warning(f"Batch size ({batch_size}) is not positive. Setting batch size to {num_train_samples}.")
            batch_size = num_train_samples

        n_batches = max(1, num_train_samples // batch_size) # Ensure at least one batch

        # --- Training Loop ---
        for epoch in range(epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0

            # Shuffle training data at the beginning of each epoch
            if shuffle:
                indices = np.random.permutation(num_train_samples)
                X_train = X_train[indices]
                y_train = y_train[indices]

            # Iterate over mini-batches
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                # Handle the last batch which might be smaller
                end_idx = min(start_idx + batch_size, num_train_samples)
                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]

                if len(X_batch) > 0: # Ensure batch is not empty
                    batch_loss = self.train_batch(X_batch, y_batch, learning_rate, loss_type, l2_lambda)
                    epoch_loss += batch_loss * len(X_batch) # Weight loss by batch size for accurate epoch average

            # Calculate average epoch loss
            epoch_loss /= num_train_samples
            epoch_time = time.time() - epoch_start_time

            # Calculate validation loss if validation set exists
            val_loss = None
            if X_val is not None and y_val is not None and len(X_val) > 0:
                val_outputs = self.predict(X_val)
                # Use compute_loss to get validation loss (gradient is ignored)
                val_loss, _ = self.compute_loss(val_outputs, y_val, loss_type=loss_type, l2_lambda=0.0) # No regularization for validation loss

            # Record history for this epoch
            self.training_history['epoch'].append(epoch)
            self.training_history['loss'].append(epoch_loss)
            self.training_history['val_loss'].append(val_loss) # Will be None if no validation
            self.training_history['learning_rate'].append(learning_rate)
            self.training_history['batch_size'].append(batch_size)
            self.training_history['time_per_epoch'].append(epoch_time)

            # Logging progress
            if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
                msg = f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.5f}"
                if val_loss is not None:
                    msg += f" - val_loss: {val_loss:.5f}"
                msg += f" - time: {epoch_time:.2f}s"
                # Use print instead of logging.info for potentially long training runs
                # to avoid excessive log file size, unless logging is configured differently.
                print(msg)
                # logging.info(msg) # Use this if prefer logging

        logging.info("Training finished.")
        return self.training_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generates predictions for the input data X.

        Args:
            X: Input data (num_samples, input_dim).

        Returns:
            Network predictions (num_samples, output_dim).
        """
        if X.ndim == 1:
            # Handle single sample input
            X = X.reshape(1, -1)
        elif X.ndim != 2:
             raise ValueError(f"Input X must be a 1D or 2D array, got {X.ndim}D.")

        # Perform forward pass without storing gradients etc. (though current implementation does)
        # For pure prediction, a separate method could avoid storing intermediate values.
        return self.forward(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray, loss_type: str = 'mse') -> Dict[str, float]:
        """
        Evaluates the network performance on the given data using specified loss.

        Calculates loss and R-squared (for regression).
        Note: For classification tasks, additional metrics like accuracy, precision,
              recall, F1-score should be calculated separately based on predictions.

        Args:
            X: Input data (num_samples, input_dim).
            y: True labels/values (num_samples, output_dim).
            loss_type: Loss function identifier to calculate the primary loss metric.

        Returns:
            A dictionary containing evaluation metrics (e.g., 'loss', 'rmse', 'r2').
        """
        predictions = self.predict(X)
        num_samples = X.shape[0]

        # Calculate the primary loss value
        loss, _ = self.compute_loss(predictions, y, loss_type=loss_type, l2_lambda=0.0) # No regularization for evaluation loss

        eval_metrics = {'loss': loss}

        # --- Regression Metrics (Example) ---
        # These are suitable if the task is regression.
        if y.ndim == predictions.ndim and y.shape == predictions.shape: # Basic check
             try:
                 eval_metrics['rmse'] = np.sqrt(mse_loss(predictions, y)[0]) # Calculate MSE specifically for RMSE

                 # R-squared calculation
                 if num_samples > 1:
                     ss_res = np.sum((y - predictions) ** 2)
                     ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
                     if ss_tot == 0: # Avoid division by zero if y is constant
                         r2 = 0.0 if ss_res == 0 else -np.inf # Or undefined, depends on convention
                     else:
                         r2 = 1.0 - (ss_res / ss_tot)
                     eval_metrics['r2'] = r2
                 else:
                     eval_metrics['r2'] = np.nan # R2 not well-defined for single sample
             except Exception as e:
                 logging.warning(f"Could not compute regression metrics (RMSE, R2): {e}")


        # --- Classification Metrics Placeholder ---
        # If loss_type suggests classification (e.g., cross_entropy, binary_cross_entropy),
        # you would typically calculate accuracy etc. here.
        # Example:
        if loss_type in ["cross_entropy", "binary_cross_entropy"]:
            logging.info("Evaluation for classification task. Consider adding accuracy, precision, recall, F1 metrics.")
            # Example Accuracy (assuming classification output)
            try:
                if loss_type == "cross_entropy": # Multi-class
                    pred_labels = np.argmax(predictions, axis=1)
                    true_labels = np.argmax(y, axis=1) # Assumes y is one-hot
                    accuracy = np.mean(pred_labels == true_labels)
                    eval_metrics['accuracy'] = accuracy
                elif loss_type == "binary_cross_entropy": # Binary/Multi-label
                    # Thresholding needed for accuracy (e.g., at 0.5 for binary)
                    pred_labels = (predictions > 0.5).astype(int)
                    true_labels = y.astype(int)
                    # Accuracy might need careful definition for multi-label (e.g., exact match ratio)
                    # Simple element-wise accuracy:
                    accuracy = np.mean(pred_labels == true_labels)
                    eval_metrics['accuracy'] = accuracy # Or a more suitable multi-label metric
            except Exception as e:
                 logging.warning(f"Could not compute classification accuracy: {e}")

        return eval_metrics


    def save_weights(self, filename: str):
        """
        Saves the network's weights, biases, and architecture details to a compressed .npz file.

        Args:
            filename: Path to the file where weights will be saved. '.npz' extension is recommended.
        """
        save_dict = {
            'layer_sizes': np.array( [self.layers[0].input_size] + [l.output_size for l in self.layers] ),
            'use_bias': np.array(self.layers[0].use_bias), # Assuming uniform use_bias across layers for simplicity
            'weight_init': self.layers[0].weight_init if hasattr(self.layers[0], 'weight_init') else 'unknown' # Store original init strategy if available
        }

        activation_names = []
        for i, layer in enumerate(self.layers):
            save_dict[f'layer_{i}_weights'] = layer.weights
            save_dict[f'layer_{i}_biases'] = layer.biases
            activation_names.append(layer.activation_fn.__class__.__name__)

        # Store activation names as a single string array (requires object dtype)
        save_dict['activation_names'] = np.array(activation_names, dtype=object)

        if not filename.endswith('.npz'):
            filename += '.npz'

        try:
            np.savez_compressed(filename, **save_dict)
            logging.info(f"Network weights and configuration saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving weights to {filename}: {e}")

    @classmethod
    def load_weights(cls, filename: str) -> 'Network':
        """
        Loads network weights, biases, and configuration from an .npz file
        and creates a new Network instance.

        Args:
            filename: Path to the .npz file containing the saved network state.

        Returns:
            A new Network instance initialized with the loaded state.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file structure is incompatible or incomplete.
        """
        try:
            data = np.load(filename, allow_pickle=True) # Allow pickle for object array (activation names)
        except FileNotFoundError:
            logging.error(f"Weight file not found: {filename}")
            raise
        except Exception as e:
            logging.error(f"Error loading weight file {filename}: {e}")
            raise ValueError(f"Could not load data from {filename}") from e

        try:
            layer_sizes = data['layer_sizes'].tolist()
            activation_names = data['activation_names'].tolist()
            use_bias = bool(data['use_bias'].item()) # Use .item() for 0-dim array
            weight_init = str(data['weight_init'].item()) if 'weight_init' in data else 'unknown' # Load if saved

            if len(activation_names) != len(layer_sizes) - 1:
                 raise ValueError("Mismatch between number of activations and layer sizes in loaded file.")

            # Prepare initial weights and biases lists
            initial_weights = []
            initial_biases = []
            num_layers = len(layer_sizes) - 1
            for i in range(num_layers):
                w_key = f'layer_{i}_weights'
                b_key = f'layer_{i}_biases'
                if w_key not in data or b_key not in data:
                     raise ValueError(f"Missing weights ({w_key}) or biases ({b_key}) for layer {i} in file.")
                initial_weights.append(data[w_key])
                initial_biases.append(data[b_key])

            # Create the network instance
            network = cls(
                layer_sizes=layer_sizes,
                activations=activation_names, # Pass names to constructor
                use_bias=use_bias,
                weight_init=weight_init, # Pass original init type if needed
                initial_layer_weights=initial_weights,
                initial_layer_biases=initial_biases
            )
            logging.info(f"Network loaded successfully from {filename}")

        except KeyError as e:
            logging.error(f"Missing expected key in weight file {filename}: {e}")
            raise ValueError(f"Incompatible or incomplete weight file: {filename}") from e
        except Exception as e:
            logging.error(f"An error occurred during network reconstruction from {filename}: {e}")
            raise ValueError(f"Failed to reconstruct network from {filename}") from e
        finally:
            data.close() # Ensure the file handle is closed

        return network

    def summary(self) -> str:
        """
        Generates a text summary of the network architecture and parameters.

        Returns:
            A string containing the network summary.
        """
        summary_str = "\n" + "="*50 + "\n"
        summary_str += "Neural Network Summary\n"
        summary_str += "="*50 + "\n"
        total_params = 0
        for i, layer in enumerate(self.layers):
            layer_params = layer.weights.size + (layer.biases.size if layer.use_bias else 0)
            total_params += layer_params
            summary_str += f"Layer {i}: {layer.__class__.__name__} (ID: {layer.id})\n"
            summary_str += f"  Input Shape: ({layer.input_size},)\n"
            summary_str += f"  Output Shape: ({layer.output_size},)\n"
            summary_str += f"  Activation: {layer.activation_fn.__class__.__name__}\n"
            summary_str += f"  Use Bias: {layer.use_bias}\n"
            summary_str += f"  Weight Shape: {layer.weights.shape}\n"
            summary_str += f"  Bias Shape: {layer.biases.shape if layer.use_bias else 'N/A'}\n"
            summary_str += f"  Parameters: {layer_params}\n"
            summary_str += "-"*50 + "\n"

        summary_str += f"Total Parameters: {total_params}\n"
        summary_str += "="*50 + "\n"
        return summary_str
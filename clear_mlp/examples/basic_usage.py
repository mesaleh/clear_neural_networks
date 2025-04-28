import os
import sys
import time
import logging # <-- Import standard logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons # Using this for the main example

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from network import Network

# --- Plotting Function ---

def plot_decision_boundary(X: np.ndarray, y_raw: np.ndarray, model: Network):
    """Plots the decision boundary of a trained model.

    Args:
        X: Input features used for training (for axis limits and plotting points).
           Shape (n_samples, 2). Assumed to be normalized if model was trained on normalized data.
        y_raw: True integer class labels for the input features. Shape (n_samples,).
        model: Trained Network instance.
    """
    h = 0.02 # Step size in the mesh

    # Define bounds of the plot, based on data range
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict classifications for each point in mesh grid
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z_probs = model.predict(mesh_points)

    # Convert probabilities to class labels
    if Z_probs.shape[1] > 1:
        # Multi-class case
        Z = np.argmax(Z_probs, axis=1)
    else:
        # Binary case
        Z = (Z_probs >= 0.5).astype(int).ravel()

    # Reshape Z to match the mesh grid shape
    try:
        Z = Z.reshape(xx.shape)
    except ValueError as e:
        print(f"Error reshaping predictions: {e}")
        return

    # Plot the contour and training examples
    plt.figure(figsize=(10, 8))
    background_cmap = plt.cm.Spectral # Or plt.cm.RdBu, plt.cm.coolwarm
    plt.contourf(xx, yy, Z, cmap=background_cmap, alpha=0.8)

    # Plot scatter points
    scatter_cmap = background_cmap # Match background
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y_raw, cmap=scatter_cmap,
                          edgecolor='k', s=35)
    plt.xlabel("Feature 1 (Normalized)")
    plt.ylabel("Feature 2 (Normalized)")
    plt.title("Decision Boundary")

    # Add legend for scatter points
    if y_raw is not None:
         n_classes = len(np.unique(y_raw))
         if n_classes > 1:
             legend_elements = scatter.legend_elements()[0]
             if n_classes == 2:
                 legend_labels = ['Class 0', 'Class 1']
             else:
                 legend_labels = [f'Class {i}' for i in range(n_classes)]

             if len(legend_elements) == len(legend_labels):
                  plt.legend(handles=legend_elements, labels=legend_labels, title="Data Classes")
             else:
                 print(f"Warning: Mismatch in legend elements ({len(legend_elements)}) and labels ({len(legend_labels)}). Skipping data legend.")

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.grid(True, alpha=0.2)


# --- Make Moons Example ---

def MakeMoonsExample():
    """Demonstrates training on the 'make_moons' dataset."""
    # --- Setup ---
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)    
    logger = logging.getLogger("MakeMoonsExample")
    logger.setLevel(logging.INFO)

    # --- Data Preparation ---
    logger.info("Generating make_moons dataset...")
    X_original, y_raw = make_moons(n_samples=300, noise=0.1, random_state=42) # Keep noise low

    # Visualize raw data
    plt.figure("Make Moons Raw Data", figsize=(8, 6)) # Give figure a name
    plt.scatter(X_original[:, 0], X_original[:, 1], c=y_raw, cmap='viridis', edgecolor='k')
    plt.title("Make Moons Dataset (noise=0.1)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True, alpha=0.3)

    # Normalize features
    X_mean = X_original.mean(axis=0)
    X_std = X_original.std(axis=0)
    X = (X_original - X_mean) / (X_std + 1e-8) # Normalized data for training

    # Reshape y for binary cross-entropy (needs shape (n_samples, 1))
    y_train_nn = y_raw.reshape(-1, 1)
    logger.info(f"Data shapes - X: {X.shape}, y_train_nn: {y_train_nn.shape}, y_raw: {y_raw.shape}")

    # --- Network Definition ---
    logger.info("Creating network...")
    # Using architecture that worked well in previous discussions
    network = Network(
        layer_sizes=[2, 16, 16, 1],
        activations=['relu', 'relu', 'sigmoid'],
        weight_init='xavier', # Standard Xavier ('fan_in + fan_out') - might need tuning
        # Consider testing 'old_xavier' here if standard doesn't perform well initially
        use_bias=True
    )
    print(network.summary())

    # --- Training ---
    logger.info("Starting training...")
    start_time = time.time()
    history = network.fit(
        X=X,
        y=y_train_nn,
        epochs=3000,             # Adjusted epochs (might need tuning)
        batch_size=32,
        learning_rate=0.2,     # Adjusted LR (might need tuning)
        validation_split=0.2,
        shuffle=True,
        verbose=True,
        log_every=100,          # Log less frequently for long training
        loss_type='binary_cross_entropy',
        l2_lambda=0.0           # Start with no regularization
    )
    end_time = time.time()
    logger.info(f"Training finished. Total training time: {end_time - start_time:.2f} seconds")

    # --- Plotting Training History ---
    logger.info("Plotting training history...")
    plt.figure("Make Moons Training History", figsize=(12, 5)) # Give figure a name

    plt.subplot(1, 2, 1)
    plt.plot(history['epoch'], history['loss'], label='Training Loss')
    plt.plot(history['epoch'], history['val_loss'], label='Validation Loss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (BCE)')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0) # Start y-axis at 0 for loss

    plt.subplot(1, 2, 2)
    plt.plot(history['epoch'], history['time_per_epoch'], label='Time per Epoch (s)')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Epoch Training Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)

    plt.tight_layout()

    # --- Plotting Decision Boundary ---
    logger.info("Plotting decision boundary...")
    plot_decision_boundary(X, y_raw, network) # Use normalized X for boundary, y_raw for colors

    # --- Evaluation ---
    logger.info("Evaluating model...")
    # Evaluate on the full dataset (or ideally a separate test set)
    metrics = network.evaluate(X, y_train_nn, loss_type='binary_cross_entropy')
    print("\nEvaluation Metrics (on training+validation data):")
    print(f"  Loss: {metrics.get('loss', 'N/A'):.4f}")
    print(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.4f}") # Check if evaluate returns accuracy

    # --- Saving Model ---
    model_filename = os.path.join(".", "make_moons_model_weights.npz") # Unique filename
    logger.info(f"Saving trained model weights to {model_filename}...")
    network.save_weights(model_filename)


# --- XOR Example ---

def xor_example():
    """Example of training on the XOR problem."""
    # --- Setup ---
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("XORExample")
    logger.setLevel(logging.INFO)

    logger.info("--- Running XOR Example ---")
    # --- Data ---
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]]) # Target shape (4, 1)
    y_raw = y.ravel() # For plotting

    # --- Network ---
    network = Network(
        layer_sizes=[2, 4, 1], # Simple network for XOR
        activations=['tanh', 'sigmoid'],
        weight_init='xavier',
        use_bias=True
    )
    logger.info(f"XOR Network Summary:\n{network.summary()}")

    # --- Training ---
    logger.info("Starting XOR training...")
    history = network.fit(
        X=X,
        y=y,
        epochs=5000,
        batch_size=4, # Full batch for XOR
        learning_rate=0.1,
        verbose=True,
        log_every=500,
        loss_type='binary_cross_entropy',
        l2_lambda=0.0
    )
    logger.info("XOR Training finished.")

    # --- Evaluation ---
    predictions = network.predict(X)
    logger.info("XOR Final Predictions:")
    correct = 0
    for inputs, target, pred in zip(X, y, predictions):
        pred_class = (pred[0] >= 0.5)
        is_correct = (pred_class == target[0])
        if is_correct: correct += 1
        logger.info(f"Input: {inputs}, Target: {target[0]}, Prediction: {pred[0]:.4f} -> Class: {int(pred_class)} {'(Correct)' if is_correct else '(Incorrect)'}")
    logger.info(f"XOR Accuracy: {correct / len(X):.2%}")

    # --- Plotting History ---
    logger.info("Plotting XOR training history...")
    plt.figure("XOR Training History", figsize=(8, 5)) # Give figure a name
    plt.plot(history['epoch'], history['loss'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (BCE)')
    plt.title('XOR Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    plt.tight_layout()

    # --- Plotting Decision Boundary ---
    logger.info("Plotting XOR decision boundary...")
    plot_decision_boundary(X, y_raw, network) # Use original X, y_raw

# --- Script Execution ---

if __name__ == "__main__":

    print("\n" + "="*40)
    print("--- Running XOR Classification Example ---")
    print("="*40)
    xor_example()

    print("\n" + "="*40)
    print("--- Running Make Moons Classification Example ---")
    print("="*40)
    MakeMoonsExample()

    # Keep plots open until manually closed
    print("\nDisplaying plots. Close plot windows to exit.")
    # keep the plots open
    # plt.ioff() # Disable interactive mode
    plt.show() # Show all created figures if interactive mode
    print("Exiting.")
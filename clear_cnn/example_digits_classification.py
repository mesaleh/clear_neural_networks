# clean_cnn/example_digits_classification.py
"""
Digit Classification using CNN - Educational Implementation

This script demonstrates how to build and train a Convolutional Neural Network (CNN)
for classifying handwritten digits (0-9) from the sklearn digits dataset
using the custom CNN implementation in model.py.

Main steps:
1. Load the sklearn digits dataset (8x8 pixel images)
2. Preprocess the data (reshape, normalize)
3. Define a CNN model architecture with convolution, pooling, and dense layers
4. Train the model over multiple epochs with batch processing
5. Evaluate the model's accuracy on test data
6. Visualize training loss and test accuracy over time

The implementation uses the custom neural network framework defined in model.py
rather than established libraries to demonstrate CNN principles from scratch.
"""

import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt # Added for plotting

from model import Sequential, Conv2D, ReLU, MaxPool2D, Flatten, Dense, compute_cross_entropy_loss

# --- Data Loading and Preprocessing (Same as before) ---

def load_sklearn_digits():
    try:
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("Error: scikit-learn is required. pip install scikit-learn")
        sys.exit(1)
    print("Loading Scikit-learn digits dataset...")
    digits = load_digits()
    X, y = digits.data, digits.target
    print(f"Dataset loaded. X shape: {X.shape}, y shape: {y.shape}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Split into Train: {X_train.shape}, Test: {X_test.shape}")
    return (X_train, y_train), (X_test, y_test)

def preprocess_digits_data(X, y):
    print(f"Preprocessing data: X shape={X.shape}, y shape={y.shape}")
    X_reshaped = X.reshape(X.shape[0], 1, 8, 8) # (N, C, H, W)
    X_normalized = X_reshaped.astype(np.float32) / 16.0 # Digits pixel values are 0-16
    y_int = y
    print(f"Preprocessing complete: X shape={X_normalized.shape}, y shape={y_int.shape}")
    return X_normalized, y_int

def create_batches(X, y, batch_size, shuffle=True):
    N = X.shape[0]
    if shuffle:
        permutation = np.random.permutation(N)
        X, y = X[permutation], y[permutation]
    num_batches = N // batch_size
    for i in range(num_batches):
        yield X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size]
    if N % batch_size != 0:
        yield X[num_batches*batch_size:], y[num_batches*batch_size:]

# --- Main Training Script ---

if __name__ == "__main__":

    # --- Configuration ---
    EPOCHS = 20
    BATCH_SIZE = 64
    LEARNING_RATE = 0.01 # Adagrad is less sensitive, but this is a good start
    NUM_CLASSES = 10
    INPUT_CHANNELS = 1
    INPUT_HEIGHT = 8
    INPUT_WIDTH = 8
    PRINT_EVERY_N_BATCHES = 5

    # --- 1. Load Data ---
    (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = load_sklearn_digits()

    # --- 2. Preprocess Data ---
    X_train, y_train = preprocess_digits_data(X_train_raw, y_train_raw)
    X_test, y_test = preprocess_digits_data(X_test_raw, y_test_raw)

    # --- 3. Define and Initialize Model using Sequential and Layers ---
    print("\nInitializing CNN model for 8x8 Digits using Sequential API...")

    model = Sequential()

    # Layer 1: Convolutional Layer
    # Input: (N, 1, 8, 8)
    # Conv2D(in_channels, out_channels, kernel_size, stride, padding_type)
    # Kernel 3x3, Stride 1, 'valid' padding: (8-3)/1 + 1 = 6. Output: (N, 8, 6, 6)
    model.add(Conv2D(in_channels=INPUT_CHANNELS, out_channels=8, kernel_size=3, stride=1, padding_type='valid'))
    model.add(ReLU())
    # Layer 2: Max Pooling Layer
    # Pool size 2x2, Stride 2: (6-2)/2 + 1 = 3. Output: (N, 8, 3, 3)
    model.add(MaxPool2D(pool_size=2, stride=2))
    
    # Layer 3: Convolutional Layer (Optional - for deeper models, ensure dimensions work)
    # Input: (N, 8, 3, 3)
    # Kernel 2x2, Stride 1, 'valid' padding: (3-2)/1+1 = 2. Output: (N, 16, 2, 2)
    # model.add(Conv2D(in_channels=8, out_channels=16, kernel_size=2, stride=1, padding_type='valid'))
    # model.add(ReLU())
    # Layer 4: Max Pooling (Optional)
    # Pool 2x2, Stride 2: (2-2)/2+1 = 1. Output: (N, 16, 1, 1)
    # model.add(MaxPool2D(pool_size=2, stride=2))

    # Layer 5: Flatten Layer
    # Input from MaxPool2D: (N, 8, 3, 3) -> Flattened: N x (8*3*3) = N x 72
    # Or if second Conv+Pool used: (N, 16, 1, 1) -> N x 16
    model.add(Flatten())
    flattened_features = 8 * 3 * 3 # Calculate based on the actual last conv/pool output
    # If using the optional second conv+pool: flattened_features = 16 * 1 * 1

    # Layer 6: Dense Layer
    model.add(Dense(input_dim=flattened_features, output_dim=NUM_CLASSES))
    # Softmax is handled by model.forward() for probabilities,
    # and its derivative is part of the dProbs passed to model.backward()

    print("\nModel Architecture:")
    current_shape = (BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    print(f"  Input Shape: {current_shape}")
    for i, layer in enumerate(model.layers):
        # To print output shapes, we'd need to do a dummy forward or have layers store output_shape
        print(f"  Layer {i+1}: {type(layer).__name__}")
        if hasattr(layer, 'params') and layer.params:
            for p_name, p_val in layer.params.items():
                 print(f"    {p_name} shape: {p_val.shape}")
        # A simple way to trace shapes (approximate, doesn't run the layer)
        if isinstance(layer, Conv2D):
            h_out = (current_shape[2] - layer.K_h + 2*layer.pad_h_before) // layer.S_h + 1
            w_out = (current_shape[3] - layer.K_w + 2*layer.pad_w_before) // layer.S_w + 1
            current_shape = (current_shape[0], layer.C_out, h_out, w_out)
            print(f"    Output shape ~: {current_shape}")
        elif isinstance(layer, MaxPool2D):
            h_out = (current_shape[2] - layer.K_h) // layer.S_h + 1
            w_out = (current_shape[3] - layer.K_w) // layer.S_w + 1
            current_shape = (current_shape[0], current_shape[1], h_out, w_out)
            print(f"    Output shape ~: {current_shape}")
        elif isinstance(layer, Flatten):
            current_shape = (current_shape[0], np.prod(current_shape[1:]))
            print(f"    Output shape ~: {current_shape}")
        elif isinstance(layer, Dense):
            current_shape = (current_shape[0], layer.D_out)
            print(f"    Output shape ~: {current_shape}")


    # --- 4. Training Loop ---
    print("\n--- Starting Training ---")
    total_batches = (len(X_train) + BATCH_SIZE - 1) // BATCH_SIZE
    start_time_total = time.time()

    train_losses = []
    test_accuracies = []

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        batch_count = 0
        model.set_training_mode(True)

        batch_generator = create_batches(X_train, y_train, BATCH_SIZE, shuffle=True)

        for i, (X_batch, y_batch_indices) in enumerate(batch_generator):
            batch_start_time = time.time()
            
            # Convert y_batch to one-hot
            y_batch_one_hot = np.eye(NUM_CLASSES)[y_batch_indices]

            # Forward pass
            probs = model.forward(X_batch)
            
            # Compute loss
            loss = compute_cross_entropy_loss(probs, y_batch_one_hot)
            epoch_loss += loss
            batch_count += 1
            
            # Backward pass
            # Gradient of Cross-Entropy Loss + Softmax w.r.t. Z_before_softmax is (Probs - Y_one_hot)
            # We need to scale by N if the loss is an average.
            # If loss is sum, dL/dZ = P-Y. If loss is avg (as computed), dL/dZ = (P-Y)/N.
            dZ_softmax = (probs - y_batch_one_hot) / X_batch.shape[0]
            model.backward(dZ_softmax)
            
            # Update parameters
            model.update_params(learning_rate=LEARNING_RATE, optimizer_type='adagrad')
            
            batch_duration = time.time() - batch_start_time
            if (i + 1) % PRINT_EVERY_N_BATCHES == 0 or (i + 1) == total_batches:
                 print(f"  Epoch {epoch+1}/{EPOCHS} | Batch {i+1}/{total_batches} | "
                       f"Batch Loss: {loss:.4f} | Time/Batch: {batch_duration:.3f}s")

        epoch_duration = time.time() - epoch_start_time
        average_epoch_loss = epoch_loss / batch_count
        train_losses.append(average_epoch_loss)

        print(f"\nEpoch {epoch+1} completed.")
        print(f"  Average Training Loss: {average_epoch_loss:.4f}")
        print(f"  Epoch Duration: {epoch_duration:.2f}s")

        # Evaluate on the test set
        print("  Evaluating on test set...")
        model.set_training_mode(False)
        test_probs = model.forward(X_test)
        test_predictions = np.argmax(test_probs, axis=1)
        test_accuracy = np.mean(test_predictions == y_test)
        test_accuracies.append(test_accuracy)
        print(f"  Test Accuracy: {test_accuracy * 100:.2f}%")
        print("-" * 30)

    total_training_time = time.time() - start_time_total
    print("\n--- Training Finished ---")
    print(f"Total Training Time: {total_training_time:.2f}s")

    # --- 6. Final Evaluation & Summary ---
    print("\n--- Final Evaluation ---")
    model.set_training_mode(False)
    final_probs = model.forward(X_test)
    final_predictions = np.argmax(final_probs, axis=1)
    final_accuracy = np.mean(final_predictions == y_test)
    print(f"Final Test Accuracy: {final_accuracy * 100:.2f}%")

    print("\nExample Predictions (first 10 test samples):")
    print(f"  Predicted: {final_predictions[:10]}")
    print(f"  Actual:    {y_test[:10]}")

    # --- Plotting ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Training Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss over Epochs')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS + 1), test_accuracies, label='Test Accuracy', color='orange', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.05) # Accuracy is between 0 and 1
    plt.legend()
    plt.title('Test Accuracy over Epochs')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
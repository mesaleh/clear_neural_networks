# clean_cnn/example_url_classification.py
"""
URL Classification using 1D CNN - Educational Implementation

This script demonstrates how to build and train a 1D Convolutional Neural Network (CNN)
for URL classification (phishing vs. legitimate) using a custom CNN implementation.
It serves as an educational example of applying CNNs to text data.

Main steps:
1. Load URL data and preprocess it using one-hot encoding (each character → one-hot vector)
2. Split data into training and testing sets
3. Define a 1D CNN model architecture with convolution, pooling, and dense layers
4. Train the model over multiple epochs with batch processing
5. Evaluate the model's accuracy on test data
6. Visualize training loss and test accuracy

The implementation uses a custom neural network framework defined in model.py
rather than established libraries to demonstrate CNN principles from scratch.
"""

import numpy as np
import time
import sys
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

from model import Sequential, Conv1D, ReLU, MaxPool1D, Flatten, Dense, compute_cross_entropy_loss

# --- Character Processing and Data Loading ---

VALID_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789.-/:_#?=&%" # Keep this simple
CHAR_TO_INT = {char: i for i, char in enumerate(VALID_CHARS)} # Indices from 0 to len-1
# Padding token will effectively be a vector of all zeros if index not in CHAR_TO_INT
VOCAB_SIZE = len(VALID_CHARS) # This is the length of the one-hot vector
MAX_URL_LENGTH = 100 # Reduce for faster iteration initially, can increase later

def is_ip_address(url_string):
    ip_pattern = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?(/.*)?$")
    host_part = url_string.split('/')[0]
    return bool(ip_pattern.match(host_part))

def url_to_one_hot_sequence(url, char_to_int_map, max_len, vocab_size):
    """Converts a single URL to a one-hot encoded sequence."""
    one_hot_sequence = np.zeros((max_len, vocab_size), dtype=np.float32)
    for i, char in enumerate(url[:max_len]):
        if char in char_to_int_map:
            char_index = char_to_int_map[char]
            one_hot_sequence[i, char_index] = 1.0
    return one_hot_sequence

def load_and_preprocess_urls_one_hot(csv_path, max_len=MAX_URL_LENGTH, vocab_size=VOCAB_SIZE, char_map=CHAR_TO_INT, max_rows=None):
    print(f"Loading URLs from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        if max_rows is not None:
            if len(df) > max_rows:
                df = df.sample(n=max_rows, random_state=42)
            else:
                print(f"Requested max_rows={max_rows} but CSV has only {len(df)} rows. Using all.")
    except pd.errors.EmptyDataError:
        print("Error: CSV file is empty.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        sys.exit(1)
    if 'url' not in df.columns or 'status' not in df.columns:
        print("Error: CSV must contain 'url' and 'status' columns.")
        sys.exit(1)

    one_hot_sequences = []
    labels = []
    ip_count = 0
    original_count = len(df)

    print("Processing URLs and converting to one-hot sequences...")
    for index, row in df.iterrows():
        url = str(row['url']).lower().strip()
        status = int(row['status'])

        if is_ip_address(url):
            ip_count += 1
            continue

        one_hot_seq = url_to_one_hot_sequence(url, char_map, max_len, vocab_size)
        one_hot_sequences.append(one_hot_seq)
        labels.append(status)

        if (index + 1) % 5000 == 0:
            print(f"  Processed {index+1}/{original_count} URLs...")

    # X_processed shape: (N, max_len, vocab_size)
    X_processed = np.array(one_hot_sequences, dtype=np.float32)

    # Transpose for Conv1D if it expects (N, C_in, L_in) where C_in = vocab_size, L_in = max_len
    # (N, vocab_size, max_len)
    X_processed = X_processed.transpose(0, 2, 1)

    y_labels = np.array(labels, dtype=np.int32)

    print(f"URL processing complete.")
    print(f"  Original URLs loaded: {original_count}")
    print(f"  IP addresses filtered: {ip_count}")
    print(f"  URLs kept: {len(one_hot_sequences)}")
    print(f"  X_processed final shape for Conv1D: {X_processed.shape}, y_labels shape: {y_labels.shape}")

    return X_processed, y_labels

def create_batches(X, y, batch_size, shuffle=True):
    N = X.shape[0]
    if shuffle:
        permutation = np.random.permutation(N)
        X, y = X[permutation], y[permutation]
    num_batches = N // batch_size
    for i in range(num_batches):
        yield X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size]
    if N % batch_size != 0 and N > num_batches*batch_size : # Ensure there's a remainder
        yield X[num_batches*batch_size:], y[num_batches*batch_size:]


# --- Main Training Script ---
if __name__ == "__main__":
    CSV_FILE_PATH = "data/url_phishing_legit.csv"

    # --- Configuration ---
    EPOCHS = 20
    BATCH_SIZE = 64
    LEARNING_RATE = 0.005
    NUM_URL_CLASSES = 2 # Phishing vs Legit
    PRINT_EVERY_N_BATCHES = 10 # Print more often with smaller batches

    # --- 1. Load and Preprocess Data ---
    # Use a larger subset of data if available and training time permits
    X_urls, y_urls = load_and_preprocess_urls_one_hot(
        CSV_FILE_PATH,
        max_len=MAX_URL_LENGTH,
        vocab_size=VOCAB_SIZE,
        char_map=CHAR_TO_INT,
        max_rows=5000 # Increased dataset size for better learning
    )

    if X_urls.shape[0] < BATCH_SIZE : # Check if we have enough data for at least one batch
        print(f"Warning: Number of samples ({X_urls.shape[0]}) is less than batch size ({BATCH_SIZE}).")
        print("Adjust BATCH_SIZE or use more data.")
        if X_urls.shape[0] == 0:
          sys.exit(1)

    try:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_urls, y_urls, test_size=0.2, random_state=42, stratify=y_urls
        )
    except ImportError:
        print("scikit-learn not found, using simple manual split.")
        # ... (manual split logic) ...
        num_samples = X_urls.shape[0]
        permutation = np.random.permutation(num_samples)
        X_urls_shuffled, y_urls_shuffled = X_urls[permutation], y_urls[permutation]
        split_idx = int(0.8 * num_samples)
        X_train, y_train = X_urls_shuffled[:split_idx], y_urls_shuffled[:split_idx]
        X_test, y_test = X_urls_shuffled[split_idx:], y_urls_shuffled[split_idx:]


    print(f"\nData Split: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Data Split: X_test: {X_test.shape}, y_test: {y_test.shape}")
    if len(y_train) > 0: print(f"Class distribution in y_train: {np.bincount(y_train)}")
    if len(y_test) > 0: print(f"Class distribution in y_test: {np.bincount(y_test)}")

    # --- 3. Define and Initialize Model ---
    print("\nInitializing 1D CNN model with one-hot encoded URLs...")
    model = Sequential()

    # Input to Conv1D: (N, VOCAB_SIZE, MAX_URL_LENGTH)
    # VOCAB_SIZE acts as input_channels for Conv1D
    num_filters_conv1 = 32  # Number of filters (output channels of Conv1D)
    kernel_len_conv1 = 5    # Length of 1D kernel (e.g., looks at 5 characters at a time)
    model.add(Conv1D(in_channels=VOCAB_SIZE, out_channels=num_filters_conv1,
                     kernel_length=kernel_len_conv1, stride=1, padding_type='valid'))
    model.add(ReLU())

    pool_len1 = 2
    model.add(MaxPool1D(pool_length=pool_len1, stride=pool_len1))

    # Optional second conv/pool block
    # num_filters_conv2 = 64
    # kernel_len_conv2 = 3
    # l_after_pool1_approx = (MAX_URL_LENGTH - kernel_len_conv1 + 1) // pool_len1
    # model.add(Conv1D(in_channels=num_filters_conv1, out_channels=num_filters_conv2,
    #                  kernel_length=kernel_len_conv2, stride=1, padding_type='valid'))
    # model.add(ReLU())
    # model.add(MaxPool1D(pool_length=2, stride=2))


    model.add(Flatten())

    # Calculate flattened_features:
    l_out_conv1 = (MAX_URL_LENGTH - kernel_len_conv1) // 1 + 1
    l_out_pool1 = (l_out_conv1 - pool_len1) // pool_len1 + 1
    flattened_features = num_filters_conv1 * l_out_pool1
    # If using second conv/pool, continue calculation:
    # l_out_conv2 = (l_out_pool1 - kernel_len_conv2) // 1 + 1
    # l_out_pool2 = (l_out_conv2 - 2) // 2 + 1 # Assuming pool_len2=2, stride2=2
    # flattened_features = num_filters_conv2 * l_out_pool2
    print(f"Calculated flattened features: {flattened_features}")

    model.add(Dense(input_dim=flattened_features, output_dim=NUM_URL_CLASSES))

    print("\nModel Architecture (1D CNN with One-Hot Input):")
    print(f"  Input: (N, {VOCAB_SIZE}, {MAX_URL_LENGTH})")

    # --- 4. Training Loop ---
    print("\n--- Starting Training ---")
    total_batches = (len(X_train) + BATCH_SIZE - 1) // BATCH_SIZE if len(X_train) > 0 else 0
    start_time_total = time.time()

    train_losses = []
    test_accuracies = []

    if total_batches == 0 and len(X_train) > 0 : # If train data exists but not enough for a batch
        print(f"Warning: Training data ({len(X_train)} samples) less than batch size ({BATCH_SIZE}). Adjust BATCH_SIZE.")
        total_batches = 1 # Force at least one batch if data exists
    elif total_batches == 0 and len(X_train) == 0:
        print("No training data available. Skipping training.")

    if len(X_train) > 0 :
        for epoch in range(EPOCHS):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            batch_count = 0
            model.set_training_mode(True)

            batch_generator = create_batches(X_train, y_train, BATCH_SIZE, shuffle=True)

            actual_batches_this_epoch = 0
            for i, (X_batch, y_batch_indices) in enumerate(batch_generator):
                actual_batches_this_epoch += 1
                batch_start_time = time.time()
                y_batch_one_hot = np.eye(NUM_URL_CLASSES)[y_batch_indices]

                probs = model.forward(X_batch)
                loss = compute_cross_entropy_loss(probs, y_batch_one_hot)
                
                # Check for NaN/Inf loss
                if np.isnan(loss) or np.isinf(loss):
                    print(f"!!! NaN or Inf loss detected at Epoch {epoch+1}, Batch {i+1}. Stopping. !!!")
                    print(f"Probs sample: {probs[0] if len(probs)>0 else 'N/A'}")
                    train_losses.append(float('inf')) # Record inf loss
                    # sys.exit("Stopping due to NaN/Inf loss.")
                    break # Break from batch loop for this epoch

                epoch_loss += loss
                batch_count += 1
                
                dZ_softmax = (probs - y_batch_one_hot) / X_batch.shape[0]
                model.backward(dZ_softmax)
                model.update_params(learning_rate=LEARNING_RATE, optimizer_type='adagrad')
                
                batch_duration = time.time() - batch_start_time
                if (i + 1) % PRINT_EVERY_N_BATCHES == 0 or (i + 1) == actual_batches_this_epoch and actual_batches_this_epoch==total_batches : # Print on last batch of epoch
                    print(f"  Epoch {epoch+1}/{EPOCHS} | Batch {i+1}/{total_batches} | " # Use total_batches for display
                        f"Batch Loss: {loss:.4f} | Time/Batch: {batch_duration:.3f}s")
            
            if np.isnan(epoch_loss) or np.isinf(epoch_loss): # If NaN/Inf encountered in epoch
                 print(f"Epoch {epoch+1} terminated early due to NaN/Inf loss.")
                 if not test_accuracies : test_accuracies.append(0) # Add placeholder if no eval done
                 # break # Break from epoch loop
                 continue # Or try next epoch if you want to see if it recovers


            epoch_duration = time.time() - epoch_start_time
            average_epoch_loss = epoch_loss / batch_count if batch_count > 0 else float('inf') # Avoid div by zero
            train_losses.append(average_epoch_loss)

            print(f"\nEpoch {epoch+1} completed.")
            print(f"  Average Training Loss: {average_epoch_loss:.4f}")
            print(f"  Epoch Duration: {epoch_duration:.2f}s")

            if len(X_test) > 0:
                print("  Evaluating on test set...")
                model.set_training_mode(False)
                test_probs = model.forward(X_test)
                test_predictions = np.argmax(test_probs, axis=1)
                test_accuracy = np.mean(test_predictions == y_test)
                test_accuracies.append(test_accuracy)
                print(f"  Test Accuracy: {test_accuracy * 100:.2f}%")
            else:
                print("  Skipping test set evaluation (no test data).")
                test_accuracies.append(0.0) # Placeholder if no test data
            print("-" * 30)
            if np.isnan(average_epoch_loss) or np.isinf(average_epoch_loss):
                print("Stopping training due to persistent NaN/Inf loss.")
                break


    total_training_time = time.time() - start_time_total
    print("\n--- Training Finished ---")
    print(f"Total Training Time: {total_training_time:.2f}s")

    if len(X_test) > 0:
        print("\n--- Final Evaluation ---")
        # ... (final evaluation code - keep as is) ...
        model.set_training_mode(False)
        final_probs = model.forward(X_test)
        final_predictions = np.argmax(final_probs, axis=1)
        final_accuracy = np.mean(final_predictions == y_test)
        print(f"Final Test Accuracy: {final_accuracy * 100:.2f}%")

        print("\nExample Predictions (first 10 test samples):")
        for i in range(min(10, len(X_test))):
            print(f"  Sample {i+1}: Predicted: {final_predictions[i]}, Actual: {y_test[i]}")

    else:
        print("No test data for final evaluation.")

    # --- Plotting ---
    if train_losses : # Check if lists are not empty
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        # Filter out inf values for plotting if any epoch was skipped
        plottable_train_losses = [l for l in train_losses if not (np.isnan(l) or np.isinf(l))]
        epochs_plotted_loss = range(1, len(plottable_train_losses) + 1)
        if plottable_train_losses:
             plt.plot(epochs_plotted_loss, plottable_train_losses, label='Training Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Loss over Epochs')
        plt.grid(True)

        if test_accuracies:
            plt.subplot(1, 2, 2)
            plottable_test_accuracies = [a for a in test_accuracies if not (np.isnan(a) or np.isinf(a))]
            epochs_plotted_acc = range(1, len(plottable_test_accuracies) + 1)
            if plottable_test_accuracies:
                plt.plot(epochs_plotted_acc, plottable_test_accuracies, label='Test Accuracy', color='orange', marker='o')

            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            min_acc_val = 0.0
            max_acc_val = 1.0
            if plottable_test_accuracies: # Check if list is not empty
                 min_acc_val = min(0.0, min(plottable_test_accuracies)-0.05)
                 max_acc_val = max(1.0, max(plottable_test_accuracies)+0.05)

            plt.ylim(min_acc_val, max_acc_val)
            plt.legend()
            plt.title('Test Accuracy over Epochs')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
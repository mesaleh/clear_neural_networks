# train.py
import numpy as np
import time
from model import RNN # Import the RNN class from model.py

# --- 1. Configuration and Hyperparameters ---
data_path = 'data/linux_man.txt' # Path to the dataset file
hidden_size = 128      # Size of the hidden state vector
seq_length = 20        # Length of sequence chunks for BPTT
learning_rate = 1e-1   # Learning rate for Adagrad
max_iterations = 10000 # Number of training iterations (increase for better results)
sample_every = 1000    # How often to sample text and print loss
sample_length = 200    # Length of the generated sample text

# --- 2. Data Loading and Preprocessing ---
print("Loading and preprocessing data...")
try:
    with open(data_path, 'r', encoding='utf-8') as f:
        data = f.read()
except FileNotFoundError:
    print(f"Error: Data file not found at '{data_path}'")
    print("Please download the Tiny Shakespeare dataset and place it in a 'data' folder or update the path.")
    exit()


chars = sorted(list(set(data)))
data_size, vocab_size = len(data), len(chars)
print(f"Dataset has {data_size} characters, {vocab_size} unique.")

# Create character-to-index and index-to-character mappings
char_to_ix = { ch: i for i, ch in enumerate(chars) }
ix_to_char = { i: ch for i, ch in enumerate(chars) }
print("Vocabulary created.")
#print(f"Chars: {''.join(chars)}")

# --- 3. Model Initialization ---
print("Initializing RNN model...")
model = RNN(hidden_size=hidden_size, vocab_size=vocab_size, learning_rate=learning_rate)
print(f"Model initialized with hidden_size={hidden_size}, vocab_size={vocab_size}")

# --- 4. Training Loop ---
print(f"Starting training for {max_iterations} iterations...")
n = 0                      # Iteration counter
p = 0                      # Data pointer (index in the dataset)
smooth_loss = -np.log(1.0/vocab_size) * seq_length # Loss smoothed over iterations, initialized to random prediction loss

# Initial hidden state (zeros) for the very first sequence chunk
hprev = np.zeros((hidden_size, 1))

start_time = time.time()

while n <= max_iterations:
    # --- Prepare Input/Target Chunk ---
    # Reset pointer and hidden state if we reach the end of the data
    # (or at the very beginning)
    if p + seq_length + 1 >= data_size or n == 0:
        hprev = np.zeros((hidden_size, 1)) # Reset RNN memory state
        p = 0                              # Go back to the start of the data

    # Get the input and target sequences for this chunk
    # Inputs: characters from p to p + seq_length - 1
    # Targets: characters from p + 1 to p + seq_length
    inputs_chars = data[p : p + seq_length]
    targets_chars = data[p + 1 : p + seq_length + 1]

    # Convert characters to integer indices
    inputs = [char_to_ix[ch] for ch in inputs_chars]
    targets = [char_to_ix[ch] for ch in targets_chars]

    # --- Perform one training step ---
    # This calls the forward pass, backward pass (BPTT), and Adagrad update
    # inside the model. It returns the loss for this chunk and the next hidden state.
    loss, hprev = model.train_step(inputs, targets, hprev)

    # --- Update Smoothed Loss ---
    # Use exponential moving average (EMA) for smoother loss tracking.
    # The reason of using smooth loss is to avoid spikes in the loss due to noise.
    # A smoother curve makes it easier to identify trends and potential problems like overfitting or slow convergence.
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    # --- Periodically Sample and Print Progress ---
    if n % sample_every == 0:
        print(f'\n---- Iteration {n}, Smoothed Loss: {smooth_loss:.4f} ----')
        # Generate sample text using the current model state
        # Seed the sampling with the first character of the current input chunk
        # Use the 'hprev' from the *end* of the last training step as the initial hidden state for sampling
        sample_indices = model.sample(h=hprev, seed_ix=inputs[0], n=sample_length)
        sample_text = ''.join(ix_to_char[ix] for ix in sample_indices)
        print(f'Sample:\n```\n{sample_text}\n```')
        print('-----------------------------------------')

    # --- Move to the next chunk and increment counter ---
    p += seq_length # Move the data pointer forward by the sequence length
    n += 1          # Increment iteration counter

# --- End of Training ---
end_time = time.time()
print("\n=========================================")
print("Training completed.")
print(f"Total training time: {end_time - start_time:.2f} seconds")
print(f"Final smoothed loss: {smooth_loss:.4f}")
print("=========================================")

# You could add code here to save the trained model parameters (model.Wxh, model.Whh, etc.)
# e.g., using np.savez('rnn_model.npz', Wxh=model.Wxh, Whh=model.Whh, ...)
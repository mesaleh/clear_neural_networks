import numpy as np
import os
from model import LSTM # Import the LSTM class

# --- Configuration ---
DATA_FILE = 'data/linux_man.txt' # Relative path to data
MODEL_SAVE_PATH = 'lstm_text_model.npz' # File to save model parameters
LOAD_EXISTING_MODEL = False # Set to True to load parameters from MODEL_SAVE_PATH

# --- Hyperparameters ---
hidden_size = 100        # Size of hidden layer of neurons (LSTM units)
seq_length = 25         # Length of sequence chunks for BPTT
learning_rate = 1e-1     # Learning rate for Adagrad
sample_every = 100       # How often to generate sample text (iterations)
print_every = 10        # How often to print loss (iterations)
save_every = 1000       # How often to save the model (iterations)
max_iterations = 500000  # Maximum number of training iterations

# --- Data Loading and Preprocessing ---
try:
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = f.read()
    print(f"Data loaded successfully from {DATA_FILE}.")
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_FILE}")
    print("Please ensure the 'data' directory exists relative to where you run train.py")
    print("and contains the 'linux_man.txt' file.")
    exit(1)

chars = sorted(list(set(data))) # Get unique characters
data_size, vocab_size = len(data), len(chars)
print(f"Data has {data_size} characters, {vocab_size} unique.")

# Create character-to-index and index-to-character mappings
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# --- Model Initialization ---
# Use the updated LSTM signature (input_size, hidden_size, output_size)
# For single-layer char model, input_size=vocab_size and output_size=vocab_size
lstm_model = LSTM(input_size=vocab_size, hidden_size=hidden_size, output_size=vocab_size,
                  seq_length=seq_length, learning_rate=learning_rate)

# --- Load Existing Model (Optional) ---
if LOAD_EXISTING_MODEL and os.path.exists(MODEL_SAVE_PATH):
    try:
        # Add allow_pickle=True for potentially saved dicts like ix_to_char
        saved_data = np.load(MODEL_SAVE_PATH, allow_pickle=True)
        # Load parameters (ensure keys match saved keys)
        lstm_model.Wf = saved_data['Wf']
        lstm_model.Wi = saved_data['Wi']
        lstm_model.Wc = saved_data['Wc']
        lstm_model.Wo = saved_data['Wo']
        lstm_model.Wy = saved_data['Wy']
        lstm_model.bf = saved_data['bf']
        lstm_model.bi = saved_data['bi']
        lstm_model.bc = saved_data['bc']
        lstm_model.bo = saved_data['bo']
        lstm_model.by = saved_data['by']

        # Load Adagrad memory as well
        lstm_model.mWf = saved_data['mWf']
        lstm_model.mWi = saved_data['mWi']
        lstm_model.mWc = saved_data['mWc']
        lstm_model.mWo = saved_data['mWo']
        lstm_model.mWy = saved_data['mWy']
        lstm_model.mbf = saved_data['mbf']
        lstm_model.mbi = saved_data['mbi']
        lstm_model.mbc = saved_data['mbc']
        lstm_model.mbo = saved_data['mbo']
        lstm_model.mby = saved_data['mby']

        # Optionally restore hyperparameters if saved (check keys)
        # loaded_hidden_size = saved_data['hidden_size']
        # loaded_input_size = saved_data['input_size'] # If saved with new keys
        # loaded_output_size = saved_data['output_size'] # If saved with new keys
        # loaded_vocab_size = saved_data['vocab_size'] # If saved with old key

        # Restore mappings if saved
        # char_to_ix = saved_data['char_to_ix'].item() # .item() if saved as 0-d array
        # ix_to_char = saved_data['ix_to_char'].item()

        print(f"Model parameters loaded from {MODEL_SAVE_PATH}")
        # Ensure current hyperparameters match loaded ones if necessary
        # assert hidden_size == loaded_hidden_size
        # assert vocab_size == loaded_vocab_size # or input/output size

    except Exception as e:
        print(f"Error loading model from {MODEL_SAVE_PATH}: {e}")
        print("Starting with fresh parameters.")
elif LOAD_EXISTING_MODEL:
    print(f"Warning: {MODEL_SAVE_PATH} not found. Starting training from scratch.")


# --- Training Loop ---
n = 0  # Iteration counter
p = 0  # Data pointer (index into the start of the sequence)

# Initialize hidden state (h_prev) and cell state (c_prev) for the first iteration
h_prev = np.zeros((hidden_size, 1))
c_prev = np.zeros((hidden_size, 1))

print("\nStarting Training...\n")

try:
    while n <= max_iterations:
        # --- Prepare Input and Target Data ---
        # Prevent pointer from exceeding dataset bounds
        if p + seq_length + 1 >= data_size or n == 0:
            # Reset pointer and hidden/cell state when wrapping around or at start
            p = 0
            h_prev = np.zeros((hidden_size, 1))
            c_prev = np.zeros((hidden_size, 1))

        # --- Prepare Input and Target Data ---
        inputs_str = data[p : p + seq_length]
        targets_str = data[p + 1 : p + seq_length + 1]

        # Convert targets to indices (needed for loss)
        targets = [char_to_ix[ch] for ch in targets_str]

        # Convert input characters to list of one-hot vectors
        inputs_vectors = []
        for ch in inputs_str:
            vec = np.zeros((vocab_size, 1)) # Create vector of size vocab_size (==input_size)
            vec[char_to_ix[ch]] = 1
            inputs_vectors.append(vec)

        # --- Sample Generation (Periodically) ---
        if n % sample_every == 0:
            sample_h = np.zeros((hidden_size, 1))
            sample_c = np.zeros((hidden_size, 1))

            # Create one-hot seed vector for sample method
            # Use first char of the current input sequence as seed index
            sample_seed_ix = char_to_ix[inputs_str[0]]
            seed_vector = np.zeros((vocab_size, 1)) # Vector shape: (input_size, 1)
            seed_vector[sample_seed_ix] = 1

            # Call sample with the seed vector
            sample_ixes = lstm_model.sample(seed_vector, sample_h, sample_c, 200)
            sample_txt = ''.join(ix_to_char[ix] for ix in sample_ixes)
            print(f'----\n Iteration {n}, Sample:\n {sample_txt} \n----')

        # --- Perform Training Step ---
        # Pass the list of input vectors to train_step
        loss, h_prev, c_prev = lstm_model.train_step(inputs_vectors, targets, h_prev, c_prev)

        # --- Print Progress ---
        if n % print_every == 0:
            # Use the smoothed loss from the model object
            print(f'Iter: {n}, Smoothed Loss: {lstm_model.smooth_loss:.4f}')

        # --- Save Model (Periodically) ---
        if n % save_every == 0 and n > 0:
            try:
                 # Save using consistent keys, maybe add input/output size
                 np.savez(MODEL_SAVE_PATH,
                         # Model parameters
                         Wf=lstm_model.Wf, Wi=lstm_model.Wi, Wc=lstm_model.Wc, Wo=lstm_model.Wo, Wy=lstm_model.Wy,
                         bf=lstm_model.bf, bi=lstm_model.bi, bc=lstm_model.bc, bo=lstm_model.bo, by=lstm_model.by,
                         # Adagrad memory
                         mWf=lstm_model.mWf, mWi=lstm_model.mWi, mWc=lstm_model.mWc, mWo=lstm_model.mWo, mWy=lstm_model.mWy,
                         mbf=lstm_model.mbf, mbi=lstm_model.mbi, mbc=lstm_model.mbc, mbo=lstm_model.mbo, mby=lstm_model.mby,
                         # Hyperparameters & Mappings (save actual model sizes)
                         hidden_size=lstm_model.hidden_size,
                         input_size=lstm_model.input_size,
                         output_size=lstm_model.output_size,
                         ix_to_char=ix_to_char, char_to_ix=char_to_ix
                         )
                 print(f"Model parameters saved to {MODEL_SAVE_PATH} at iteration {n}")
            except Exception as e:
                 print(f"Error saving model at iteration {n}: {e}")

        p += seq_length
        n += 1

except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving final model...")
    try:
        # Use same keys for final save
        np.savez(MODEL_SAVE_PATH,
                 Wf=lstm_model.Wf, Wi=lstm_model.Wi, Wc=lstm_model.Wc, Wo=lstm_model.Wo, Wy=lstm_model.Wy,
                 bf=lstm_model.bf, bi=lstm_model.bi, bc=lstm_model.bc, bo=lstm_model.bo, by=lstm_model.by,
                 mWf=lstm_model.mWf, mWi=lstm_model.mWi, mWc=lstm_model.mWc, mWo=lstm_model.mWo, mWy=lstm_model.mWy,
                 mbf=lstm_model.mbf, mbi=lstm_model.mbi, mbc=lstm_model.mbc, mbo=lstm_model.mbo, mby=lstm_model.mby,
                 hidden_size=lstm_model.hidden_size, input_size=lstm_model.input_size, output_size=lstm_model.output_size,
                 ix_to_char=ix_to_char, char_to_ix=char_to_ix
                 )
        print(f"Final model parameters saved to {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Error saving final model: {e}")

print("\nTraining finished.")
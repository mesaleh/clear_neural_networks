import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse
from model import LSTM 

# --- Configuration ---
MODEL_SAVE_PATH = 'lstm_stocks_model.npz'
LOAD_EXISTING_MODEL = True  # Always try to load the model

# --- Hyperparameters ---
hidden_size = 200        # Size of hidden layer
seq_length = 30          # Use 30 past days to predict the next day
learning_rate = 1e-2     # Learning rate for regression
max_iterations = 10000   # Number of training iterations
print_every = 100        # How often to print loss
save_every = 1000        # How often to save the model

def load_stock_data():
    """Load stock prices from CSV file"""
    stock_data = pd.read_csv('data/MSFT_1d_data.csv')
    stock_prices = stock_data['Close'].values
    stock_prices = [round(float(p), 2) for p in stock_prices]
    print(f"Loaded {len(stock_prices)} data points.")
    return stock_prices

def preprocess_data(stock_prices, seq_length):
    """
    Preprocess stock data:
    1. Standardize the prices (mean=0, std=1)
    2. Create sequences for LSTM input
    """
    # Standardization (Z-score normalization)
    mean_price = np.mean(stock_prices)
    std_price = np.std(stock_prices)
    normalized_prices = (stock_prices - mean_price) / std_price
    print(f"Data standardized. Mean: {mean_price:.2f}, Std Dev: {std_price:.2f}")
    
    # Create sequences for LSTM input and targets
    data_X = [] # List of input sequences
    data_Y = [] # List of target values
    num_sequences = len(normalized_prices) - seq_length
    
    for i in range(num_sequences):
        # Input: sequence of length 'seq_length'
        input_seq = normalized_prices[i : i + seq_length]
        # Target: the price immediately following the input sequence
        target_val = normalized_prices[i + seq_length]
        
        # Create list of 2D column vectors for input sequence
        input_vectors_seq = [np.array([[val]]) for val in input_seq] # Shape: (seq_length, 1)
        data_X.append(input_vectors_seq)
        
        # Create 2D column vector for target
        target_vector = np.array([[target_val]]) # Shape: (1, 1)
        data_Y.append(target_vector)
    
    print(f"Created {len(data_X)} sequences of length {seq_length}.")
    return data_X, data_Y, mean_price, std_price

def init_or_load_model(input_size, hidden_size, output_size, seq_length, learning_rate):
    """Initialize a new model or load an existing one"""
    # Initialize model
    lstm_model = LSTM(input_size=input_size,
                      hidden_size=hidden_size,
                      output_size=output_size,
                      seq_length=seq_length,
                      learning_rate=learning_rate,
                      task_type='regression')
    
    print("LSTM model initialized for regression.")
    
    # Try loading existing model if requested
    if LOAD_EXISTING_MODEL and os.path.exists(MODEL_SAVE_PATH):
        try:
            saved_data = np.load(MODEL_SAVE_PATH, allow_pickle=True)
            
            # Load parameters
            parameter_names = ['Wf', 'Wi', 'Wc', 'Wo', 'Wy', 
                              'bf', 'bi', 'bc', 'bo', 'by',
                              'mWf', 'mWi', 'mWc', 'mWo', 'mWy',
                              'mbf', 'mbi', 'mbc', 'mbo', 'mby']
            
            for param in parameter_names:
                if param in saved_data:
                    setattr(lstm_model, param, saved_data[param])
                    
            print(f"Model parameters loaded from {MODEL_SAVE_PATH}")
        except Exception as e:
            print(f"Error loading model from {MODEL_SAVE_PATH}: {e}")
            print("Starting with fresh parameters.")
    elif LOAD_EXISTING_MODEL:
        print(f"Warning: {MODEL_SAVE_PATH} not found. Starting training from scratch.")
        
    return lstm_model

def prepare_targets_for_bptt(target_vector, seq_length, output_size):
    """
    When using BPTT in a regression task focused on final prediction:
    - Create dummy targets (zeros) for all steps except the last one
    """
    targets_for_bptt = [np.zeros((output_size, 1)) for _ in range(seq_length - 1)] # Dummy targets
    targets_for_bptt.append(target_vector) # Actual target vector for final step
    return targets_for_bptt

def train_model(lstm_model, data_X, data_Y, hidden_size, max_iterations):
    """Train the LSTM model on the prepared sequences"""
    n = 0  # Iteration counter
    p = 0  # Pointer to current sequence
    
    # Initialize states
    h_prev = np.zeros((hidden_size, 1))
    c_prev = np.zeros((hidden_size, 1))
    
    print("\nStarting Training (Regression Task)...\n")
    losses = [] # To store loss values for plotting
    
    try:
        while n <= max_iterations:
            # Reset pointer and states at the end of epoch
            if p + 1 > len(data_X):
                p = 0
                h_prev = np.zeros((hidden_size, 1))
                c_prev = np.zeros((hidden_size, 1))
            
            inputs_vectors = data_X[p]
            target_vector = data_Y[p]
            
            # Prepare targets for BPTT (zeros except last step)
            targets_for_bptt = prepare_targets_for_bptt(target_vector, seq_length, output_size=1)
            
            # Perform training step
            loss, h_prev, c_prev = lstm_model.train_step(inputs_vectors, targets_for_bptt, h_prev, c_prev)
            current_smooth_loss = lstm_model.smooth_loss if lstm_model.smooth_loss is not None else loss
            
            if current_smooth_loss is not None:
                losses.append(current_smooth_loss)
            
            # Print progress
            if n % print_every == 0:
                current_loss = lstm_model.smooth_loss if lstm_model.smooth_loss is not None else loss
                print(f'Iter: {n}, Smoothed Loss (MSE): {current_loss:.6f}')
            
            # Save model periodically
            if n % save_every == 0 and n > 0:
                save_model(lstm_model, mean_price, std_price)
            
            p += 1
            n += 1
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    print("\nTraining finished.")
    return h_prev, c_prev, losses

def save_model(lstm_model, mean_price, std_price):
    """Save model parameters to file"""
    try:
        np.savez(MODEL_SAVE_PATH,
                # Model parameters
                Wf=lstm_model.Wf, Wi=lstm_model.Wi, Wc=lstm_model.Wc, Wo=lstm_model.Wo, Wy=lstm_model.Wy,
                bf=lstm_model.bf, bi=lstm_model.bi, bc=lstm_model.bc, bo=lstm_model.bo, by=lstm_model.by,
                # Adagrad memory
                mWf=lstm_model.mWf, mWi=lstm_model.mWi, mWc=lstm_model.mWc, mWo=lstm_model.mWo, mWy=lstm_model.mWy,
                mbf=lstm_model.mbf, mbi=lstm_model.mbi, mbc=lstm_model.mbc, mbo=lstm_model.mbo, mby=lstm_model.mby,
                # Hyperparameters and normalization values
                input_size=lstm_model.input_size, hidden_size=lstm_model.hidden_size,
                output_size=lstm_model.output_size, task_type=lstm_model.task_type,
                mean_price=mean_price, std_price=std_price
                )
        print(f"Model saved to {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Error saving model: {e}")

def evaluate_model(lstm_model, data_X, data_Y, h_state, c_state, mean_price, std_price):
    """Evaluate the model on a specific sequence"""
    # Get the sequence and actual next price
    input_sequence_vectors = data_X[-1]
    actual_next_normalized_price_vector = data_Y[-1]
    actual_next_normalized_price = actual_next_normalized_price_vector[0][0]
    
    # Forward pass with the given hidden/cell state
    outputs, _, _ = lstm_model.forward(input_sequence_vectors, h_state, c_state)
    
    # Get prediction from last time step
    predicted_normalized_price_vector = outputs['ys'][seq_length - 1]
    predicted_normalized_price = predicted_normalized_price_vector[0, 0]
    
    # Convert normalized prices back to actual prices
    predicted_price = predicted_normalized_price * std_price + mean_price
    actual_next_price = actual_next_normalized_price * std_price + mean_price
    
    # Print evaluation results
    print(f"Predicted Normalized Price: {predicted_normalized_price:.4f}")
    print(f"Actual Normalized Price:   {actual_next_normalized_price:.4f}")
    print("-" * 20)
    print(f"Predicted Actual Price: {predicted_price:.2f}")
    print(f"Actual Next Price:      {actual_next_price:.2f}")
    print(f"Prediction Error:       {abs(predicted_price - actual_next_price):.2f}")
    
    return predicted_price, actual_next_price

def generate_predictions(lstm_model, data_X, hidden_size, mean_price, std_price):
    """Generate predictions for all sequences in the dataset"""
    all_predicted_normalized_prices = []
    h_predict = np.zeros((hidden_size, 1))
    c_predict = np.zeros((hidden_size, 1))
    
    for input_sequence_vectors in data_X:
        # Forward pass
        outputs, h_predict, c_predict = lstm_model.forward(input_sequence_vectors, h_predict, c_predict)
        # Get prediction from last step
        predicted_normalized_price = outputs['ys'][seq_length - 1][0, 0]
        all_predicted_normalized_prices.append(predicted_normalized_price)
    
    # Convert to actual prices
    all_predicted_prices = [p * std_price + mean_price for p in all_predicted_normalized_prices]
    print(f"Generated {len(all_predicted_prices)} predictions.")
    return all_predicted_prices

def plot_results(stock_prices, all_predicted_prices, losses, seq_length):
    """Plot training loss and predictions"""
    plt.figure(figsize=(15, 6))
    
    # Plot 1: Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Smoothed Training Loss (MSE)')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    
    # Plot 2: Actual vs Predicted Prices
    plt.subplot(1, 2, 2)
    time_steps = np.arange(len(stock_prices))
    plt.plot(time_steps, stock_prices, label='Actual Prices', alpha=0.7)
    plt.plot(time_steps[seq_length:], all_predicted_prices, 
             label='Predicted Prices', linestyle='--', alpha=0.8)
    plt.title('Stock Price Prediction')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Main execution workflow
if __name__ == "__main__":
    # Add command line argument for evaluation-only mode
    parser = argparse.ArgumentParser(description='LSTM Stock Price Prediction')
    parser.add_argument('--eval-only', action='store_true', 
                        help='Run in evaluation mode only (no training)')
    args = parser.parse_args()
    
    # 1. Load and preprocess data
    stock_prices = load_stock_data()
    data_X, data_Y, mean_price, std_price = preprocess_data(stock_prices, seq_length)
    
    # 2. Initialize model
    input_size = 1      # Single price input at each time step
    output_size = 1     # Single price output (prediction)
    lstm_model = init_or_load_model(input_size, hidden_size, output_size, seq_length, learning_rate)
    
    # Check if the model was loaded successfully
    model_loaded = os.path.exists(MODEL_SAVE_PATH) and LOAD_EXISTING_MODEL
    
    # 3. Train model (skip if in evaluation-only mode)
    if not args.eval_only:
        h_final, c_final, losses = train_model(lstm_model, data_X, data_Y, hidden_size, max_iterations)
        print("\nTraining completed.")
    else:
        if model_loaded:
            print("\nRunning in evaluation-only mode, using loaded model.")
            h_final = np.zeros((hidden_size, 1))  # Start with fresh states for evaluation
            c_final = np.zeros((hidden_size, 1))
            losses = []  # Empty list since we didn't train
        else:
            print("\nERROR: Cannot run in evaluation-only mode - no model file found.")
            print(f"Please ensure {MODEL_SAVE_PATH} exists or run without --eval-only.")
            exit(1)
    
    # 4. Evaluate on last sequence
    print("\nEvaluating model on the last sequence...")
    evaluate_model(lstm_model, data_X, data_Y, h_final, c_final, mean_price, std_price)
    
    # 5. Generate predictions for all sequences
    print("\nGenerating predictions for the entire dataset...")
    all_predicted_prices = generate_predictions(lstm_model, data_X, hidden_size, mean_price, std_price)
    
    # 6. Plot results
    print("\nPlotting predictions...")
    if not args.eval_only and losses:
        plot_results(stock_prices, all_predicted_prices, losses, seq_length)
    else:
        # Create a simplified plot without loss curve when in eval-only mode
        plt.figure(figsize=(10, 6))
        time_steps = np.arange(len(stock_prices))
        plt.plot(time_steps, stock_prices, label='Actual Prices', alpha=0.7)
        plt.plot(time_steps[seq_length:], all_predicted_prices, 
                 label='Predicted Prices', linestyle='--', alpha=0.8)
        plt.title('Stock Price Prediction (Evaluation Only)')
        plt.xlabel('Time Step')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    print("Done.")
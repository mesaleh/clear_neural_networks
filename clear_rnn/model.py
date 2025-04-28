# model.py
import numpy as np

class RNN:
    """
    A simple Vanilla Recurrent Neural Network (RNN) model for character-level text generation.

    This class encapsulates the model parameters, the forward and backward pass logic
    (including Backpropagation Through Time - BPTT), and a method for sampling
    new sequences.
    """
    def __init__(self, hidden_size, vocab_size, learning_rate=1e-1):
        """
        Initializes the RNN model parameters and hyperparameters.

        Args:
            hidden_size (int): The number of units in the hidden layer (memory capacity).
            vocab_size (int): The size of the vocabulary (number of unique characters).
                               This determines the input and output layer sizes.
            learning_rate (float): The step size for parameter updates during training.
        """
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate

        # --- Model Parameters ---
        # Weights for input-to-hidden connections (shape: hidden_size x vocab_size)
        self.Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
        # Weights for hidden-to-hidden connections (the recurrent connection) (shape: hidden_size x hidden_size)
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        # Weights for hidden-to-output connections (shape: vocab_size x hidden_size)
        self.Why = np.random.randn(vocab_size, hidden_size) * 0.01

        # Biases for the hidden layer (shape: hidden_size x 1)
        self.bh = np.zeros((hidden_size, 1))
        # Biases for the output layer (shape: vocab_size x 1)
        self.by = np.zeros((vocab_size, 1))

        # --- Adagrad Memory ---
        # Memory variables for Adagrad optimizer, initialized to zeros.
        # These store the sum of squared gradients for each parameter.
        self.mWxh = np.zeros_like(self.Wxh)
        self.mWhh = np.zeros_like(self.Whh)
        self.mWhy = np.zeros_like(self.Why)
        self.mbh = np.zeros_like(self.bh)
        self.mby = np.zeros_like(self.by)

    def _loss_fun(self, inputs, targets, hprev):
        """
        Performs a full forward and backward pass for a sequence of inputs and targets.
        Calculates the loss, gradients, and the final hidden state.

        Args:
            inputs (list): List of integers representing input character indices.
            targets (list): List of integers representing target character indices (inputs shifted by one).
            hprev (np.array): Initial hidden state (from the previous sequence chunk). Shape: (hidden_size, 1).

        Returns:
            loss (float): The total cross-entropy loss for the sequence.
            dWxh, dWhh, dWhy, dbh, dby: Gradients for the model parameters.
            hnext (np.array): The final hidden state after processing the sequence. Shape: (hidden_size, 1).
        """
        # --- Stores for forward pass intermediates ---
        # Using dictionaries to store values at each time step 't'
        xs, hs, os, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)  # Store initial hidden state at index -1 for convenience
        loss = 0

        # === Forward Pass ===
        for t in range(len(inputs)):
            # 1. Encode input character: Convert index to one-hot vector
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1

            # 2. Calculate hidden state: ht = tanh(Wxh*xt + Whh*h_{t-1} + bh)
            #    This is the core recurrent calculation. hs[t] depends on hs[t-1].
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh)

            # 3. Calculate output logits: ot = Why*ht + by
            #    Logits are the raw scores before applying softmax.
            os[t] = np.dot(self.Why, hs[t]) + self.by

            # 4. Calculate output probabilities: pt = softmax(ot)
            #    Subtract max for numerical stability during exponentiation.
            exp_os = np.exp(os[t] - np.max(os[t]))
            ps[t] = exp_os / np.sum(exp_os)

            # 5. Calculate cross-entropy loss for this time step
            #    Loss = -log(probability of the correct target character)
            loss += -np.log(ps[t][targets[t], 0])

        # === Backward Pass (Backpropagation Through Time - BPTT) ===

        # Initialize gradients for this sequence chunk
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)

        # Initialize gradient of the loss wrt the hidden state for the *next* time step
        # This 'dhnext' carries the gradient information back through time.
        dhnext = np.zeros_like(hs[0])

        # Iterate backwards through the time steps (from last to first)
        for t in reversed(range(len(inputs))):
            # 1. Calculate gradient of loss wrt output logits (dL/dot)
            #    For softmax + cross-entropy loss, this derivative is simple: pt - yt
            #    where yt is the one-hot encoded target vector.
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1  # Subtract 1 from the probability of the correct class

            # 2. Calculate gradients for the output layer (Why, by)
            #    dL/dWhy = dL/dot * dot/dWhy = dy * hs[t].T
            #    dL/dby = dL/dot * dot/dby = dy
            dWhy += np.dot(dy, hs[t].T)
            dby += dy

            # 3. Backpropagate gradient to the hidden state (dL/dht)
            #    The hidden state hs[t] influences the loss via two paths:
            #    a) Directly through the output layer (Why * hs[t])
            #    b) Indirectly through the next hidden state (Whh * hs[t] -> hs[t+1] ...)
            #    So, dL/dht = (gradient from output) + (gradient from next hidden state)
            #    dL/dht = np.dot(Why.T, dy) + dhnext
            dh = np.dot(self.Why.T, dy) + dhnext

            # 4. Backpropagate through the tanh activation function
            #    dL/d(tanh_input) = dL/dht * dht/d(tanh_input)
            #    dht/d(tanh_input) = (1 - tanh^2(input)) = (1 - hs[t]^2)
            dhraw = (1 - hs[t] * hs[t]) * dh  # Gradient before the tanh nonlinearity

            # 5. Calculate gradients for the hidden layer parameters (Wxh, Whh, bh)
            #    dL/dbh = dL/d(tanh_input) * d(tanh_input)/dbh = dhraw * 1
            #    dL/dWxh = dL/d(tanh_input) * d(tanh_input)/dWxh = dhraw * xs[t].T
            #    dL/dWhh = dL/d(tanh_input) * d(tanh_input)/dWhh = dhraw * hs[t-1].T
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T) # Uses hidden state from *previous* step

            # 6. Calculate the gradient to pass to the *previous* time step (t-1)
            #    This becomes 'dhnext' for the next iteration of the backward loop.
            #    dL/dh_{t-1} = dL/d(tanh_input) * d(tanh_input)/dh_{t-1} = dhraw.T @ Whh
            dhnext = np.dot(self.Whh.T, dhraw)

        # --- Gradient Clipping ---
        # Prevent exploding gradients by clipping values outside a specified range [-5, 5].
        # Modifies the gradient arrays in-place.
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        # Return the loss, calculated gradients, and the final hidden state of this sequence
        hnext = hs[len(inputs)-1]
        return loss, dWxh, dWhh, dWhy, dbh, dby, hnext

    def sample(self, h, seed_ix, n):
        """
        Generates a sequence of character indices from the model.

        Args:
            h (np.array): Initial hidden state. Shape: (hidden_size, 1).
            seed_ix (int): Index of the first character to seed the generation.
            n (int): Number of characters to generate.

        Returns:
            ixes (list): List of generated character indices.
        """
        # Create a one-hot vector for the initial seed character
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1

        ixes = []  # List to store the generated indices

        # Generate 'n' characters one by one
        for t in range(n):
            # Calculate the hidden state using the current input 'x' and previous hidden state 'h'
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            # Calculate output logits
            y = np.dot(self.Why, h) + self.by
            # Calculate probabilities using softmax
            p = np.exp(y - np.max(y)) / np.sum(np.exp(y - np.max(y)))

            # Sample the *next* character index based on the calculated probabilities 'p'.
            # This introduces randomness, preventing deterministic (and often repetitive) output.
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())

            # Prepare the *sampled* character as the input for the *next* time step.
            # Create a one-hot vector for the sampled character index 'ix'.
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1

            # Append the sampled index to the list of generated indices
            ixes.append(ix)

        return ixes

    def train_step(self, inputs, targets, hprev):
        """
        Performs one training step: forward pass, backward pass, and parameter update using Adagrad.

        Args:
            inputs (list): List of integers representing input character indices for the sequence chunk.
            targets (list): List of integers representing target character indices.
            hprev (np.array): Initial hidden state for this chunk. Shape: (hidden_size, 1).

        Returns:
            loss (float): The loss calculated for this sequence chunk.
            hnext (np.array): The final hidden state after processing the chunk. Shape: (hidden_size, 1).
        """
        # 1. Calculate loss and gradients using the internal loss function
        loss, dWxh, dWhh, dWhy, dbh, dby, hnext = self._loss_fun(inputs, targets, hprev)

        # 2. Perform parameter update using Adagrad optimizer
        # Adagrad adapts the learning rate for each parameter based on historical gradients.
        # Formula: param += -learning_rate * gradient / sqrt(memory + epsilon)
        epsilon = 1e-8 # Small value to prevent division by zero
        for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
                                      [dWxh, dWhh, dWhy, dbh, dby],
                                      [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]):
            mem += dparam * dparam  # Accumulate the square of the gradient
            param += -self.learning_rate * dparam / np.sqrt(mem + epsilon) # Update parameter

        return loss, hnext
import numpy as np

# --- Activation Functions ---
def sigmoid(x):
    """ Numerically stable sigmoid function. """
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))

def softmax(x):
    """ Compute softmax values for each set of scores in x. """
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / e_x.sum(axis=0, keepdims=True)

# --- LSTM Model Class ---
class LSTM:
    """
    A simple Long Short-Term Memory (LSTM) network implemented from scratch using NumPy.
    This class expects input sequences as lists of vectors.

    Handles the model architecture, forward pass, backward pass (BPTT),
    sampling (for classification tasks), and parameter updates using Adagrad.
    Can be used for 'classification' (cross-entropy loss, softmax output)
    or 'regression' (MSE loss, linear output).
    """
    def __init__(self, input_size, hidden_size, output_size, seq_length, learning_rate, task_type='classification'):
        """
        Initializes the LSTM model.

        Args:
            input_size (int): Dimension of the input vectors x_t.
            hidden_size (int): Number of units in the hidden layer (h_t, c_t dimension).
            output_size (int): Dimension of the output vectors y_t.
                               (vocab size for classification, number of regression values for regression).
            seq_length (int): Length of the input/output sequences processed during BPTT.
            learning_rate (float): Learning rate for the Adagrad optimizer.
            task_type (str): 'classification' or 'regression'. Determines loss function,
                             output activation handling, and initial gradient calculation.
        """
        assert task_type in ['classification', 'regression'], "task_type must be 'classification' or 'regression'"

        # Hidden size determines the size of the internal state vectors h_t and c_t (number of features/neurons)
        # Vocab size determines the size of the input and output layers
        # Sequence length determines how many time steps we unroll the LSTM for during training
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        self.task_type = task_type

        concat_size = hidden_size + input_size

        # Forget gate parameters (Controls what to forget from cell state)
        self.Wf = np.random.randn(hidden_size, concat_size) * 0.01
        self.bf = np.zeros((hidden_size, 1))

        # Input gate parameters (Controls what new information to store in cell state)
        self.Wi = np.random.randn(hidden_size, concat_size) * 0.01
        self.bi = np.zeros((hidden_size, 1))

        # Candidate cell state parameters (Proposes new values to add to cell state)
        self.Wc = np.random.randn(hidden_size, concat_size) * 0.01
        self.bc = np.zeros((hidden_size, 1))

        # Output gate parameters (Controls what to output from cell state)
        self.Wo = np.random.randn(hidden_size, concat_size) * 0.01
        self.bo = np.zeros((hidden_size, 1))

        # Output layer parameters (Maps hidden state to vocabulary space)
        self.Wy = np.random.randn(output_size, hidden_size) * 0.01
        self.by = np.zeros((output_size, 1))

        # Store all parameters in a list for easier access during updates/gradient checking
        self.params = [self.Wf, self.Wi, self.Wc, self.Wo, self.Wy,
                       self.bf, self.bi, self.bc, self.bo, self.by]

        # --- Adagrad Memory ---
        # Stores sum of squared gradients for adaptive learning rate.
        self.mWf, self.mWi, self.mWc, self.mWo, self.mWy = [np.zeros_like(p) for p in [self.Wf, self.Wi, self.Wc, self.Wo, self.Wy]]
        self.mbf, self.mbi, self.mbc, self.mbo, self.mby = [np.zeros_like(b) for b in [self.bf, self.bi, self.bc, self.bo, self.by]]
        self.adagrad_mem = [self.mWf, self.mWi, self.mWc, self.mWo, self.mWy,
                            self.mbf, self.mbi, self.mbc, self.mbo, self.mby]

        # Loss Tracking - Initial value more meaningful for classification
        if self.task_type == 'classification':
            self.smooth_loss = -np.log(1.0 / output_size) * seq_length
        else:
            # For regression, initialize differently, perhaps None or a placeholder
            self.smooth_loss = None # Or a large number like 1e6

    def forward(self, inputs_vectors, h_prev, c_prev):
        """
        Performs the forward pass of the LSTM for a sequence of input vectors.

        Calculates hidden states, cell states, gate activations, and output
        values/probabilities for each time step. Stores intermediate values needed
        for the backward pass.

        Args:
            inputs_vectors (list): List of input vectors. Each vector shape: [input_size, 1].
            h_prev (np.ndarray): Previous hidden state (shape: [hidden_size, 1]).
            c_prev (np.ndarray): Previous cell state (shape: [hidden_size, 1]).

        Returns:
            tuple: Contains:
                - outputs (dict): Dictionary storing intermediate values (xs, hs, cs, ys, gates, xc).
                                  Also contains 'ps' (probabilities) if task_type is 'classification'.
                - h_next (np.ndarray): Final hidden state.
                - c_next (np.ndarray): Final cell state.
        """
        outputs = {
            'xs': {}, 'hs': {-1: h_prev}, 'cs': {-1: c_prev},
            'ys': {}, # Raw outputs before final activation/softmax
            'f_gates': {}, 'i_gates': {}, 'c_tilde': {}, 'o_gates': {},
            'xc': {}
        }
        # Add 'ps' dictionary only if needed for classification
        if self.task_type == 'classification':
            outputs['ps'] = {}

        seq_len = len(inputs_vectors)

        for t in range(seq_len):
            outputs['xs'][t] = inputs_vectors[t]

            # The implementation in this code, which concatenates h_{t-1} (previous hidden state) and x_t (current input)
            # into a single larger vector xc, and then multiplies this concatenated vector by the weight matrices (like Wf),
            # is the standard and mathematically correct approach used in almost all practical LSTM implementations (including frameworks like TensorFlow, PyTorch, etc.).
            # z_f = Wf · concat(h_t-1, x_t) + b
            # this is mathematically equivalent to the approach commonly on LSTM diagrams of
            # z_f = WWf · [h_t-1, x_t] + b_f, since
            # (hidden_size x concat_size) x (concat_size x 1) + (hidden_size x 1) = (hidden_size x 1)
            # is equivalent to
            # (hidden_size x hidden_size) x (hidden_size x 1) + (hidden_size x input_size) x (input_size x 1) + (hidden_size x 1) = (hidden_size x 1)
            # So it's easier and more efficient to use one bigger weight matrix than using two weight matrices.
            outputs['xc'][t] = np.concatenate((outputs['hs'][t-1], outputs['xs'][t]), axis=0)

            # Forget gate
            outputs['f_gates'][t] = sigmoid(np.dot(self.Wf, outputs['xc'][t]) + self.bf)
            # Input gate
            outputs['i_gates'][t] = sigmoid(np.dot(self.Wi, outputs['xc'][t]) + self.bi)
            # Candidate cell state
            outputs['c_tilde'][t] = np.tanh(np.dot(self.Wc, outputs['xc'][t]) + self.bc)
            # Output gate
            outputs['o_gates'][t] = sigmoid(np.dot(self.Wo, outputs['xc'][t]) + self.bo)
            # Cell state update
            outputs['cs'][t] = outputs['f_gates'][t] * outputs['cs'][t-1] + \
                               outputs['i_gates'][t] * outputs['c_tilde'][t]
            # Hidden state update
            # outputs['hs'][t] dimension is (hidden_size, 1)
            outputs['hs'][t] = outputs['o_gates'][t] * np.tanh(outputs['cs'][t])

            # Output layer (Dense layer): Calculate raw output 'ys' (logits)
            # ys shape: (output_size, 1)
            outputs['ys'][t] = np.dot(self.Wy, outputs['hs'][t]) + self.by

            # Apply final activation only if classification
            if self.task_type == 'classification':
                outputs['ps'][t] = softmax(outputs['ys'][t])
            # For regression, 'ys' is the final prediction value

        h_next = outputs['hs'][seq_len-1]
        c_next = outputs['cs'][seq_len-1]

        return outputs, h_next, c_next

    def backward(self, targets, outputs):
        """
        Performs the backward pass (BPTT) for the LSTM.

        Calculates gradients for all parameters based on the outputs
        and the target sequence (indices for classification, values for regression).

        Args:
            targets (list): List of target values.
                            - For classification: integers (indices).
                            - For regression: numpy arrays of shape (output_size, 1).
            outputs (dict): Dictionary containing intermediate values from the forward pass.

        Returns:
            tuple: Gradients for all parameters (dWf, ...), dh_prev, dc_prev.
        """
        dWf, dWi, dWc, dWo, dWy = np.zeros_like(self.Wf), np.zeros_like(self.Wi), np.zeros_like(self.Wc), np.zeros_like(self.Wo), np.zeros_like(self.Wy)
        dbf, dbi, dbc, dbo, dby = np.zeros_like(self.bf), np.zeros_like(self.bi), np.zeros_like(self.bc), np.zeros_like(self.bo), np.zeros_like(self.by)

        # Initialize gradients for hidden and cell state to pass back through time
        dh_next = np.zeros_like(outputs['hs'][0]) # Gradient w.r.t h at t+1
        dc_next = np.zeros_like(outputs['cs'][0]) # Gradient w.r.t c at t+1
        seq_len = len(targets)

        # Iterate backwards through time (from last step to first)
        for t in reversed(range(seq_len)):
            # --- 1. Gradient from Loss (w.r.t. y_t or p_t) ---
            # Calculate dy = dLoss/dys (gradient w.r.t raw output ys)
            
            if self.task_type == 'classification':
                # Gradient for Softmax + Cross-Entropy
                dy = np.copy(outputs['ps'][t])
                dy[targets[t]] -= 1 # Assumes targets[t] is the correct index
            else: # Regression
                if t == seq_len - 1: # Only apply loss gradient at the final step since it is where we get the final output
                    # Gradient for MSE Loss: dL/dys = ys - target
                    # Assumes targets[t] is a numpy array of shape (output_size, 1)
                    dy = outputs['ys'][t] - targets[t]
                    # Optional: Scale by 1/N or 2/N if needed, but often absorbed by learning rate
                    # dy /= seq_len # Example scaling if using average MSE in backward pass
                else:
                    # For intermediate steps, there is no direct loss contribution
                    # The gradient dh will propagate back from the next step (t+1)
                    dy = np.zeros((self.output_size, 1)) # No loss gradient for this step

            # --- 2. Gradients for Output Layer (Wy, by) --- (Same calculation)
            dWy += np.dot(dy, outputs['hs'][t].T)
            dby += dy

            # --- 3. Backpropagate to Hidden State h_t ---
            # Gradient w.r.t h_t = (gradient from output layer) + (gradient from next step h_{t+1})
            dh = np.dot(self.Wy.T, dy) + dh_next

            # --- 4. Backpropagate through Hidden State calculation: h_t = o_t * tanh(c_t) ---
            # Gradient w.r.t o_t = dh * tanh(c_t)
            do = dh * np.tanh(outputs['cs'][t])
            # Gradient w.r.t tanh(c_t) = dh * o_t
            # Gradient w.r.t c_t (through tanh) = dh * o_t * (1 - tanh(c_t)^2)
            dc_from_h = dh * outputs['o_gates'][t] * (1 - np.tanh(outputs['cs'][t])**2)

            # --- 5. Backpropagate to Cell State c_t ---
            # Gradient w.r.t c_t = (gradient from hidden state h_t) + (gradient from next step c_{t+1})
            dc = dc_from_h + dc_next

            # --- 6. Backpropagate through Cell State update: c_t = f_t * c_{t-1} + i_t * c_tilde_t ---
            # Gradient w.r.t f_t = dc * c_{t-1}
            df = dc * outputs['cs'][t-1]
            # Gradient w.r.t c_{t-1} = dc * f_t  (This becomes dc_next for the previous step)
            dc_prev = dc * outputs['f_gates'][t]
            # Gradient w.r.t i_t = dc * c_tilde_t
            di = dc * outputs['c_tilde'][t]
            # Gradient w.r.t c_tilde_t = dc * i_t
            dc_tilde = dc * outputs['i_gates'][t]

            # --- 7. Backpropagate through Gate Activations (using derivatives) ---
            # Derivative of sigmoid: sig * (1 - sig)
            # Derivative of tanh: 1 - tanh^2
            # Gradient w.r.t pre-activation (raw) values (e.g., Wf*xc + bf)
            do_raw = do * outputs['o_gates'][t] * (1 - outputs['o_gates'][t])
            dc_tilde_raw = dc_tilde * (1 - outputs['c_tilde'][t]**2)
            di_raw = di * outputs['i_gates'][t] * (1 - outputs['i_gates'][t])
            df_raw = df * outputs['f_gates'][t] * (1 - outputs['f_gates'][t])

            # --- 8. Gradients for Gate Parameters (Wf, bf, Wi, bi, Wc, bc, Wo, bo) ---
            # Use the raw gate gradients and the concatenated input xc_t
            # Gradient w.r.t Wx = d(raw) * xc_t^T
            # Gradient w.r.t bx = d(raw)
            dWf += np.dot(df_raw, outputs['xc'][t].T)
            dbf += df_raw
            dWi += np.dot(di_raw, outputs['xc'][t].T)
            dbi += di_raw
            dWc += np.dot(dc_tilde_raw, outputs['xc'][t].T)
            dbc += dc_tilde_raw
            dWo += np.dot(do_raw, outputs['xc'][t].T)
            dbo += do_raw

            # --- 9. Backpropagate Gradient to Concatenated Input xc_t ---
            # Gradient w.r.t xc_t is the sum of gradients from all four gates
            # dxc = Wf.T*df_raw + Wi.T*di_raw + Wc.T*dc_tilde_raw + Wo.T*do_raw
            dxc = np.dot(self.Wf.T, df_raw) + \
                  np.dot(self.Wi.T, di_raw) + \
                  np.dot(self.Wc.T, dc_tilde_raw) + \
                  np.dot(self.Wo.T, do_raw)

            # --- 10. Backpropagate Gradient to Previous Hidden State h_{t-1} ---
            # Extract the part of dxc corresponding to h_{t-1}
            # This becomes dh_next for the previous time step
            dh_prev = dxc[:self.hidden_size, :]

            # --- 11. Update Gradients for Next Iteration (t-1) ---
            dh_next = dh_prev
            dc_next = dc_prev # Already calculated in step 6

        # Clip gradients to mitigate exploding gradients
        grads = [dWf, dWi, dWc, dWo, dWy, dbf, dbi, dbc, dbo, dby]
        grads = self._clip_gradients(grads)
        # Clip gradients to mitigate exploding gradients
        # Return clipped gradients and the gradients w.r.t the initial hidden/cell states
        # dh_next and dc_next now hold the gradients w.r.t. h_prev and c_prev
        return grads, dh_next, dc_next


    def _clip_gradients(self, grads, clip_value=5.0):
        """ Clips gradients element-wise to mitigate exploding gradients. """
        clipped_grads = []
        for grad in grads:
            # Use np.clip with out=grad for in-place modification
            np.clip(grad, -clip_value, clip_value, out=grad)
            clipped_grads.append(grad)
        return clipped_grads

    def _update_params_adagrad(self, grads, epsilon=1e-8):
        """ Updates model parameters using the Adagrad optimization algorithm. """
        for param, grad, mem in zip(self.params, grads, self.adagrad_mem):
            mem += grad * grad # Accumulate squared gradients
            # Update parameter: Scale learning rate by sqrt of accumulated squared gradients
            param -= self.learning_rate * grad / (np.sqrt(mem) + epsilon)

    def calculate_loss(self, targets, outputs):
        """
        Calculates the loss for the sequence based on task_type.

        Args:
            targets (list): List of target values (indices or vectors).
            outputs (dict): Dictionary from the forward pass.

        Returns:
            float: Average loss per time step.
        """
        loss = 0
        seq_len = len(targets)
        if seq_len == 0: return 0

        if self.task_type == 'classification':
            # Cross-Entropy Loss
            for t in range(seq_len):
                # targets[t] is the index of the correct class
                prob_of_target = outputs['ps'][t][targets[t], 0]
                loss += -np.log(prob_of_target + 1e-12) # Add epsilon for stability
        else: # Regression
            # Mean Squared Error (MSE) Loss
            last_step_index = seq_len - 1
            # targets[last_step_index] is the target vector/value array, shape (output_size, 1)
            # outputs['ys'][last_step_index] is the predicted vector/value array, shape (output_size, 1)
            loss = np.sum((outputs['ys'][last_step_index] - targets[last_step_index])**2)

            # Calculate mean over all elements and all time steps
            loss /= self.output_size # Mean squared error

        return loss

    def train_step(self, inputs_vectors, targets, h_prev, c_prev):
        """
        Performs a single training step: forward pass, loss calculation,
        backward pass (BPTT), gradient clipping, and parameter update.

        Args:
            inputs_vectors (list): Input sequence (list of vectors).
            targets (list): Target sequence (indices or vectors).
            h_prev (np.ndarray): Previous hidden state.
            c_prev (np.ndarray): Previous cell state.

        Returns:
            tuple: Contains:
                - loss (float): The average loss for this training step.
                - h_next (np.ndarray): The final hidden state.
                - c_next (np.ndarray): The final cell state.
        """
        # --- Forward Pass ---
        outputs, h_next, c_next = self.forward(inputs_vectors, h_prev, c_prev)
        # --- Calculate Loss ---
        loss = self.calculate_loss(targets, outputs)

        # --- Backward Pass (BPTT) ---
        # Returns clipped gradients and gradients w.r.t initial h/c (not needed here)
        grads, _, _ = self.backward(targets, outputs)

        # --- Update Parameters (Adagrad) ---
        self._update_params_adagrad(grads)

        # Update smooth loss (handle None case for regression)
        if self.smooth_loss is None:
            self.smooth_loss = loss
        else:
            self.smooth_loss = self.smooth_loss * 0.999 + loss * 0.001

        return loss, h_next, c_next

    def sample(self, seed_vector, h_prev, c_prev, n):
        """
        Generates a sequence of indices starting from a seed vector.
        This method is primarily intended for classification tasks.
        """
        # Add check for task type
        if self.task_type != 'classification':
            raise NotImplementedError("Sampling is only implemented for classification tasks.")

        x = seed_vector
        h, c = h_prev, c_prev
        ixes = []

        for _ in range(n):
            xc = np.concatenate((h, x), axis=0)
            f_gate = sigmoid(np.dot(self.Wf, xc) + self.bf)
            i_gate = sigmoid(np.dot(self.Wi, xc) + self.bi)
            c_tilde = np.tanh(np.dot(self.Wc, xc) + self.bc)
            o_gate = sigmoid(np.dot(self.Wo, xc) + self.bo)
            c = f_gate * c + i_gate * c_tilde
            h = o_gate * np.tanh(c)
            y = np.dot(self.Wy, h) + self.by
            p = softmax(y) # Use softmax for probabilities

            # Sample index from the probability distribution (over output_size)
            ix = np.random.choice(range(self.output_size), p=p.ravel())
            ixes.append(ix)

            # --- Prepare input vector for the next time step ---
            # Create a one-hot vector of dimension input_size.
            # Assumes for char-RNN sampling that input_size == output_size.
            if self.input_size != self.output_size:
                 print("Warning: Sampling assumes input_size == output_size for creating next input.")
            x = np.zeros((self.input_size, 1))
            if ix < self.input_size:
                x[ix] = 1
            else:
                print(f"Warning: Sampled index {ix} >= input_size {self.input_size}. Using zero vector.")

        return ixes
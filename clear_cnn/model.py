# clean_cnn/model.py

"""
NumPy-based Convolutional Neural Network Implementation - Educational Framework

This module provides a from-scratch implementation of core CNN building blocks
using only NumPy. It serves as an educational tool to understand the inner workings
of neural networks without relying on deep learning frameworks.

Main components:
1. Layer-based architecture with common building blocks:
   - Convolutional layers (Conv2D, Conv1D)
   - Activation functions (ReLU)
   - Pooling layers (MaxPool2D, MaxPool1D) 
   - Fully connected layers (Dense)
   - Reshaping operations (Flatten)
2. Forward and backward propagation implementation for each layer
3. Gradient-based optimization with AdaGrad
4. Sequential model container for stacking layers
5. Basic loss function implementations (cross-entropy)

The implementation deliberately exposes the mathematical operations of 
forward/backward passes to help understand the gradient flow in CNNs.
"""

import numpy as np

# --- Base Layer Class ---
class Layer:
    """
    Abstract base class for all layers in the neural network.
    """
    def __init__(self):
        self.params = {}  # Stores learnable parameters (e.g., weights, biases)
        self.grads = {}   # Stores gradients of learnable parameters
        self.optimizer_state = {} # Stores state for optimizers like Adagrad
        self.training_mode = True # For layers like Dropout (not used here yet)
        self.input_shape = None   # To store the shape of the input during forward pass

    def forward(self, input_data):
        """Performs the forward pass for the layer."""
        raise NotImplementedError("Each layer must implement its own forward pass.")

    def backward(self, doutput):
        """
        Performs the backward pass for the layer.
        Computes gradients with respect to its inputs and parameters.
        """
        raise NotImplementedError("Each layer must implement its own backward pass.")

    def update_params(self, learning_rate, optimizer_type='adagrad', adagrad_epsilon=1e-8):
        """
        Updates learnable parameters. Only implemented by layers with parameters.
        For simplicity, Adagrad logic is embedded here.
        """
        if not self.params: # No parameters to update
            return

        if optimizer_type == 'adagrad':
            for param_name in self.params:
                if param_name not in self.optimizer_state: # Initialize Adagrad memory
                    self.optimizer_state[param_name] = np.zeros_like(self.params[param_name])

                self.optimizer_state[param_name] += self.grads[param_name]**2
                self.params[param_name] -= learning_rate * self.grads[param_name] / \
                                          (np.sqrt(self.optimizer_state[param_name]) + adagrad_epsilon)
        else:
            raise NotImplementedError(f"Optimizer '{optimizer_type}' not implemented.")

    def set_training_mode(self, mode: bool):
        """Sets the training mode (e.g., for dropout or batch normalization)."""
        self.training_mode = mode

# --- Convolutional Layers ---

class Conv2D(Layer):
    """
    2D Convolutional Layer.

    Assumes input_data shape: (N, C_in, H_in, W_in)
            N: Batch size
            C_in: Number of input channels
            H_in: Height of input feature map
            W_in: Width of input feature map

    Kernel shape: (C_out, C_in, K_h, K_w)
            C_out: Number of output channels (filters)
            K_h: Kernel height
            K_w: Kernel width
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding_type='valid'):
        super().__init__()
        self.C_in = in_channels
        self.C_out = out_channels # Number of filters
        if isinstance(kernel_size, int):
            self.K_h = self.K_w = kernel_size
        else:
            # in case it's a tuple, e.g., (K_h, K_w)
            self.K_h, self.K_w = kernel_size
        self.S_h = self.S_w = stride if isinstance(stride, int) else stride

        self.padding_type = padding_type
        # the value of padding
        self.pad_h_before, self.pad_h_after = 0, 0
        self.pad_w_before, self.pad_w_after = 0, 0

        # Initialize weights (W) and biases (b)
        # He initialization for weights
        scale = np.sqrt(2.0 / (self.C_in * self.K_h * self.K_w))
        # weights are of four dimensions: (num of output kernels, num input channels, kernel height, kernel width)
        # biases are of one dimension: (num of output kernels,)
        self.params['W'] = np.random.randn(self.C_out, self.C_in, self.K_h, self.K_w) * scale
        self.params['b'] = np.zeros(self.C_out) # One bias per output channel/filter

        self.cache = {} # To store values needed for backpropagation

    def _calculate_output_dims_and_padding(self, H_in, W_in):
        """Calculates output dimensions and padding values."""
        if self.padding_type == 'valid':
            self.pad_h_before, self.pad_h_after = 0, 0
            self.pad_w_before, self.pad_w_after = 0, 0
            H_out = (H_in - self.K_h) // self.S_h + 1
            W_out = (W_in - self.K_w) // self.S_w + 1
        elif self.padding_type == 'same':
            # For 'same' padding, output size aims to be ceil(input_size / stride)
            H_out = int(np.ceil(H_in / self.S_h))
            W_out = int(np.ceil(W_in / self.S_w))

            # Calculate total padding needed
            # P_total = (H_out - 1) * Stride + KernelSize - H_in
            # calculates number of rows to add to the left and right
            total_pad_h = max(0, (H_out - 1) * self.S_h + self.K_h - H_in)
            # calculates number of columns to add to the left and right
            total_pad_w = max(0, (W_out - 1) * self.S_w + self.K_w - W_in)

            self.pad_h_before = total_pad_h // 2
            self.pad_h_after = total_pad_h - self.pad_h_before
            self.pad_w_before = total_pad_w // 2
            self.pad_w_after = total_pad_w - self.pad_w_before
        else:
            raise ValueError(f"Unknown padding type: {self.padding_type}")
        return H_out, W_out

    def forward(self, A_prev):
        """
        Performs the forward pass of the convolution.
        A_prev shape: (N, C_in, H_in, W_in)
        Output Z shape: (N, C_out, H_out, W_out)
        """
        self.input_shape = A_prev.shape
        N, C_in_data, H_in, W_in = A_prev.shape
        assert C_in_data == self.C_in, "Input channels mismatch"

        W = self.params['W']
        b = self.params['b']

        H_out, W_out = self._calculate_output_dims_and_padding(H_in, W_in)
        Z = np.zeros((N, self.C_out, H_out, W_out))

        # Apply padding to A_prev
        # np.pad format: ((before_ax0, after_ax0), (before_ax1, after_ax1), ...)
        A_prev_padded = np.pad(A_prev,
                               ((0, 0), (0, 0), (self.pad_h_before, self.pad_h_after), (self.pad_w_before, self.pad_w_after)),
                               mode='constant', constant_values=(0, 0))

        # Perform convolution
        for i in range(N):                      # Loop over batch
            a_prev_padded_sample = A_prev_padded[i]
            for f in range(self.C_out):         # Loop over filters (output channels)
                kernel = W[f, :, :, :]          # Shape: (C_in, K_h, K_w)
                bias = b[f]
                for h_out in range(H_out):      # Loop over output height
                    for w_out in range(W_out):  # Loop over output width
                        # Define the slice in the padded input
                        vert_start = h_out * self.S_h
                        vert_end = vert_start + self.K_h
                        horiz_start = w_out * self.S_w
                        horiz_end = horiz_start + self.K_w

                        # Extract the receptive field from the padded input
                        a_slice_padded = a_prev_padded_sample[:, vert_start:vert_end, horiz_start:horiz_end]

                        # Element-wise product between the slice and the kernel, sum over all C_in, K_h, K_w
                        # and add bias.
                        Z[i, f, h_out, w_out] = np.sum(a_slice_padded * kernel) + bias

        self.cache['A_prev_padded'] = A_prev_padded
        self.cache['W'] = W
        return Z

    def backward(self, dZ):
        """
        Performs the backward pass of the convolution.
        dZ (gradient of loss w.r.t. output Z) shape: (N, C_out, H_out, W_out)

        Computes:
        dA_prev (gradient of loss w.r.t. input A_prev)
        dW (gradient of loss w.r.t. weights W)
        db (gradient of loss w.r.t. biases b)
        """
        A_prev_padded = self.cache['A_prev_padded']
        W = self.cache['W'] # Shape: (C_out, C_in, K_h, K_w)
        N, C_in, H_in_padded, W_in_padded = A_prev_padded.shape
        _, C_out_dz, H_out, W_out = dZ.shape # dZ has C_out channels
        assert C_out_dz == self.C_out

        dA_prev_padded = np.zeros_like(A_prev_padded)
        self.grads['W'] = np.zeros_like(W)
        self.grads['b'] = np.zeros_like(self.params['b'])

        # --- Calculate gradient of bias (db) ---
        # db is sum of dZ over batch, height, and width dimensions for each filter.
        # we didn't consider axis=1 (C_out) because we want the gradient for each individual filter.
        self.grads['b'] = np.sum(dZ, axis=(0, 2, 3))

        # --- Calculate gradients dW and dA_prev_padded ---
        for i in range(N): # Loop over batch
            a_prev_padded_sample = A_prev_padded[i] # (C_in, H_in_pad, W_in_pad)
            da_prev_padded_sample = dA_prev_padded[i] # (C_in, H_in_pad, W_in_pad)

            for f in range(self.C_out): # Loop over filters
                for h_out in range(H_out): # Loop over output height
                    for w_out in range(W_out): # Loop over output width
                        # Define the slice in the padded input
                        vert_start = h_out * self.S_h
                        vert_end = vert_start + self.K_h
                        horiz_start = w_out * self.S_w
                        horiz_end = horiz_start + self.K_w

                        # Current gradient from dZ for this output element
                        dZ_curr = dZ[i, f, h_out, w_out]

                        # --- a note about calculating dW ---
                        # dW[f] accumulates the gradient for filter f.
                        # The gradient of a filter's weights (often denoted dW) is a tensor that has the same dimensions as the filter (kernel)
                        # itself. For example, for a 2D convolutional filter with shape (C_in, K_h, K_w)
                        # (where C_in is the number of input channels, K_h is kernel height, and K_w is kernel width), 
                        # its gradient dW will also have the shape (C_in, K_h, K_w).                        

                        # Extract the receptive field used for this output (input patch). The first dimension is the channel dimension.
                        a_slice = a_prev_padded_sample[:, vert_start:vert_end, horiz_start:horiz_end]

                        # --- Calculate dW ---
                        # dW[f] += a_slice * dZ_curr
                        # This sums contributions from all input samples (N) and all spatial locations (H_out, W_out)
                        # self.grads['W'] has dimensions (filter index, input dimensions, kernel height, kernel width)
                        # The accumulation += is done here to accumulate the gradient for different weights of the same filter 
                        # as they move by a stride step and the same pixel gets multiplied by different weights.
                        self.grads['W'][f, :, :, :] += a_slice * dZ_curr

                        # --- Calculate dA_prev_padded ---
                        # The gradient dZ_curr is passed back through the weights W[f]
                        # to the corresponding input slice.
                        # W[f] has shape (C_in, K_h, K_w)
                        # The accumulation += is done for all spatial locations (H_out, W_out) for multiple filters.
                        # This is the gradient of the loss w.r.t. the input A_prev_padded
                        da_prev_padded_sample[:, vert_start:vert_end, horiz_start:horiz_end] += W[f, :, :, :] * dZ_curr
        
        # --- Unpad dA_prev_padded to get dA_prev ---
        # dA_prev should have the shape of the original A_prev (before padding)
        original_H_in = self.input_shape[2]
        original_W_in = self.input_shape[3]
        
        # Extract the central part of dA_prev_padded corresponding to the original A_prev
        # If pad_h_before = 0 and pad_h_after = 0, then original_H_in = H_in_padded, so this slice is the whole array.
        dA_prev = dA_prev_padded[:, :, self.pad_h_before : self.pad_h_before + original_H_in,
                                        self.pad_w_before : self.pad_w_before + original_W_in]
        return dA_prev

class Conv1D(Layer):
    """
    1D Convolutional Layer.
    Assumes input_data shape: (N, C_in, L_in)
    Kernel shape: (C_out, C_in, K_len)
    """
    def __init__(self, in_channels, out_channels, kernel_length, stride=1, padding_type='valid'):
        super().__init__()
        self.C_in = in_channels
        self.C_out = out_channels
        self.K_len = kernel_length
        self.S_len = stride
        self.padding_type = padding_type
        self.pad_len_before, self.pad_len_after = 0, 0

        scale = np.sqrt(2.0 / (self.C_in * self.K_len))
        self.params['W'] = np.random.randn(self.C_out, self.C_in, self.K_len) * scale
        self.params['b'] = np.zeros(self.C_out)
        self.cache = {}

    def _calculate_output_dims_and_padding(self, L_in):
        if self.padding_type == 'valid':
            self.pad_len_before, self.pad_len_after = 0, 0
            L_out = (L_in - self.K_len) // self.S_len + 1
        elif self.padding_type == 'same':
            L_out = int(np.ceil(L_in / self.S_len))
            total_pad_len = max(0, (L_out - 1) * self.S_len + self.K_len - L_in)
            self.pad_len_before = total_pad_len // 2
            self.pad_len_after = total_pad_len - self.pad_len_before
        else:
            raise ValueError(f"Unknown padding type: {self.padding_type}")
        return L_out

    def forward(self, A_prev):
        self.input_shape = A_prev.shape
        N, C_in_data, L_in = A_prev.shape
        assert C_in_data == self.C_in

        W, b = self.params['W'], self.params['b']
        L_out = self._calculate_output_dims_and_padding(L_in)
        Z = np.zeros((N, self.C_out, L_out))

        A_prev_padded = np.pad(A_prev,
                               ((0,0), (0,0), (self.pad_len_before, self.pad_len_after)),
                               mode='constant', constant_values=(0,0))

        for i in range(N):
            a_prev_padded_sample = A_prev_padded[i]
            for f in range(self.C_out):
                kernel, bias = W[f], b[f]
                for l_out in range(L_out):
                    start = l_out * self.S_len
                    end = start + self.K_len
                    a_slice = a_prev_padded_sample[:, start:end]
                    Z[i, f, l_out] = np.sum(a_slice * kernel) + bias
        
        self.cache['A_prev_padded'] = A_prev_padded
        self.cache['W'] = W
        return Z

    def backward(self, dZ):
        A_prev_padded = self.cache['A_prev_padded']
        W = self.cache['W']
        N, C_in, L_in_padded = A_prev_padded.shape
        _, _, L_out = dZ.shape

        dA_prev_padded = np.zeros_like(A_prev_padded)
        self.grads['W'] = np.zeros_like(W)
        self.grads['b'] = np.zeros_like(self.params['b'])

        self.grads['b'] = np.sum(dZ, axis=(0, 2))

        for i in range(N):
            a_prev_padded_sample = A_prev_padded[i]
            da_prev_padded_sample = dA_prev_padded[i]
            for f in range(self.C_out):
                for l_out in range(L_out):
                    start = l_out * self.S_len
                    end = start + self.K_len
                    dZ_curr = dZ[i, f, l_out]
                    a_slice = a_prev_padded_sample[:, start:end]
                    self.grads['W'][f] += a_slice * dZ_curr
                    da_prev_padded_sample[:, start:end] += W[f] * dZ_curr
        
        original_L_in = self.input_shape[2]
        dA_prev = dA_prev_padded[:, :, self.pad_len_before : self.pad_len_before + original_L_in]
        return dA_prev

# --- Activation Layers ---

class ReLU(Layer):
    """Rectified Linear Unit activation layer."""
    def __init__(self):
        super().__init__()
        self.cache = {}

    def forward(self, Z):
        """Z can be any shape."""
        self.input_shape = Z.shape
        self.cache['Z'] = Z
        A = np.maximum(0, Z)
        return A

    def backward(self, dA):
        """dA has the same shape as Z."""
        Z = self.cache['Z']
        dZ = dA * (Z > 0) # Gradient is dA where Z > 0, else 0
        return dZ

# --- Pooling Layers ---

class MaxPool2D(Layer):
    """
    Max Pooling layer for 2D inputs.
    Input shape: (N, C, H_in, W_in)
    Output shape: (N, C, H_out, W_out)
    """
    def __init__(self, pool_size, stride=None):
        super().__init__()
        if isinstance(pool_size, int):
            self.K_h = self.K_w = pool_size
        else:
            self.K_h, self.K_w = pool_size

        if stride is None: # Default stride is same as pool_size
            self.S_h, self.S_w = self.K_h, self.K_w
        elif isinstance(stride, int):
            self.S_h = self.S_w = stride
        else:
            self.S_h, self.S_w = stride
        self.cache = {}

    def forward(self, A_prev):
        self.input_shape = A_prev.shape
        N, C, H_in, W_in = A_prev.shape

        H_out = (H_in - self.K_h) // self.S_h + 1
        W_out = (W_in - self.K_w) // self.S_w + 1
        A_new = np.zeros((N, C, H_out, W_out))
        # Mask to store positions of max values for backprop
        mask = np.zeros_like(A_prev, dtype=bool)

        for i in range(N):
            for c in range(C):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        vert_start = h_out * self.S_h
                        vert_end = vert_start + self.K_h
                        horiz_start = w_out * self.S_w
                        horiz_end = horiz_start + self.K_w

                        a_prev_slice = A_prev[i, c, vert_start:vert_end, horiz_start:horiz_end]
                        max_val = np.max(a_prev_slice)
                        A_new[i, c, h_out, w_out] = max_val
                        
                        # Create mask for this window
                        window_mask = (a_prev_slice == max_val)
                        # If multiple elements have max_val, this mask will have multiple True.
                        # In standard backprop, usually only one is chosen (e.g., the first one).
                        # For simplicity, this implementation might distribute gradient to all max values.
                        # To pick one (often first):
                        # temp_mask = np.zeros_like(window_mask, dtype=bool)
                        # r_max, c_max = np.unravel_index(np.argmax(a_prev_slice), a_prev_slice.shape)
                        # temp_mask[r_max, c_max] = True
                        # mask[i, c, vert_start:vert_end, horiz_start:horiz_end] = temp_mask
                        mask[i, c, vert_start:vert_end, horiz_start:horiz_end] = \
                            mask[i, c, vert_start:vert_end, horiz_start:horiz_end] | window_mask


        self.cache['A_prev'] = A_prev
        self.cache['mask'] = mask
        return A_new

    def backward(self, dA_new):
        A_prev = self.cache['A_prev']
        mask = self.cache['mask']
        N, C, H_in, W_in = A_prev.shape
        _, _, H_out, W_out = dA_new.shape

        dA_prev = np.zeros_like(A_prev)

        for i in range(N):
            for c in range(C):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        vert_start = h_out * self.S_h
                        vert_end = vert_start + self.K_h
                        horiz_start = w_out * self.S_w
                        horiz_end = horiz_start + self.K_w

                        # Get the gradient for the current output element
                        grad = dA_new[i, c, h_out, w_out]
                        # Get the mask for the corresponding input window
                        a_prev_window_mask = mask[i, c, vert_start:vert_end, horiz_start:horiz_end]
                        
                        # Distribute the gradient to the positions where the max was taken
                        dA_prev[i, c, vert_start:vert_end, horiz_start:horiz_end] += a_prev_window_mask * grad
        return dA_prev


class MaxPool1D(Layer):
    """
    Max Pooling layer for 1D inputs.
    Input shape: (N, C, L_in)
    Output shape: (N, C, L_out)
    """
    def __init__(self, pool_length, stride=None):
        super().__init__()
        self.K_len = pool_length
        self.S_len = stride if stride is not None else pool_length # Default stride is pool_length
        self.cache = {}
        # self.input_shape will be set in forward pass (already in Layer base)

    def forward(self, A_prev):
        self.input_shape = A_prev.shape
        N, C, L_in = A_prev.shape

        L_out = (L_in - self.K_len) // self.S_len + 1
        A_new = np.zeros((N, C, L_out))
        mask = np.zeros_like(A_prev, dtype=bool) # For storing positions of max values

        for i in range(N):
            for c in range(C):
                for l_out in range(L_out):
                    start = l_out * self.S_len
                    end = start + self.K_len
                    a_prev_slice = A_prev[i, c, start:end]
                    max_val = np.max(a_prev_slice)
                    A_new[i, c, l_out] = max_val
                    
                    window_mask = (a_prev_slice == max_val)
                    mask[i, c, start:end] = mask[i, c, start:end] | window_mask
        
        self.cache['A_prev'] = A_prev
        self.cache['mask'] = mask
        return A_new

    def backward(self, dA_new):
        A_prev = self.cache['A_prev']
        mask = self.cache['mask']
        # N, C, L_in = A_prev.shape # Not strictly needed if using self.input_shape
        # _, _, L_out = dA_new.shape # Not strictly needed
        N, C, L_in = self.input_shape # Use stored input_shape

        dA_prev = np.zeros_like(A_prev) # Or np.zeros(self.input_shape)

        # Loop dimensions for dA_new (output of pooling)
        N_dA, C_dA, L_out_dA = dA_new.shape
        assert C_dA == C, "Channel mismatch in MaxPool1D backward"


        for i in range(N_dA): # Iterate up to N_dA (batch size of incoming gradient)
            for c_channel in range(C_dA): # Iterate up to C_dA (channels of incoming gradient)
                for l_out in range(L_out_dA): # Iterate up to L_out_dA (length of incoming gradient)
                    start = l_out * self.S_len
                    end = start + self.K_len
                    grad = dA_new[i, c_channel, l_out]
                    # Ensure mask slicing corresponds to current channel and sample
                    a_prev_window_mask = mask[i, c_channel, start:end] 
                    dA_prev[i, c_channel, start:end] += a_prev_window_mask * grad
        return dA_prev
    

# --- Reshaping Layers ---

class Flatten(Layer):
    """
    Flattens the input from (N, C, H, W) or (N, C, L) to (N, C*H*W) or (N, C*L).
    """
    def __init__(self):
        super().__init__()
        self.original_shape = None # To store shape for unflattening in backward pass
        self.cache = {} 
        
    def forward(self, A_prev):
        self.input_shape = A_prev.shape
        self.original_shape = A_prev.shape
        N = self.original_shape[0]
        # Reshape to (N, -1), where -1 infers the product of remaining dimensions
        A_flat = A_prev.reshape(N, -1)
        return A_flat

    def backward(self, dA_flat):
        """dA_flat has shape (N, flattened_dim)."""
        # Reshape back to the original multi-dimensional shape
        dA_prev = dA_flat.reshape(self.original_shape)
        return dA_prev

# --- Fully Connected Layer ---

class Dense(Layer):
    """
    Fully connected (Dense) layer.
    Input shape: (N, D_in)
    Output shape: (N, D_out)
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.D_in = input_dim
        self.D_out = output_dim

        # He initialization for weights
        scale = np.sqrt(2.0 / self.D_in)
        self.params['W'] = np.random.randn(self.D_in, self.D_out) * scale
        self.params['b'] = np.zeros(self.D_out) # Bias per output neuron
        self.cache = {}

    def forward(self, A_prev):
        """A_prev shape: (N, D_in)"""
        self.input_shape = A_prev.shape
        W = self.params['W']
        b = self.params['b']

        Z = np.dot(A_prev, W) + b # (N, D_in) @ (D_in, D_out) -> (N, D_out)
        self.cache['A_prev'] = A_prev
        self.cache['W'] = W
        return Z

    def backward(self, dZ):
        """dZ shape: (N, D_out)"""
        A_prev = self.cache['A_prev'] # Shape (N, D_in)
        W = self.cache['W']         # Shape (D_in, D_out)
        N = A_prev.shape[0]

        # Gradient of loss w.r.t. weights W
        self.grads['W'] = np.dot(A_prev.T, dZ) # (D_in, N) @ (N, D_out) -> (D_in, D_out)
        # Gradient of loss w.r.t. biases b
        self.grads['b'] = np.sum(dZ, axis=0)   # Sum over batch, shape (D_out,)
        # Gradient of loss w.r.t. previous activation A_prev
        dA_prev = np.dot(dZ, W.T) # (N, D_out) @ (D_out, D_in) -> (N, D_in)

        return dA_prev

# --- Output Layer / Activation ---
class Softmax: # Not a Layer subclass as it's often combined with loss
    """Softmax activation, usually the last step before loss calculation."""
    def __init__(self):
        self.cache = {}

    def forward(self, Z):
        """Z shape: (N, num_classes)"""
        # Shift Z for numerical stability (subtract max to prevent overflow in exp)
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        A = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        self.cache['A'] = A # Store probs for potential direct use in loss derivative
        return A

    # Backward pass for Softmax is typically integrated with Cross-Entropy loss.
    # If dL/dA is given, dL/dZ can be complex.
    # However, if dL/dZ_softmax_output = (Probs - Y_one_hot), that's simpler.

# --- Sequential Model ---

class Sequential:
    """
    A container for a linear stack of layers.
    """
    def __init__(self, layers=None):
        self.layers = layers if layers is not None else []
        self.softmax_activation = Softmax() # For the final output probabilities

    def add(self, layer):
        """Adds a layer to the model."""
        if not self.layers and not isinstance(layer, (Conv2D, Conv1D, Dense, Flatten)):
             # First layer needs to know input_dim to infer its own params if not Flatten
            print(f"Warning: First layer added is {type(layer)}. Ensure its input_dim is correctly inferred or set.")
        elif self.layers and hasattr(self.layers[-1], 'output_shape_for_next_layer'):
            # Future: Layers could explicitly pass output_shape to next layer for auto-config
            pass
        self.layers.append(layer)

    def forward(self, X):
        """
        Performs a full forward pass through all layers.
        Returns final probabilities and a list of caches from each layer.
        """
        A = X
        self.layer_caches = [] # Store intermediate activations/values needed for backprop by layers
                               # This is different from `layer.cache` which is internal to the layer
        for layer in self.layers:
            A = layer.forward(A)
            self.layer_caches.append(layer.cache) # Store internal cache of the layer
        
        # Apply Softmax to the output of the last Dense layer
        probs = self.softmax_activation.forward(A)
        return probs

    def backward(self, dProbs):
        """
        Performs a full backward pass through all layers.
        dProbs is the gradient of the loss with respect to the softmax output probabilities.
        Typically, for Cross-Entropy loss and Softmax, this is (Probs - Y_one_hot).
        Note: The dProbs here is actually dZ_softmax (gradient before softmax).
        """
        dA_prev = dProbs # This is dL/dZ_before_softmax
        
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            # layer.cache should have been populated during forward pass and stored by the layer itself
            dA_prev = layer.backward(dA_prev)
        return dA_prev # Gradient with respect to the input X of the model

    def update_params(self, learning_rate, optimizer_type='adagrad', adagrad_epsilon=1e-8):
        """Updates parameters for all learnable layers."""
        for layer in self.layers:
            if hasattr(layer, 'params') and layer.params: # Check if layer has parameters
                layer.update_params(learning_rate, optimizer_type, adagrad_epsilon)
    
    def set_training_mode(self, mode: bool):
        """Sets training mode for all layers."""
        for layer in self.layers:
            layer.set_training_mode(mode)

# --- Loss Function ---
def compute_cross_entropy_loss(probs, y_one_hot):
    """
    Computes the Cross-Entropy loss.
    probs: (N, num_classes) - output probabilities from Softmax
    y_one_hot: (N, num_classes) - one-hot encoded true labels
    """
    N = probs.shape[0]
    epsilon = 1e-9 # To prevent log(0)
    loss = - (1/N) * np.sum(y_one_hot * np.log(probs + epsilon))
    return loss

# --- Main Test (can be removed or moved to a test file) ---
if __name__ == '__main__':
    print("\n--- Basic Test of Modular CNN Components ---")

    BATCH_SIZE = 2
    INPUT_CHANNELS = 1
    INPUT_HEIGHT = 8 # Smaller for quick test
    INPUT_WIDTH = 8
    NUM_CLASSES = 3

    dummy_X = np.random.rand(BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    dummy_y_indices = np.random.randint(0, NUM_CLASSES, size=BATCH_SIZE)
    dummy_y_one_hot = np.eye(NUM_CLASSES)[dummy_y_indices]

    print(f"Dummy X shape: {dummy_X.shape}")
    print(f"Dummy y_one_hot shape: {dummy_y_one_hot.shape}")

    # --- Define a simple model ---
    model = Sequential()
    # Conv1: 1 input channel, 4 output filters, kernel 3x3, stride 1, 'valid' padding
    # Output: (8-3)/1 + 1 = 6x6. So (N, 4, 6, 6)
    model.add(Conv2D(in_channels=INPUT_CHANNELS, out_channels=4, kernel_size=3, stride=1, padding_type='valid'))
    model.add(ReLU())
    # Pool1: 2x2 pool, stride 2
    # Output: (6-2)/2 + 1 = 3x3. So (N, 4, 3, 3)
    model.add(MaxPool2D(pool_size=2, stride=2))
    model.add(Flatten()) # Output: N x (4*3*3) = N x 36
    # Dense: 36 input features, NUM_CLASSES output features
    model.add(Dense(input_dim=4*3*3, output_dim=NUM_CLASSES))
    # Softmax is applied implicitly during forward for probs, and its gradient is part of dProbs

    print("\nModel Architecture:")
    for i, layer in enumerate(model.layers):
        print(f"  Layer {i+1}: {type(layer).__name__}")
        if hasattr(layer, 'params') and layer.params:
            for p_name, p_val in layer.params.items():
                print(f"    {p_name} shape: {p_val.shape}")

    # --- Test Single Training Step ---
    print("\n--- Testing Single Train Step ---")
    try:
        # 1. Forward pass
        probs = model.forward(dummy_X)
        print(f"Probs shape: {probs.shape}, expected ({BATCH_SIZE}, {NUM_CLASSES})")

        # 2. Compute loss
        loss = compute_cross_entropy_loss(probs, dummy_y_one_hot)
        print(f"Loss after one forward pass: {loss:.4f}")

        # 3. Backward pass
        # Gradient of Cross-Entropy Loss w.r.t. Z_before_softmax is (Probs - Y_one_hot)
        # We scale by 1/N often, but let's assume it's handled if loss is averaged.
        # Or, for simplicity, if the loss is sum, dL/dZ = P-Y. If loss is avg, dL/dZ = (P-Y)/N
        dZ_softmax = (probs - dummy_y_one_hot) / BATCH_SIZE # Gradient w.r.t output of last Dense layer
        
        model.backward(dZ_softmax) # This dZ_softmax is dL/d(Output of Dense)

        # Check if gradients were computed for learnable layers
        print("Gradients computed (showing norms):")
        for layer in model.layers:
            if hasattr(layer, 'grads') and layer.grads:
                for name, grad_array in layer.grads.items():
                    if grad_array is not None:
                        print(f"  Layer {type(layer).__name__} ||grad_{name}||: {np.linalg.norm(grad_array):.4e}")
                    else:
                        print(f"  Layer {type(layer).__name__} grad_{name} is None")
        
        # 4. Update parameters
        print("Updating parameters...")
        initial_conv_w_mean = np.mean(model.layers[0].params['W'])
        initial_dense_w_mean = np.mean(model.layers[-1].params['W'])
        
        model.update_params(learning_rate=0.01)
        
        final_conv_w_mean = np.mean(model.layers[0].params['W'])
        final_dense_w_mean = np.mean(model.layers[-1].params['W'])

        print(f"Mean of Conv W before update: {initial_conv_w_mean:.6f}")
        print(f"Mean of Conv W after update:  {final_conv_w_mean:.6f}")
        if not np.isclose(initial_conv_w_mean, final_conv_w_mean):
             print("  Conv parameters were updated.")
        else:
             print("  WARN: Conv parameters were NOT updated.")

        print(f"Mean of Dense W before update: {initial_dense_w_mean:.6f}")
        print(f"Mean of Dense W after update:  {final_dense_w_mean:.6f}")
        if not np.isclose(initial_dense_w_mean, final_dense_w_mean):
             print("  Dense parameters were updated.")
        else:
             print("  WARN: Dense parameters were NOT updated.")

        print("\nFull Class Test Completed Successfully (basic functionality).")

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
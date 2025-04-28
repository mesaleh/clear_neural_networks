# Clear Multi-Layer Perceptron (MLP) Implementation

This directory contains the `clear_` implementation of a foundational Feedforward Neural Network, also known as a Multi-Layer Perceptron (MLP).

Our goal with this implementation is to provide a highly readable and commented codebase that demonstrates the core mechanics of an MLP using basic Python and NumPy.

## What is Included?

The code here allows you to build, train, and evaluate a simple dense neural network. Key features include:

*   Vectorized forward and backward passes using NumPy.
*   Support for various activation functions (ReLU, Tanh, Sigmoid, Softmax, Linear).
*   Common weight initialization strategies (Xavier/Glorot, simple random).
*   Support for bias terms.
*   Standard loss functions (MSE, Binary Cross-Entropy, Categorical Cross-Entropy).
*   L2 regularization.
*   A basic training loop with mini-batch gradient descent.
*   Methods for prediction, evaluation, and saving/loading weights.

## File Overview

*   [`network.py`](./network.py): Defines the `Network` class, which orchestrates layers, manages the training loop (`fit`), prediction (`predict`), evaluation (`evaluate`), loss calculation, and overall forward/backward passes.
*   [`layer.py`](./layer.py): Defines the `Layer` class. This is where the vectorized operations for a single dense layer's forward and backward pass occur. It conceptually represents a group of neurons but performs calculations efficiently using matrix operations. Handles weight/bias storage and gradient accumulation for its parameters.
*   [`activations.py`](./activations.py): Contains implementations of various activation functions (ReLU, Tanh, Sigmoid, etc.) and their derivatives, used within the `Layer` class.
*   `logs/`: A directory (created during execution) for log files generated during training.
*   `README.md`: This file.

*(Note: Utility functions like logging setup and plotting are located in the top-level `clear_utils/` directory)*

## How to Use

A comprehensive example demonstrating how to load data, create a network, train it, plot the training history and decision boundary, and save/load weights is provided in the `examples/` directory at the root of the repository.

1.  Make sure you have cloned the main repository and installed dependencies (see the [main README](../README.md)).
2.  Navigate to the `examples/` directory:
    ```bash
    cd ../examples/
    ```
3.  Run the basic usage script:
    ```bash
    python basic_usage.py
    ```
    This script will run training examples on datasets like XOR and Make Moons, print progress and evaluation metrics, and display plots.

Feel free to modify the `basic_usage.py` script to experiment with different network architectures, activations, learning rates, epochs, and regularization strengths.

## Code for Learning

This code is designed to be read! Dive into the `.py` files. The comments aim to explain:

*   The purpose of classes and methods.
*   The shape of data and matrices at each step.
*   The mathematical operations being performed.
*   The flow of data and gradients during forward and backward passes.

I hope this provides a clear window into the inner workings of a neural network!
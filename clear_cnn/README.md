# Clear Convolutional Neural Network (CNN)

Welcome to the `clear_cnn` section of the `clear_neural_networks` repository! This part of the project provides a from-scratch implementation of a Convolutional Neural Network (CNN), built primarily using NumPy, to demystify its inner workings for educational purposes.

We explore the core components of CNNs, including convolutional layers (both 1D and 2D), activation functions (ReLU), pooling layers (Max Pooling for 1D and 2D), flattening, and dense (fully connected) layers, all tied together in a sequential model structure.

## Purpose

The goal here is to understand:

*   **Convolution Operations:** How filters extract features from input data (images or sequences).
*   **Pooling:** The concept of downsampling and feature aggregation.
*   **Hierarchical Feature Learning:** How stacked convolutional and pooling layers learn increasingly complex patterns.
*   **Spatial vs. Sequential Data:** Demonstrating CNNs on both 2D image data (digits classification) and 1D sequence data (URL classification).
*   **Backpropagation through CNN Layers:** The specific gradient calculations required for convolutional and pooling layers.
*   **Modular Design:** How to structure a neural network using reusable layer components.

## Implementation Details

*   **`model.py`:** Contains the core, modular implementation:
    *   `Layer`: An abstract base class for all network layers.
    *   `Conv2D`, `Conv1D`: Implementations for 2D and 1D convolutions, supporting 'valid' and 'same' padding.
    *   `ReLU`: Rectified Linear Unit activation.
    *   `MaxPool2D`, `MaxPool1D`: Max pooling layers for 2D and 1D data.
    *   `Flatten`: Reshapes multi-dimensional input into a 1D vector.
    *   `Dense`: Standard fully connected layer.
    *   `Softmax`: Output activation for classification (used by the `Sequential` model).
    *   `Sequential`: A container to build models by stacking layers linearly. It handles the forward pass, backward pass, and parameter updates for the entire stack.
    *   `compute_cross_entropy_loss`: A standalone function for loss calculation.
    *   Optimizer: Adagrad is implemented within each layer's `update_params` method.
*   **Educational Focus:** The code is heavily commented, especially the `forward` and `backward` methods of each layer, to explain the operations and the flow of data and gradients.

## Examples

This directory includes two primary examples to demonstrate the CNN's capabilities on different types of tasks:

### 1. Handwritten Digits Classification (2D CNN)

*   **File:** `example_digits_classification.py`
*   **Task:** Classifying 8x8 grayscale images of handwritten digits (0-9) from the Scikit-learn `digits` dataset.
*   **Demonstrates:**
    *   Using `Conv2D` and `MaxPool2D` layers for image feature extraction.
    *   A typical CNN architecture for image classification.
    *   Data preprocessing for image inputs.
*   **To Run:**
    ```bash
    cd /path/to/your/clear_neural_networks/clear_cnn
    python example_digits_classification.py
    ```
    This will load the data, train the 2D CNN, and plot the training loss and test accuracy over epochs. You'll need `numpy`, `scikit-learn`, and `matplotlib` installed.

### 2. URL Phishing Classification (1D CNN)

*   **File:** `example_url_classification.py`
*   **Task:** Classifying URLs as either legitimate or phishing, based on their character sequences.
*   **Dataset:** Requires a CSV file named `url_phishing_legit.csv` in the `clear_cnn` directory with 'url' and 'status' (0 or 1) columns. The dataset was obtained from here: https://www.kaggle.com/datasets/harisudhan411/phishing-and-legitimate-urls/data
*   **Demonstrates:**
    *   Using `Conv1D` and `MaxPool1D` layers for sequence feature extraction.
    *   One-hot encoding of characters as input to the 1D CNN.
    *   Applying CNNs to non-image, sequential data.
*   **To Run:**
    1.  Ensure you have `url_phishing_legit.csv` in the `clear_cnn` directory.
    2.  Install `pandas` if you haven't already (`pip install pandas`).
    ```bash
    cd /path/to/your/clear_neural_networks/clear_cnn
    python example_url_classification.py
    ```
    This will load the URL data, preprocess it, train the 1D CNN, and plot its performance.

## Key Learnings from this Module

*   The mechanics of convolution and how filter weights and biases are learned.
*   The role of pooling in reducing dimensionality and creating invariance.
*   How to adapt CNN architectures for 1D sequential data versus 2D spatial data.
*   The flow of gradients backward through these specialized layers.
*   The importance of input data representation (e.g., one-hot encoding for character sequences).

## Contributing

This project is primarily for educationl purpose and to serve as an educational resource. However, if you spot bugs, have suggestions for clarity, or want to discuss the implementations, feel free to open an issue!

---
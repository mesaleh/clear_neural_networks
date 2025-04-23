# Clear Neural Network Implementations

Ilcome to the `clear_neural_nets` repository!

This project is an educational resource designed to help you truly understand how foundational neural network architectures work under the hood. Instead of relying on high-level frameworks that abstract away the details, I build these models using clear, readable Python code, primarily leveraging **NumPy** for numerical operations where appropriate.
I might also use `PyTorch` only for basic objects such as `tensor` or modules such as `nn.Linear` in advanced architectures like Transformers since I would have already covered how those are implemented in `clear_mlp`.

My goal is to provide implementations that are easy to follow, heavily commented, and directly illustrate the mathematical operations involved in forward passes, backpropagation, and optimization.

This repository will grow over time to include various fundamental architectures.

## Why "Clear"?

The name "Clear" emphasizes:

1.  **Code Readability:** I prioritize straightforward code over optimization tricks (initially).
2.  **Conceptual Transparency:** Each part of the network is implemented explicitly to show the underlying mechanics.
3.  **Educational Focus:** The code is richly commented to explain the *why* behind the implementation.

## Features of the Series

*   Implementations of various popular neural network architectures.
*   Code written in standard Python, using minimal external libraries (primarily NumPy).
*   Focus on the core algorithms: forward pass, backward pass (gradient calculation), and parameter updates.
*   Heavily commented code to serve as an educational guide.
*   Basic examples demonstrating how to use each implementation.

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/clear_neural_nets.git
    cd clear_neural_nets
    ```
2.  **Install dependencies:**
    I aim for minimal dependencies, but you will need `numpy`, `matplotlib` (for plotting), `scikit-learn` (for datasets), and maybe `PyTorch` in the future, not to use it for full implementation but for just getting basic objects out of the way.
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Create a `requirements.txt` file in the root directory containing `numpy`, `matplotlib`, `PyTorch`)*

## Project Structure

This repository is organized as a monorepo, with each neural network architecture residing in its own subdirectory:
```
.
├── README.md <-- You are here (Overall project description)
├── requirements.txt <-- Project dependencies
├── clear_mlp/ <-- Files for the Multi-Layer Perceptron implementation
│ └── basic_usage.py 
├── clear_rnn/ <-- (Future) Files for a Recurrent Neural Network
├── clear_transformer/ <-- (Future) Files for a Transformer
```


## Available Implementations

*   **Multi-Layer Perceptron (MLP):** A fundamental feedforward neural network.
    *   Navigate to [`./clear_mlp/`](./clean_mlp/) to find the code and its dedicated README.md for details on this implementation.
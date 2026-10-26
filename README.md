# 🧠 Neural Network From Scratch

### Building a Neural Network and Automatic Differentiation Engine in Pure Python

A neural network implementation built from scratch in Python, without using deep learning frameworks such as PyTorch or TensorFlow.

This project explores the mathematical foundations of deep learning, including **automatic differentiation, computational graphs, backpropagation, neurons, layers, and Multi-Layer Perceptrons (MLPs)**.

> Inspired by Andrej Karpathy's [micrograd tutorial](https://www.youtube.com/watch?v=VMj-3S1tku0).

---

## 🚀 Features

- **Automatic Differentiation (Autograd)** — Custom scalar-based differentiation engine.
- **Computational Graphs** — Track mathematical operations and their dependencies.
- **Backpropagation** — Calculate gradients using the chain rule.
- **Activation Functions** — Implementations of `tanh` and `exp`.
- **Custom Neurons** — Weighted inputs, biases, and activation functions.
- **Neural Network Layers** — Combine multiple neurons into layers.
- **Multi-Layer Perceptron (MLP)** — Connect multiple layers to build a neural network.
- **Gradient Computation** — Calculate gradients with respect to trainable parameters.

---

## 🏗️ Project Architecture

```text
Neural Network From Scratch
│
├── Autograd Engine (Nombre)
│   ├── Scalar operations
│   ├── Computational graph
│   ├── Activation functions
│   └── Backpropagation
│
├── Neuron
│   ├── Weights
│   ├── Bias
│   └── tanh activation
│
├── Layer
│   └── Multiple neurons
│
└── MLP
    ├── Multiple layers
    ├── Forward propagation
    └── Parameter collection
```

### Example Network Architecture

```text
Input Layer        Hidden Layer       Output Layer

    x1 ───────────── h1 ──────────┐
       ╲          ╱               │
        ╲        ╱                │
         ╲      ╱                 ▼
          ╲    h2 ─────────────── y
          ╱    ╱                  ▲
         ╱    ╱                   │
    x2 ───────────── h3 ──────────┘

   2 inputs         3 neurons       1 output
```

---

## 🔬 1. Automatic Differentiation

The core of this project is a custom class called `Nombre`, which represents a scalar value and tracks its gradient.

Each mathematical operation creates a computational graph that enables automatic differentiation.

```python
a = Nombre(2)
b = Nombre(3)

c = a * b
d = c + a

d._retropropager()

print(a.b_val)  # 4.0
print(b.b_val)  # 2.0
```

The gradients are calculated using the chain rule:

$$
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial y}
\cdot
\frac{\partial y}{\partial x}
$$

---

## 🧩 2. Neuron Implementation

A neuron computes a weighted sum of its inputs, adds a bias, and applies an activation function.

$$
z = \sum_{i=1}^{n} w_i x_i + b
$$

The activation function is:

$$
a = \tanh(z)
$$

### Example

```python
neurone = Neuron(
    poids=[0.5, -1.0],
    biais=0.2
)

y = neurone.propager([2.0, 3.0])

print("Output:", y.h_val)

y._retropropager()

print("Weight gradients:", [p.b_val for p in neurone.poids])
print("Bias gradient:", neurone.biais.b_val)
```

---

## 🧠 3. Multi-Layer Perceptron (MLP)

Multiple neurons are grouped into layers, and multiple layers form a Multi-Layer Perceptron.

### Example

```python
reseau = MLP([
    couche([
        Neuron([0.5, -0.2], 0.1),
        Neuron([0.3, 0.7], -0.2),
        Neuron([-0.4, 0.6], 0.0)
    ]),
    couche([
        Neuron([0.2, -0.5, 0.8], 0.1)
    ])
])

sortie = reseau.propager([2.0, 1.0])[0]

print("Prediction:", sortie.h_val)

sortie._retropropager()

print("Gradients:", [p.b_val for p in reseau.parameters()])
```

The network supports forward propagation and automatic gradient computation through multiple layers.

---

## ⚙️ Technologies

- Python 3
- Python `math` library
- Object-Oriented Programming (OOP)
- Automatic Differentiation
- Backpropagation
- Neural Network Mathematics

**No PyTorch or TensorFlow required.**

---

## 📚 Key Concepts

| Concept | Description |
|---|---|
| Computational Graph | Tracks mathematical operations and dependencies |
| Automatic Differentiation | Automatically computes derivatives |
| Chain Rule | Propagates derivatives through operations |
| Backpropagation | Calculates gradients from outputs to inputs |
| Topological Sorting | Determines the correct order of gradient computation |
| Activation Functions | Introduces nonlinearity into neural networks |
| MLP | Combines multiple layers of neurons |

---

## 🛣️ Future Improvements

- [ ] Implement a complete training loop
- [ ] Add Mean Squared Error (MSE) loss
- [ ] Implement gradient descent
- [ ] Visualize training loss
- [ ] Compare gradients against PyTorch
- [ ] Train the network on a classification dataset

---

## 🎯 Learning Objectives

The main objective of this project is to understand how neural networks work internally instead of treating deep learning frameworks as black boxes.

By implementing everything from scratch, this project provides a deeper understanding of:

- Mathematical operations and derivatives
- Computational graphs
- Automatic differentiation
- Backpropagation
- Neural network architectures
- The mathematical foundations of deep learning

---

## 🙏 Acknowledgments

This project was inspired by **Andrej Karpathy's micrograd**, an educational implementation of automatic differentiation and neural networks.

- 🎥 [Micrograd Tutorial — Andrej Karpathy](https://www.youtube.com/watch?v=VMj-3S1tku0)
- 💻 [Original Micrograd GitHub Repository](https://github.com/karpathy/micrograd)

---

⭐ If you find this project useful, consider giving it a star!

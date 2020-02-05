# XABY: Functional Machine Learning

I've been wanting to experiment with functional programming in Python, specifically for machine learning and neural networks. Neural networks are, in general, collections of sequences of operations. I thought it'd be interesting to see if I could construct network architectures out of only function calls for these operations.

I can't be purely functional here in the sense that pure functions have exactly the same output for the same input. Models are updated while training, so even though the model is a single function, the output will change given a static input. Technically you could return a new model for each update, but this would eat up a lot of memory.

I also had some fun messing with Python's operators. One problem with chaining functions is the first function you read is the last function that is called. For example, consider a network with one hidden layer trained on MNIST. If each operation is a function, then you'd calculate the digit probabilities like this:

```python
log_softmax(linear(256, 10)(relu(linear(784, 256)(flatten(inputs)))))
```

Instead, XABY uses the `>>` operator to call functions in succession:
```python
inputs >> flatten >> linear(784, 256) >> relu >> linear(256, 10) >> log_softmax
```

This way the code reads in the same order as execution.

I've also been wanting to explore [JAX](https://github.com/google/jax), a new library that provides GPU/TPU acceleration, automatic differentiation, and function vectorization. Each XABY operation is a function compiled with JAX. The sequence of operations that make a model are also compiled with JAX. So the model is one big compiled function that runs on the GPU automatically. It's pretty cool.

Creating a model is just chaining together operations. To update a model, you create a backpropagation function and a model update function (with stochastic gradient descent for example).

```python
model = flatten >> linear(784, 256) >> relu \
                >> linear(256, 10) >> log_softmax

# Create functions
backprop = model << nlloss
update = sgd(model, lr=0.003)
...
# In the training loop
loss, grads = backprop(inputs, targets)
update(grads)
```

Inference is as simple as running a tensor through a model. This won't calculate gradients so it's faster!

```python
class_probabilities = inputs >> model >> exp
# To calculate the loss without backpropagtion
loss = inputs >> model >> nlloss << targets
```

## Notice!

This is all experimental and rough. The API is likely to change. And things you want to do probably won't work. I don't know if XABY will be feasible in the long run. I haven't implemented convolutional operations. I haven't tried to implement branching networks, things like skip connections in ResNets. I'm going to try though!

I also don't have any documentation yet. Or tests.

## Dependencies
- Python 3.6+
- [JAX](https://github.com/google/jax)
- For the example notebooks:
    - Jupyter Notebook
    - PyTorch and Torchvision 

## Installation

Clone the repo and install with pip

```bash
git clone https://github.com/mcleonard/xaby.git
cd xaby
pip install -e .
```

## Demonstration!

For a demonstration of using XABY to train on MNIST, check out  `examples/MNIST.ipynb`. Just a fully-connected network, no convolutions yet. I also compare XABY with PyTorch. XABY appears to be a little slower sometimes, but overall it's comparable to PyTorch on GPUs.



# XABY: Functional Machine Learning

I've been wanting to experiment with functional programming in Python, specifically for machine learning and neural networks. Neural networks are, in general, collections of sequences of operations. I thought it'd be interesting to see if I could construct network architectures out of only function calls for these operations.

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

Creating a model is just chaining together operations. To update a model, you create a loss function and a model update function (with stochastic gradient descent for example). The loss function can return gradients which you use to update the model.

```python
import xaby as xb
import xaby.nn as xn

### Compose a model, in two lines for readability
model = xb.flatten(axis=0) >> xn.linear(784, 256) >> xn.relu \
                           >> xn.linear(256, 10) >> xn.log_softmax(axis=0)

# Compose a loss function
loss = xb.split(model, xb.skip) >> xn.losses.nll_loss()

# Update function
update = xb.optim.sgd(lr=0.003)

...
# In the training loop
# Wrap up our input data
for images, labels in data_loader:
    inputs = xb.pack(images, labels)
    
    # Get the gradients
    train_loss, grads = loss << inputs
    
    # Then, update the function with the gradients
    update(loss, grads)
```

## Notice!

This is all experimental and rough. The API is likely to change. And things you want to do probably won't work. I don't know if XABY will be feasible in the long run.

I also don't have any documentation yet. Or tests. These are coming soon.

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

## Demonstrations!

For demonstrations of using XABY check out the notebooks in the `examples` directory.

- `XABY Features.ipynb` is a good starting point to learn about XABY
- `MNIST.ipynb` shows you how to train a fully-connected network to classify handwritten digits from images.
- `MNIST-CNN.ipynb` shows you how to train a convolutional network to classify handwritten digits from images.




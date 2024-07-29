# Learning by doing

Heavily inspired by [Andrej Karpathy's](https://karpathy.ai/) philosophy of learning by actually implementing an algorithm from scratch. This is an assortment of algorithms/methods I wanted to understand and visualize. 

All the implemented algorithms / learning methods have production grade (and significantly better) implementations. The objective here is to solidify my understanding of these methods.


## What's in this repository?
### fundamentals
Contains the implementation of linear and logistic regression. To train the model, I used stochastic gradient descent.
[Linear Regression](https://github.com/varun-suresh/ml-practice/fundamentals/linearRegression.py)
[Logistic Regression](https://github.com/varun-suresh/ml-practice/fundamentals/logisticRegression.py)

### Autograd
Almost identical to [Andrej Karpathy's micrograd implementation](https://github.com/karpathy/micrograd). Added Conv1D implementation (It is slow!) but verified that the calculated gradients are identical to using PyTorch. (TODO: Add a notebook to verify my conv1D implementation and PyTorch's implementation give identical results)

### Transformers
Implemented the attention module (almost identical to Karpathy's implementation in [minGPT](https://github.com/karpathy/minGPT)). To build this, I used PyTorch.


## Local Development Setup
Install pyenv - Follow instructions [here](https://github.com/pyenv/pyenv).

Create a virtual environment with python 3.10+
```
pyenv virtualenv ml_practice
```
Activate the virtual environment using
```
pyenv activate ml_practice
```
Go to the root of where you have this repository cloned. Run 
```
pip install -r requirements.txt 
```
This installs all the packages required to run scripts in this repo. To reference paths, set
```
export PYTHONPATH="$PYTHONPATH:/directory/where/this/repo/is/cloned"
```
# Implementation of ML Algorithms from Scratch

Hand-coded implementations of classic machine-learning algorithms — no ML frameworks, just NumPy — including KNN, Gaussian Naive Bayes, a multi-layer perceptron with backpropagation, RBF networks, and bagging random forests.

## Overview

This repository demonstrates a deep understanding of how machine-learning algorithms actually work by implementing them from first principles instead of calling `sklearn` directly. Each folder contains a self-contained Jupyter notebook (plus supporting scripts where useful) that builds an algorithm from the math up and evaluates it on a standard dataset such as Iris or Salary Data.

## Features

- **K-Nearest Neighbors** (`implementation_KNN`): distance-based classification with configurable K and distance metric
- **Gaussian Naive Bayes** (`implementation_GaussianNaiveBayes`): Bayes classifier with per-class Gaussian likelihoods
- **Multi-Layer Perceptron** (`implementationDynamicMLP`): flexible MLP with configurable layer sizes and activation functions (sigmoid, tanh, ReLU, linear), trained by backpropagation — includes reusable `MLP.py` module and Iris/Salary demos
- **RBF Networks** (`implementationRBF`): radial-basis-function classifier and regressor trained with gradient descent
- **Bagging Random Forest** (`bagging_randomForest`): decision trees with impurity-based splitting + bagging ensemble voting
- Every experiment reports accuracy with a train/test split

## Tech Stack

- Python 3
- NumPy, Pandas, SciPy
- scikit-learn (used only for datasets and evaluation metrics, not for the algorithms themselves)
- Matplotlib, Seaborn (visualization)
- Jupyter Notebook

## Installation

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn jupyter
```

## Usage

```bash
jupyter notebook
```

Open any notebook and run all cells. For the dynamic MLP, a reusable module is also provided:

```bash
cd implementationDynamicMLP
python iris.py     # train a 4-5-3 MLP on the Iris dataset
python salary.py   # train a 2-layer MLP for salary regression
```

### MLP configuration

The MLP is configured via a `PARAMS` dict:

```python
PARAMS = {
    'LEARNING_RATE': 0.01,
    'CODE_OF_ACTIVATION_FUNCTIONS': [4, 1, 3],  # 1=sigmoid 2=tanh 3=relu 4=linear
    'NUMBER_OF_PERCEPTRONS_FOR_HIDDEN_LAYERS': [5],
}
model = MLP(X_train, y_train_onehot, PARAMS)
model.train(epochs=100)
model.predict_row(sample)
```

## Project Structure

```
bagging_randomForest/            # decision tree + bagging ensemble on Iris
implementation_KNN/              # KNN classifier on Iris
implementation_GaussianNaiveBayes/  # Gaussian NB on Iris
implementationDynamicMLP/        # MLP from scratch (module + Iris/Salary demos)
    MLP.py                       # the MLP / Layer / Perceptron implementation
    iris.py, salary.py           # runnable demos
implementationRBF/               # RBF network classifier + regressor (Q4/Q5 data)
```

## Development

The code is intentionally dependency-light. To contribute, extend an existing implementation (e.g., add softmax cross-entropy to the MLP) and add a comparison notebook against the `sklearn` equivalent.

## Roadmap

- Add softmax output layer + cross-entropy loss to the MLP
- Add a from-scratch linear regression and SVM
- Compare each implementation against its `sklearn` counterpart in a single summary notebook

## License

MIT

## Author

Seyed Farhad Hosseini

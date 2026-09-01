# Dynamic Multi-Layer Perceptron

> A flexible multi-layer perceptron with backpropagation, implemented from scratch in NumPy.

A configurable neural network whose **architecture, activation functions, and learning rate** are set at construction time. Supports multi-class classification and regression, and is evaluated on **Iris** (classification) and **Salary** (regression) datasets.

## Highlights

- Dynamic architecture: any number of layers, any size per layer.
- Four activation functions per layer: **sigmoid, tanh, ReLU, linear**.
- Online gradient descent with the **delta rule** and full **backpropagation**.
- Learnable **bias** per neuron (constant-1 input branch).
- Pure NumPy — no deep-learning framework.
- Deterministic training with an optional random seed.

## How it works

1. **Architecture** — the input layer size equals the number of features, the output layer equals the number of labels, and hidden layers are sized from the configuration.
2. **Forward pass** — activations propagate layer by layer through the chosen activation functions.
3. **Backpropagation** — the output error is propagated backwards to compute a delta per neuron.
4. **Weight update** — each connection weight moves against the gradient scaled by the learning rate (delta rule).

### Configuration

```python
PARAMS = {
    "LEARNING_RATE": 0.01,
    "CODE_OF_ACTIVATION_FUNCTIONS": [4, 1, 3],  # linear -> sigmoid -> relu
    "NUMBER_OF_PERCEPTRONS_FOR_HIDDEN_LAYERS": [5],
}

model = MLP(X_train, y_train, PARAMS, seed=42)
model.train(epochs=100)
model.predict_row(sample)      # output vector
model.predict_class(sample)    # argmax class index
```

| Code | Activation |
|------|------------|
| 1    | Sigmoid    |
| 2    | Tanh       |
| 3    | ReLU       |
| 4    | Linear     |

## Project structure

```
implementationDynamicMLP/
├── implementationDynamicMLP.ipynb   # original notebook (reference, untouched)
├── README.md
├── requirements.txt
├── data/
│   └── Salary_Data.csv              # regression dataset
├── scripts/
│   ├── train_iris.py                # Iris classification demo
│   └── train_salary.py              # Salary regression demo
├── src/
│   └── models/
│       ├── mlp.py                   # MLP (network orchestration)
│       ├── layer.py                 # Layer
│       ├── perceptron.py            # Perceptron
│       ├── input_branch.py          # InputBranch (weighted connections)
│       └── activation.py            # ActivationFunction + derivatives
└── tests/
    └── test_mlp.py                  # pytest smoke tests
```

## Getting started

```bash
cd implementationDynamicMLP
pip install -r requirements.txt

# Multi-class classification on Iris
python scripts/train_iris.py

# Regression on Salary data
python scripts/train_salary.py

# Run the test suite
pytest
```

## Results

| Task          | Dataset | Epochs | Metric                    | Result  |
|---------------|---------|--------|---------------------------|---------|
| Classification| Iris    | 100    | Test accuracy             | 96.67 % |
| Regression    | Salary  | 100    | Train / test MSE          | 32.6M / 49.0M |

## Running the notebook

```bash
jupyter notebook implementationDynamicMLP.ipynb
```

---

**[← Back to repository home](../README.md)**

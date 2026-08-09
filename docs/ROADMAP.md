# Atlas Model Lab Roadmap

This repository progresses from the smallest trainable model to a GPT-style transformer. Each stage exists to make the next abstraction understandable.

## Stage 1: linear unit

Build and understand:

- one weight;
- one bias;
- mean squared error;
- analytical gradients;
- gradient descent;
- deterministic training.

Exit condition: the model learns a known linear relationship and the tests prove that loss decreases.

## Stage 2: multilayer network primitives

Implement with NumPy:

- dense layers;
- ReLU;
- numerically stable softmax;
- cross-entropy;
- manual backward passes;
- finite-difference gradient checking.

Exit condition: analytical gradients match numerical gradients within a declared tolerance.

## Stage 3: synthetic classification

Compose the primitives into a small classifier and intentionally overfit a tiny deterministic dataset.

Exit condition: the network can memorise a bounded dataset, making forward, backward, loss, and parameter updates observable.

## Stage 4: MNIST

Train a multilayer perceptron on MNIST.

Measure:

- training loss;
- validation loss;
- training accuracy;
- validation accuracy;
- elapsed time;
- seed and configuration.

Exit condition: the result is reproducible and the experiment record explains what changed between runs.

## Stage 5: optimisers

Implement and compare:

- stochastic gradient descent;
- momentum;
- Adam;
- AdamW where appropriate.

The optimiser state and update equations should remain visible rather than hidden behind an ML framework.

## Stage 6: controlled experiments

Compare a small evidence-driven matrix of:

- learning rates;
- batch sizes;
- hidden widths;
- network depth;
- activation behaviour;
- initialisation schemes;
- optimiser choices.

Every experiment should answer a specific question.

## Stage 7: tiny transformer

Move to PyTorch for tensor execution and automatic differentiation while implementing the transformer architecture directly.

Start with:

- token embeddings;
- positional embeddings;
- query, key, and value projections;
- causal self-attention;
- multi-head attention;
- feed-forward blocks;
- residual connections;
- normalisation;
- autoregressive generation.

## Stage 8: GPT-2-style architecture

Reproduce a GPT-2-style decoder architecture from primary technical references, then measure training and inference behaviour on local hardware.

Architecture reproduction, training-pipeline reproduction, and historical-result reproduction must remain separate claims.

# Image Captioning with RNN, LSTM, and Attention

A deep learning coursework project focused on implementing and comparing recurrent neural network architectures for **image caption generation** using PyTorch.

The project combines convolutional neural networks (CNNs) for visual feature extraction with recurrent language models to generate natural-language descriptions of images. It explores three decoder architectures: Vanilla RNN, LSTM, and Attention LSTM.

## Overview

Image captioning is a task that combines computer vision and natural language processing. Given an image, a model extracts visual features and generates a sequence of words describing its content.

This project implements the main components of an image captioning pipeline, including recurrent network operations, word embeddings, temporal loss computation, attention, and autoregressive caption generation.

## Key Features

* Manual implementation of Vanilla RNN forward and backward passes
* LSTM cell and sequence processing
* Attention LSTM with scaled dot-product attention
* CNN-based image feature extraction using a pretrained RegNet-X-400MF backbone
* Learnable word embeddings
* Temporal cross-entropy loss with padding-token masking
* Training-time loss computation and test-time caption generation
* Numerical checks for model outputs and gradients
* Training experiments on small subsets and larger training samples
* Attention-weight visualization for inspecting spatial image regions

## Model Architectures

### 1. Vanilla RNN

The Vanilla RNN processes the embedded caption sequence using recurrent hidden states. The image features are projected to initialize the decoder's hidden state.

The implementation covers:

* Single-step forward pass
* Single-step backward pass
* Sequence-level forward propagation
* Backpropagation through time (BPTT)
* Image-conditioned caption generation

### 2. LSTM

The LSTM decoder uses input, forget, and output gates, together with a cell state, to process caption sequences.

The implementation includes recurrent state updates, sequence processing, and autoregressive caption sampling.

### 3. Attention LSTM

The Attention LSTM uses spatial CNN features to help the decoder focus on different image regions while generating each word.

At each decoding step, scaled dot-product attention computes weights over the spatial feature map. The weighted visual representation is then incorporated into the LSTM update.

The notebook also includes attention-weight visualization to inspect which image regions receive higher attention during caption generation.

## Architecture

The overall pipeline is:

```text
Input Image
     |
     v
Pretrained RegNet-X-400MF
     |
     v
Visual Feature Extraction
     |
     v
Feature Projection
     |
     +-----------------------------+
     |                             |
     v                             v
 Vanilla RNN / LSTM          Attention LSTM
     |                             |
     |                      Spatial Attention
     |                             |
     +--------------+--------------+
                    |
                    v
             Hidden Representations
                    |
                    v
           Vocabulary Score Layer
                    |
                    v
          Next-Word Prediction
                    |
                    v
          Generated Image Caption
```

The Vanilla RNN and standard LSTM use a projected global image representation to initialize the decoder. The Attention LSTM instead uses projected spatial image features to calculate attention during decoding.

## Implementation Details

### Image Encoder

The project uses a pretrained RegNet-X-400MF model from Torchvision to extract visual features.

The classifier is bypassed to obtain convolutional feature maps for the captioning decoder. Image features are normalized using ImageNet mean and standard deviation.

### Word Embeddings

Each word is represented by a learnable embedding vector. The embedded caption is processed sequentially by the selected recurrent decoder.

### Training Objective

The model predicts the next word at each timestep. Training uses temporal cross-entropy loss, ignoring padding tokens so that they do not contribute to the loss.

### Caption Generation

During inference, the decoder starts with a start token and repeatedly predicts the next word based on the image representation and previously generated words.

For the Attention LSTM, the implementation also returns attention weights that can be used to visualize spatial attention.

## Experiments and Validation

The notebook contains several stages of experimentation:

* Numerical checks for recurrent forward and backward computations
* Gradient comparisons for the Vanilla RNN
* Validation of LSTM and attention operations against expected outputs
* Small-subset overfitting experiments
* Caption generation on training and validation samples
* Visualization of attention weights for generated captions

The notebook includes recorded training outputs for the implemented models. These experiments are useful for checking implementation correctness and observing learning behavior, but they should not be interpreted as evidence of strong generalization without a systematic evaluation.

## Technology Stack

* **Python**
* **PyTorch**
* **Torchvision**
* **NumPy**
* **Matplotlib**
* **Jupyter Notebook / Google Colab**

## Repository Contents

```text
.
├── rnn_lstm_captioning.py
├── rnn_lstm_captioning.ipynb
└── README.md
```

* `rnn_lstm_captioning.py` — model components and core implementations.
* `rnn_lstm_captioning.ipynb` — implementation walkthrough, numerical checks, training experiments, caption sampling, and attention visualization.
* `README.md` — project overview and documentation.

## Getting Started

### Prerequisites

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Jupyter Notebook or Google Colab

Install the core libraries in an environment compatible with your Python and CUDA setup:

```bash
pip install torch torchvision numpy matplotlib jupyter
```

### Running the Notebook

Open `rnn_lstm_captioning.ipynb` in Jupyter Notebook or Google Colab and execute the cells in order.

**Important:** The provided notebook references additional course utility modules, including `a5_helper.py` and `eecs598` utilities, as well as prepared image-captioning data and vocabulary mappings. These resources are not included in the two files provided here. The complete original environment and dataset setup are therefore required to reproduce the notebook's training and visualization experiments.

The pretrained RegNet weights may also need to be downloaded by Torchvision.

## Learning Outcomes

This project provided practical experience with:

* Recurrent neural network implementation
* Backpropagation through time
* LSTM gating mechanisms
* Attention mechanisms for visual feature selection
* CNN feature extraction
* Sequence modeling and word embeddings
* Autoregressive generation
* Numerical gradient checking
* Training and evaluating deep learning models with PyTorch

## Project Context

This repository documents an academic deep learning project focused on understanding and implementing image captioning models. It is intended for educational and experimentation purposes rather than as a novel image captioning method.

If the implementation is based on a course assignment or starter code, the original course and source materials should be credited here.

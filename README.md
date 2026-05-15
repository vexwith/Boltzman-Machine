# Boltzman-Machine
Model based on Hopfield network type memory. Trained on [MNIST database](https://en.wikipedia.org/wiki/MNIST_database)

Given a partial image of some digit it should generate a completed image of the closest number.

## Project Overview

This project implements a neural network model in theory capable of:
- **Pattern recognition** on handwritten digits (0-9)
- **Image completion** - given a partial or corrupted image, the model generates a completed image of the closest recognized digit
- **Energy minimization** using simulated annealing to find optimal configurations

The model is based on Hopfield networks and uses stochastic update rules with temperature-based probability adjustments.
Unfortunately it does not work as expected. When trained on a small sample it always converges to 1 max 2 different digits and when trained on a larger sample it converges to a weird amalgamation of all digits simultaneously.

## Features

- GPU support (CUDA) with automatic CPU fallback
- MNIST dataset integration for training
- Energy-based learning and inference
- Interactive visualization during generation

## Installation

### Prerequisites
- Python 3.7+
- CUDA-capable GPU (optional, CPU will work but is slower)

### Setup

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from main import Network

# Initialize network with 784 neurons (28x28 pixel image)
nn = Network(784)

# Train on MNIST images
nn.embed_img(num_images=10, num_labels=1)

# Load and embed a test image
nn.embed_test('test_2.png')

# Display the embedded image
nn.decode_img()

# Generate/complete the image using energy minimization
nn.generate(find_minimum=True)

# Display the result
nn.decode_img()
```

## Main Components

### Network Class

#### Key Methods

| Method | Description |
|--------|-------------|
| `embed_img()` | Embed MNIST training images into the network weights |
| `embed_test(png_path)` | Load and embed a test image (must be 28x28 grayscale) |
| `generate()` | Run the generative process using simulated annealing |
| `decode_img()` | Display the current network state as an image |
| `save_model(filepath)` | Save trained weights to disk |
| `load_model(filepath)` | Load previously trained model |
| `train()` | Full training pipeline with multiple epochs |

#### Energy Functions

- `get_neuron_energy()` - Calculate energy contribution of a single neuron
- `get_system_energy()` - Calculate total system energy (higher = better pattern match)

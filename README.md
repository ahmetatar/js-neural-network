# JS Neural Network

A simple JavaScript neural network implementation from scratch, designed to learn logical gates (AND, OR, XOR) and recognize handwritten digits (MNIST dataset).

## Features

- **Single-layer Perceptron**: For simple linearly separable gates (AND, OR)
- **Multi-Layer Perceptron (MLP)**: For complex patterns like XOR and MNIST digit recognition
- **Custom Matrix Library**: Basic matrix operations for neural network computations
- **Pre-built Training Models**: Ready-to-use training configurations for different tasks
- **Web API**: Express server with training endpoint and static file serving

## Installation

```bash
npm install
```

## Quick Start

Run the default example (MNIST digit recognition):

```bash
npm start
```

Start the web API server:

```bash
npm run start:api
```

## Project Structure

```
src/
├── index.js                 # Entry point
├── sl-neural-network.js     # Single-layer perceptron
├── ml-neural-network.js     # Multi-layer perceptron
├── server.js                # Express API server
├── training/
│   ├── and-gate-model.js    # AND gate training
│   ├── xor-gate-model.js    # XOR gate training
│   └── mnist-model.js       # MNIST digit recognition training
└── utils/
    ├── matrix.js            # Matrix operations library
    └── training-metrics.js  # Training metrics tracking
models/
└── mnist-model.json         # Pre-trained MNIST model
public/
└── index.html               # Web interface
```

## How to Use the Models

### 1. AND Gate Model (Single-Layer Perceptron)

The AND gate is a simple linearly separable problem that can be solved with a single-layer perceptron.

```javascript
import { getTrainedModelForANDGateLogic } from "./training/and-gate-model.js";

// Get a trained AND gate model
const nn = getTrainedModelForANDGateLogic();

// Make predictions
console.log(nn.predict(0, 0)); // ~0 (expected: 0)
console.log(nn.predict(0, 1)); // ~0 (expected: 0)
console.log(nn.predict(1, 0)); // ~0 (expected: 0)
console.log(nn.predict(1, 1)); // ~1 (expected: 1)
```

### 2. XOR Gate Model (Multi-Layer Perceptron)

XOR is not linearly separable and requires a hidden layer to learn the pattern.

```javascript
import { getTrainedModelForXORGate } from "./training/xor-gate-model.js";

// Get a trained XOR gate model
const nn = getTrainedModelForXORGate();

// Make predictions (returns a 2D matrix)
console.log(nn.feedforward([0, 0])); // ~[[0]] (expected: 0)
console.log(nn.feedforward([0, 1])); // ~[[1]] (expected: 1)
console.log(nn.feedforward([1, 0])); // ~[[1]] (expected: 1)
console.log(nn.feedforward([1, 1])); // ~[[0]] (expected: 0)
```

### 3. MNIST Digit Recognition Model

Recognizes handwritten digits (0-9) using a 784→64→10 neural network architecture.

```javascript
import { getTrainedModelForMNIST } from "./training/mnist-model.js";

// Get trained model and test data
const { nn, testSet } = getTrainedModelForMNIST();

// Test with a sample from the test set
const output = nn.feedforward(testSet[0].input);

console.log("Expected Output:", testSet[0].output); // One-hot encoded label
console.log("Neural Network Output:", output);       // Network predictions

// Find the predicted digit (index of highest value)
const predicted = output.indexOf(Math.max(...output.flat()));
console.log("Predicted Digit:", predicted);
```

## Web API

The project includes an Express server for training models via HTTP.

### Start the server

```bash
npm run start:api
```

### Endpoints

- `GET /` - Serves the web interface from `public/index.html`
- `GET /train` - Trains the MNIST model and returns training metrics

### Example

```bash
curl http://localhost:3000/train
```

## Creating Custom Models

### Using SlNeuralNetwork (Simple Perceptron)

```javascript
import { SlNeuralNetwork } from "./sl-neural-network.js";

const nn = new SlNeuralNetwork();

// Train the network
for (let i = 0; i < 20000; i++) {
  nn.train(input1, input2, target);
}

// Make predictions
const result = nn.predict(input1, input2);
```

### Using MlNeuralNetwork (Multi-Layer Perceptron)

```javascript
import { MlNeuralNetwork } from "./ml-neural-network.js";

// Create network: inputNodes, hiddenNodes, outputNodes
const nn = new MlNeuralNetwork(2, 4, 1);

// Adjust learning rate (default: 0.1)
nn.learningRate = 0.05;

// Train the network
for (let i = 0; i < 50000; i++) {
  nn.train(inputArray, targetArray);
}

// Make predictions
const output = nn.feedforward(inputArray);
```

## Neural Network Architecture

### SlNeuralNetwork
- **Input**: 2 values
- **Output**: 1 value (sigmoid activated, 0-1)
- **Activation**: Sigmoid
- **Use case**: AND, OR gates

### MlNeuralNetwork
- **Layers**: Input → Hidden → Output
- **Activation**: Sigmoid
- **Training**: Backpropagation with gradient descent
- **Use case**: XOR gate, MNIST, complex patterns

## Dependencies

- `express`: Web server for API endpoints
- `mnist`: MNIST dataset for digit recognition training

## License

ISC

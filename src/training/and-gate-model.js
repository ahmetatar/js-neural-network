import { SlNeuralNetwork } from '../sl-neural-network.js';

export function trainAndGateModel() {
  const nn = new SlNeuralNetwork();

  // Training data for AND gate
  const trainingData = [
    { inputs: [0, 0], target: 0 },
    { inputs: [0, 1], target: 0 },
    { inputs: [1, 0], target: 0 },
    { inputs: [1, 1], target: 1 },
  ];

  // Training
  for (let i = 0; i < 20000; i++) {
    const data = trainingData[Math.floor(Math.random() * trainingData.length)];
    nn.train(data.inputs[0], data.inputs[1], data.target);
  }

  return nn;
}

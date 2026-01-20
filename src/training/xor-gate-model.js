import { MlNeuralNetwork } from "../ml-neural-network.js";

export function trainXorGateModel() {
  const nn = new MlNeuralNetwork(2, 3, 1);

  const trainingData = [
    { inputs: [0, 0], target: [0] },
    { inputs: [0, 1], target: [1] },
    { inputs: [1, 0], target: [1] },
    { inputs: [1, 1], target: [0] },
  ];

  // Training
  for (let i = 0; i < 50000; i++) {
    let d = trainingData[Math.floor(Math.random() * 4)];
    nn.train(d.inputs, d.target);
  }

  return nn;
}

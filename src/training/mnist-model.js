import MNIST from "mnist";
import { MlNeuralNetwork } from "../ml-neural-network.js";

export function trainMnistModel() {
  const set = MNIST.set(8000, 2000);
  const trainingSet = set.training;

  const nn = new MlNeuralNetwork(784, 64, 10);
  const restored = nn.restore();

  if (restored) {
    console.log("Model restored from mnist-model.json");
    return nn;
  }

  // Training
  nn.metrics.startTraining();
  console.log("Training new MNIST model...");
  for (let epoch = 0; epoch < 5; epoch++) {
    for (let data of trainingSet) {
      nn.train(data.input, data.output);
    }

    // Record epoch metrics
    const loss = nn.metrics.losses[nn.metrics.losses.length - 1];
    const accuracy = nn.metrics.accuracies.length > 0 ? nn.metrics.accuracies[nn.metrics.accuracies.length - 1] : null;

    nn.metrics.recordEpoch(epoch + 1, loss, accuracy);
    console.log(`Epoch ${epoch + 1} completed.`);
  }

  nn.metrics.endTraining();
  nn.save();

  return nn;
}

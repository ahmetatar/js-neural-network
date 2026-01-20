import fs from "fs";
import { Matrix } from "./utils/matrix.js";
import { TrainingMetrics } from "./utils/training-metrics.js";

/**
 * A simple Multi-Layer Perceptron (MLP) neural network for logical gates.
 */
export class MlNeuralNetwork {
  constructor(inputNodes, hiddenNodes, outputNodes) {
    this.weightsInputHiddens = Matrix.random(hiddenNodes, inputNodes);
    this.weightsOutputHiddens = Matrix.random(outputNodes, hiddenNodes);
    this.biasHidden = Matrix.random(hiddenNodes, 1);
    this.biasOutput = Matrix.random(outputNodes, 1);
    this.learningRate = 0.1;
    this.metrics = new TrainingMetrics();
  }

  static PATH() {
    return "models/mnist-model.json";
  }

  feedForward(inputArray) {
    // 1. Input -> Hidden
    const inputs = Matrix.fromArray(inputArray);
    let hidden = Matrix.multiply(this.weightsInputHiddens, inputs);
    hidden = this.addBiasAndActivate(hidden, this.biasHidden);

    // 2. Hidden -> Output
    let output = Matrix.multiply(this.weightsOutputHiddens, hidden);
    output = this.addBiasAndActivate(output, this.biasOutput);

    return output;
  }

  backForward() {}

  sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  sigmoidDerivative(x) {
    return x * (1 - x);
  }

  addBiasAndActivate(matrix, bias) {
    return matrix.map((row, i) => row.map((val) => this.sigmoid(val + bias[i][0])));
  }

  recordMetrics(errors, targets, outputs) {
    // Calculate and record loss (Mean Squared Error)
    const errorValues = Matrix.toArray(errors).flat();
    const loss = this.metrics.calculateMSE(errorValues);
    this.metrics.recordLoss(loss);

    // Calculate and record accuracy (threshold-based for classification)
    const outputValues = Matrix.toArray(outputs).flat();
    const targetValues = targets.flat();
    const isCorrect = outputValues.every((val, i) => Math.round(val) === targetValues[i]);
    this.metrics.recordPrediction(isCorrect);

    // Record running accuracy
    const accuracy = this.metrics.getCurrentAccuracy();
    this.metrics.recordAccuracy(accuracy);
  }

  save() {
    const modelData = {
      weights_in: this.weightsInputHiddens,
      weights_out: this.weightsOutputHiddens,
      bias_hidden: this.biasHidden,
      bias_output: this.biasOutput,
      learning_rate: this.learningRate,
    };
    fs.writeFileSync(MlNeuralNetwork.PATH(), JSON.stringify(modelData), "utf-8");
  }

  restore() {
    const path = MlNeuralNetwork.PATH();
    if (!fs.existsSync(path)) {
      return false;
    }

    const rawData = fs.readFileSync(path, "utf-8");
    const modelData = JSON.parse(rawData);

    this.weightsInputHiddens = modelData.weights_in;
    this.weightsOutputHiddens = modelData.weights_out;
    this.biasHidden = modelData.bias_hidden;
    this.biasOutput = modelData.bias_output;
    this.learningRate = modelData.learning_rate;
    return true;
  }

  train(inputArray, targetArray) {
    // Feedforward
    const output = this.feedForward(inputArray);
    // Convert target array to matrix
    let targets = Matrix.fromArray(targetArray);
    // Calculate output errors
    let outputErrors = Matrix.subtract(targets, output);
    this.recordMetrics(outputErrors, targetArray, output);

    // Backpropagation
    // Calculate gradients
    const inputs = Matrix.fromArray(inputArray);
    const hidden = Matrix.multiply(this.weightsInputHiddens, inputs);
    let gradients = output.map((row) => row.map((val) => this.sigmoidDerivative(val)));
    gradients = Matrix.hadamard(gradients, outputErrors);
    gradients = gradients.map((row) => row.map((val) => val * this.learningRate));

    // Calculate hidden -> output deltas
    let transposedHiddenLayer = Matrix.transpose(hidden);
    let weightOutputHiddenDeltas = Matrix.multiply(gradients, transposedHiddenLayer);

    // Update weights and biases for hidden -> output
    this.weightsOutputHiddens = Matrix.add(this.weightsOutputHiddens, weightOutputHiddenDeltas);
    this.biasOutput = Matrix.add(this.biasOutput, gradients);

    // Calculate hidden layer errors
    let transposedOutputHiddenWeights = Matrix.transpose(this.weightsOutputHiddens);
    let hiddenErrors = Matrix.multiply(transposedOutputHiddenWeights, outputErrors);

    // Calculate hidden gradients
    let hiddenGradients = hidden.map((row) => row.map((val) => this.sigmoidDerivative(val)));
    hiddenGradients = Matrix.hadamard(hiddenGradients, hiddenErrors);
    hiddenGradients = hiddenGradients.map((row) => row.map((val) => val * this.learningRate));

    // Calculate input -> hidden deltas
    let transposedInputs = Matrix.transpose(inputs);
    let weightInputHiddenDeltas = Matrix.multiply(hiddenGradients, transposedInputs);

    // Update weights and biases for input -> hidden
    this.weightsInputHiddens = Matrix.add(this.weightsInputHiddens, weightInputHiddenDeltas);
    this.biasHidden = Matrix.add(this.biasHidden, hiddenGradients);
  }
}

/**
 * A simple neural network to learn logical gates (AND, OR, NOT).
 */
export class SlNeuralNetwork {
  constructor() {
    this.weight1 = Math.random();
    this.weight2 = Math.random();
    this.bias = Math.random();
    this.learningRate = 0.1;
  }

  sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  sigmoidDerivative(x) {
    return x * (1 - x);
  }

  predict(input1, input2) {
    const z = input1 * this.weight1 + input2 * this.weight2 + this.bias;
    return this.sigmoid(z);
  }

  train(input1, input2, target) {
    const output = this.predict(input1, input2);
    const error = target - output;
    const adjustment = error * this.sigmoidDerivative(output);

    this.weight1 += input1 * adjustment * this.learningRate;
    this.weight2 += input2 * adjustment * this.learningRate;
    this.bias += adjustment * this.learningRate;
  }
}

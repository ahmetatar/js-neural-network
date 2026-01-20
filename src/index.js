import { Matrix } from "./utils/matrix.js";
import { trainMnistModel } from "./training/mnist-model.js";
import MNIST from "mnist";

const model = trainMnistModel();
const testSet = MNIST.set(0, 100).test;

const output = model.feedForward(testSet[0].input);
const arrOut = Matrix.toArray(output);

console.log("Predicted Output: ", arrOut.indexOf(Math.max(...arrOut)));
console.log("Expected Output: ", testSet[0].output.indexOf(1));

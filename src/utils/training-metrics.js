/**
 * A class to collect and track metrics during the training phase of a neural network model.
 */
export class TrainingMetrics {
  constructor() {
    this.reset();
  }

  /**
   * Resets all collected metrics to their initial state.
   */
  reset() {
    this.epochs = 0;
    this.losses = [];
    this.accuracies = [];
    this.startTime = null;
    this.endTime = null;
    this.totalSamples = 0;
    this.correctPredictions = 0;
    this.epochMetrics = [];
  }

  /**
   * Starts the training timer.
   */
  startTraining() {
    this.startTime = Date.now();
  }

  /**
   * Stops the training timer.
   */
  endTraining() {
    this.endTime = Date.now();
  }

  /**
   * Records the loss for the current epoch.
   * @param {number} loss - The loss value to record.
   */
  recordLoss(loss) {
    this.losses.push(loss);
  }

  /**
   * Records the accuracy for the current epoch.
   * @param {number} accuracy - The accuracy value to record (0-1).
   */
  recordAccuracy(accuracy) {
    this.accuracies.push(accuracy);
  }

  /**
   * Records metrics for a completed epoch.
   * @param {number} epochNumber - The epoch number.
   * @param {number} loss - The loss for this epoch.
   * @param {number} accuracy - The accuracy for this epoch (0-1).
   */
  recordEpoch(epochNumber, loss, accuracy = null) {
    this.epochs = epochNumber;
    this.losses.push(loss);
    if (accuracy !== null) {
      this.accuracies.push(accuracy);
    }
    this.epochMetrics.push({
      epoch: epochNumber,
      loss: loss,
      accuracy: accuracy,
      timestamp: Date.now(),
    });
  }

  /**
   * Records a single sample prediction result.
   * @param {boolean} isCorrect - Whether the prediction was correct.
   */
  recordPrediction(isCorrect) {
    this.totalSamples++;
    if (isCorrect) {
      this.correctPredictions++;
    }
  }

  /**
   * Calculates the Mean Squared Error from an array of errors.
   * @param {number[]} errors - Array of error values.
   * @returns {number} The mean squared error.
   */
  calculateMSE(errors) {
    if (errors.length === 0) return 0;
    const sumSquaredErrors = errors.reduce((sum, error) => sum + error * error, 0);
    return sumSquaredErrors / errors.length;
  }

  /**
   * Returns the average loss across all recorded losses.
   * @returns {number} The average loss.
   */
  getAverageLoss() {
    if (this.losses.length === 0) return 0;
    return this.losses.reduce((sum, loss) => sum + loss, 0) / this.losses.length;
  }

  /**
   * Returns the average accuracy across all recorded accuracies.
   * @returns {number} The average accuracy.
   */
  getAverageAccuracy() {
    if (this.accuracies.length === 0) return 0;
    return this.accuracies.reduce((sum, acc) => sum + acc, 0) / this.accuracies.length;
  }

  /**
   * Returns the current accuracy based on recorded predictions.
   * @returns {number} The current accuracy (0-1).
   */
  getCurrentAccuracy() {
    if (this.totalSamples === 0) return 0;
    return this.correctPredictions / this.totalSamples;
  }

  /**
   * Returns the total training duration in milliseconds.
   * @returns {number} Training duration in milliseconds.
   */
  getTrainingDuration() {
    if (!this.startTime) return 0;
    const end = this.endTime || Date.now();
    return end - this.startTime;
  }

  /**
   * Returns all collected metrics.
   * @returns {Object} An object containing all training metrics.
   */
  getMetrics() {
    return {
      epochs: this.epochs,
      totalSamples: this.totalSamples,
      correctPredictions: this.correctPredictions,
      currentAccuracy: this.getCurrentAccuracy(),
      averageAccuracy: this.getAverageAccuracy(),
      averageLoss: this.getAverageLoss(),
      finalLoss: this.losses.length > 0 ? this.losses[this.losses.length - 1] : null,
      trainingDurationMs: this.getTrainingDuration(),
      trainingDurationSeconds: this.getTrainingDuration() / 1000,
      losses: [...this.losses],
      accuracies: [...this.accuracies],
      epochMetrics: [...this.epochMetrics],
    };
  }
}

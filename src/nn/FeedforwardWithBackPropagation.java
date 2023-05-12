package nn;

import java.util.Random;

public class FeedforwardWithBackPropagation extends SimpleFeedforwardNeuralNetwork{
    private int inputDim;
    private int hiddenDim;
    private int outputDim;
    private double[][] inputWeights;
    private double[] hiddenBiases;
    private double[][] outputWeights;
    private double[] outputBiases;
    private double[] hidden;
    private double[] output;

    public FeedforwardWithBackPropagation(int inputDim, int hiddenDim, int outputDim) {
        super(inputDim, hiddenDim, outputDim);
        this.inputWeights = new double[inputDim][hiddenDim];
        this.hiddenBiases = new double[hiddenDim];
        this.outputWeights = new double[hiddenDim][outputDim];
        this.outputBiases = new double[outputDim];
        this.hidden = new double[hiddenDim];
        this.output = new double[outputDim];
    }

    // Rest of the constructor and other methods

    public void train(double[] input, double[] target, double learningRate, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Forward propagation
            forward(input);

            // Backpropagation
            // Compute gradients for output layer
            double[] outputGradients = new double[outputDim];
            for (int i = 0; i < outputDim; i++) {
                outputGradients[i] = output[i] - target[i];
            }

            // Update output weights and biases
            for (int i = 0; i < hiddenDim; i++) {
                for (int j = 0; j < outputDim; j++) {
                    outputWeights[i][j] -= learningRate * outputGradients[j] * hidden[i];
                }
            }
            for (int i = 0; i < outputDim; i++) {
                outputBiases[i] -= learningRate * outputGradients[i];
            }

            // Compute gradients for hidden layer
            double[] hiddenGradients = new double[hiddenDim];
            for (int i = 0; i < hiddenDim; i++) {
                double sum = 0.0;
                for (int j = 0; j < outputDim; j++) {
                    sum += outputGradients[j] * outputWeights[i][j];
                }
                hiddenGradients[i] = sum * (hidden[i] > 0 ? 1 : 0);
            }

            // Update input weights and biases
            for (int i = 0; i < inputDim; i++) {
                for (int j = 0; j < hiddenDim; j++) {
                    inputWeights[i][j] -= learningRate * hiddenGradients[j] * input[i];
                }
            }
            for (int i = 0; i < hiddenDim; i++) {
                hiddenBiases[i] -= learningRate * hiddenGradients[i];
            }
        }
    }

}

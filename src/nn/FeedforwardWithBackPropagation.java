/*
package nn;

import java.sql.SQLOutput;
import java.util.Random;

public class FeedforwardWithBackPropagation {
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
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        this.outputDim = outputDim;
        this.inputWeights = new double[inputDim][hiddenDim];
        this.hiddenBiases = new double[hiddenDim];
        this.outputWeights = new double[hiddenDim][outputDim];
        this.outputBiases = new double[outputDim];
        this.hidden = new double[hiddenDim];
        this.output = new double[outputDim];
    }

    public FeedforwardWithBackPropagation(int inputDim, int hiddenDim, int outputDim, double[] values) {
        this(inputDim, hiddenDim, outputDim);
        int offset = 0;
        for (int i = 0; i < inputDim; i++) {
            for (int j = 0; j < hiddenDim; j++) {
                inputWeights[i][j] = values[i * hiddenDim + j];
            }
        }
        offset = inputDim * hiddenDim;
        for (int i = 0; i < hiddenDim; i++) {
            hiddenBiases[i] = values[offset + i];
        }
        offset += hiddenDim;
        for (int i = 0; i < hiddenDim; i++) {
            for (int j = 0; j < outputDim; j++) {
                outputWeights[i][j] = values[offset + i * outputDim + j];
            }
        }
        offset += hiddenDim * outputDim;
        for (int i = 0; i < outputDim; i++) {
            outputBiases[i] = values[offset + i];
        }

    }

    public int getChromossomeSize() {
        return inputWeights.length * inputWeights[0].length + hiddenBiases.length
                + outputWeights.length * outputWeights[0].length + outputBiases.length;
    }

    public double[] getChromossome() {
        double[] chromossome = new double[getChromossomeSize()];
        int offset = 0;
        for (int i = 0; i < inputDim; i++) {
            for (int j = 0; j < hiddenDim; j++) {
                chromossome[i * hiddenDim + j] = inputWeights[i][j];
            }
        }
        offset = inputDim * hiddenDim;
        for (int i = 0; i < hiddenDim; i++) {
            chromossome[offset + i] = hiddenBiases[i];
        }
        offset += hiddenDim;
        for (int i = 0; i < hiddenDim; i++) {
            for (int j = 0; j < outputDim; j++) {
                chromossome[offset + i * outputDim + j] = outputWeights[i][j];
            }
        }
        offset += hiddenDim * outputDim;
        for (int i = 0; i < outputDim; i++) {
            chromossome[offset + i] = outputBiases[i];
        }

        return chromossome;

    }

    public void initializeWeights() {
        // Randomly initialize weights and biases
        Random random = new Random();
        for (int i = 0; i < inputDim; i++) {
            for (int j = 0; j < hiddenDim; j++) {
                inputWeights[i][j] = random.nextDouble() - 0.5;
            }
        }
        for (int i = 0; i < hiddenDim; i++) {
            hiddenBiases[i] = random.nextDouble() - 0.5;
            for (int j = 0; j < outputDim; j++) {
                outputWeights[i][j] = random.nextDouble() - 0.5;
            }
        }
        for (int i = 0; i < outputDim; i++) {
            outputBiases[i] = random.nextDouble() - 0.5;
        }
    }


    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public double[] forward(double[] input) {
        // Verify input dimensions
        if (input.length != inputDim) {
            System.out.println(input.length);
            System.out.println(inputDim);
            throw new IllegalArgumentException("Invalid input dimensions");
        }


        // Compute output given input
        double[] hidden = new double[hiddenDim];
        for (int i = 0; i < hiddenDim; i++) {
            double sum = 0.0;
            for (int j = 0; j < inputDim; j++) {
                double d = input[j];
                sum += d * inputWeights[j][i];
            }
            hidden[i] = sigmoid(sum + hiddenBiases[i]);
        }
        double[] output = new double[outputDim];
        for (int i = 0; i < outputDim; i++) {
            double sum = 0.0;
            for (int j = 0; j < hiddenDim; j++) {
                sum += hidden[j] * outputWeights[j][i];
            }
            output[i] = Math.exp(sum + outputBiases[i]);
        }
        double sum = 0.0;
        for (int i = 0; i < outputDim; i++) {
            sum += output[i];
        }
        for (int i = 0; i < outputDim; i++) {
            output[i] /= sum;
        }
        return output;
    }



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
*/

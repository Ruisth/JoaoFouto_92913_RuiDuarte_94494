package nn;

public class NeuralNetwork {
    private int inputSize;
    private int hiddenSize;
    private int outputSize;
    private double[][] inputWeights;
    private double[][] outputWeights;


    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.inputWeights = new double[inputSize][hiddenSize];
        this.outputWeights = new double[hiddenSize][outputSize];
    }

    public double[] feedForward(double[] inputs) {
        double[] hiddenLayer = new double[hiddenSize];
        double[] outputs = new double[outputSize];

        for (int i = 0; i < hiddenSize; i++) {
            double sum = 0;
            for (int j = 0; j < inputSize; j++) {
                sum += inputs[j] * inputWeights[j][i];
            }
            hiddenLayer[i] = Math.tanh(sum);
        }

        for (int i = 0; i < outputSize; i++) {
            double sum = 0;
            for (int j = 0; j < hiddenSize; j++) {
                sum += hiddenLayer[j] * outputWeights[j][i];
            }
            outputs[i] = Math.tanh(sum);
        }

        return outputs;
    }

    // Getter and setter methods for inputWeights and outputWeights
    public double[][] getInputWeights() {
        return inputWeights;
    }

    public void setInputWeights(double[][] inputWeights) {
        this.inputWeights = inputWeights;
    }

    public double[][] getOutputWeights() {
        return outputWeights;
    }

    public void setOutputWeights(double[][] outputWeights) {
        this.outputWeights = outputWeights;
    }
}

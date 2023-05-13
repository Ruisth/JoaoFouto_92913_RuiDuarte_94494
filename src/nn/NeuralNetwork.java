package nn;

import controllers.GameController;
import space.Board;

import java.util.Arrays;
import java.util.Random;

public class NeuralNetwork implements GameController {
    private int inputDim;
    private int hiddenDim;
    private int outputDim;
    private double[][] inputWeights;
    private double[] hiddenBiases;
    private double[][] outputWeights;
    private double[] outputBiases;
    private static double fitness;

    public NeuralNetwork(int inputDim, int hiddenDim, int outputDim) {
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        this.outputDim = outputDim;
        this.inputWeights = new double[inputDim][hiddenDim];
        this.hiddenBiases = new double[hiddenDim];
        this.outputWeights = new double[hiddenDim][outputDim];
        this.outputBiases = new double[outputDim];
        initializeWeights();
    }

    public NeuralNetwork(int inputDim, int hiddenDim, int outputDim, double[][] inputWeights, double[] hiddenBiases, double[][] outputWeights, double[] outputBiases) {
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        this.outputDim = outputDim;

        this.inputWeights = inputWeights;
        this.hiddenBiases = hiddenBiases;
        this.outputWeights = outputWeights;
        this.outputBiases = outputBiases;
    }


    public void initializeWeights(){
        Random random = new Random();

        //Inicialização dos pesos de cada camada
        inputWeights = new double[hiddenDim][inputDim];
        for (int i = 0; i < hiddenDim; i++){
            for (int j = 0; j < inputDim; j++){
                inputWeights[i][j] = random.nextDouble() - 0.5;
            }
        }

        //Inicialização das biases
        hiddenBiases = new double[hiddenDim];
        for (int i = 0; i < hiddenDim; i++){
            hiddenBiases[i] = random.nextDouble() - 0.5;
        }

        //Inicaliza os pesos da camada oculta para a camada de saída
        outputWeights = new double[outputDim][hiddenDim];
        for (int i = 0; i < outputDim; i++){
            for (int j = 0; j < hiddenDim; j++){
                outputWeights[i][j] = random.nextDouble() - 0.5;
            }
        }

        //Inicialização das biases de saída
        outputBiases = new double[outputDim];
        for (int i = 0; i < outputDim; i++){
            outputBiases[i] = random.nextDouble() - 0.5;
        }
    }

    //Função sigmoide
    private double sigmoid(double x){
        return 1.0 / (1.0 + Math.exp(-x));
    }


    @Override
    public double[] nextMove(double[] currentState) {

        //Cálculo da ativação da camada oculta
        double[] hiddenActivations = new double[hiddenDim];
        for(int i = 0; i < hiddenDim; i++){
            double activation = 0.0;
            for (int j = 0; j < inputDim; j++){
                activation += inputWeights[i][j] * currentState[j];
            }
            activation += hiddenBiases[i];
            hiddenActivations[i] = activation;
        }

        //Cálculo da ativação da camada de saída
        double[] outputActivations = new double[outputDim];
        for (int i = 0; i < outputDim; i++){
            double activation = 0.0;
            for (int j = 0; j < hiddenDim; j++){
                activation += outputWeights[i][j] * sigmoid(hiddenActivations[j]);
            }
            activation += outputBiases[i];
            outputActivations[i] += activation;
        }

        //Aplicação da função de ativação final
        double[] output = new double[outputDim];
        for (int i = 0; i < outputDim; i++){
            output[i] = outputActivations[i];
        }

        return output;
    }

    public int getInputDim() {
        return inputDim;
    }

    public void setInputDim(int inputDim) {
        this.inputDim = inputDim;
    }

    public int getHiddenDim() {
        return hiddenDim;
    }

    public void setHiddenDim(int hiddenDim) {
        this.hiddenDim = hiddenDim;
    }

    public int getOutputDim() {
        return outputDim;
    }

    public void setOutputDim(int outputDim) {
        this.outputDim = outputDim;
    }

    public double[][] getInputWeights() {
        return inputWeights;
    }

    public void setInputWeights(double[][] inputWeights) {
        this.inputWeights = inputWeights;
    }

    public double[] getHiddenBiases() {
        return hiddenBiases;
    }

    public void setHiddenBiases(double[] hiddenBiases) {
        this.hiddenBiases = hiddenBiases;
    }

    public double[][] getOutputWeights() {
        return outputWeights;
    }

    public void setOutputWeights(double[][] outputWeights) {
        this.outputWeights = outputWeights;
    }

    public double[] getOutputBiases() {
        return outputBiases;
    }

    public void setOutputBiases(double[] outputBiases) {
        this.outputBiases = outputBiases;
    }

    public double getFitness() {
        return fitness;
    }

    public void setFitness(double fitness) {
        this.fitness = fitness;
    }

    @Override
    public String toString() {
        return "NeuralNetwork{" +
                "inputDim=" + inputDim +
                ", hiddenDim=" + hiddenDim +
                ", outputDim=" + outputDim +
                '}';
    }
}
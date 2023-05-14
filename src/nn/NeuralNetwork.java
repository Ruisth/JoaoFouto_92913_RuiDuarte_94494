package nn;

import controllers.GameController;
import space.Board;

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

    }


    public void initializeWeights(){

        //Inicialização dos pesos de cada camada
        inputWeights = new double[inputDim][hiddenDim];
        for (int i = 0; i < inputDim; i++){
            for (int j = 0; j < hiddenDim; j++){
                inputWeights[i][j] = Math.random() - 0.5;
            }
        }

        //Inicialização das biases
        hiddenBiases = new double[hiddenDim];
        for (int i = 0; i < hiddenDim; i++){
            hiddenBiases[i] = Math.random() - 0.5;
        }

        //Inicaliza os pesos da camada oculta para a camada de saída
        outputWeights = new double[hiddenDim][outputDim];
        for (int i = 0; i < hiddenDim; i++){
            for (int j = 0; j < outputDim; j++){
                outputWeights[i][j] = Math.random() - 0.5;
            }
        }

        //Inicialização das biases de saída
        outputBiases = new double[outputDim];
        for (int i = 0; i < outputDim; i++){
            outputBiases[i] = Math.random() - 0.5;
        }
    }

    public void setupBoard(){
        Board b = new Board(this);
        b.setSeed(5);
        b.run();
        fitness = b.getFitness();
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
                double state = currentState[j];
                activation += inputWeights[j][i] * state;
            }
            activation += hiddenBiases[i];
            hiddenActivations[i] = activation;
        }

        //Cálculo da ativação da camada de saída
        double[] outputActivations = new double[outputDim];
        for (int i = 0; i < outputDim; i++){
            double activation = 0.0;
            for (int j = 0; j < hiddenDim; j++){
                activation += hiddenActivations[j] * outputWeights[j][i];
            }

            outputActivations[i] = sigmoid(activation + outputBiases[i]);
        }

        //Aplicação da função de ativação final
        double[] output = new double[outputDim];
        for (int i = 0; i < outputDim; i++){
            output[i] = sigmoid(outputActivations[i]);
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

    public int getTotalParams(){
        return inputDim * hiddenDim + hiddenDim + hiddenDim * outputDim + outputDim;
    }

    public double[] getNode(){
        double[] params = new double[getTotalParams()];

        int index = 0;
        for (int i = 0; i < inputDim; i++){
            for (int j = 0; j < hiddenDim; j++){
                params[index++] = inputWeights[j][i];
            }
        }

        for (int i = 0; i < hiddenDim; i++){
            params[index++] = hiddenBiases[i];
        }

        for (int i = 0; i < hiddenDim; i++) {
            for (int j = 0; j < outputDim; j++){
                params[index++] = outputWeights[i][j];
            }
        }

        for (int i = 0; i < outputDim; i++){
            params[index++] = outputBiases[i];
        }

        return params;
    }

    public void setNode(double[] params) {
        if (params.length != getTotalParams()) {
            throw new IllegalArgumentException("O tamanho do array de parâmetros não corresponde ao número esperado de parâmetros.");
        }

        int index = 0;
        for (int i = 0; i < inputDim; i++) {
            for (int j = 0; j < hiddenDim; j++) {
                inputWeights[i][j] = params[index++];
            }
        }

        for (int i = 0; i < hiddenDim; i++) {
            hiddenBiases[i] = params[index++];
        }

        for (int i = 0; i < hiddenDim; i++) {
            for (int j = 0; j < outputDim; j++) {
                outputWeights[i][j] = params[index++];
            }
        }

        for (int i = 0; i < outputDim; i++) {
            outputBiases[i] = params[index++];
        }
    }

    public void setFitness(double fitness) {
        this.fitness = fitness;
    }

   @Override
    public String toString() {
        int init = 0;
        int hid = 0;
        int out = 0;
        double[][] inwei = null;
        double[] hidbia = null;
        double[] outbia = null;
        double[][] outwei = null;
        double fit = 0.0;
        for (NeuralNetwork nn : GeneticAlgorithm.population) {
            init = nn.getInputDim();
            hid = nn.getHiddenDim();
            out = nn.getOutputDim();
            inwei = nn.getInputWeights();
            hidbia = nn.getHiddenBiases();
            outbia = nn.getOutputBiases();
            outwei = nn.getOutputWeights();
            fit = nn.getFitness();
        }

        return "Neural Network: " + init + "- Input Dim | " + hid + "- Hidden Dim | " + out + "- Output Dim | " + inwei + "- Input Weights | " + hidbia + "- Hidden Biases | " + outbia + "- Output Biases | " + outwei + "- Output Weights | " + fit + "- Fitness | \n";
    }
}
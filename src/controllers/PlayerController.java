package controllers;

import nn.FeedforwardWithBackPropagation;
import space.Commons;

public class PlayerController implements GameController {
    private FeedforwardWithBackPropagation neuralNetwork;

    public PlayerController() {
        // Initialize the neural network with the desired dimensions
        int inputDim = Commons.STATE_SIZE;
        int hiddenDim = 50;
        int outputDim = Commons.NUM_ACTIONS;
        neuralNetwork = new FeedforwardWithBackPropagation(inputDim, hiddenDim, outputDim);

        // Randomly initialize the weights and biases
        neuralNetwork.initializeWeights();
    }

    public void train(double[][] inputs, double[][] targets, double learningRate, int epochs) {
        // Train the neural network using the provided inputs and targets
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                double[] input = inputs[i];
                double[] target = targets[i];
                neuralNetwork.train(input, target, learningRate, 1);
            }
        }
    }

    @Override
    public double[] nextMove(double[] currentState) {
        // Make a move based on the current game state represented by the currentState
        return neuralNetwork.forward(currentState);
    }
}
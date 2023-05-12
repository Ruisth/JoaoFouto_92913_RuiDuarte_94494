package controllers;

import java.util.Arrays;
import java.util.Random;

import controllers.GameController;
import controllers.RandomController;
import nn.SimpleFeedforwardNeuralNetwork;
import space.Board;
import space.Commons;
import space.SpaceInvaders;

public class PlayRandomController {
	public static void main(String[] args) {

		PlayerController controller = new PlayerController();

		// Training data
		double[][] inputs =  {
				{0.2, 0.5},   // Example input 1
				{0.7, 0.3},   // Example input 2
				// Add more input values as needed
		};

		double[][] targets = {
				{1.0, 0.0},   // Example target 1
				{0.0, 1.0},   // Example target 2
				// Add more target values as needed
		};

		// Train the neural network
		double learningRate = 0.1;
		int epochs = 500;
		controller.train(inputs, targets, learningRate, epochs);

		// Make a move based on the current game state
		double[] currentState = new double[Commons.STATE_SIZE];
		double[] nextMove = controller.nextMove(currentState);
		System.out.println("Next move: " + Arrays.toString(nextMove));

		// Evaluate the neural network controller using the provided API methods
		int seed = 123; // Specify the seed value for reproducibility
		Board b = new Board(controller);
		b.setSeed(seed);
		b.run();
		double fitness = b.getFitness();
		System.out.println("Fitness score: " + fitness);


		SpaceInvaders.showControllerPlaying(controller,seed);
	}
}

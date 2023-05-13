/*
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

		// Create an array with 112 input entries
		int numSamples = 50;
		double[][] inputData = new double[numSamples][112];
		for (int i = 0; i < numSamples; i++) {
			for (int j = 0; j < 112; j++) {
				inputData[i][j] = (j + 1) * 0.01; // Example: assign values ranging from 0.01 to 1.12
			}
		}

		// Create an array with 112 target entries
		double[][] targetData = new double[numSamples][4];
		for (int i = 0; i < numSamples; i++) {
			targetData[i][0] = (i % 4 == 0) ? 1 : 0;
			targetData[i][1] = (i % 4 == 1) ? 1 : 0;
			targetData[i][2] = (i % 4 == 2) ? 1 : 0;
			targetData[i][3] = (i % 4 == 3) ? 1 : 0;
		}

		// Training data
		double[][] inputs = inputData;
		double[][] targets = targetData;

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
*/

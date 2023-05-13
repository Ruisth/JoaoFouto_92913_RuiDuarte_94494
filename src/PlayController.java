import nn.NeuralNetwork;
import nn.GeneticAlgorithm;
import space.Commons;
import space.SpaceInvaders;
import nn.GeneticAlgorithm;
import java.util.ArrayList;
import java.util.List;


public class PlayController {
    public static void main(String[] args) {


        List<NeuralNetwork> population = new ArrayList<>();
        for (int i = 0; i < populationSize; i++) {
            int inputSize = Commons.STATE_SIZE;
            int hiddenSize = 0;
            int outputSize = Commons.NUM_ACTIONS;

            NeuralNetwork nn = new NeuralNetwork(inputSize, hiddenSize, outputSize);
            
            // Initialize nn's weights randomly
            population.add(nn);
        }

        for (NeuralNetwork nn : population) {
            double fitness = geneticAlgorithm.calculateFitness(nn, seed);
            // Store the fitness value for nn
        }

        int generations = 100;

        for (int generation = 0; generation < generations; generation++) {
            // Select parents based on their fitness
            List<NeuralNetwork> parents = geneticAlgorithm.selection(population);

            // Perform crossover to generate offspring
            List<NeuralNetwork> offspring = geneticAlgorithm.crossover(parents);

            // Apply mutation to the offspring
            geneticAlgorithm.mutation(offspring, mutationRate);

            // Evaluate the fitness of the new population (parents + offspring)
            for (NeuralNetwork nn : offspring) {
                double fitness = geneticAlgorithm.calculateFitness(nn, seed);
                // Store the fitness value for nn
            }

            // Update the population for the next generation
            population = geneticAlgorithm.createNewPopulation(parents, offspring);

            // Optionally: Print the best fitness value in each generation
        }

        // Get the best-performing neural network from the final population
        NeuralNetwork bestNN = geneticAlgorithm.getBestNeuralNetwork(population);

        // Visualize the behavior of the best neural network
        SpaceInvaders.showControllerPlaying(new NNController(bestNN), seed);

    }
}


package nn;

import space.Board;

import java.util.*;
import java.util.stream.Collectors;

import static space.Commons.*;

public class GeneticAlgorithm {

    private static final int POPULATION_SIZE = 100;
    private static final int GENERATIONS = 100;
    private static final double MUTATION_RATE = 0.01;
    private static Random random = new Random();
    private List<NeuralNetwork> population = new ArrayList<>();
    private double fitness;
    double[]fitnessList = new double[POPULATION_SIZE + 1];

    public GeneticAlgorithm() {

    }

    public void comecar(){
        GeneticAlgorithm ga = new GeneticAlgorithm();

        for (int i = 0; i < POPULATION_SIZE; i++) {
            NeuralNetwork neuralNetwork = new NeuralNetwork(STATE_SIZE, STATE_SIZE, NUM_ACTIONS);
            population.add(neuralNetwork);
            Board b = new Board(neuralNetwork);
            b.setSeed(5);
            b.run();
            //fitness = b.getFitness();
           fitnessList[i] = b.getFitness();

        }
        // Execute o algoritmo genético por um determinado número de gerações
       // System.out.println(population.toString());
        for (int i = 0; i < GENERATIONS; i++) {

            List<NeuralNetwork> newPopulation = new ArrayList<>();


            // Selecione os pais e crie a nova população com cruzamento e mutação
            for (int j = 0; j < POPULATION_SIZE; j++) {
                NeuralNetwork parent1 = selectParent(population);
                NeuralNetwork parent2 = selectParent(population);
                NeuralNetwork child = crossover(parent1, parent2);
                mutation(child);
                newPopulation.add(child);
            }

            // Atualize a população
            population = newPopulation;
           /* for (NeuralNetwork nn:population) {
                System.out.println(nn.toString());
            }*/

            // If it's not the first generation, use the best weights from the previous generation

            if (i > 0) {
                NeuralNetwork bestNN = selectParent(population);
                double[][] bestInputWeights = bestNN.getInputWeights();
                double[] bestHiddenBiases = bestNN.getHiddenBiases();
                double[][] bestOutputWeights = bestNN.getOutputWeights();
                double[] bestOutputBiases = bestNN.getOutputBiases();

                for (int j = 0; j < POPULATION_SIZE; j++) {
                    NeuralNetwork neuralNetwork = new NeuralNetwork(STATE_SIZE, STATE_SIZE, NUM_ACTIONS, bestInputWeights, bestHiddenBiases, bestOutputWeights, bestOutputBiases);
                    population.set(j, neuralNetwork);
                    Board b = new Board(neuralNetwork);
                    b.setSeed(5);
                    b.run();
                    fitnessList[i] = b.getFitness();
                }
            }

            // Imprima a geração atual e a pontuação média de fitness
            double totalFitness = 0;
            /*for (NeuralNetwork nn : population) {
                totalFitness += nn.getFitness();
            }*/

            for (int x = 0; x < fitnessList.length - 1; x++) {
                totalFitness += fitnessList[x];
                //System.err.printf("DENTRO DO FOR -- Generation %d - TotalFitness: %f \n", i + 1, totalFitness);
            }
            double avgFitness = totalFitness / population.size();
            System.err.printf("Generation %d - TotalFitness: %f Average Fitness: %.2f\n", i + 1, totalFitness, avgFitness);
        }

        // Encontre o NeuralNetwork com a maior pontuação de fitness
        NeuralNetwork bestNN = selectParent(population);
        System.out.printf("Best NeuralNetwork Fitness: %.2f\n", bestNN.getFitness());
    }


/*
    public void comecar(){
        GeneticAlgorithm ga = new GeneticAlgorithm();

        int populationSize = POPULATION_SIZE;
        int generations = GENERATIONS;
        double mutationRate = MUTATION_RATE;

        for (int i = 0; i < POPULATION_SIZE; i++) {
            NeuralNetwork neuralNetwork = new NeuralNetwork(STATE_SIZE, STATE_SIZE, NUM_ACTIONS);
            neuralNetwork.initializeWeights();
            population.add(neuralNetwork);
            Board b = new Board(neuralNetwork);
            b.setSeed(5);
            b.run();
            //fitness = b.getFitness();

            fitnessList[i] = b.getFitness();
        }

        for (int i = 0; i < generations; i++) {
            // Set the population size, mutation rate, etc. for this generation
            // For example, you could decrease the population size and mutation rate over time
            if (i > 0) {
                populationSize = (int) Math.max(populationSize * 0.9, 10);
                mutationRate *= 0.95;
            }

            List<NeuralNetwork> newPopulation = new ArrayList<>(populationSize);

            // Create the new population by selecting parents and performing crossover and mutation
            for (int j = 0; j < populationSize; j++) {
                NeuralNetwork parent1 = selectParent(population);
                NeuralNetwork parent2 = selectParent(population);
                NeuralNetwork child = crossover(parent1, parent2);
                mutation(child);
                newPopulation.add(child);
            }



            // Update the population
            population = newPopulation;

            // Print the current generation and average fitness
            double totalFitness = 0;
            for (NeuralNetwork nn : population) {
                totalFitness += nn.getFitness();
            }

            for (int x = 0; x < populationSize; x++) {
                totalFitness = fitnessList[x];
            }

                double avgFitness = totalFitness / population.size();
                System.err.printf("Generation %d - Population Size: %d, Mutation Rate: %.4f, Total Fitness: %f, Average Fitness: %.2f\n", i + 1, populationSize, mutationRate, totalFitness, avgFitness);

            }

        // Find the NeuralNetwork with the highest fitness score
        NeuralNetwork bestNN = selectParent(population);
        System.out.printf("Best NeuralNetwork Fitness: %.2f\n", bestNN.getFitness());
    }*/

    public NeuralNetwork selectParent(List<NeuralNetwork> population) {
        NeuralNetwork bestParent = population.get(0);

        for (int i = 1; i < population.size(); i++) {
            NeuralNetwork currentParent = population.get(i);
            if (currentParent.getFitness() > bestParent.getFitness()) {
                bestParent = currentParent;
            }
        }

        return bestParent;
    }

    public NeuralNetwork crossover(NeuralNetwork parent1, NeuralNetwork parent2) {
        NeuralNetwork child = new NeuralNetwork(parent1.getInputDim(), parent1.getHiddenDim(), parent1.getOutputDim());

        // Perform crossover for input weights
        for (int i = 0; i < parent1.getHiddenDim(); i++) {
            for (int j = 0; j < parent1.getInputDim(); j++) {
                if (Math.random() < 0.5) {
                    child.getInputWeights()[i][j] = parent1.getInputWeights()[i][j];
                } else {
                    child.getInputWeights()[i][j] = parent2.getInputWeights()[i][j];
                }
            }
        }

        // Perform crossover for hidden biases
        for (int i = 0; i < parent1.getHiddenDim(); i++) {
            if (Math.random() < 0.5) {
                child.getHiddenBiases()[i] = parent1.getHiddenBiases()[i];
            } else {
                child.getHiddenBiases()[i] = parent2.getHiddenBiases()[i];
            }
        }

        // Perform crossover for output weights
        for (int i = 0; i < parent1.getOutputDim(); i++) {
            for (int j = 0; j < parent1.getHiddenDim(); j++) {
                if (Math.random() < 0.5) {
                    child.getOutputWeights()[i][j] = parent1.getOutputWeights()[i][j];
                } else {
                    child.getOutputWeights()[i][j] = parent2.getOutputWeights()[i][j];
                }
            }
        }

        // Perform crossover for output biases
        for (int i = 0; i < parent1.getOutputDim(); i++) {
            if (Math.random() < 0.5) {
                child.getOutputBiases()[i] = parent1.getOutputBiases()[i];
            } else {
                child.getOutputBiases()[i] = parent2.getOutputBiases()[i];
            }
        }

        return child;
    }

    private void mutation(NeuralNetwork nn) {
        Random random = new Random();

        for (int i = 0; i < nn.getHiddenDim(); i++) {
            if (random.nextDouble() < MUTATION_RATE) {
                nn.getHiddenBiases()[i] += random.nextDouble() - 0.5;
            }
        }
    }

    /*@Override
    public String toString() {
        return "Geraçao " + this.population;
    }*/

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Population:\n");

        int counter = 1;
        for (NeuralNetwork nn : population) {
            sb.append("Neural Network ").append(counter).append(":\n");
            sb.append(nn.toString()).append("\n");
            counter++;
        }

        return sb.toString();
    }
}

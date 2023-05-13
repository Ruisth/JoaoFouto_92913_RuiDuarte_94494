package nn;

import controllers.GameController;
import space.Board;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

public class GeneticAlgorithm{

    private int populationSize;
    private double mutationRate;
    private double crossoverRate;
    private Random random;


    public GeneticAlgorithm(int populationSize, double mutationRate, double crossoverRate) {
        this.populationSize = populationSize;
        this.mutationRate = mutationRate;
        this.crossoverRate = crossoverRate;
    }


    public List<Board> selection(List<Board> population) {

        // Assuming that the fitness values are already stored in the neural networks
        population.sort(Comparator.comparing(Board::getFitness).reversed());

        List<Board> selected = new ArrayList<>();
        for (int i = 0; i < populationSize / 2; i++) {
            selected.add(population.get(i));
        }
        return selected;
    }

    public List<NeuralNetwork> crossover(List<NeuralNetwork> parents) {
        List<NeuralNetwork> offspring = new ArrayList<>();

        for (int i = 0; i < populationSize / 2; i += 2) {
            NeuralNetwork parent1 = parents.get(i);
            NeuralNetwork parent2 = parents.get(i + 1);

            if (random.nextDouble() < crossoverRate) {
                NeuralNetwork child1 = parent1.crossover(parent2);
                NeuralNetwork child2 = parent2.crossover(parent1);

                offspring.add(child1);
                offspring.add(child2);
            } else {
                offspring.add(parent1.copy());
                offspring.add(parent2.copy());
            }
        }
        return offspring;
    }

    public void mutation(List<NeuralNetwork> offspring, double mutationRate) {
        for (NeuralNetwork nn : offspring) {
            nn.mutate(mutationRate);
        }
    }

    // Implement selection, mutation, and crossover methods

    public double calculateFitness(NeuralNetwork nn, long seed) {
        GameController controller = new NNController(nn);
        Board b = new Board(controller);
        b.setSeed(seed);
        b.run();
        double fitness = b.getFitness();
        return fitness;
    }

}

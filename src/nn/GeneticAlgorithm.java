package nn;

import space.Board;
import space.SpaceInvaders;

import java.util.*;
import java.util.stream.Collectors;

import static space.Commons.*;

public class GeneticAlgorithm {

    private static final int POPULATION_SIZE = 100;
    private static final int GENERATIONS = 100;
    private static final double MUTATION_RATE = 0.01;
    private static Random random = new Random();
    public static List<NeuralNetwork> population = new ArrayList<>();
    public static List<NeuralNetwork> bestPop = new ArrayList<>();
    public NeuralEntity[] nePopulation = new NeuralEntity[POPULATION_SIZE];
    public NeuralEntity[] topEntity = new NeuralEntity[25];
    public double fitness;
    public double bestFitness;
    public NeuralNetwork highestFitnessPlay;


    public GeneticAlgorithm() {

    }


    public void comecar(){

        //Inicialização da população de NeuralEntity


        for (int i = 0; i < GENERATIONS; i++) {
            if (i == 0) {

                //Inicialização da primeira geração com initializeWeights();
                for (int j = 0; j < POPULATION_SIZE; j++) {

                    NeuralNetwork nn = new NeuralNetwork(STATE_SIZE, STATE_SIZE, NUM_ACTIONS);
                    nn.initializeWeights();
                    Board b = new Board(nn);
                    b.setSeed(5);
                    b.run();
                    nn.setFitness(b.getFitness());
                    population.add(nn);
                    population.sort(Comparator.comparing(NeuralNetwork::getFitness).reversed());






                    NeuralEntity ne = new NeuralEntity();
                    ne.setFitness(nn.getFitness());
                    ne.setNode(nn.getNode());
                    nePopulation[j] = ne;




                }
                bestPop.add(population.get(0));
                Arrays.sort(nePopulation);



            }else{
                // Seleciona os 25 melhores da geração anterior
                for (int j = 0; j < 25; j++) {
                    topEntity[j] = nePopulation[j];
                }
                Arrays.sort(topEntity);


                for (int j = 0; j < POPULATION_SIZE; j++) {

                    //Criação da geração seguinte com o melhor pai
                    double[] parent = topEntity[0].getNode();


                    //Criação do filho através da junção dos melhores pais
                    double[] child = new double[parent.length];
                    for (int k = 0; k < parent.length; k++) {
                        child[k] = parent[k];
                    }

                    mutation(child);


                    //Nova NeuralNetwork com as mutações
                    NeuralNetwork nn = new NeuralNetwork(STATE_SIZE, STATE_SIZE, NUM_ACTIONS);
                    nn.setNode(child);
                    Board b = new Board(nn);
                    b.setSeed(5);
                    b.run();
                    nn.setFitness(b.getFitness());
                    population.add(nn);
                    population.sort(Comparator.comparing(NeuralNetwork::getFitness).reversed());






                    NeuralEntity ne = new NeuralEntity();
                    ne.setFitness(nn.getFitness());
                    ne.setNode(nn.getNode());
                    nePopulation[j] = ne;
                    Arrays.sort(nePopulation);

                    for (int k = 0; k < 25; k++) {
                        topEntity[k] = nePopulation[k];
                    }
                    Arrays.sort(topEntity);


                }
                NeuralNetwork nn = new NeuralNetwork(STATE_SIZE, STATE_SIZE, NUM_ACTIONS);
                double[] best = new double[topEntity[0].getNode().length];
                nn.setNode(best);
                Board b = new Board(nn);
                b.setSeed(5);
                b.run();
                nn.setFitness(b.getFitness());
                population.add(nn);
                population.sort(Comparator.comparing(NeuralNetwork::getFitness).reversed());
                bestFitness = topEntity[0].getFitness();
                bestPop.add(population.get(0));
            }

        }

        bestPop.sort(Comparator.comparing(NeuralNetwork::getFitness).reversed());
        //System.out.println(bestPop.toString() + "\n");
        highestFitnessPlay = bestPop.get(0);


        System.out.println("Melhor Fitness Score: " + bestFitness + "\n");
        System.out.println(bestPop.size());
        //System.out.println(highestFitnessPlay.getFitness());
        //SpaceInvaders.showControllerPlaying(highestFitnessPlay, 5);
    }


    private void mutation(double[] node) {

        for (int i = 0; i < node.length; i++) {
            if (Math.random() < MUTATION_RATE) {
                node[i] += (Math.random() - 0.5);
            }
        }
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
        for (NeuralNetwork nn : population) {
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

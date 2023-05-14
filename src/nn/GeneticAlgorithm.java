package nn;

import space.Board;
import space.SpaceInvaders;

import java.util.*;
import java.util.stream.Collectors;

import static space.Commons.*;

public class GeneticAlgorithm {

    private static final int POPULATION_SIZE = 100;
    private static final int GENERATIONS = 25;
    private static final double MUTATION_RATE = 0.1;
    private static Random random = new Random();
    public static List<NeuralNetwork> population = new ArrayList<>();
    public NeuralEntity[] nePopulation = new NeuralEntity[POPULATION_SIZE];
    public NeuralEntity[] topEntity = new NeuralEntity[25];
    public double fitness;
    public double bestFitness;


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
                    nn.setupBoard();
                    fitness = nn.getFitness();
                    population.add(nn);
                    double[] node = nn.getNode();


                    NeuralEntity ne = new NeuralEntity();
                    ne.setFitness(fitness);
                    ne.setNode(node);
                    nePopulation[j] = ne;


                }
                Arrays.sort(nePopulation);



            }else{

                //while( i >0 && i < GENERATIONS) {
                    // Seleciona os 25 melhores da geração anterior
                    for (int j = 0; j < 25; j++) {
                        topEntity[j] = nePopulation[j];
                    }


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
                        nn.setupBoard();
                        fitness = nn.getFitness();

                        population.add(nn);

                        NeuralEntity ne = new NeuralEntity();
                        ne.setFitness(fitness);
                        ne.setNode(child);
                        nePopulation[j] = ne;
                        Arrays.sort(nePopulation);

                        for (int k = 0; k < 25; k++) {
                            topEntity[k] = nePopulation[k];
                        }
                        Arrays.sort(topEntity);
                        bestFitness = topEntity[0].getFitness();
                        //population = nePopulation;


                    //}
                }
            }
        }
        System.out.println("População ne: " + nePopulation + "\n");
        System.out.println("TopList: " + topEntity + "\n");
        System.out.println("Melhor Fitness Score: " + bestFitness);
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

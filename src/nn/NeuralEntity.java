package nn;

public class NeuralEntity implements Comparable<NeuralEntity> {
    public double fitness;
    public double[] node;

    public NeuralEntity(){

    }

    public double getFitness() {
        return fitness;
    }

    public void setFitness(double fitness) {
        this.fitness = fitness;
    }

    @Override
    public int compareTo(NeuralEntity other) {
        return Double.compare(other.getFitness(), this.getFitness());
    }


    public double[] getNode() {
        return node;
    }

    public void setNode(double[] node) {
        this.node = node;
    }

    @Override
    public String toString(){
        int init = 0;
        int hid = 0;
        int out = 0;
        double[][] inwei = null;
        double[] hidbia = null;
        double[] outbia = null;
        double[][] outwei = null;
        double fit = 0.0;
        for (NeuralNetwork nn : GeneticAlgorithm.bestPop) {
            init = nn.getInputDim();
            hid = nn.getHiddenDim();
            out = nn.getOutputDim();
            inwei = nn.getInputWeights();
            hidbia = nn.getHiddenBiases();
            outbia = nn.getOutputBiases();
            outwei = nn.getOutputWeights();
        }
        fit = getFitness();

        return "Neural Network: " + init + "- Input Dim | " + hid + "- Hidden Dim | " + out + "- Output Dim | " + inwei + "- Input Weights | " + hidbia + "- Hidden Biases | " + outbia + "- Output Biases | " + outwei + "- Output Weights | " + fit + "- Fitness | \n";
    }
}

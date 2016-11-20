/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ffnn;

public class Layer {
    private Neuron[] neurons;

    //constructor for hidden and output layer
    public Layer(int n, int nprev) {
        neurons = new Neuron[n];
        for (int i=0 ; i<n ; i++) {
            neurons[i] = new Neuron(nprev);
        }
    }

    //constructor for input layer
    public Layer(int n) {
        neurons = new Neuron[n];
        for (int i=0 ; i<n ; i++) {
            neurons[i] = new Neuron();
        } 
    }

    public Neuron[] getNeurons() {
        return neurons;
    }

    //sets output for inputlayer
    public void setOutputs(double[] input) {
        assert(neurons.length == input.length);
        for (int i=0 ; i<neurons.length ; i++) {
            neurons[i].setOutput(input[i]);
        }
    }

    public void printLayer() {
        for (int i=0 ; i<neurons.length ; i++) {
            //System.out.println("Neuron " + i);
            neurons[i].printNeuron();
        }
    }
    
    public void printLayerNonDebug() {
        for (int i=0 ; i<neurons.length ; i++) {
            System.out.println("Neuron " + i);
            neurons[i].printNeuronNonDebug();
        }
    }

    
    public void generateOutput(Neuron[] prev) {
        for (int i=0 ; i<neurons.length ; i++) {
            neurons[i].calculateOutput(prev);        
        }
    }


    public void setOutputError(double[] desired) {
        for (int i=0 ; i<neurons.length ; i++) {
            neurons[i].calculateOutputError(desired[i]);
        }
    }

    public void setHiddenError(Neuron[] next) {
        for (int i=0 ; i<neurons.length ; i++) {
            neurons[i].calculateHiddenError(i, next);
        }
    }

    public void setNewWeight(Neuron[] prev) {
        for (int i=0 ; i<neurons.length ; i++) {
            neurons[i].calculateWeight(prev);
        }
    }

    public void generateErrorThreshold(double[] desired) {
        for (int i=0; i< neurons.length; i++) {
            neurons[i].calculateErrorThreshold(desired[i]);
        }
    }

    public double getErrorThreshold() {
        double errorThreshold = 0;
        for (int i=0; i< neurons.length; i++) {
            errorThreshold += neurons[i].getErrorThreshold();
        }
        return errorThreshold;
    }
}
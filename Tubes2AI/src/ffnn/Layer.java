/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ffnn;

import java.io.Serializable;

public class Layer implements Serializable{
    private Neuron[] neurons;

    //konstruktor untuk hidden dan output layer
    public Layer(int n, int nprev) {
        neurons = new Neuron[n];
        for (int i=0 ; i<n ; i++) {
            neurons[i] = new Neuron(nprev);
        }
    }

    //konstruktor untuk input layer
    public Layer(int n) {
        neurons = new Neuron[n];
        for (int i=0 ; i<n ; i++) {
            neurons[i] = new Neuron();
        } 
    }

    public Neuron[] getNeurons() {
        return neurons;
    }

    //sets output (untuk inputlayer pada khususnya)
    public void setOutputs(double[] input) {
        assert(neurons.length == input.length);
        for (int i=0 ; i<neurons.length ; i++) {
            neurons[i].setOutput(input[i]);
        }
    }
    
    public void printLayer() {
        for (int i=0 ; i<neurons.length ; i++) {            
            neurons[i].printNeuron();
        }
    }
    
    public void printLayerNonDebug() {
        for (int i=0 ; i<neurons.length ; i++) {
            System.out.println("Neuron " + i);
            neurons[i].printNeuronNonDebug();
        }
    }

    //menghitung output untuk seluruh neuron pada layer
    public void generateOutput(Neuron[] prev) {
        for (int i=0 ; i<neurons.length ; i++) {
            neurons[i].calculateOutput(prev);        
        }
    }

    //menghitung error untuk seluruh neuron output layer
    public void setOutputError(double[] desired) {
        for (int i=0 ; i<neurons.length ; i++) {
            neurons[i].calculateOutputError(desired[i]);
        }
    }

    //menghitung error untuk seluruh neuron hidden layer
    public void setHiddenError(Neuron[] next) {
        for (int i=0 ; i<neurons.length ; i++) {
            neurons[i].calculateHiddenError(i, next);
        }
    }

    //menghitung weight baru untuk seluruh neuron pada layer
    public void setNewWeight(Neuron[] prev) {
        for (int i=0 ; i<neurons.length ; i++) {
            neurons[i].calculateWeight(prev);
        }
    }

    //menghitung error threshold pada layer
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
        return errorThreshold / neurons.length;
    }
}
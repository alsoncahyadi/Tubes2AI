/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ffnn;

import java.io.Serializable;
import java.util.Random;

public class Neuron implements Serializable{

    private double output;      //output yang dihasilkan neuron
    private double[] weights;   //weight untuk setiap neuron pada layer sebelumnya
    private double error;       //error hasil perhitungan pada backpropagate
    private double bias;        //weight dari bias (value bias = 1)
    private double errorThreshold;  //error threshold un

    public Neuron() {

    }

    public Neuron(int nprev) {
        Random r = new Random();
        weights = new double[nprev];
        //inisiasi weight secara random antara -0.5 sampai 0.5
        for (int i = 0; i < nprev; i++) {
            weights[i] = r.nextDouble() - 0.5;
            weights[i] = (double) java.lang.Math.round(weights[i] * 100d) / 100d;
        }
        //inisiasi bias secara random antara -0.05 sampai 0.05
        bias = r.nextDouble() / 10 - 0.05;
        bias = (double) java.lang.Math.round(bias * 100d) / 100d;
        error = 0;
    }

    /* Getter Setter */
    public void setOutput(double out) {
        output = out;
    }

    public void setError(double err) {
        error = err;
    }

    public void setWeight(int i, double w) {
        weights[i] = w;
    }

    public void setBias(double b) {
        bias = b;
    }

    public void setErrorThreshold(double errt) {
        errorThreshold = errt;
    }

    public double getOutput() {
        return output;
    }

    public double getWeight(int i) {
        return weights[i];
    }

    public double getError() {
        return error;
    }

    public double getBias() {
        return bias;
    }

    public double getErrorThreshold() {
        return errorThreshold;
    }
    
    public void printNeuron() {
        
    }
    
    //print detil neuron
    public void printNeuronNonDebug() {
        System.out.println("Output : " + output);

        if (weights != null) {
            for (int i = 0; i < weights.length; i++) {
                System.out.println("Weight " + i + " : " + weights[i]);
            }
        } else {
            System.out.println("Weight null");
        }

        System.out.println("Error : " + error);

        System.out.println("Bias : " + bias);

        System.out.println("Error Threshold: " + errorThreshold);

        System.out.println();
    }

    //fungsi aktivasi sigmoid
    public double activationFunction(double sigma) {
        return 1 / (1 + java.lang.Math.pow(java.lang.Math.E, -sigma));
    }

    //menghitung output berdasarkan output dari neuron pada layer sebelumnya
    public void calculateOutput(Neuron[] prev) {
        double sigma = 0;
        for (int j = 0; j < prev.length; j++) {
            sigma += prev[j].getOutput() * getWeight(j);
        }
        sigma += getBias();
        setOutput(activationFunction(sigma));
    }

    //menghitung error untuk neuron pada output layer
    public void calculateOutputError(double desired) {
        double out = getOutput();
        double err = out * (1 - out) * (desired - out);
        setError(err);
    }

    //menghitung error untuk neuron pada hidden layer
    public void calculateHiddenError(int idx, Neuron[] next) {
        double out = getOutput();
        double sigma = 0;
        for (int i = 0; i < next.length; i++) {
            sigma += (next[i].getError() * next[i].getWeight(idx));
        }
        double err = out * (1 - out) * sigma;
        setError(err);
    }

    //menghitung dan set weight baru berdasarkan error
    public void calculateWeight(Neuron[] prev) {
        for (int j = 0; j < prev.length; j++) {
            double w = getWeight(j);
            w = w + getError() * prev[j].getOutput() * Cons.getLearningRate();   //LEARNING RATE!!!!
            setWeight(j, w);
        }
        double b = getBias();
        b = b + getError();
        setBias(b);
    }

    //menghitung error threshold
    public void calculateErrorThreshold(double desired) {
        double out = getOutput();
        double err = (desired - out) * (desired-out);
        setErrorThreshold(err); 
    }
}
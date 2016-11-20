/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ffnn;

import java.io.Serializable;
import java.util.Random;

public class Neuron implements Serializable{

    private double output;
    //private Link[] input;
    private double[] weights;
    private double error;
    private double bias;
    private double errorThreshold;

    private static final double learningrate = 0.01;

    public Neuron() {

    }

    public Neuron(int nprev) {
        Random r = new Random();
        weights = new double[nprev];
        for (int i = 0; i < nprev; i++) {
            weights[i] = r.nextDouble() / 10 - 0.05;
            weights[i] = (double) java.lang.Math.round(weights[i] * 100d) / 100d;
        }
        bias = r.nextDouble() / 10 - 0.05;
        bias = (double) java.lang.Math.round(bias * 100d) / 100d;
        error = 0;
    }

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
        //System.out.println("Output : " + output);

        if (weights != null) {
            for (int i = 0; i < weights.length; i++) {
                //System.out.println("Weight " + i + " : " + weights[i]);
            }
        } else {
            //System.out.println("Weight null");
        }

        //System.out.println("Error : " + error);

        //System.out.println("Bias : " + bias);

        //System.out.println("Error Threshold: " + errorThreshold);

        //System.out.println();
    }
    
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

    public double activationFunction(double sigma) {
        return 1 / (1 + java.lang.Math.pow(java.lang.Math.E, -sigma));
    }

    public void calculateOutput(Neuron[] prev) {
        double sigma = 0;
        for (int j = 0; j < prev.length; j++) {
            sigma += prev[j].getOutput() * getWeight(j);
        }
        sigma += getBias();
        setOutput(activationFunction(sigma));
    }

    public void calculateOutputError(double desired) {
        double out = getOutput();
        double err = out * (1 - out) * (desired - out);
        setError(err);
    }

    public void calculateHiddenError(int idx, Neuron[] next) {
        double out = getOutput();
        double sigma = 0;
        for (int i = 0; i < next.length; i++) {
            sigma += (next[i].getError() * next[i].getWeight(idx));
        }
        double err = out * (1 - out) * sigma;
        setError(err);
    }

    public void calculateWeight(Neuron[] prev) {
        for (int j = 0; j < prev.length; j++) {
            double w = getWeight(j);
            w = w + getError() * prev[j].getOutput() * learningrate;   //LEARNING RATE!!!!
            setWeight(j, w);
        }
        double b = getBias();
        b = b + getError();
        setBias(b);
    }

    public void calculateErrorThreshold(double desired) {
        double out = getOutput();
        double err = java.lang.Math.pow((desired - out), 2);
        setErrorThreshold(err); 
    }
}

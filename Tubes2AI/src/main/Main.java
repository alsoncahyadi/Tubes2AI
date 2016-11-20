/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package main;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import weka.core.Instances;
import ffnn.FeedForwardNN;
import weka.classifiers.Evaluation;

/**
 *
 * @author alson
 */
public class Main {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        BufferedReader breader = null;
        breader = new BufferedReader(new FileReader("C:\\Program Files\\Weka-3-8\\data\\iris.arff"));

        Instances ins = new Instances(breader);
        ins.setClassIndex(ins.attribute("class").index());

        breader.close();

        double[] dataTest0 = {5.1, 3.5, 1.4, 0.2};
        double[] dataTest1 = {7.0, 3.2, 4.7, 1.4};
        double[] dataTest2 = {6.3, 3.3, 6.0, 2.5};

        int numOutput = ins.numClasses();
        if (numOutput == 2) {
            numOutput = 1;
        }

        FeedForwardNN ffnn = new FeedForwardNN(ins.numAttributes() - 1, 1, numOutput);
        ffnn.buildClassifier(ins);
        System.out.println("Classify 0: " + ffnn.classify(dataTest0));
        System.out.println("Classify 1: " + ffnn.classify(dataTest1));
        System.out.println("Classify 2: " + ffnn.classify(dataTest2));

        
        Evaluation eval = new Evaluation(ins);
        eval.evaluateModel(ffnn, ins);
        //OUTPUT
        
        System.out.println(eval.toSummaryString("=== Stratified cross-validation ===\n" + "=== Summary ===", true));
        System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ==="));
        System.out.println(eval.toMatrixString("===Confusion matrix==="));
        System.out.println(eval.fMeasure(1) + " " + eval.recall(1));
    }
}

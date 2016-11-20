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
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import weka.classifiers.Classifier;
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
        //breader = new BufferedReader(new FileReader("C:\\Program Files\\Weka-3-8\\data\\iris.arff"));
        breader = new BufferedReader(new FileReader("C:\\Users\\alson\\Desktop\\tb2ai\\dataset\\team.arff"));

        Instances ins = new Instances(breader);
        //ins.setClassIndex(ins.attribute("class").index());
        ins.setClassIndex(ins.numAttributes() - 1);
        breader.close();

        int numOutput = ins.numClasses();
        if (numOutput == 2) {
            numOutput = 1;
        }
        int iterations = 100000;
        int nInputLayer = ins.numAttributes() - 1;
        int nHiddenLayer = 100;
        int nOutputLayer = numOutput;
        System.out.println("numOutput: " + numOutput);
        FeedForwardNN ffnn = new FeedForwardNN(nInputLayer, nHiddenLayer, nOutputLayer, iterations);
        ffnn.buildClassifier(ins);

        Evaluation eval = new Evaluation(ins);
        eval.evaluateModel(ffnn, ins);
        //OUTPUT

        System.out.println(eval.toSummaryString("=== Stratified cross-validation ===\n" + "=== Summary ===", true));
        System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ==="));
        System.out.println(eval.toMatrixString("===Confusion matrix==="));
        System.out.println(eval.fMeasure(1) + " " + eval.recall(1));
        
        System.out.println("SAVE MODEL");
        System.out.println("> Start Saving Model");
        String path = System.getProperty("user.home") + "/Desktop/tb2ai/classifier/" + ffnn.getnHid() + "-" + ffnn.getIterations() + ".clf";
        weka.core.SerializationHelper.write(path, ffnn);
        System.out.println("> Done Saving Model");
        System.out.println("> Model saved at: " + path);
    }
}

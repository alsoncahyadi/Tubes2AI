/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package main;

import ffnn.Cons;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import weka.core.Instances;
import ffnn.FeedForwardNN;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.text.DecimalFormat;
import java.util.Date;
import java.util.Scanner;
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
        Scanner sc = new Scanner(System.in);
        ////READ INSTANCES
        BufferedReader breader = null;
        String dataSetName = "team";
        //breader = new BufferedReader(new FileReader(System.getProperty("user.home") + "\\Desktop\\tb2ai\\dataset\\" + dataSetName + ".arff"));
        breader = new BufferedReader(new FileReader("D:\\Team.arff"));

        Instances ins = new Instances(breader);
        int clsIndex = ins.numAttributes() - 1;
        System.out.print("Insert class index: ");
        clsIndex = sc.nextInt();
        ins.setClassIndex(clsIndex);
        breader.close();

        ////CHECK IF BOOLEAN
        int numOutput = ins.numClasses();
        if (numOutput == 2) {
            numOutput = 1;
        }

        ////INITIALIZATION PARAMETERS
        int iterations = 200000;
        int nInputLayer = ins.numAttributes() - 1;
        int nHiddenLayer = 100;
        int nOutputLayer = numOutput;

        double learningRate = 0.3;
        double decreaseConst = 0.1;
        
        
        System.out.print("Input number of iterations: ");
        iterations = sc.nextInt();
        System.out.print("Input number of neurons in hidden layer: ");
        nHiddenLayer = sc.nextInt();
        System.out.print("Input learning rate: ");
        learningRate = sc.nextDouble();
        System.out.print("Input decrease constant: ");
        decreaseConst = sc.nextDouble();
        

        ////INITIALIZATION
        Cons.setLearningRate(learningRate);
        Cons.setDecreaseConst(decreaseConst);
        FeedForwardNN ffnn;
        if (nHiddenLayer == 0) {
            ffnn = new FeedForwardNN(nInputLayer, nOutputLayer, iterations);
        } else {
            ffnn = new FeedForwardNN(nInputLayer, nHiddenLayer, nOutputLayer, iterations);
        }
        ////LOAD OR BUILD
        Date start = new Date();
        ffnn.buildClassifier(ins);
        Date end = new Date();
        //String loadFileName = "team-100-200000-0.2";
        //ffnn = (FeedForwardNN) weka.core.SerializationHelper.read("C:\\Users\\alson\\Desktop\\tb2ai\\classifier\\" + loadFileName + ".model");

        //EVALUATE
        Evaluation eval = new Evaluation(ins);
        eval.evaluateModel(ffnn, ins);

        System.out.println(eval.toSummaryString("=== Stratified cross-validation ===\n" + "=== Summary ===", true));
        System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ==="));
        System.out.println(eval.toMatrixString("===Confusion matrix==="));
        System.out.println(eval.fMeasure(1) + " " + eval.recall(1));

        System.out.println("SAVE MODEL");
        System.out.println("> Start Saving Model");
        //String path = System.getProperty("user.home") + "/Desktop/tb2ai/classifier/" + dataSetName + "-" + ffnn.getnHid() + "-" + ffnn.getIterations() + "-" + learningRate + "-" + "acc" + ".model";
        String path = "D:\\Model\\" + dataSetName + "-" + ffnn.getnHid() + "-" + ffnn.getIterations() + "-" + learningRate + "-" + "acc" + ".model";
        weka.core.SerializationHelper.write(path, ffnn);
        System.out.println("> Done Saving Model");
        System.out.println("> Model saved at: " + path);

        DecimalFormat df = new DecimalFormat();
        df.setMaximumFractionDigits(2);
        System.out.println("TIME TAKEN: " + df.format((end.getTime() - start.getTime()) / 1000) + "s");
    }
}
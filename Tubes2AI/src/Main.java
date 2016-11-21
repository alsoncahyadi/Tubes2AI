import weka.core.Instance;
import weka.core.Instances;
import weka.core.DenseInstance;
import weka.core.converters.ArffLoader;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;

import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;


import java.io.File;
import java.util.*;

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

public class Main {
    private static Instances train; //dataset yang digunakan
    private static Classifier classifier;
    private static Evaluation eval;
    private static Discretize filter;

    public static void load(String filename) throws Exception {
        //membaca dataset dari file dengan nama filename
        ArffLoader loader = new ArffLoader();
        loader.setFile(new File(filename));
        train = loader.getDataSet();
        train.setClassIndex(train.numAttributes() - 1);
    }

    public static void discretize() throws Exception {
        //setup filter discretize
        filter = new Discretize();
        filter.setInputFormat(train);

        //apply discretize
        Instances filtered = Filter.useFilter(train, filter);
        train = filtered;
    }

    public static void naiveBayes() throws Exception {
        
		/*
		//train NaiveBayes
        classifier = new NaiveBayes();
        classifier.buildClassifier(train);
		*/
    }
	
    public static void ffnn() throws Exception {
        Scanner sc = new Scanner(System.in);
        System.out.print("Insert class index: ");
        int clsIndex = sc.nextInt();
        train.setClassIndex(clsIndex);


        ////CHECK IF BOOLEAN
        int numOutput = train.numClasses();
        if (numOutput == 2) {
            numOutput = 1;
        }

        ////INITIALIZATION PARAMETERS
        int iterations = 200000;
        int nInputLayer = train.numAttributes() - 1;
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
        if (nHiddenLayer == 0) {
            classifier = new FeedForwardNN(nInputLayer, nOutputLayer, iterations);
        } else {
            classifier = new FeedForwardNN(nInputLayer, nHiddenLayer, nOutputLayer, iterations);
        }
        classifier.buildClassifier(train);
    }

    public static void fulltraining() throws Exception {
        //melakukan pembelajaran dengan skema full training
        eval = new Evaluation(train);
        eval.evaluateModel(classifier, train);
    }

    public static void crossValidate() throws Exception {
        //melakukan pembelajaran dengan skema 10-fold cross validation
        eval = new Evaluation(train);
        eval.crossValidateModel(classifier, train, 10, new Random(1));
    }

    public static void printEvalResult() throws Exception {
        //mencetak hasil evaluation ke layar
        System.out.println(classifier);
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
    }

    public static void saveModel(String modelname) throws Exception {
        //menyimpan model hasil pembelajaran ke file eksternal
        String outname = "";
        outname = outname.concat(modelname);
        outname = outname.concat(".model");
        weka.core.SerializationHelper.write(outname, classifier);
    }


    public static void loadModel(String modelfile) throws Exception {
        //membaca model dari file eksternal
        classifier = (FeedForwardNN) weka.core.SerializationHelper.read(modelfile);
    }

    public static Instance makeNewInstance() {
        //membaca sebuah instance baru kemudian melakukan klasifikasi
        Scanner sc = new Scanner(System.in);
        int n = train.numAttributes();
        Instance newins = new DenseInstance(n);
        newins.setDataset(train);
        for (int i=0 ; i<n-1 ; i++) {
            System.out.print("Input attribute " + train.attribute(i).name() + ": ");
            float val = sc.nextFloat();
            newins.setValue(i, val);
        }
        return newins;
    }
    
    public static void classify(boolean discretized) throws Exception {
        Instance newins = makeNewInstance();
        if (discretized) {
            if (filter.input(newins)) {
                newins = filter.output();
            }
        }
        double result = classifier.classifyInstance(newins);
        System.out.println("Hasil klasifikasi = " + train.classAttribute().value((int) result));
   }

    public static void main(String args[]) throws Exception {
		System.out.println("-----------------------------------------");
		System.out.println("-                                       -");
		System.out.println("-    TUBES 2 Artificial Intelligence    -");
		System.out.println("-      Alson Elvina Alvin Michael       -");
        System.out.println("-                                       -");
		System.out.println("-----------------------------------------");
		System.out.println();
		System.out.print("Please input path to dataset : ");
        Scanner sc = new Scanner(System.in);
        String filename = sc.next();
		
        // load data
        load(filename);
        
        //memilih load atau pembelajaran baru
		System.out.println();
        System.out.println("Train new model or load model : 1.New   2.Load");
        System.out.print("Select mode : ");
        String mvar = sc.next();
        if ("1".equals(mvar)) {
            //setup filter
            System.out.println("Filter 0. None 1. Discretisize");
            System.out.print("Use filter : ");
            String fvar = sc.next();
            if ("1".equals(fvar)) {
                discretize();
            } 

            // train dataset
			System.out.println();
			System.out.println("Learning method : 1.NaiveBayes   2.FFNN");
			System.out.print("Select method : ");
			mvar = sc.next();
			
			if ("1".equals(mvar)) {
				naiveBayes();
			}
			else if ("2".equals(mvar)) {
				ffnn();
			}
			
            
        } 
        else if ("2".equals(mvar)) {
            //load model
			System.out.println();
            System.out.println("[Load model]");
            System.out.print("Model name : ");
            String mnvar = sc.next();
            loadModel(mnvar);
        }
        
        //use schema
		System.out.println();
        System.out.println("Schema 1. 10-fold Cross Validate 2. Full Training");
        System.out.print("Use schema : ");
        String svar = sc.next();
        if ("1".equals(svar)) {
            crossValidate();
        } 
        else if ("2".equals(svar)) {
            fulltraining();
        }

        //Mencetak hasil eval
        printEvalResult();

        //save model
		System.out.println();
        System.out.print("Want to save model?(Y/N) : ");
        String savevar = sc.next();
        if ("Y".equals(savevar)) {
                System.out.print("model name : ");
                String modelname = sc.next();
                saveModel(modelname);
                System.out.println("model saved");
        }

        //classify new instance
		System.out.println();
        System.out.print("Want to classify new instance?(Y/N) : ");
        String clvar = sc.next();
        if ("Y".equals(clvar)) {
			System.out.println();
            System.out.print("Want to discretize new instance?(Y/N) : ");
            String dvar = sc.next();
            boolean disc;
            disc = "Y".equals(dvar);
            classify(disc);
        }
    }
}
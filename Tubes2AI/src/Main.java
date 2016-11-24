
import weka.core.Instance;
import weka.core.Instances;
import weka.core.DenseInstance;
import weka.core.converters.ArffLoader;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;

import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

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
import naivebayes.NBayes;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.filters.unsupervised.attribute.Remove;

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
        //Filtering jadi nominal semua
        Instances disc_dataset = null;
        String[] options = new String[2];
        options[0]="-R";
        options[1]="first-last";
        
       
        //Discretize filtering
        Discretize disc = new Discretize();
        disc.setOptions(options);
        disc.setInputFormat(train);
        disc_dataset = Filter.useFilter(train, disc);
        
        train = disc_dataset;
    }

    public static void naiveBayes() throws Exception {
        Scanner sc = new Scanner(System.in);
        System.out.print("Insert class index: ");
        int clsIndex = sc.nextInt();
        train.setClassIndex(clsIndex);
        System.out.println("Selected class attribute: " + train.attribute(train.classIndex()));

        System.out.print("Do you want remove attribute ? (Y/N) ");
        String rvar = sc.next();
        Instances newDataset = null;
        if ("Y".equals(rvar)) {
            Remove remove = new Remove();
            System.out.print("Remove attribute : ");
            String atr = sc.next();
            remove.setAttributeIndices(atr);
            int a = Integer.parseInt(atr);
            if(a<=clsIndex) {
                clsIndex -= 1;
            }
            remove.setInvertSelection(false);
            remove.setInputFormat(train);
            newDataset = Filter.useFilter(train, remove);
            newDataset.setClassIndex(clsIndex);
        }
        train = newDataset;

        classifier = new NBayes();
        classifier.buildClassifier(train);
    }

    public static FeedForwardNN ffnn() throws Exception {
        Scanner sc = new Scanner(System.in);
        System.out.print("Insert class index: ");
        int clsIndex = sc.nextInt();
        train.setClassIndex(clsIndex);
        System.out.println("Selected class attribute: " + train.attribute(train.classIndex()));

        //DELETE 26 or 27
        if (clsIndex == 26) {
            System.out.println("> Removing index 27:" + train.attribute(27).name());
            Remove remove = new Remove();
            remove.setAttributeIndices("28");
            remove.setInputFormat(train);
            train = Filter.useFilter(train, remove);
            System.out.println("> Index 27 deleted");
        } else if (clsIndex == 27) {
            System.out.println("> Removing index 26:" + train.attribute(26).name());
            Remove remove = new Remove();
            remove.setAttributeIndices("27");
            remove.setInputFormat(train);
            train = Filter.useFilter(train, remove);
            System.out.println("> Index 26 deleted");
        }

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
        System.out.print("Input number of hidden layer: ");
        if (sc.nextInt() == 1) {
            System.out.print("Input number of neurons in hidden layer: ");
            nHiddenLayer = sc.nextInt();
        }
        else {
            nHiddenLayer = 0;
        }
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
        return (FeedForwardNN) classifier;
    }

    public static FeedForwardNN ffnn(int clsIndex, int iter, int hid, double lr, double dc) throws Exception {

        ////CHECK IF BOOLEAN
        int numOutput = train.numClasses();
        if (numOutput == 2) {
            numOutput = 1;
        }

        ////INITIALIZATION PARAMETERS
        int iterations = iter;
        int nInputLayer = train.numAttributes() - 1;
        int nHiddenLayer = hid;
        int nOutputLayer = numOutput;

        double learningRate = lr;
        double decreaseConst = dc;

        ////INITIALIZATION
        Cons.setLearningRate(learningRate);
        Cons.setDecreaseConst(decreaseConst);
        if (nHiddenLayer == 0) {
            classifier = new FeedForwardNN(nInputLayer, nOutputLayer, iterations);
        } else {
            classifier = new FeedForwardNN(nInputLayer, nHiddenLayer, nOutputLayer, iterations);
        }
        classifier.buildClassifier(train);
        return (FeedForwardNN) classifier;
    }

    public static void batchFfnn(int batchSize, Instances test) throws Exception {
        //ASSUMPTION: ErrT never exceeds 100
        double minErrT = 100;
        FeedForwardNN bestFfnn = null;
        FeedForwardNN currFfnn = null;
        double[] errT = new double[batchSize];

        Scanner sc = new Scanner(System.in);

        System.out.print("Input class index: ");
        int clsIndex = sc.nextInt();

        //SET CLASS INDEX
        train.setClassIndex(clsIndex);
        System.out.println("Selected class attribute: " + train.attribute(train.classIndex()));

        //DELETE 26 or 27
        if (clsIndex == 26) {
            System.out.println("> Removing index 27:" + train.attribute(27).name());
            Remove remove = new Remove();
            remove.setAttributeIndices("28");
            remove.setInputFormat(train);
            train = Filter.useFilter(train, remove);
            System.out.println("> Index 27 deleted");
        } else if (clsIndex == 27) {
            System.out.println("> Removing index 26:" + train.attribute(26).name());
            Remove remove = new Remove();
            remove.setAttributeIndices("27");
            remove.setInputFormat(train);
            train = Filter.useFilter(train, remove);
            System.out.println("> Index 26 deleted");
        }

        System.out.print("Input number of iterations: ");
        int iterations = sc.nextInt();
        System.out.print("Input number of neurons in hidden layer: ");
        int nHiddenLayer = sc.nextInt();
        System.out.print("Input learning rate: ");
        double learningRate = sc.nextDouble();
        System.out.print("Input decrease constant: ");
        double decreaseConst = sc.nextDouble();

        for (int i = 0; i < batchSize; i++) {
            System.out.println("*****************************");
            System.out.println("*********" + i + " iteration **********");
            System.out.println("*****************************");
            System.out.println("");

            currFfnn = ffnn(clsIndex, iterations, nHiddenLayer, learningRate, decreaseConst);

            ////TEST the datatest
            int numInstances = test.numInstances();

            double[][] in = new double[numInstances][test.numAttributes() - 1];
            double[][] out = new double[numInstances][test.numClasses()];
            double[] classes = null;

            //READ OUTPUTS
            System.out.println("READ OUTPUTS");
            System.out.println("> Start Reading Outputs");
            classes = test.attributeToDoubleArray(test.classIndex());
            for (int j = 0; j < classes.length; j++) {
                out[j][(int) classes[j]] = 1.0;
            }
            System.out.println("> Done Reading Outputs");

            //READ INPUTS
            System.out.println("READ INPUTS");
            System.out.println("> Start Reading Inputs");
            for (int k = 0; k < in.length; k++) {
                int j = 0;
                int cnt = 0;
                while (cnt <= in[k].length) {
                    if (cnt != test.classIndex()) {
                        in[k][j] = test.instance(k).value(cnt);
                        j++;
                    }
                    cnt++;
                }
            }
            System.out.println("> Done Reading Inputs");
            
            //CALCULATE TEST ERROR
            double sumErrorThreshold = 0;
            for (int j = 0; j < test.numInstances(); j++) {
                currFfnn.feedForward(in[j]);
                currFfnn.generateErrorThreshold(out[j]);
                sumErrorThreshold += currFfnn.getErrorThreshold();
            }
            double currErrorThreshold = (sumErrorThreshold / test.numInstances());
            
            System.out.println(">> Err Threshold Test(" + i + "): " + currErrorThreshold);
            
            //STORE ERROR
            errT[i] = currErrorThreshold;
            if (errT[i] < minErrT) {
                minErrT = errT[i];
                bestFfnn = currFfnn;
            }
        }
        System.out.println("> Errors in batch:");
        for (int i = 0; i < batchSize; i++) {
            System.out.println("   >" + i + ": " + errT[i]);
        }
        classifier = bestFfnn;
        System.out.println("Best ET in batch: " + minErrT);
    }

    public static void fulltraining(Instances test) throws Exception {
        //melakukan pembelajaran dengan skema full training
        eval = new Evaluation(test);
        eval.evaluateModel(classifier, test);
    }

    public static void crossValidate(Instances test) throws Exception {
        //melakukan pembelajaran dengan skema 10-fold cross validation
        eval = new Evaluation(test);
        eval.crossValidateModel(classifier, test, 10, new Random(1));
    }

    public static void splitTest(Instances test, int a) throws Exception {
        train.randomize(new java.util.Random(0));
        int trainSize = (int) Math.round(test.numInstances()*a/100);
        int testSize = test.numInstances() - trainSize;
        Instances training = new Instances(test, 0, trainSize);
        Instances testing = new Instances(test, trainSize, testSize);
        
        classifier.buildClassifier(training);
        
        eval = new Evaluation(training);
        eval.evaluateModel(classifier,testing);
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
        classifier = (Classifier) weka.core.SerializationHelper.read(modelfile);
    }

    public static Instance makeNewInstance() {
        //membaca sebuah instance baru kemudian melakukan klasifikasi
        Scanner sc = new Scanner(System.in);
        int n = train.numAttributes();
        Instance newins = new DenseInstance(n);
        newins.setDataset(train);
        for (int i = 0; i < n; i++) {
            if (i != train.classIndex()) {
                System.out.print("Input attribute " + train.attribute(i).name() + ": ");
                float val = sc.nextFloat();
                newins.setValue(i, val);
            }
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

        //READ DATATEST
        System.out.print("Path to test dataset: ");
        filename = sc.next();
        ArffLoader loader = new ArffLoader();
        loader.setFile(new File(filename));
        Instances test = loader.getDataSet();
        test.setClassIndex(test.numAttributes() - 1);
        System.out.print("Insert class index: ");
        int clsIndex = sc.nextInt();
        test.setClassIndex(clsIndex);
        
        //start to descretize
        Instances disc_dataset = null;
        String[] options = new String[2];
        options[0]="-R";
        options[1]="first-last";
       
        //Discretize filtering
        filter = new Discretize();
        filter.setOptions(options);
        filter.setInputFormat(test);
        disc_dataset = Filter.useFilter(test, filter);
        
        test = disc_dataset;
        test.setClassIndex(clsIndex);
        
        System.out.println("Selected class attribute: " + test.attribute(test.classIndex()));

        System.out.print("Do you want remove attribute ? (Y/N) ");
        String rvar = sc.next();
        Instances newDataset = test;
        if ("Y".equals(rvar)) {
            Remove remove = new Remove();
            System.out.print("Remove attribute : ");
            String atr = sc.next();
            remove.setAttributeIndices(atr);
            int a = Integer.parseInt(atr);
            if(a<=clsIndex) {
                clsIndex -= 1;
            }
            remove.setInvertSelection(false);
            remove.setInputFormat(test);
            newDataset = Filter.useFilter(test, remove);
            newDataset.setClassIndex(clsIndex);
        }
        test = newDataset;
        
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
            } else if ("2".equals(mvar)) {
                System.out.println("Batch or Not: 1.Batch   2.Not Batch");
                System.out.print("Select option: ");
                mvar = sc.next();
                if ("1".equals(mvar)) {
                    System.out.print("Insert Number of batch size: ");
                    int batchSize = sc.nextInt();
                    batchFfnn(batchSize, test);
                } else if ("2".equals(mvar)) {
                    ffnn();
                }
            }

        } else if ("2".equals(mvar)) {
            //load model
            System.out.println();
            System.out.println("[Load model]");
            System.out.print("Model name : ");
            String mnvar = sc.next();
            loadModel(mnvar);
        }

        //use schema
        System.out.println();

        
        System.out.println("Schema 1. 10-fold Cross Validate 2. Full Training 3. Split Test");
        System.out.print("Use schema : ");
        String svar = sc.next();
        if ("1".equals(svar)) {
            crossValidate(test);
        } else if ("2".equals(svar)) {
            fulltraining(test);
        }
        else if ("3".equals(svar)) {
            System.out.print("Jumlah percentase yang diinginkan : ");
            int percent = sc.nextInt();
            splitTest(test, percent);
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

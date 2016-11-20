/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ffnn;

import weka.classifiers.AbstractClassifier;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

public class FeedForwardNN extends AbstractClassifier {

    private Layer inputLayer;
    private Layer outputLayer;
    private Layer hiddenLayer;
    private double errorThreshold;
    private double[] desiredOutputs;
    private Normalize normalize = new Normalize();
    private Instances lastBuiltInstances;
    private int classIndex;


    /*public FeedForwardNN(int in, int out, int nhid, int hid) {
        inputLayer = new Layer(in);
        outputLayer = new Layer(out, in);
        if (nhid > 0) {
            hidenLayer = new Layer[nhid];
            for (int i=0 ; i<nhid ; i++) {
                hiddenLayer[i] = new Layer(hid);
            }
        }
    }*/
    //constructor for 0 hidden layer
    public FeedForwardNN(int in, int out, int classIndex) {
        inputLayer = new Layer(in);
        outputLayer = new Layer(out, in);
        this.classIndex = classIndex;
    }

    //constructor for 1 hidden layer
    public FeedForwardNN(int in, int hid, int out, int classIndex) {
        inputLayer = new Layer(in);
        hiddenLayer = new Layer(hid, in);
        outputLayer = new Layer(out, hid);
        this.classIndex = classIndex;
    }

    public Layer getOutputLayer() {
        return outputLayer;
    }

    public double getErrorThreshold() {
        return errorThreshold;
    }

    //hitung output untuk setiap neuron dengan input inputs    
    public void feedForward(double[] inputs) {
        //System.out.println("====== Feed Forward ======");
        inputLayer.setOutputs(inputs);
        //System.out.println("=== Input layer ===");
        inputLayer.printLayer();
        if (hiddenLayer == null) {
            outputLayer.generateOutput(inputLayer.getNeurons());
            //System.out.println("=== Output layer ===");
            outputLayer.printLayer();
        } else {
            hiddenLayer.generateOutput(inputLayer.getNeurons());
            //System.out.println("=== Hidden layer ===");
            hiddenLayer.printLayer();
            outputLayer.generateOutput(hiddenLayer.getNeurons());
            //System.out.println("=== Output layer ===");
            outputLayer.printLayer();
        }
    }

    //update weight berdasarkan output yang sudah dihasilkan dan target adalah outputs
    //harus melakukan feedforward dulu untuk menghitung output
    public void backPropagate(double[] outputs) {
        //System.out.println("====== Back Propagate ======");
        desiredOutputs = outputs;
        outputLayer.setOutputError(desiredOutputs);
        //System.out.println("=== Output layer ===");
        outputLayer.printLayer();
        if (hiddenLayer == null) {
            outputLayer.setNewWeight(inputLayer.getNeurons());
            //System.out.println("=== Output layer updated ===");
            outputLayer.printLayer();
        } else {
            hiddenLayer.setHiddenError(outputLayer.getNeurons());
            //System.out.println("=== Hidden layer ===");
            hiddenLayer.printLayer();
            outputLayer.setNewWeight(hiddenLayer.getNeurons());
            //System.out.println("=== Output layer updated ===");
            outputLayer.printLayer();
            hiddenLayer.setNewWeight(inputLayer.getNeurons());
            //System.out.println("=== Hidden layer updated ===");
            hiddenLayer.printLayer();
        }
    }

    //untuk pembelajaran: ulang feedforward dan backpropagate untuk setiap instance, berhenti ketika ?
    public int classify(double[] inputs) {
        assert (inputs.length == inputLayer.getNeurons().length);
        feedForward(inputs);
        double[] result = new double[outputLayer.getNeurons().length];

        int ret;
        if (outputLayer.getNeurons().length == 1) {
            if (outputLayer.getNeurons()[0].getOutput() > 0.5) {
                ret = 0;
            } else {
                ret = 1;
            }
        } else {
            double max = 0;
            int maxidx = 0;
            int i;
            for (i = 0; i < result.length; i++) {
                result[i] = outputLayer.getNeurons()[i].getOutput();
                //System.out.print("res[" + i + "]: " + result[i] + ", ");
                if (result[i] > max) {
                    max = result[i];
                    maxidx = i;
                }
            }
            ret = maxidx;
        }
        return ret;
    }

    public void generateErrorThreshold(double[] outputs) {
        //System.out.println("====== Calculate Error Threshold on Output Layer ======");
        desiredOutputs = outputs;
        outputLayer.generateErrorThreshold(outputs);
        errorThreshold = outputLayer.getErrorThreshold();
    }

    @Override
    public void buildClassifier(Instances insNonNormalized) throws Exception {

        //System.out.println("INITIAL:");
        //outputLayer.printLayerNonDebug();
        //NORMALIZE
        System.out.println("NORMALIZE INSTANCES");
        System.out.println("> Start Normalizing");
        normalize.setInputFormat(insNonNormalized);
        Instances ins = Filter.useFilter(insNonNormalized, normalize);
        System.out.println("> Done Normalizing");

        int numInstances = ins.numInstances();

        double[][] in = new double[numInstances][ins.numAttributes()];
        double[][] out = new double[numInstances][ins.numClasses()];
        System.out.println(ins.numAttributes());
        double[] classes = null;

        //READ ALGORITHM
        System.out.println("READ OUTPUTS");
        System.out.println("> Start Reading Outputs");
        classes = ins.attributeToDoubleArray(classIndex);
        for (int i = 0; i < classes.length; i++) {
            out[i][(int) classes[i]] = 1.0;
        }
        System.out.println("> Done Reading Outputs");

        //OUTPUT OUTPUTS
        System.out.println("OUTPUTS:");
        for (int i = 0; i < out.length; i++) {
            System.out.print("  " + (i + 73) + ") ");
            for (int j = 0; j < out[i].length; j++) {
                System.out.print(out[i][j] + " | ");
            }
            System.out.println("");
        }

        //READ INPUTS
        System.out.println("READ INPUTS");
        System.out.println("> Start Reading Inputs");
        for (int i = 0; i < in.length; i++) {
            int j = 0;
            int cnt = 0;
            while (cnt < in[i].length) {
                if (cnt != classIndex) {
                    //if (ins.instance(i).attribute(cnt).name() != "class") {
                    in[i][j] = ins.instance(i).value(cnt);
                    j++;
                }
                cnt++;
            }
        }
        System.out.println("> Done Reading Inputs");

        //OUTPUT INPUTS
        System.out.println("INPUTS: ");
        for (int i = 0; i < in.length; i++) {
            System.out.print("  " + (i + 73) + ") ");
            for (int j = 0; j < in[i].length; j++) {
                System.out.print(in[i][j] + " | ");
            }
            System.out.println("");
        }

        //DOING ANN
        for (int i = 0; i < 50000; i++) {
            for (int j = 0; j < ins.numInstances(); j++) {
                this.feedForward(in[j]);
                this.backPropagate(out[j]);
                this.generateErrorThreshold(out[j]);
            }
            if (i % 5000 == 0) {
                System.out.println("ET-" + i + ": " + getErrorThreshold());
            }
        }

        outputLayer.printLayerNonDebug();

        //COBA CLASSIFY
        System.out.println("CLASSIFY: ");
        for (int i = 0; i < in.length; i++) {
            System.out.println("> Classify-" + (i + 73) + ": " + classify(in[i]));
        }

        System.out.println("NORMALIZE INFO");
        System.out.println("Max Array: " + normalize.getMaxArray().toString());
        System.out.println("Min ARray: " + normalize.getMinArray().toString());
        System.out.println("Scale: " + normalize.getScale());
        System.out.println("Translation: " + normalize.getTranslation());
        System.out.println(normalize.getRevision());
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        /*int nAttr = instance.numAttributes();
        for (int i = 0; i < instance.numAttributes(); i++) {
            if (instance.attribute(i).name().equals("class")) {
                nAttr--;
            }
        }
        double[] dataTest = new double[nAttr];

        int count = 0;
        int j = 0;
        while (count < nAttr) {
            if (!instance.attribute(count).name().equals("class")) {
                dataTest[j] = instance.value(count);
                j++;
            };
            count++;
        }*/
        if (normalize.input(instance)) {
            instance = normalize.output();
            double[] dataTest = new double[instance.numAttributes()];
            for (int i = 0; i < instance.numAttributes(); i++) {
                dataTest[i] = instance.value(i);
            }
            return (double) classify(dataTest);
        } else {
            return -1;
        }
    }
    /*
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return null;
    }*/
}

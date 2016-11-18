public class FeedForwardNN {
    private Layer inputLayer;
    private Layer outputLayer;
    private Layer hiddenLayer;
    private double[] desiredOutputs;


    
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
    public FeedForwardNN(int in, int out) {
        inputLayer = new Layer(in);
        outputLayer = new Layer(out, in);
    }

    //constructor for 1 hidden layer
    public FeedForwardNN(int in, int hid, int out) {
        inputLayer = new Layer(in);
        hiddenLayer = new Layer(hid, in);
        outputLayer = new Layer(out, hid);
    }    

    public Layer getOutputLayer() {
        return outputLayer;
    }

    //hitung output untuk setiap neuron dengan input inputs    
    public void feedForward(double[] inputs) {
        System.out.println("====== Feed Forward ======");
        inputLayer.setOutputs(inputs);
        System.out.println("=== Input layer ===");
        inputLayer.printLayer();
        if (hiddenLayer == null) {
            outputLayer.generateOutput(inputLayer.getNeurons());
            System.out.println("=== Output layer ===");
            outputLayer.printLayer();
        } else {
            hiddenLayer.generateOutput(inputLayer.getNeurons());
            System.out.println("=== Hidden layer ===");
            hiddenLayer.printLayer();
            outputLayer.generateOutput(hiddenLayer.getNeurons());
            System.out.println("=== Output layer ===");
            outputLayer.printLayer();
        }
    }

    //update weight berdasarkan output yang sudah dihasilkan dan target adalah outputs
    //harus melakukan feedforward dulu untuk menghitung output
    public void backPropagate(double[] outputs) {
        System.out.println("====== Back Propagate ======");
        desiredOutputs = outputs;
        outputLayer.setOutputError(desiredOutputs);
        System.out.println("=== Output layer ===");
        outputLayer.printLayer();
        if (hiddenLayer == null) {
            outputLayer.setNewWeight(inputLayer.getNeurons());
            System.out.println("=== Output layer updated ===");
            outputLayer.printLayer();
        } else {
            hiddenLayer.setHiddenError(outputLayer.getNeurons());
            System.out.println("=== Hidden layer ===");
            hiddenLayer.printLayer();
            outputLayer.setNewWeight(hiddenLayer.getNeurons());
            System.out.println("=== Output layer updated ===");
            outputLayer.printLayer();
            hiddenLayer.setNewWeight(inputLayer.getNeurons());
            System.out.println("=== Hidden layer updated ===");
            hiddenLayer.printLayer();
        }
    }

    //untuk pembelajaran: ulang feedforward dan backpropagate untuk setiap instance, berhenti ketika ?

    public int classify(double[] inputs) {
        assert(inputs.length == inputLayer.getNeurons().length);
        feedForward(inputs);
        double[] result = new double[outputLayer.getNeurons().length];
        double max = 0;
        int maxidx = 0;
        int i;
        for (i=0 ; i<result.length ; i++) {
            result[i] = outputLayer.getNeurons()[i].getOutput();
            if (result[i] > max) {
                max = result[i];
                maxidx = i;
            }
        }
        return maxidx;
    }


}
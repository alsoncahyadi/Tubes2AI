public class Main {
    public static void main(String[] args) {
        FeedForwardNN f = new FeedForwardNN(3,2,2);
        double[] in = {1.0, 2.0, 3.0};
        double[] out = {1, 0};
        for (int i=0 ; i<100 ; i++) {
            f.feedForward(in);
            f.backPropagate(out);
            //f.feedForward(in);
        }

        System.out.println("Classify : " +f.classify(in));
    }
}
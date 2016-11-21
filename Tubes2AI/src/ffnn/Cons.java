/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ffnn;

/**
 *
 * @author alson
 */
public class Cons {
    private static double learningRate = 0.1;
    private static double decreaseConst = 0.0001;
    
    public static double getLearningRate() {
        return learningRate;
    }

    public static void setLearningRate(double learningRate) {
        Cons.learningRate = learningRate;
    }

    public static double getDecreaseConst() {
        return decreaseConst;
    }

    public static void setDecreaseConst(double decreaseConst) {
        Cons.decreaseConst = decreaseConst;
    }
    
    public static double calculateScale(int i, int iterations) {
        final int steepnessConst = 1;
        return (1 + Cons.getDecreaseConst() * i * steepnessConst / iterations);
    }
}

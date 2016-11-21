/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package naivebayes;

import java.util.Vector;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.*;

public class NBayes extends AbstractClassifier {
    Vector<Data> data = new Vector();
    int classidx = 0;
    static double[] prob;
    @Override
    public void buildClassifier(Instances ins) throws Exception {
        
         //Mendelete instance dengan atribut missing
            for (int i = 0; i<ins.numAttributes(); i++) {
                ins.deleteWithMissing(i);
            }
            ins.deleteWithMissingClass();   // Mendelete instance dengan missing class

            //Klo numeric convert ke nominal
            if (ins.classAttribute().isNumeric()) {
                System.out.println("It's Numeric!");
                Discretize filter = new Discretize();

                filter.setInputFormat(ins);
                ins = Filter.useFilter(ins, filter);
            }
            
            //inisiasi Vektor
            for (int i = 0; i<ins.numAttributes();i++) {
                for(int j = 0; j<ins.attribute(i).numValues(); j++) {
                    int k = 0;
                    while (k<ins.numClasses()) {
                       Data datum = new Data(i, j, k, 0);
                       data.add(datum);
                       k++; 
                    }
                }
            }
            //menghitung freq dari setiap data pada Vector yang ada pada instance
            for(int i = 0; i<ins.numInstances(); i++) {
                for (int j=0; j<ins.numAttributes(); j++) {
                    int index = getIndex(data, j, ins.instance(i).attribute(j).indexOfValue(ins.instance(i).toString(j)), ins.instance(i).attribute(ins.classIndex()).indexOfValue(ins.instance(i).toString(ins.classIndex())));
                    data.get(index).setfreq(data.get(index).getfreq() + 1);
                }
            }
            
            classidx = ins.classIndex();
            int a=0;
            if(classidx>0) {
                for (int i =0; i<classidx; i++) {
                    a += ins.attribute(i).numValues();
                }
            }
            for (int i = classidx; i<ins.numClasses(); i++) {
                while (data.get(i).getfreq() == 0) {
                    data.remove(i);
                }
            }        

            //Menghitung total pembagi setiap attribut
            for (int i = 0; i < data.size(); i++) {
                for (int j = classidx; j<ins.classIndex(); j++) {
                    if(data.get(i).getResult() == data.get(j).getResult() && data.get(j).getResult() == data.get(j).getValue()) {
                        data.get(i).setTotal(data.get(j).getfreq());
                    }
                }
            }
            
            prob = new double[data.size()];
        
            for(int i=0; i<data.size(); i++) {
                if(data.get(i).getcolumn() == 0) {
                    prob[i] = data.get(i).getfreq() / ins.numInstances();
                }
                else {
                    prob[i] = data.get(i).getfreq() / data.get(i).getTotal();
                }   
            }
    }
    
    public double classifyInstance(Instance tes) {
        double [] probMax = new double [tes.numClasses()];
        int idx = 0;
            
        for (int k = 0; k<tes.numClasses(); k++) {
            if (tes.classIndex() == 0) {
                for(int i=0; i<tes.numClasses(); i++) {
                    probMax[i] = prob[i];
                }
            }
            else {
                int a = 0;
                for (int i =0; i<classidx; i++) {
                    a += tes.attribute(i).numValues();
                }
                for(int i = 0; i<tes.numClasses(); i++) {
                    probMax[i] = prob[a];
                    a++;
                }

            }
            for(int j = 0; j<tes.numAttributes();j++) {
                if(j!=tes.classIndex()) {
                    idx = getIndex(data, j, tes.attribute(j).indexOfValue(tes.toString(j)), tes.attribute(k).indexOfValue(tes.attribute(k).value(k)));
                    probMax[k] *= prob[idx];
                }
            }
        }
            
        
        int x = searchMax(probMax);
        
        return x;
        
    }
    
     public static int searchMax(double [] array) {
        int imax = 0;
        for (int i=0; i<array.length; i++) {
            if(array[i] > array[imax]) {
                imax = i;
            }
        }
        
        return imax;
    }
     
     public static double[] initializeProbClass(Instances dataset) {
        double [] probClass = new double [dataset.numClasses()];
        int a = dataset.classIndex();
        int ix = 0;
        if(a==0) {
            for (int i=0; i<dataset.numClasses(); i++) {
                probClass[i] = prob[i];
            }
        }
        else {
            for (int i =0; i<a; i++) {
                a += dataset.attribute(i).numValues();
            }
            for (int i= 0; i< dataset.numClasses(); i++) {
                probClass[i] = prob[a];
                System.out.println("a : " + a);
                a++;
            }
        }
        
        return probClass;
    }
    
    public static int getIndex(Vector<Data> input, int attr, int nilaiAttr, int namaKls) {
        int i = 0;
        int index=9999;
        while(i<input.size()) {
            if(input.get(i).getValue() == nilaiAttr && input.get(i).getResult() == namaKls && input.get(i).getcolumn() == attr) {
                index=i;
                i=9999;
            }
            else
                i++;
        }
        return index;
    }
    

    
    ////Probability Table
    // 1st => attribute
    // 2nd => num values
    // 3rd => class
    /*
    private double[][][] probT;
    private double[] classes;
    private int classIndex = 0;

    @Override
    public void buildClassifier(Instances ins) throws Exception {
        classIndex = ins.classIndex();
        int maxNumVal = 0;
        
        for (int i = 0; i < ins.numAttributes(); i++) {
            if (maxNumVal < ins.attribute(i).numValues()) {
                maxNumVal = ins.attribute(i).numValues();
            }
        }

        probT = new double[ins.numAttributes() - 1][maxNumVal][ins.numClasses()];
        classes = new double[ins.numClasses()];

        for (int i = 0; i < ins.numInstances(); i++) {
            for (int j = 0; j < ins.numAttributes(); j++) {
                
            }
        }
    }*/
}

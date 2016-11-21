
import java.util.Enumeration;
import java.util.Scanner;
import java.util.Vector;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.*;

public class Main {
    
    static double[] prob;
    
    public static void main(String args[]) {
        // Baca File nya
        Vector<Data> data = new Vector();
        Instances newDataset=null;
        try {
            //load dataset
            ConverterUtils.DataSource source = new ConverterUtils.DataSource("C:\\Users\\User\\Documents\\Elvina\\Semester 5\\AI\\tubes\\mush.arff");
            Instances dataset = source.getDataSet();
            int i = 0;
            boolean found = false;
            while (i<dataset.numAttributes() && ! found) {
                if (dataset.attribute(i).name().equals("class")) {
                    dataset.setClassIndex(i);
                    found = true;
                }
                else
                    i++;
            }
            
            //Mendelete instance dengan atribut missing
            for (i = 0; i < dataset.numAttributes(); i++) {
                dataset.deleteWithMissing(i);
            }
            dataset.deleteWithMissingClass();   // Mendelete instance dengan missing class
            newDataset = new Instances(dataset);

            //Klo numeric convert ke nominal
            if (dataset.classAttribute().isNumeric()) {
                System.out.println("It's Numeric!");
                Discretize filter = new Discretize();

                filter.setInputFormat(dataset);
                newDataset = Filter.useFilter(dataset, filter);
            }
            
            //inisiasi Vektor
            for (i = 0; i<newDataset.numAttributes();i++) {
                for(int j = 0; j<newDataset.attribute(i).numValues(); j++) {
                    int k = 0;
                    while (k<newDataset.attribute(newDataset.classIndex()).numValues()) {
                       Data datum = new Data(newDataset.attribute(i).name(), newDataset.attribute(i).value(j), newDataset.attribute(newDataset.classIndex()).value(k), 0);
                       data.add(datum);
                       k++; 
                    }
                }
            }
            //menghitung freq dari setiap data pada Vector yang ada pada instance
            for(i = 0; i<newDataset.numInstances(); i++) {
                for (int j=0; j<newDataset.numAttributes(); j++) {
                    Data datum = new Data(newDataset.attribute(j).name(),newDataset.instance(i).toString(j), newDataset.instance(i).toString(0),1);
                    int index = getIndex(data, newDataset.attribute(j).name(), newDataset.instance(i).toString(j), newDataset.instance(i).toString(0));
                    data.get(index).setfreq(data.get(index).getfreq() + datum.getfreq());
                }
            }
        } catch (Exception e) {
            System.out.println("Exception: " + e.getMessage());
        }
        
        //Menghapus nilai kls yg salah pada data model
        int a = newDataset.classIndex();
        if(a>0) {
            for (int i =0; i<a; i++) {
                a += newDataset.attribute(i).numValues();
            }
        }
        for (int i = a; i<Math.pow(newDataset.numClasses(), 2); i++) {
            while (data.get(i).getfreq() == 0) {
                data.remove(i);
            }
        }        
        
        //Menghitung total pembagi setiap attribut
        for (int i = 0; i < data.size(); i++) {
            for (int j = a; j<Math.pow(newDataset.attribute(newDataset.classIndex()).numValues(), 2); j++) {
                if(data.get(i).getResult().equals(data.get(j).getResult()) && data.get(j).getResult().equals(data.get(j).getValue())) {
                    data.get(i).setTotal(data.get(j).getfreq());
                }
            }
        }
        
        // Membuat classifier <- probabilitas
       buildClassifier(data, newDataset);
       /*for (int i=0; i<data.size(); i++) {
            System.out.println(data.get(i).getcolumn() + " " + data.get(i).getValue() + " " + data.get(i).getResult() +  " " + prob[i]);
       }*/
       
       // Melakukan klasifikasi untuk confussion matrix
       double x = classifyInstances(data, newDataset);
    }

    public static void buildClassifier(Vector<Data> model, Instances dataset) {
        //double prob_true = 1, prob_false = 1;
       prob = new double[model.size()];
        
        for(int i=0; i<model.size(); i++) {
            if(model.get(i).getcolumn().equalsIgnoreCase("class")) {
                prob[i] = model.get(i).getfreq() / dataset.numInstances();
            }
            else {
                prob[i] = model.get(i).getfreq() / model.get(i).getTotal();
            }   
        }
    }
    
    public static double classifyInstances(Vector<Data> data, Instances dataset) {
        double [] probclass = new double[dataset.numClasses()];
        int [] falseInstances = new int [dataset.numInstances()];
        int countTrue = 0;
        int countFalse = 0;
        int [] cfalseClass = new int[dataset.numClasses()];
        int a = dataset.classIndex();
        int ix = 0;
        
        for (int i = 0; i<dataset.numClasses(); i++) {
            cfalseClass[i] = 0;
        }
        
        int idx;
        for (int i = 0; i<dataset.numInstances(); i++) {
            probclass = initializeProbClass(dataset);
            for (int j = 0; j<dataset.numAttributes(); j++) {
                for (int k = 0; k < dataset.numClasses(); k++) {
                    if (j != dataset.classIndex()) {
                        idx = getIndex(data, dataset.attribute(j).name(), dataset.instance(i).toString(j), dataset.attribute(dataset.classIndex()).value(k));
                        probclass[k] *= prob[idx];
                    }
                }
            }
            int imax = searchMax(probclass);
            if (dataset.attribute(dataset.classIndex()).value(imax).equals(dataset.instance(i).toString(dataset.classIndex()))) {
                countTrue++;
                //System.out.println("true : " + countTrue);
            }
            else {
                countFalse++;
                falseInstances[ix] = i;
                ix++;
                for (int xx = 0; xx<dataset.numClasses(); xx++) {
                    if(dataset.instance(i).toString(dataset.classIndex()).equals(dataset.attribute(dataset.classIndex()).value(xx))) {
                        cfalseClass[xx]++;
                    }
                }
                //System.out.println("false : " + countFalse);
            }
        }
        double total = (double) dataset.numInstances();
        double accurate = (double) countTrue / total;
        double falseCls = (double) countFalse / total;
        System.out.println(countFalse);
        for (int xx = 0; xx<dataset.numClasses(); xx++) {
            System.out.println("False klasifikasi kelas " + dataset.attribute(dataset.classIndex()).value(xx) + " : " + cfalseClass[xx]);
        }
        
        System.out.println("Akurasi : " + accurate);
        System.out.println("False klasifikasi : " + falseCls);
        /*for(int i = 0; i<ix; i++) {
            int j = falseInstances[i];
            System.out.println(dataset.get(j));
        }*/
        
        return 0;
    }
    
    public static double classifyInstance(Vector<Data> data, Instances tes) {
        double [] probMax = new double [tes.numClasses()];
        probMax = initializeProbClass(tes);
        
        for (int i = 0; i<tes.numClasses(); i++) {
            for(int j = 0; j<tes.numAttributes();j++) {
                if(j!=tes.classIndex()) {
                    System.out.println(tes.attribute(j).name() + " " +  tes.instance(2).toString(j) + " " + tes.attribute(tes.classIndex()).value(i));
                    int idx = getIndex(data, tes.attribute(j).name(), tes.instance(2).toString(j), tes.attribute(tes.classIndex()).value(i));
                    probMax[i] *= prob[idx];
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
    
    public static int getIndex(Vector<Data> input, String attr, String nilaiAttr, String namaKls) {
        int i = 0;
        int index=9999;
        while(i<input.size()) {
            if(input.get(i).getValue().equals(nilaiAttr)&& input.get(i).getResult().equals(namaKls) && input.get(i).getcolumn().equals(attr)) {
                index=i;
                i=9999;
            }
            else
                i++;
        }
        return index;
    }
}


import java.util.Enumeration;
import java.util.Scanner;
import java.util.Vector;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.*;

public class Main {

    public static void main(String args[]) {
        // Baca File nya
        Vector<Data> data = new Vector();
        try {
            //load dataset
            ConverterUtils.DataSource source = new ConverterUtils.DataSource("mush.arff");
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
            //System.out.println("index class : " + dataset.classIndex());
            /*//Mendelete instance dengan atribut missing
            for (int i = 0; i < dataset.numAttributes(); i++) {
                dataset.deleteWithMissing(i);
            }
            dataset.deleteWithMissingClass();   // Mendelete instance dengan missing class*/
            Instances newDataset = new Instances(dataset);

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
        
        for (int i=0; i<data.size(); i++) {
            System.out.println(data.get(i).getcolumn() + " " + data.get(i).getValue() + " " + data.get(i).getResult() + " " + data.get(i).getfreq());
        }

        //Hitung pembagi tiap kolom dan tiap result
        for (int i = 0; i < data.size(); i++) {
            int count = 0;
            for (int j = 0; j < data.size(); j++) {
                if ((data.get(i).getcolumn().equals(data.get(j).getcolumn())) && (data.get(i).getResult() == data.get(j).getResult())) {
                    count += data.get(j).getfreq();
                }
            }
            data.get(i).setTotal(count) ;
        }

        //Klasifikasi
        //boolean train_result = NB(input, yes, no, "sunny", "cool", "high", "true");
        //System.out.println(train_result);
    }

    public static boolean NB(Vector<Data> model, Instances dataset, Instances tes) {
        //double prob_true = 1, prob_false = 1;
        int ixclass = dataset.classIndex();
        int sumAtrClass = dataset.attribute(ixclass).numValues();
        double [] prob = new double[sumAtrClass];
        
        for(int i=0; i<sumAtrClass; i++) {
            prob[i] = 1;
        }
        
        for (int k=0; k<tes.numInstances(); k++){
            for(int j=0; j<dataset.numAttributes(); j++) {
                if (k!=dataset.classIndex()) {
                    for (int l = 0; l<model.size(); l++) {
                        if(tes.get(k).attribute(j).equals(model.get(k).getValue())) {
                            int i = 0;
                            while(i<sumAtrClass) {
                                if(model.get(i).getResult().equals(dataset.attribute(ixclass).value(i))) {
                                    prob[i] = prob[i]*(model.get(i).getfreq()/model.get(i).getTotal());
                                }
                                else
                                    i++;
                            }
                        }
                    }
                }
            }
        }

        //untuk kolom 1
        /*for (int i = 0; i < model.size(); i++) {
            if (coll1.equals(model.get(i).getValue())) {        //mencari yang nilai atr = nilai model.get(i).getValue)
                if (model.get(i).isResult() == true) {
                    prob_true = prob_true * model.get(i).getfreq() / model.get(i).getTotal();
                } else {
                    prob_false = prob_false * model.get(i).getfreq() / model.get(i).getTotal();
                }
            }
        }

        //untuk kolom 2
        for (int i = 0; i < model.size(); i++) {
            if (coll2.equals(model.get(i).getValue())) {
                if (model.get(i).isResult() == true) {
                    prob_true 
    = prob_true * model.get(i).getfreq() / model.get(i).getTotal();
                } else {
                    prob_false = prob_false * model.get(i).getfreq() / model.get(i).getTotal();
                }
            }
        }

        //untuk kolom 3
        for (int i = 0; i < model.size(); i++) {
            if (coll3.equals(model.get(i).getValue())) {
                if (model.get(i).isResult() == true) {
                    prob_true = prob_true * model.get(i).getfreq() / model.get(i).getTotal();
                } else {
                    prob_false = prob_false * model.get(i).getfreq() / model.get(i).getTotal();
                }
            }
        }

        //untuk kolom 4
        for (int i = 0; i < model.size(); i++) {
            if (coll4.equals(model.get(i).getValue())) {
                if (model.get(i).isResult() == true) {
                    prob_true = prob_true * model.get(i).getfreq() / model.get(i).getTotal();
                } else {
                    prob_false = prob_false * model.get(i).getfreq() / model.get(i).getTotal();
                }
            }
        }

        //perhitungan probabilitas
        System.out.println("P(yes)  = " + prob_true * (yes) / (yes + no));
        System.out.println("P(no)   = " + prob_false * (no) / (yes + no));

        //hasil klasifikasi (jika nilai sama maka true)*/
        return true;
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
    
    public static void stdIO () {
        Scanner scan = new Scanner(System.in);
        double in = scan.nextDouble();
    
    }
}

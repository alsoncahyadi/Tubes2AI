/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package naivebayes;

import java.io.Serializable;
import java.util.Scanner;
import java.util.Vector;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import weka.filters.Filter;
import weka.filters.supervised.attribute.*;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Remove.*;

public class NBayes extends AbstractClassifier implements Serializable {

    Vector<Data> data = new Vector();
    int classidx = 0;
    static double[] prob;

    @Override
    public void buildClassifier(Instances ins) throws Exception {
        //Filtering jadi nominal semua
        Instances disc_dataset = null;
        String[] options = new String[2];
        options[0]="-R";
        options[1]="first-last";
        
       
        //Discretize filtering
        Discretize disc = new Discretize();
        disc.setOptions(options);
        disc.setInputFormat(ins);
        disc_dataset = Filter.useFilter(ins, disc);
        
        ins = disc_dataset;

        //Mendelete instance dengan atribut missing
        for (int i = 0; i < ins.numAttributes(); i++) {
            ins.deleteWithMissing(i);
        }
        ins.deleteWithMissingClass();   // Mendelete instance dengan missing class
        
        
        //inisiasi Vektor
        for (int i = 0; i <ins.numAttributes(); i++) {
            for (int j = 0; j <ins.attribute(i).numValues(); j++) {
                int k = 0;
                while (k < ins.numClasses()) {
                    Data datum = new Data(i, j, k, 0);
                    data.add(datum);
                    k++;
                }   
            }
        }
        
        //menghitung freq dari setiap data pada Vector yang ada pada instance
        for (int i = 0; i < ins.numInstances(); i++) {
            for (int j = 0; j < ins.numAttributes(); j++) {
                int index = getIndex(data, j, (int) ins.instance(i).value(j), (int) ins.instance(i).value(ins.classIndex()));
                data.get(index).setfreq(data.get(index).getfreq() + 1);
            }
        }

        //Untuk  mengahapus vektor yang tidak diperlukan
        //Karena adanya kombinasi pada kelas yang tidak diperlukan
        classidx = ins.classIndex();
        int a = 0;
        
        if (classidx > 0) {
            for (int i = 0; i < classidx; i++) {
                a += (ins.attribute(i).numValues()*ins.numClasses());
            }
            for (int i = a; i < a + ins.numClasses(); i++) {
                while (data.get(i).getfreq() == 0) {
                    data.remove(i);
                }
            }
        } else {
            for (int i = classidx; i < ins.numClasses(); i++) {
                while (data.get(i).getfreq() == 0) {
                    data.remove(i);
                }
            }
        }

        //Menghitung total pembagi untuk setiap attribute (jumlah kelas yang bersesuaian)
        for (int i = 0; i < data.size(); i++) {
            for (int j = a; j < a + ins.numClasses(); j++) {
                if (data.get(i).getResult() == data.get(j).getResult() && data.get(j).getResult() == data.get(j).getValue()) {
                    data.get(i).setTotal(data.get(j).getfreq());
                }
            }
        }
        
        //inisialisasi tabel untuk probabilitas
        prob = new double[data.size()];
        //menghitung probabilitas setiap attribute & kelas
        for (int i = 0; i < data.size(); i++) {
            if (data.get(i).getcolumn() == ins.classIndex()) {
                prob[i] = data.get(i).getfreq() / ins.numInstances();
            } else {
                prob[i] = data.get(i).getfreq() / data.get(i).getTotal();
            }
        }
    }

    public double classifyInstance(Instance tes) {
        double[] probMax = new double[tes.numClasses()];
        int idx = 0;
        //inisialisasi untuk mencari probabilitas dari attribut2 yang ada
        //untuk dilakukan klasifikasi
        if (tes.classIndex() == 0) {
            for (int i = 0; i < tes.numClasses(); i++) {
                probMax[i] = prob[i];
            }
        } else {
            int a = 0;
            if (classidx > 0) {
                for (int i = 0; i < classidx; i++) {
                    a += (tes.attribute(i).numValues()*tes.numClasses());
                }
                for (int i = 0; i < tes.numClasses(); i++) {
                    probMax[i] = prob[a];
                    a++;
                }
            }
        }
        
        //menghitung probabilitas setiap attribut dengan kemungkinan kelas yang ada
        for (int k = 0; k < tes.numClasses(); k++) {
            for (int j = 0; j < tes.numAttributes(); j++) {
                if (j != tes.classIndex()) {
                    idx = getIndex(data, j, (int) tes.value(j), k);
                    probMax[k] *= prob[idx];
                }
            }
        }
        //output untuk nilai probabilitas maksimal
        int x = searchMax(probMax);

        return x;

    }

    public static int searchMax(double[] array) {
        int imax = 0;
        for (int i = 0; i < array.length; i++) {
            if (array[i] > array[imax]) {
                imax = i;
            }
        }
        //mengembalikan indeks yang memiliki nilai dari probabilitas paling besar
        return imax;
    }

    public static int getIndex(Vector<Data> input, int attr, int nilaiAttr, int namaKls) {
        //digunakan untuk mnecari index pada vector 
        int i = 0;
        int index = -1;
        boolean found = false;
        while (!found && i < input.size()) {
            if (input.get(i).getValue() == nilaiAttr && input.get(i).getResult() == namaKls && input.get(i).getcolumn() == attr) {
                index = i;
                found = true;
            } else {
                i++;
            }
        }
        return index;
    }
}

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package naivebayes;

/**
 *
 * @author alson
 */
public class Data {
    private int column;      //atribut
    private int value;        //nilai dari atribut
    private int result;      //kelas
    private double freq;         //frekuensi nilai pada kolom tersebut
    private double total;        //pembagi per kolom per result

    public int getcolumn() {
        return column;
    }

    public int getValue() {
        return value;
    }

    public int getResult() {
        return result;
    }

    public double getfreq() {
        return freq;
    }

    public double getTotal() {
        return total;
    }

    public void setcolumn(int column) {
        this.column = column;
    }

    public void setValue(int value) {
        this.value = value;
    }

    public void setResult(int result) {
        this.result = result;
    }

    public void setfreq(double freq) {
        this.freq = freq;
    }

    public void setTotal(double total) {
        this.total = total;
    }
    
    public Data(int coll, int val, int rslt, double fr){
        column = coll;
        value = val;
        result = rslt;
        freq = fr;
        total = 0;
    }
}

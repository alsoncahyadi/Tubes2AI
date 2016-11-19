//File : Data.java
//Desc : Kelas Data untuk menyimpan nilai setiap data yang sudah diubah dari DataSet ke Tabel Frekuensi

public class Data {
    private String column;      //atribut
    private String value;        //nilai dari atribut
    private String result;      //kelas
    private double freq;         //frekuensi nilai pada kolom tersebut
    private double total;        //pembagi per kolom per result

    public String getcolumn() {
        return column;
    }

    public String getValue() {
        return value;
    }

    public String getResult() {
        return result;
    }

    public double getfreq() {
        return freq;
    }

    public double getTotal() {
        return total;
    }

    public void setcolumn(String column) {
        this.column = column;
    }

    public void setValue(String value) {
        this.value = value;
    }

    public void setResult(String result) {
        this.result = result;
    }

    public void setfreq(double freq) {
        this.freq = freq;
    }

    public void setTotal(double total) {
        this.total = total;
    }
    
    public Data(String coll, String val, String rslt, double fr){
        column = coll;
        value = val;
        result = new String (rslt);
        freq = fr;
        total = 0;
    }

}

package core;

enum ac_type{
    ac_ReLU         ,
    ac_ReLU6        ,
    ac_LeakyReLU    ,
    ac_Sigmoid      ,
    ac_Sigmoid_hard ,
    ac_Tanh         ,
    ac_Tanh_hard    ,
    ac_Sign         ,
    ac_pass         ,
};

public class netLyrConf{
    private int indim, outdim;
    private byte actp;
    private int dexa;
    private byte []weights;
    private byte []bias;

    static{
        System.loadLibrary("lmlpcore_tojava");
    }

    private native void alloc_W_B(int indi, int oudi);

    public netLyrConf(int indi, int oudi, byte actp, float dexa){
        this.indim = indi;
        this.outdim = oudi;
        this.actp = actp;
        this.dexa = (int)(dexa * 65536);
        alloc_W_B(indi, oudi);
    }
}

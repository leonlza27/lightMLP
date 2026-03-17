package core;
import mlib.matrixbp;

public class mlptrain{
    private netLyrConf []netsrc;
    //these will be handled by native code
    //private byte [][]fullConnData;
    //private byte [][]grad_w;
    //private byte [][]grad_tolast;
    private int train_exahandle;

    static{
        System.loadLibrary("lmlpcore_tojava");
    }

    private native void setuptrainer(netLyrConf []modeldef);

    public native void execute(matrixbp in, matrixbp out);
    public native void backward(matrixbp grad0, float lr);
    
    public mlptrain(netLyrConf []modeldef){
        this.netsrc = modeldef;
        setuptrainer(modeldef);
    }

}

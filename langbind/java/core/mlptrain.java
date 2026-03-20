package core;
import mlib.matrixbp;

public class mlptrain{
    private netdef netsrc;
    //these will be handled by native code
    //private byte [][]fullConnData;
    //private byte [][]grad_w;
    //private byte [][]grad_tolast;
    private int train_exahandle;

    static{
        System.loadLibrary("lmlpcore_tojava");
    }

    private native void setuptrainer(netdef modeldef);

    public native void execute(matrixbp in, matrixbp out);
    public native void backward(matrixbp grad0, float lr);
    
    public mlptrain(netdef modeldef){
        this.netsrc = modeldef;
        setuptrainer(modeldef);
    }

}

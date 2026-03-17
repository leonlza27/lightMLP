package core;
import mlib.matrixbp;

public class mlptrain{
    private netLyrConf []netsrc;
    //this will be handled by native code
    //private byte [][]fullConnData;
    private int exec_exahandle;

    static{
        System.loadLibrary("lmlpcore_tojava");
    }

    private native void setupexec(netLyrConf []modeldef);

    public native void execute(matrixbp in, matrixbp out);
    
    public mlpexec(netLyrConf []modeldef){
        this.netsrc = modeldef;
        setupexec(modeldef);
    }

}

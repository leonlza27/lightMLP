package core;
import mlib.matrixbp;

public class mlpexec{
    private netdef netsrc;
    //this will be handled by native code
    //private byte [][]fullConnData;
    private int exec_exahandle;

    static{
        System.loadLibrary("lmlpcore_tojava");
    }

    private native void setupexec(netdef modeldef);

    public native void execute(matrixbp in, matrixbp out);
    
    public mlpexec(netdef modeldef){
        this.netsrc = modeldef;
        setupexec(modeldef);
    }

}

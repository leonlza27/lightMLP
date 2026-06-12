package core;
import mlib.matrixbp;

public class mlpexec{
    private netdef netsrc;
    //this will be handled by native code
    //private byte [][]fullConnData;
    private long exec_exahandle;

    static{
        System.loadLibrary("lmlpcore_tojava");
    }

    private native void setupexec(netdef modeldef);

    public native void execute(matrixbp in, matrixbp out);
    
    private native void deallocexec();

    public void clearup(){
        deallocexec();
        netsrc = null;
    }

    public mlpexec(netdef modeldef){
        this.netsrc = modeldef;
        setupexec(modeldef);
    }

}

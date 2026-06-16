package core;
import mlib.matrixbp;

public class mlptrain{
    private netdef netsrc;
    //these will be handled by native code
    //private byte [][]fullConnData;
    //private byte [][]grad_w;
    //private byte [][]grad_tolast;
    private long train_exahandle;

    static{
        System.loadLibrary("lmlpcore_tojava");
    }

    private native void setuptrainer(netdef modeldef);
    private native void setuptrainer_asGradCollector(netdef modeldef);

    private native void dealloctrainer();

    public native void execute(matrixbp in, matrixbp out);
    public native void backward(matrixbp grad0, float lr);
    public native void savegrads_to(mlptrain gradscap);
    public native void backward_from_totalgrads(mlptrain gradscap, float lr);
    public native void get_grad2last(matrixbp gradtolast);

    public native boolean isGradSaver();

    public void clearup(){
        dealloctrainer();
        netsrc = null;
    }
    
    public mlptrain(netdef modeldef, boolean forSaveGrads){
        this.netsrc = modeldef;
        if(!forSaveGrads){
            setuptrainer(modeldef);
        } else{
            setuptrainer_asGradCollector(modeldef);
        }
    }

}

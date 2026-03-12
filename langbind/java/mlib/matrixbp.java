package mlib;

public class matrixbp{
    private byte []data;
    
    static{
        System.loadLibrary("mbp_tpdef_java");
    }

    private native void setupmbp(int r, int c);
    private native String printmatrix();

    public matrixbp(int r, int c){
        setupmbp(r, c);
    }

    public native void fromlist(float []lstin) throws IllegalArgumentException;
    public native void fromrand();

    @Override
    public String toString(){
        return printmatrix();
    }
    
}

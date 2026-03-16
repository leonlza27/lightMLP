package mlib;

public class matrixbp{
    private byte []data;

    static{
        System.loadLibrary("mbp_tojava");
    }

    private native void setupmbp(int r, int c);
    private native String printmatrix();

    public matrixbp(int r, int c){
        setupmbp(r, c);
    }

    public native void fromlist(float []lstin) throws IllegalArgumentException;
    public native void fromrand();
    
    public native void toarr_cpy(float []lstin) throws IllegalArgumentException;
    public native float[] toarr_new();
    
    public native int rows();
    public native int cols();

    @Override
    public String toString(){
        return printmatrix();
    }

}

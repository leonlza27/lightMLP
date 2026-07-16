package core;
import mlib.matrixbp;

public class convop{
    static{
        System.loadLibrary("lmlpcore_tojava");
    }

    public static native void maxpool(matrixbp arrin, short pool_rows, short pool_cols, short stepr, short stepc, matrixbp arrout, short[] idx_of_maxs);
    public static native void averpool(matrixbp arrin, short pool_rows, short pool_cols, short stepr, short stepc, matrixbp arrout);

    public static native void maxpool_backward(matrixbp arrin, short[] idx_of_maxs, short pool_rows, short pool_cols, short stepr, short stepc, matrixbp arrout);
    public static native void averpool_backward(matrixbp arrin, short pool_rows, short pool_cols, short stepr, short stepc, matrixbp arrout);

    public static native void padding(matrixbp arrin, short r_extend, short c_extend, matrixbp arrout);
    public static native void curedges(matrixbp arrin, short r_cut, short c_cut, matrixbp arrout);

    public static native void divelem(matrixbp arrin, short pool_rows, short pool_cols, short expand, short stepr, short stepc, matrixbp[] arrouts);
    public static native void mergeelem(matrixbp[] arrins, short expand, short stepr, short stepc, matrixbp arrout);

    public static native void conv(matrixbp arrin, matrixbp convcore, short expand, short stepr, short stepc, matrixbp arrout);
}
package mlib;

public class ops{
    static{
        System.loadLibrary("mbp_tojava");
    }

    public static native void madd(matrixbp m1, matrixbp m2, matrixbp mr);
    public static native void msub(matrixbp m1, matrixbp m2, matrixbp mr);
    public static native void mscale(matrixbp m1, float scaler, matrixbp mr);
    public static native void mmul(matrixbp m1, matrixbp m2, matrixbp mr);
    public static native void mmul_byelem(matrixbp m1, matrixbp m2, matrixbp mr);
    public static native void mtpose(matrixbp m1, matrixbp mr);
}

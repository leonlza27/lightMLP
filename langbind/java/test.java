import mlib.matrixbp;
import mlib.ops;

public class test{
    public static void main(String []args){
        matrixbp m1 = new matrixbp(2,3);
        matrixbp m2 = new matrixbp(2,3);
        matrixbp mr = new matrixbp(2,3);
        matrixbp mt = new matrixbp(3,2);
        //m1.fromrand();
        float []lstin = {(float)1.,(float)2.,(float)3.,(float)4.,(float)5.,(float)6.};
        m1.fromlist(lstin);
        m2.fromlist(lstin);
        ops.msub(m1,m2,mr);
        System.out.println(mr);
        ops.mtpose(m1,mt);
        System.out.println(mt);
    }
}

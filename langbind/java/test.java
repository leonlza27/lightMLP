import mlib.matrixbp;

public class test{
    public static void main(String []args){
        matrixbp m1 = new matrixbp(2,3);
        //m1.fromrand();
        float []lstin = {(float)1.,(float)2.,(float)3.,(float)4.,(float)5.,(float)6.};
        m1.fromlist(lstin);
        System.out.println(m1);
        float []lstin_excp = {(float)1.,(float)2.,(float)3.,(float)4.,(float)5.};
        try{
            m1.fromlist(lstin_excp);
        }catch(Exception e){
            System.out.println("output for incorrect input test\n");
        }

    }
}

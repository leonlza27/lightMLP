import mlib.matrixbp;
import mlib.ops;
import core.netdef;
import core.netlyr;
import core.mlptrain;

public class test_train{
    public static void main(String []args){
        netlyr []modeldef = {new netlyr(2,3,netlyr.actp.ac_Sigmoid,0.0f), new netlyr(3,1,netlyr.actp.ac_Sigmoid,0.0f)};
	System.out.println("Setup netdef\n");
        netdef ndef = new netdef();
        ndef.setup_ndef(modeldef);
	System.out.println("Setup mlptrain\n");
        mlptrain ntrain = new mlptrain(ndef, false);
        matrixbp m_in = new matrixbp(2,1);
        matrixbp m_out = new matrixbp(1,1);
	System.out.println("fwd\n");
        ntrain.execute(m_in, m_out);
	System.out.println("back\n");
        ntrain.backward(m_out, 0.0f);
    }
}
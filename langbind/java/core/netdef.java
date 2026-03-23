package core;

public class netdef{
    private long _addr;
    private int netlen;
    
    static{
        System.loadLibrary("lmlpcore_tojava");
    }

    private native void setup_ndef(netlyr []modeldef);

    public netdef(netlyr []modeldef){
        setup_ndef(modeldef);
    }
}

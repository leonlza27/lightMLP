package core;

public class netdef{
    private long _addr;
    private int netlen;
    
    static{
        System.loadLibrary("lmlpcore_tojava");
    }

    public native void setup_ndef(netlyr []modeldef);

    public native void setup_fromfile(String fpath);

    public native void freemodel();

    public native void savemodel(String fpath);

}

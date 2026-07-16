package core;

public class netlyr{
    private int indim;
    private int outdim;
    private byte _actp;
    private float dexa;

    public enum actp{
        ReLU            ,
        ReLU6           ,
        LeakyReLU       ,
        Sigmoid         ,
        Sigmoid_hard    ,
        Tanh            ,
        Tanh_hard       ,
        TripleLine      ,
        pass            ,
    };

    public netlyr(int indim, int outdim, actp ac_type, float dexa){
        this.indim = indim;
        this.outdim = outdim;
        this.dexa = dexa;
        this._actp = ac_type.oridnal();
    }
}

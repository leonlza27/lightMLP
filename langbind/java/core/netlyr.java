package core;

public class netlyr{
    private int indim;
    private int outdim;
    private byte _actp;
    private float dexa;

    enum actp{
        ac_ReLU         ,
        ac_ReLU6        ,
        ac_LeakyReLU    ,
        ac_Sigmoid      ,
        ac_Sigmoid_hard ,
        ac_Tanh         ,
        ac_Tanh_hard    ,
        ac_Sign         ,
        ac_pass         ,
    };

    public netlyr(int indim, int outdim, actp ac_type, float dexa){
        this.indim = indim;
        this.outdim = outdim;
        this.dexa = dexa;
        switch(ac_type){
            case ac_ReLU:
                this._actp = 0;
                break;
            case ac_ReLU6:
                this._actp = 1;
                break;
            case ac_LeakyReLU:
                this._actp = 2;
                break;
            case ac_Sigmoid:
                this._actp = 3;
                break;
            case ac_Sigmoid_hard:
                this._actp = 4;
                break;
            case ac_Tanh:
                this._actp = 5;
                break;
            case ac_Tanh_hard:
                this._actp = 6;
                break;
            case ac_Sign:
                this._actp = 7;
                break;
            case ac_pass:
                this._actp = 8;
                break;
        }
    }
}

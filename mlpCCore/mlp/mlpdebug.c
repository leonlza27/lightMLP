#include "ac_callback.h"
#include "mlp.h"
#include <stdio.h>
//这个证明windows在连续调用malloc会出现非0无效地址问题而代码逻辑并无问题<line:14>
int main(){
    qfix w0[6];
    qfix b0[3];
    qfix w1[3];
    qfix b1[1];
    netLyrConf net[] = {{ac_Tanh,2,3,w0,b0,0},{ac_Tanh,3,1,w1,b1,0}};

    mlpTrainStatus ntrain;
#ifndef manual
    mlptrainer_setup(2, net, &ntrain);
#else
    ntrain.calclyrs = 2;
    ntrain.modelsrc = net;

    qfix *fullconndatas[3];
    qfix fullconndata1[3];
    qfix fullconndata2[1];
    fullconndatas[1] = fullconndata1;
    fullconndatas[2] = fullconndata2;
    ntrain.fullConnData = fullconndatas;

    qfix *w_ds[2];
    qfix w_d1[6];
    qfix w_d2[3];
    w_ds[0] = w_d1;
    w_ds[1] = w_d2;
    ntrain.w_grad = w_ds;

    qfix *x_ls[2];
    qfix x_l1[2];
    qfix x_l2[3];
    x_ls[0] = x_l1;
    x_ls[1] = x_l2;
    ntrain.lyrinput_grad = x_ls;
#endif
    for(int i = 0; i < 2; i++) printf("%d: fullconndata:%p w_grad:%p gradx_tolast:%p\n",i,ntrain.fullConnData[i], ntrain.w_grad[i], ntrain.lyrinput_grad[i]);

    mlptrainer_execute(&ntrain, w0);
    qfix simresu = 3;
    mlptrainer_backward(&ntrain, &simresu, 0);
}
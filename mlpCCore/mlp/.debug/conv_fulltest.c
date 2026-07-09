#include "../convop.h"
#include "../../qfix_ops/qfix_op.h"
#include <stdio.h>
#include "../../mlp/mlp.h"

int main(){
    qfix input[16];
    for (int i = 0; i < 16; i++){ 
        input[i] = float_to_qfix(i + 1);
        printf("%f ", qfix_to_float(input[i]));
        if(i % 4 == 3) printf("\n");
    }
    printf("\n");

    qfix w[9];
    for(uint16_t i = 0; i < 9; i++){
        w[i] = float_to_qfix(1 + i);
    }

    qfix _B = 0;

    qfix resu1[9];
    qfix resu2[9];
    qfix resu3[9];
    qfix resu4[9];

    qfix *resus[4] = {resu1, resu2, resu3, resu4};

    conv_divelem(input, 4, 4, 3, 3, 0, 1, 1, resus);

    netLyrConf convcore;
    convcore.acTp = ac_pass;
    convcore.bias = &_B;
    convcore.in_dim = 9;
    convcore.out_dim = 1;
    convcore.weights = w;
    convcore.dExtra = 0;

    mlpTrainStatus cc;
    mlptrainer_setup(1, &convcore, &cc);

    qfix resu[4];
    for(int i = 0; i < 4; i++){
        mlptrainer_execute(&cc, resus[i]);
        qfix *res = mlptrain_resu_ref(&cc);
        resu[i] = *res;
    }

    for (int i = 0; i < 4; i++){ 
        printf("%f ", qfix_to_float(resu[i]));
        if(i % 2 == 1) printf("\n");
    }
    printf("\n");

    conv_fwdcalc(input, w, 4, 4, 3, 3, 0, 1, 1, resu);

    for (int i = 0; i < 4; i++){ 
        printf("%f ", qfix_to_float(resu[i]));
        if(i % 2 == 1) printf("\n");
    }
    printf("\n");
}

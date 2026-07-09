#include "../convop.h"
#include "../../qfix_ops/qfix_op.h"
#include <stdio.h>

int main(){
    qfix input[16];
    for (int i = 0; i < 16; i++){ 
        input[i] = float_to_qfix(i + 1);
        printf("%f ", qfix_to_float(input[i]));
        if(i % 4 == 3) printf("\n");
    }
    printf("\n");

    qfix resu1[9];
    qfix resu2[9];
    qfix resu3[9];
    qfix resu4[9];

    qfix *resus[4] = {resu1, resu2, resu3, resu4};

    conv_divelem(input, 4, 4, 3, 3, 0, 1, 1, resus);

    for (int i = 0; i < 9; i++){ 
        printf("%f ", qfix_to_float(resu1[i]));
        if(i % 3 == 2) printf("\n");
    }

    printf("\n");

    for (int i = 0; i < 9; i++){ 
        printf("%f ", qfix_to_float(resu2[i]));
        if(i % 3 == 2) printf("\n");
    }

    printf("\n");

    for (int i = 0; i < 9; i++){ 
        printf("%f ", qfix_to_float(resu3[i]));
        if(i % 3 == 2) printf("\n");
    }

    printf("\n");

    for (int i = 0; i < 9; i++){ 
        printf("%f ", qfix_to_float(resu4[i]));
        if(i % 3 == 2) printf("\n");
    }
    printf("\n");
    
    for(int ii = 0; ii < 4; ii++){
        for (int i = 0; i < 9; i++){ 
            resus[ii][i] = float_to_qfix(1);
        }
    }

    qfix _grads[16] = {0};

    conv_mergeelem(resus, 4, 4, 3, 3, 0, 1, 1, _grads);

    for (int i = 0; i < 16; i++){ 
        printf("%f ", qfix_to_float(_grads[i]));
        if(i % 4 == 3) printf("\n");
    }
    printf("\n");

    return 0;
}
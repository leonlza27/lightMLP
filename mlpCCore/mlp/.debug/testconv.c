#include "../convop.h"
#include <stdio.h>
#include "../../qfix_ops/qfix_op.h"

int main(){
    qfix input[16];
    for (int i = 0; i < 16; i++){ 
        input[i] = float_to_qfix(i + 1);
        printf("%f ", qfix_to_float(input[i]));
        if(i % 4 == 3) printf("\n");
    }
    printf("\n");
    qfix output[4] = {0,0,0,0};
    conv_averpool(input, 4, 4, 2, 2, 2, 2, output);
    for(int i = 0; i < 4 ; i++) printf("%f ", qfix_to_float(output[i]));
    printf("\n");
    uint16_t maxs[4];

    conv_maxpool(input, 4, 4, 2, 2, 2, 2, output, maxs);

    for(int i = 0; i < 4 ; i++) printf("[%d]%f ", maxs[i], qfix_to_float(output[i]));
    printf("\n\n");

    output[0] = float_to_qfix(4);
    output[1] = float_to_qfix(8);
    output[2] = float_to_qfix(12);
    output[3] = float_to_qfix(16);

    qfix grads[16];
    conv_averpool_backward(output, 4, 4, 2, 2, 2, 2, grads);
    for(int i = 0; i < 16 ; i++){
        printf("%f ", qfix_to_float(grads[i]));
        if(i%4 == 3) printf("\n");
    }
    printf("\n");

    conv_maxpool_backward(output, maxs, 4, 4, 2, 2, 2, 2, grads);
    for(int i = 0; i < 16 ; i++){
        printf("%f ", qfix_to_float(grads[i]));
        if(i%4 == 3) printf("\n");
    }
    printf("\n");
    return 0;
}
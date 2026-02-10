#include "mlp.h"
#include <stdio.h>

int main(){
    qfix w1[6] = {float_to_qfix(1), float_to_qfix(2), float_to_qfix(3),float_to_qfix(1), float_to_qfix(2), float_to_qfix(3)}, b1[5] = {float_to_qfix(1), 0}, w2[10], b2[2];
    netLyrConf example[] = {{ac_pass,3,2,w1,b1,0}};
    mlpTrainStatus cap;
    qfix in[] = {float_to_qfix(1), float_to_qfix(2), float_to_qfix(3)};
    qfix excp[] = {float_to_qfix(10), float_to_qfix(30)};
    qfix grad[2];
    mlptrainer_setup(1,example, &cap);
    for(uint16_t i = 0; i < 200; i++){
    mlptrainer_execute(&cap, in);
    grad[0] = excp[0] - cap.fullConnData[1][0];
    grad[1] = excp[1] - cap.fullConnData[1][1];
    mlptrainer_backward(&cap, grad, float_to_qfix(0.125));
    printf("%f %f\n", qfix_to_float(cap.fullConnData[1][0]), qfix_to_float(cap.fullConnData[1][1]));
    }
    mlptrainer_clearup(&cap);
}
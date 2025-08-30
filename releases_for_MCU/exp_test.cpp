#include "qfloat_operation/qfloat_op.h"
#include <stdio.h>
#include <math.h>

int main(){
    printf("        input\texcepted\t  actual\t    bias\n");
    for(float i =-8.0f; i< 8.0f; i+=0.1f){
        qfloat qf = float_to_qfloat(i);
        float excepted = exp(i);
        float actual = qfloat_to_float(exp_qfloat(qf));
        float bias = fabs(actual - excepted);
        printf("exp(%8.4f)\t%8.4f\t%8.4f\t%8.4f\n",i,excepted,actual,bias);
    }
    return 0;
}
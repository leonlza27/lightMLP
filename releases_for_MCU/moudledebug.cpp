//初始验证
#include <stdio.h>
#include "qfloat_operation/qfloat_op.h"


int main(){

   qfloat num1 = float_to_qfloat(0.1f);
   qfloat num2 = float_to_qfloat(0.2f);
   
   printf("%.2f\n",qfloat_to_float(qfloat_mul(num1,num2)));


    return 0;
}
#include "f_q16_16_op.h"

f_q16_16 q16_16_add(f_q16_16 num1, f_q16_16 num2){
    return num1 + num2;
}

f_q16_16 q16_16_sub(f_q16_16 num1, f_q16_16 num2){
    return num1 - num2;
}

f_q16_16 q16_16_mul(f_q16_16 num1, f_q16_16 num2){
    return (f_q16_16)(((int64_t)num1 * num2) >> 16);
}

f_q16_16 q16_16_div(f_q16_16 num1, f_q16_16 num2){
    return (f_q16_16)(((int64_t)num1 << 16) / num2);
}

#include "qfloat_op.h"

qfloat qfloat_add(qfloat num1, qfloat num2){
    return num1 + num2;
}

qfloat qfloat_sub(qfloat num1, qfloat num2){
    return num1 - num2;
}

qfloat qfloat_mul(qfloat num1, qfloat num2){
    return (qfloat)(((_tmp_larger)num1 * num2) >> QSHIFT);
}

qfloat qfloat_div(qfloat num1, qfloat num2){
    return (qfloat)(((_tmp_larger)num1 << QSHIFT) / num2);
}

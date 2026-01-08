#ifndef _qfix_operations_
#define _qfix_operations_

#include "../CustomConf.h"


//类型转换

//虽然名字里只有浮点,但它其实同时支持浮点与整数输入 :)
#define float_to_qfix(value) (qfix)(value * (1 << QSHIFT))
#define qfix_to_float(qfvalue) ((float)qfvalue / (1 << QSHIFT))
#define qfix_to_float64(qfvalue) ((double)qfvalue / (1 << QSHIFT))
#define qfix_to_int(qfvalue) (qfvalue >> QSHIFT)

#ifdef __cplusplus
extern "C"{
#endif

#define qfmin(num1, num2) (num1 < num2)? num1: num2

#define qfmax(num1, num2) (num1 > num2)? num1: num2

//运算
static inline qfix qfix_add(qfix num1, qfix num2){
    return num1 + num2;
}

static inline qfix qfix_sub(qfix num1, qfix num2){
    return num1 - num2;
}

static inline qfix qfix_mul(qfix num1, qfix num2){
    return (qfix)(((_tmp_larger)num1 * num2) >> QSHIFT);
}

static inline qfix qfix_div(qfix num1, qfix num2){
    return (qfix)(((_tmp_larger)num1 << QSHIFT) / num2);
}

qfix exp_qfix(qfix x);

#ifdef __cplusplus
}
#endif
#endif
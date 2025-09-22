#ifndef _qfix_operations_
#define _qfix_operations_

#include "../CustomConf.h"


//类型转换

#define float_to_qfix(value) (qfix)(value * (float)(1 << QSHIFT))
#define qfix_to_float(qfvalue) ((float)qfvalue / (float)(1 << QSHIFT))

#ifdef __cplusplus
extern "C"{
#endif

//运算
qfix qfix_add(qfix num1,qfix num2);

qfix qfix_sub(qfix num1,qfix num2);

qfix qfix_mul(qfix num1,qfix num2);

qfix qfix_div(qfix num1,qfix num2);

qfix exp_qfix(qfix x);

#ifdef __cplusplus
}
#endif
#endif
#ifndef _qfloat_operations_
#define _qfloat_operations_

#include "../../CustomConf.h"


//类型转换

#define float_to_qfloat(value) (qfloat)(value * (float)(1 << QSHIFT))
#define qfloat_to_float(qfvalue) ((float)qfvalue / (float)(1 << QSHIFT))

#ifdef __cplusplus
extern "C"{
#endif

//运算
qfloat qfloat_add(qfloat num1,qfloat num2);

qfloat qfloat_sub(qfloat num1,qfloat num2);

qfloat qfloat_mul(qfloat num1,qfloat num2);

qfloat qfloat_div(qfloat num1,qfloat num2);

#ifdef __cplusplus
}
#endif
#endif
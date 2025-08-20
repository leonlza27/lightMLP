#ifndef _float_q16_16_operations_
#define _float_q16_16_operations_

#include <stdint.h>
typedef int32_t f_q16_16;

#define Q16_16_MAX 0x7FFFFFFF
#define Q16_16_MIN 0x80000000

//类型转换

#define float_to_q16_16(value) (f_q16_16)(value * (float)(1 << 16))
#define q16_16_to_float(q16value) ((float)q16value / (float)(1 << 16))

#ifdef __cplusplus
extern "C"{
#endif

//运算
f_q16_16 q16_16_add(f_q16_16 num1,f_q16_16 num2);

f_q16_16 q16_16_sub(f_q16_16 num1,f_q16_16 num2);

f_q16_16 q16_16_mul(f_q16_16 num1,f_q16_16 num2);

f_q16_16 q16_16_div(f_q16_16 num1,f_q16_16 num2);

#ifdef __cplusplus
}
#endif
#endif
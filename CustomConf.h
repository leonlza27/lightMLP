/*
兼容MCU处理器环境类型配置
*/
#ifndef _CustomConf_
#define _CustomConf_

#include <stdint.h>
#include <limits.h>

//定点数配置

typedef int16_t f_q8_8;
typedef int32_t f_q16_16;

#define e_BASE_Q16 178145
#define e_BASE_Q8 696

#define Q8_MAX INT16_MAX
#define Q8_MIN INT16_MIN
#define Q16_MAX INT32_MAX
#define Q16_MIN INT32_MIN

#ifdef CPU_PLATFORM_8BIT
typedef f_q8_8 qfloat;
#define QSHIFT 8
#define _tmp_larger int32_t
typedef int16_t qf_cur_cast;
#define e_QFBASE e_BASE_Q8
#define QF_FLOAT_MASK 0xFF
#define QF_MAX Q8_MAX
#define QF_MIN Q8_MIN
#else
typedef f_q16_16 qfloat;
#define QSHIFT 16
#define _tmp_larger int64_t
typedef int32_t qf_cur_cast;
#define e_QFBASE e_BASE_Q16
#define QF_FLOAT_MASK 0xFFFF
#define QF_MAX Q16_MAX
#define QF_MIN Q16_MIN
#endif

#endif
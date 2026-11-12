#ifndef v2_arrop_h
#define v2_arrop_h 

#include <stdint.h>
#include "../CustomConf.h"

//新API

DLLEXPORT void arr_add(const qfix *in1, const  qfix* in2, uint32_t len, qfix* out);
DLLEXPORT void arr_sub(const qfix *in1, const qfix* in2, const uint32_t len, qfix* out);
DLLEXPORT void arr_scale(const qfix *in1, const qfix scale, const uint32_t len, qfix* out);
DLLEXPORT void arr_emul(const qfix *in1, const qfix* in2, const uint32_t len, qfix* out);
DLLEXPORT void arr_mmul(const qfix *in1, const qfix* in2, const uint16_t rows1, const uint16_t c1_r2, const uint16_t cols2, qfix* out);
DLLEXPORT void arr_mtpose(const qfix *in1, const uint16_t rows1, const uint16_t cols1, qfix* out);

DLLEXPORT void arr_add_mt(const qfix *in1, const  qfix* in2, uint32_t len, qfix* out);
DLLEXPORT void arr_sub_mt(const qfix *in1, const qfix* in2, const uint32_t len, qfix* out);
DLLEXPORT void arr_scale_mt(const qfix *in1, const qfix scale, const uint32_t len, qfix* out);
DLLEXPORT void arr_emul_mt(const qfix *in1, const qfix* in2, const uint32_t len, qfix* out);
DLLEXPORT void arr_mmul_mt(const qfix *in1, const qfix* in2, const uint16_t rows1, const uint16_t c1_r2, const uint16_t cols2, qfix* out);
DLLEXPORT void arr_mtpose_mt(const qfix *in1, const uint16_t rows1, const uint16_t cols1, qfix* out);


#endif
#include "arrop.h"
#include "../qfix_ops/qfix_op.h"
#include "../CustomConf.h"

DLLEXPORT void arr_add(const qfix *in1, const qfix* in2, const uint32_t len, qfix* out){
    for(uint32_t i = 0; i < len; i++){
        out[i] = in1[i] + in2[i];
    }
}

DLLEXPORT void arr_sub(const qfix *in1, const qfix* in2, const uint32_t len, qfix* out){
    for(uint32_t i = 0; i < len; i++){
        out[i] = in1[i] - in2[i];
    }
}

DLLEXPORT void arr_scale(const qfix *in1, const qfix scale, const uint32_t len, qfix* out){
    for(uint32_t i = 0; i < len; i++){
        out[i] = qfix_mul(in1[i], scale);
    }
}

DLLEXPORT void arr_emul(const qfix *in1, const qfix* in2, const uint32_t len, qfix* out){
    for(uint32_t i = 0; i < len; i++){
        out[i] = qfix_mul(in1[i], in2[i]);
    }
}

DLLEXPORT void arr_mmul(const qfix *in1, const qfix* in2, const uint16_t rows1, const uint16_t c1_r2, const uint16_t cols2, qfix* out){
    uint32_t a_row_offset = 0;
    for (uint16_t i = 0; i < rows1; i++){
        for (uint16_t j = 0; j < cols2; j++){
            _tmp_larger sum = 0;
            // 内循环使用指针运算
            qfix *a_ptr = &in1[a_row_offset];
            qfix *b_ptr = &in2[j]; // B的第j列起始位置
            for (uint16_t kk = 0; kk < c1_r2; kk++){
                sum += ((_tmp_larger)(*a_ptr) * (*b_ptr));
                a_ptr++;
                b_ptr += cols2; // 跳到B的下一行（列优先访问）
            }
            out[i * cols2 + j] = (qfix)(sum  >> QSHIFT);
        }
        a_row_offset += c1_r2;
    }
}

DLLEXPORT void arr_mtpose(const qfix *in1, const uint16_t rows1, const uint16_t cols1, qfix* out){
    for (uint16_t i = 0; i < rows1; i++){
        qfix *start0 = in1 + i * rows1;
        qfix *destBlock0 = out + i;
        for (uint16_t j = 0; j < cols1; j++){
            *destBlock0 = *start0;
            start0++;
            destBlock0 += rows1;
        }
    }
}
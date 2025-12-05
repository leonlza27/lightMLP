#include "matrix_static.h"


void matrix_bp_add(const matrix_bp_data *madd1, const matrix_bp_data *madd2, matrix_bp_data *resu) {
    resu->cols = madd1->cols;
    resu->rows = madd1->rows;
    uint32_t size = madd1->cols * madd1->rows;
    
    bp *in1_base = madd1->data;
    bp *in2_base = madd2->data;
    bp *out_base = resu->data;
    
    uint16_t i = 0;

    for(; i < size; i+=4){
        out_base[i] = in1_base[i] + in2_base[i];
        out_base[i + 1] = in1_base[i + 1] + in2_base[i + 1];
        out_base[i + 2] = in1_base[i + 2] + in2_base[i + 2];
        out_base[i + 3] = in1_base[i + 3] + in2_base[i + 3];
    }

    for(; i < size; i++){
        out_base[i] = in1_base[i] + in2_base[i];
    }
}

void matrix_bp_mulpty(const matrix_bp_data *mmul1, const matrix_bp_data *mmul2, matrix_bp_data *resu) {
        
    const uint16_t k = mmul1->cols;
    const uint16_t m = mmul1->rows;
    const uint16_t n = mmul2->cols;
    
    resu->rows = m;
    resu->cols = n;
    
    bp *a = mmul1->data;
    bp *b = mmul2->data;
    bp *c = resu->data;
    
    // 关键优化：预计算偏移，减少乘法运算
    for(uint16_t i = 0; i < m; i++){
        // 预计算A的行偏移
        uint32_t a_row_offset = i * k;
        
        for(uint16_t j = 0; j < n; j++){
            _tmp_larger sum = 0;
            
            // 内循环使用指针运算
            bp *a_ptr = &a[a_row_offset];
            bp *b_ptr = &b[j];  // B的第j列起始位置
            
            for(uint16_t kk = 0; kk < k; kk++){
                sum += ((_tmp_larger)(*a_ptr) * (*b_ptr));
                a_ptr++;
                b_ptr += n;  // 跳到B的下一行（列优先访问）
            }
            
            c[i * n + j] = (bp)((sum + (1LL << (QSHIFT - 1))) >> QSHIFT);
        }
    }    
}

// 优化版本1：使用局部变量和指针运算
void matrix_bp_mulpty_optimized(const matrix_bp_data *A, 
                          const matrix_bp_data *B, 
                          matrix_bp_data *C) {
    const uint16_t k = A->cols;
    const uint16_t m = A->rows;
    const uint16_t n = B->cols;
    
    C->rows = m;
    C->cols = n;
    
    bp *a = A->data;
    bp *b = B->data;
    bp *c = C->data;
    
    // 关键优化：预计算偏移，减少乘法运算
    for(uint16_t i = 0; i < m; i++){
        // 预计算A的行偏移
        uint32_t a_row_offset = i * k;
        
        for(uint16_t j = 0; j < n; j++){
            _tmp_larger sum = 0;
            
            // 内循环使用指针运算
            bp *a_ptr = &a[a_row_offset];
            bp *b_ptr = &b[j];  // B的第j列起始位置
            
            for(uint16_t kk = 0; kk < k; kk++){
                sum += ((_tmp_larger)(*a_ptr) * (*b_ptr));
                a_ptr++;
                b_ptr += n;  // 跳到B的下一行（列优先访问）
            }
            
            c[i * n + j] = (bp)((sum + (1LL << (QSHIFT - 1))) >> QSHIFT);
        }
    }
}
#ifdef ON_DBG

void matrix_bp_mulpty_raw(const matrix_bp_data *mmul1, const matrix_bp_data *mmul2, matrix_bp_data *resu) {
    resu->cols = mmul2->cols;
    resu->rows = mmul1->rows;
    const uint16_t _k = mmul1->cols,
                    out_rows = mmul1->rows,
                    out_cols = mmul2->cols;

    bp *in1_base = mmul1->data,
           *in2_base = mmul2->data,
           *out_base = resu->data;
           
    for(uint16_t i = 0; i < out_rows; i++){
        for(uint16_t j = 0; j < out_cols; j++){
            _tmp_larger sum = 0;
            for(uint16_t k = 0; k < _k; k++){
                sum += ((_tmp_larger)in1_base[i * _k + k] * in2_base[k * out_cols + j]);
            }
            out_base[i * out_cols + j] = (bp)(sum + (1LL << (QSHIFT -1))) >> QSHIFT;
        }
    }
}

#endif

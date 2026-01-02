#include "matrix_gern.h"

void matrix_bp_add(const matrix_bp_data *madd1, const matrix_bp_data *madd2, matrix_bp_data *resu) {
    resu->cols = madd1->cols;
    resu->rows = madd1->rows;
    uint32_t size = madd1->cols * madd1->rows;
    
    bp *in1_base = (bp*)madd1->data;
    bp *in2_base = (bp*)madd2->data;
    bp *out_base = resu->data;
    multi_process(size,[=]wrapper_custom_start_end{
    uint32_t i = start;

    for(; i < end; i+=4){
        out_base[i] = in1_base[i] + in2_base[i];
        out_base[i + 1] = in1_base[i + 1] + in2_base[i + 1];
        out_base[i + 2] = in1_base[i + 2] + in2_base[i + 2];
        out_base[i + 3] = in1_base[i + 3] + in2_base[i + 3];
    }

    for(; i < end; i++){
        out_base[i] = in1_base[i] + in2_base[i];
    }
    });
}

void matrix_bp_scale(const matrix_bp_data *msrc, const qfix num, matrix_bp_data *resu){
        resu->cols = msrc->cols;
        resu->rows = msrc->rows;
        uint32_t size = msrc->cols * msrc->rows;
        
        bp *in1_base = (bp*)msrc->data;
        bp *out_base = resu->data;
        
        multi_process(size,[=]wrapper_custom_start_end{
        uint16_t i = start;

        for(; i < end; i+=4){
            out_base[i] = qfix_mul(in1_base[i], num);
            out_base[i + 1] = qfix_mul(in1_base[i + 1], num);
            out_base[i + 2] = qfix_mul(in1_base[i + 2], num);
            out_base[i + 3] = qfix_mul(in1_base[i + 3], num);
        }

        for(; i < end; i++){
            out_base[i] = qfix_mul(in1_base[i], num);
        }
    });
}

void matrix_bp_mulpty(const matrix_bp_data *mmul1, const matrix_bp_data *mmul2, matrix_bp_data *resu) {
        
    const uint16_t k = mmul1->cols;
    const uint16_t m = mmul1->rows;
    const uint16_t n = mmul2->cols;
    
    resu->rows = m;
    resu->cols = n;
    
    bp *a = (bp*)mmul1->data;
    bp *b = (bp*)mmul2->data;
    bp *c = resu->data;
    
    // 关键优化：预计算偏移，减少乘法运算
    if(m > n){
    multi_process(m,[=]wrapper_custom_start_end{
        // 预计算A的行偏移
        uint32_t a_row_offset = start * k;
        for(uint16_t i = start; i < end; i++){
        
        
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

        a_row_offset += k;
        }    
    });

    }else{
        multi_process(n,[=]wrapper_custom_start_end{
        for(uint16_t j = start; j < end; j++){
        // 预计算A的行偏移
        uint32_t a_row_offset = 0;
        
        for(uint16_t i = 0; i < m; i++){
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
            a_row_offset += k;
            }
        }    
    });

    }
}

void matrix_bp_transpose(const matrix_bp_data *source, matrix_bp_data *dest){
    bp *src = (bp*)source->data;
    bp *dst = dest->data;

    const uint16_t m = source->rows;
    const uint16_t n = source->cols;

    dest->cols = m;
    dest->rows = n;

    uint8_t isInplace = (source == dest);

    if(m > n){
        multi_process(m,[=]wrapper_custom_start_end{
            for(uint16_t i = start; i < end; i++){
                bp *start0 = src + i * n;
                bp *destBlock0 = dst + i;

                for(uint16_t j = 0; j < n; j++){
                    *destBlock0 = *start0;
                    start0++;
                    destBlock0 += m;
                }
            }
        });
    }else{
        multi_process(n,[=]wrapper_custom_start_end{
            for(uint16_t j = start;j < end;j++){
                bp *start0 = src + j;
                bp *destBlock0 = dst + j * m;
                for(uint16_t i = 0; i < m; i++){
                    *destBlock0 = *start0;
                    start0 += n;
                    destBlock0++;
                }
            }
        });
    }
}

void matrix_bp_sub(const matrix_bp_data *msrc, const matrix_bp_data *msub, matrix_bp_data *resu){
    resu->cols = msrc->cols;
    resu->rows = msrc->rows;
    uint32_t size = msrc->cols * msrc->rows;
    
    bp *in1_base = (bp*)msrc->data;
    bp *in2_base = (bp*)msub->data;
    bp *out_base = resu->data;
    multi_process(size,[=]wrapper_custom_start_end{
    uint16_t i = start;

    for(; i < end; i+=4){
        out_base[i] = in1_base[i] - in2_base[i];
        out_base[i + 1] = in1_base[i + 1] - in2_base[i + 1];
        out_base[i + 2] = in1_base[i + 2] - in2_base[i + 2];
        out_base[i + 3] = in1_base[i + 3] - in2_base[i + 3];
    }

    for(; i < end; i++){
        out_base[i] = in1_base[i] - in2_base[i];
    }
    });
}

void matrix_bp_mulptyByElem(const matrix_bp_data *mmul1, const matrix_bp_data *mmul2, matrix_bp_data *resu){
    resu->cols = mmul1->cols;
    resu->rows = mmul1->rows;
    uint32_t size = mmul1->cols * mmul1->rows;
    
    bp *in1_base = (bp*)mmul1->data;
    bp *in2_base = (bp*)mmul2->data;
    bp *out_base = resu->data;
    multi_process(size,[=]wrapper_custom_start_end{
    uint16_t i = start;

    for(; i < end; i+=4){
        out_base[i] = qfix_mul(in1_base[i],in2_base[i]);
        out_base[i + 1] = qfix_mul(in1_base[i + 1], in2_base[i + 1]);
        out_base[i + 2] = qfix_mul(in1_base[i + 2], in2_base[i + 2]);
        out_base[i + 3] = qfix_mul(in1_base[i + 3], in2_base[i + 3]);
    }

    for(; i < end; i++){
        out_base[i] = qfix_mul(in1_base[i],in2_base[i]);
    }
    });
}

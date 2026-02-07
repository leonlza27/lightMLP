#include "matrix_static.h"
#include "../tools/mult_proc.h"

typedef struct dExchange{
    bp *opt;
    bp *in1;
    bp *in2;
    qfix num;
}dExchange;

typedef struct dExchange2dop{
    bp *opt;
    bp *in1;
    bp *in2;
    uint16_t dimoth0;
    uint16_t dimoth1;
    uint16_t dimoth2;
}dExchange2dop;

registerTask(maddloop, {
    dExchange *datain = (dExchange*)data;
    bp *in1_base = datain->in1;
    bp *in2_base = datain->in2;
    bp *out_base = datain->opt;
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
})

registerTask(msubloop, {
    dExchange *datain = (dExchange *)data;
    bp *in1_base = datain->in1;
    bp *in2_base = datain->in2;
    bp *out_base = datain->opt;
    uint32_t i = start;
    for (; i < end; i += 4){
        out_base[i] = in1_base[i] - in2_base[i];
        out_base[i + 1] = in1_base[i + 1] - in2_base[i + 1];
        out_base[i + 2] = in1_base[i + 2] - in2_base[i + 2];
        out_base[i + 3] = in1_base[i + 3] - in2_base[i + 3];
    }
    for (; i < end; i++){
        out_base[i] = in1_base[i] - in2_base[i];
    }
})

registerTask(mscaleloop, {
    dExchange *datain = (dExchange *)data;
    bp *in1_base = datain->in1;
    bp num = datain->num;
    bp *out_base = datain->opt;
    uint32_t i = start;
    for (; i < end; i += 4){
        out_base[i] = qfix_mul(in1_base[i], num);
        out_base[i + 1] = qfix_mul(in1_base[i + 1], num);
        out_base[i + 2] = qfix_mul(in1_base[i + 2], num);
        out_base[i + 3] = qfix_mul(in1_base[i + 3], num);
    }
    for (; i < end; i++){
        out_base[i] = qfix_mul(in1_base[i], num);
    }
})

registerTask(mmulelemloop, {
    dExchange *datain = (dExchange *)data;
    bp *in1_base = datain->in1;
    bp *in2_base = datain->in2;
    bp *out_base = datain->opt;
    uint32_t i = start;
    for (; i < end; i += 4){
        out_base[i] = qfix_mul(in1_base[i], in2_base[i]);
        out_base[i + 1] = qfix_mul(in1_base[i + 1], in2_base[i + 1]);
        out_base[i + 2] = qfix_mul(in1_base[i + 2], in2_base[i + 2]);
        out_base[i + 3] = qfix_mul(in1_base[i + 3], in2_base[i + 3]);
    }
    for (; i < end; i++){
        out_base[i] = qfix_mul(in1_base[i], in2_base[i]);
    }
})

registerTask(mmul_movern, {
    dExchange2dop *datain = (dExchange2dop *)data;
    bp *a = datain->in1;
    bp *b = datain->in2;
    bp *c = datain->opt;
    uint16_t k = datain->dimoth1;
    uint16_t n = datain->dimoth2;
    uint32_t a_row_offset = start * k;
    for (uint16_t i = start; i < end; i++){
        for (uint16_t j = 0; j < n; j++){
            _tmp_larger sum = 0;
            // 内循环使用指针运算
            bp *a_ptr = &a[a_row_offset];
            bp *b_ptr = &b[j]; // B的第j列起始位置
            for (uint16_t kk = 0; kk < k; kk++){
                sum += ((_tmp_larger)(*a_ptr) * (*b_ptr));
                a_ptr++;
                b_ptr += n; // 跳到B的下一行（列优先访问）
            }
            c[i * n + j] = (bp)((sum + (1LL << (QSHIFT - 1))) >> QSHIFT);
        }
        a_row_offset += k;
    }
})

registerTask(mmul_noverm, {
    dExchange2dop *datain = (dExchange2dop *)data;
    bp *a = datain->in1;
    bp *b = datain->in2;
    bp *c = datain->opt;
    uint16_t k = datain->dimoth1;
    uint16_t m = datain->dimoth0;
    uint16_t n = datain->dimoth2;
    for (uint16_t j = start; j < end; j++){
        // 预计算A的行偏移
        uint32_t a_row_offset = 0;
        for (uint16_t i = 0; i < m; i++){
            _tmp_larger sum = 0;
            // 内循环使用指针运算
            bp *a_ptr = &a[a_row_offset];
            bp *b_ptr = &b[j]; // B的第j列起始位置
            for (uint16_t kk = 0; kk < k; kk++){
                sum += ((_tmp_larger)(*a_ptr) * (*b_ptr));
                a_ptr++;
                b_ptr += n; // 跳到B的下一行（列优先访问）
            }
            c[i * n + j] = (bp)((sum + (1LL << (QSHIFT - 1))) >> QSHIFT);
            a_row_offset += k;
        }
    }
})

registerTask(mT_movern, {
    dExchange2dop *datain = (dExchange2dop *)data;
    bp *src = datain->in1;
    bp *dst = datain->opt;
    uint16_t m = datain->dimoth0;
    uint16_t n = datain->dimoth2;
    for (uint16_t i = start; i < end; i++){
        bp *start0 = src + i * n;
        bp *destBlock0 = dst + i;
        for (uint16_t j = 0; j < n; j++)
        {
            *destBlock0 = *start0;
            start0++;
            destBlock0 += m;
        }
    }
})

registerTask(mT_noverm, {
    dExchange2dop *datain = (dExchange2dop *)data;
    bp *src = datain->in1;
    bp *dst = datain->opt;
    uint16_t m = datain->dimoth0;
    uint16_t n = datain->dimoth2;
    for (uint16_t j = start; j < end; j++){
        bp *start0 = src + j;
        bp *destBlock0 = dst + j * m;
        for (uint16_t i = 0; i < m; i++){
            *destBlock0 = *start0;
            start0 += n;
            destBlock0++;
        }
    }
})

matrix_bp alloc_matrix_bp(uint16_t m, uint16_t n){
    matrix_bp ret = (matrix_bp)malloc(sizeof(matrix_bp_data) + m * n * sizeof(qfix));
    ret->rows = m;
    ret->cols = n;
    return ret;
}

void matrix_bp_add(const matrix_bp_data *madd1, const matrix_bp_data *madd2, matrix_bp_data *resu){
    resu->cols = madd1->cols;
    resu->rows = madd1->rows;
    uint32_t size = madd1->cols * madd1->rows;
    
    dExchange para = {resu->data, madd1->data, madd2->data, 0};

    multi_process_c(size, maddloop, para)
}

void matrix_bp_sub(const matrix_bp_data *msrc, const matrix_bp_data *msub, matrix_bp_data *resu) {
    resu->cols = msrc->cols;
    resu->rows = msrc->rows;
    uint32_t size = msrc->cols * msrc->rows;

    dExchange para = {resu->data, msrc->data, msub->data, 0};

    multi_process_c(size, msubloop, para)
}

void matrix_bp_scale(const matrix_bp_data *msrc, const qfix num, matrix_bp_data *resu){
    resu->cols = msrc->cols;
    resu->rows = msrc->rows;
    uint32_t size = msrc->cols * msrc->rows;

    dExchange para = {resu->data, msrc->data, 0, num};
        
    multi_process_c(size, mscaleloop, para)
}

void matrix_bp_mulptyByElem(const matrix_bp_data *mmul1, const matrix_bp_data *mmul2, matrix_bp_data *resu){
    resu->cols = mmul1->cols;
    resu->rows = mmul1->rows;
    uint32_t size = mmul1->cols * mmul1->rows;
    
    dExchange para = {resu->data, mmul1->data, mmul2->data, 0};
        
    multi_process_c(size, mmulelemloop, para)
}

void matrix_bp_mulpty(const matrix_bp_data *mmul1, const matrix_bp_data *mmul2, matrix_bp_data *resu) {
    const uint16_t k = mmul1->cols;
    const uint16_t m = mmul1->rows;
    const uint16_t n = mmul2->cols;
    
    resu->rows = m;
    resu->cols = n;
    
    dExchange2dop para = {resu->data, mmul1->data, mmul2->data, m, k, n};
    
    if(m > n){
        multi_process_c(m, mmul_movern, para)
    }else{
        multi_process_c(n, mmul_noverm, para)
    }

}

void matrix_bp_transpose(const matrix_bp_data *source, matrix_bp_data *dest){
    const uint16_t m = source->rows;
    const uint16_t n = source->cols;

    dest->cols = m;
    dest->rows = n;

    uint8_t isInplace = (source == dest);

    dExchange2dop para = {dest->data, source->data, 0, m, 0, n};
    
    if(m > n){
        multi_process_c(m, mT_movern, para)
    }else{
        multi_process_c(n, mT_noverm, para)
    }
}

void plot_matrix_bp(matrix_bp tg, uint16_t m, uint16_t n, uint8_t flag, ...){
    tg->rows = m;
    tg->cols = n;
    uint32_t size = m * n;
    qfix *dataDst = tg->data;

    va_list elemin;
    va_start(elemin, flag);

    int64_t arg0 = va_arg(elemin, int64_t);

    switch(flag){
        case shapeOnly: goto _ret; break;
        case copyFromExisted: {
            qfix* existed = va_arg(elemin, qfix*);
            for(uint32_t i = 0; i < size; i++){
                dataDst[i] = existed[i];
            }

            goto _ret;
        }
        break;

        default: break;
    }

    for(uint32_t i = 0; i < size; i++){
        dataDst[i] = va_arg(elemin, qfix);
    }

    _ret:
    va_end(elemin);
}

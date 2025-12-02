#include "matrix_static.h"


void matrix_bp_add(const matrix_bp_data *madd1, const matrix_bp_data *madd2, matrix_bp_data *resu) {
    resu->cols = madd1->cols;
    resu->rows = madd1->rows;
    uint32_t size = madd1->cols * madd1->rows;
    
    bp *in1_base = madd1->data;
    bp *in2_base = madd2->data;
    bp *out_base = resu->data;
    
    for(uint16_t i = 0; i < size; i++){
        out_base[i] = in1_base[i] + in2_base[i];
    }
}

void matrix_bp_mulpty(const matrix_bp_data *mmul1, const matrix_bp_data *mmul2, matrix_bp_data *resu) {
    const uint16_t _k = mmul1->cols,
                    out_rows = mmul1->rows,
                    out_cols = mmul2->cols;
    
    resu->cols = out_cols;
    resu->rows = out_rows;
    
    //matrix_qf.data地址源
    bp *in1_base = mmul1->data,
           *in2_base = mmul2->data,
           *out_base = resu->data;
        
    for(uint16_t i = 0; i < out_rows; i += 2){
        for(uint16_t j = 0; j < out_cols; j += 2){
            _tmp_larger sum00 = 0, sum01 = 0, sum10 = 0, sum11 = 0;
            
            uint16_t k;
            // 内层循环4路unroll
            for(k = 0; k + 3 < _k; k += 4){
                // 第一行
                bp a0_0 = in1_base[i * _k + k];
                bp a0_1 = in1_base[i * _k + k+1];
                bp a0_2 = in1_base[i * _k + k+2];
                bp a0_3 = in1_base[i * _k + k+3];
                
                // 第二行（如果存在）
                bp a1_0 = (i+1 < out_rows) ? in1_base[(i+1) * _k + k] : 0;
                bp a1_1 = (i+1 < out_rows) ? in1_base[(i+1) * _k + k+1] : 0;
                bp a1_2 = (i+1 < out_rows) ? in1_base[(i+1) * _k + k+2] : 0;
                bp a1_3 = (i+1 < out_rows) ? in1_base[(i+1) * _k + k+3] : 0;
                
                // B矩阵的列
                bp b0_0 = in2_base[k * out_cols + j];
                bp b0_1 = (j+1 < out_cols) ? in2_base[k * out_cols + j+1] : 0;
                bp b1_0 = in2_base[(k+1) * out_cols + j];
                bp b1_1 = (j+1 < out_cols) ? in2_base[(k+1) * out_cols + j+1] : 0;
                bp b2_0 = in2_base[(k+2) * out_cols + j];
                bp b2_1 = (j+1 < out_cols) ? in2_base[(k+2) * out_cols + j+1] : 0;
                bp b3_0 = in2_base[(k+3) * out_cols + j];
                bp b3_1 = (j+1 < out_cols) ? in2_base[(k+3) * out_cols + j+1] : 0;
                
                sum00 += a0_0 * b0_0 + a0_1 * b1_0 + a0_2 * b2_0 + a0_3 * b3_0;
                if(j+1 < out_cols) sum01 += a0_0 * b0_1 + a0_1 * b1_1 + a0_2 * b2_1 + a0_3 * b3_1;
                
                if(i+1 < out_rows) {
                    sum10 += a1_0 * b0_0 + a1_1 * b1_0 + a1_2 * b2_0 + a1_3 * b3_0;
                    if(j+1 < out_cols) sum11 += a1_0 * b0_1 + a1_1 * b1_1 + a1_2 * b2_1 + a1_3 * b3_1;
                }
            }
            
            // 处理剩余k
            for(; k < _k; k++){
                bp a0 = in1_base[i * _k + k];
                bp a1 = (i+1 < out_rows) ? in1_base[(i+1) * _k + k] : 0;
                bp b0 = in2_base[k * out_cols + j];
                bp b1 = (j+1 < out_cols) ? in2_base[k * out_cols + j+1] : 0;
                
                sum00 += a0 * b0;
                if(j+1 < out_cols) sum01 += a0 * b1;
                if(i+1 < out_rows) sum10 += a1 * b0;
                if(i+1 < out_rows && j+1 < out_cols) sum11 += a1 * b1;
            }
            
            // 存储结果
            out_base[i * out_cols + j] = (bp)(sum00 + (1LL << (QSHIFT -1))) >> QSHIFT;
            if(j+1 < out_cols) out_base[i * out_cols + j+1] = (bp)(sum01 + (1LL << (QSHIFT -1))) >> QSHIFT;
            if(i+1 < out_rows) {
                out_base[(i+1) * out_cols + j] = (bp)(sum10 + (1LL << (QSHIFT -1))) >> QSHIFT;
                if(j+1 < out_cols) out_base[(i+1) * out_cols + j+1] = (bp)(sum11 + (1LL << (QSHIFT -1))) >> QSHIFT;
            }
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

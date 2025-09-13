#include "matrix_static.h"
#include <stdlib.h>

static inline uint16_t min(uint16_t num1, uint16_t num2){
    return num1 < num2? num1: num2;
}

void matrix_qfloat_init(matrix_qfloat_data *matrix, uint16_t m, uint16_t n, qfloat *data) {
    matrix->cols = n;
    matrix->rows = m;
    matrix->data = (qfloat*)malloc(m * n * sizeof(qfloat));
    
    if(matrix->data == NULL) {
        // 处理内存分配失败
        return;
    }
    
    if(data == 0) {
        for(uint32_t i = 0; i < m*n; i++) {
            matrix->data[i] = 0;
        }
        return;
    }
    for(uint32_t i = 0; i < m*n; i++) {
        matrix->data[i] = data[i];
    }
}

void matrix_qfloat_reset(matrix_qfloat_data *matrix, uint16_t m, uint16_t n, qfloat *data) {
    // 检查是否需要重新分配内存
    if(matrix->data == NULL || (m * n) != (matrix->rows * matrix->cols)) {
        // 释放原有内存
        if(matrix->data != NULL) {
            free(matrix->data);
        }
        // 分配新内存
        matrix->data = (qfloat*)malloc(m * n * sizeof(qfloat));
        if(matrix->data == NULL) {
            // 处理内存分配失败
            return;
        }
    }

    matrix->cols = n;
    matrix->rows = m;
    
    if(data == 0) {
        for(uint32_t i = 0; i < m*n; i++) {
            matrix->data[i] = 0;
        }
        return;
    }
    for(uint32_t i = 0; i < m*n; i++) {
        matrix->data[i] = data[i];
    }
}

void matrix_qfloat_add(const matrix_qfloat_data *madd1, const matrix_qfloat_data *madd2, matrix_qfloat_data *resu) {
    resu->cols = madd1->cols;
    resu->rows = madd1->rows;
    register uint32_t size = madd1->cols * madd1->rows;
    
    qfloat *in1_base = madd1->data;
    qfloat *in2_base = madd2->data;
    qfloat *out_base = resu->data;
    
    for(uint16_t i_blk = 0; i_blk < size; i_blk += BLOCK_SIZE){
        uint16_t i_in_max = min(size - i_blk, BLOCK_SIZE);
        qfloat *in1row = in1_base + i_blk;
        qfloat *in2row = in2_base + i_blk;
        qfloat *outrow = out_base + i_blk;
        for(uint16_t i = 0; i < i_in_max; i++){
            outrow[i] = in1row[i] + in2row[i];
        }
    }
}

void matrix_qfloat_mulpty(const matrix_qfloat_data *mmul1, const matrix_qfloat_data *mmul2, matrix_qfloat_data *resu) {
    resu->cols = mmul2->cols;
    resu->rows = mmul1->rows;
    const uint16_t _k = mmul1->cols,
                    out_rows = mmul1->rows,
                    out_cols = mmul2->cols;
    
    //matrix_qf.data地址源
    qfloat *in1_base = mmul1->data,
           *in2_base = mmul2->data,
           *out_base = resu->data;
    
    
    for(uint16_t i_blk = 0; i_blk < out_rows; i_blk += BLOCK_SIZE){
        for(uint16_t j_blk = 0; j_blk < out_cols; j_blk += BLOCK_SIZE){
            for(uint16_t k_blk = 0; k_blk < _k; k_blk += BLOCK_SIZE){
            
            uint16_t i_in_max = min(out_rows - i_blk, BLOCK_SIZE);
            uint16_t j_in_max = min(out_cols - j_blk, BLOCK_SIZE);
            uint16_t k_in_max = min(_k - k_blk, BLOCK_SIZE);
            
                for(uint16_t i = 0; i < i_in_max; i++){
               
                    for(uint16_t j = 0; j < j_in_max; j++){
                        
                        _tmp_larger sum = 0;
                        for(uint16_t k = 0; k < k_in_max; k++){
                            sum += ((_tmp_larger)in1_base[(i_blk + i) * _k + (k_blk + k)] * in2_base[(k_blk + k) * out_rows + (j_blk + j)]);
                        }
                        out_base[(i_blk + i) * out_cols + (j_blk + j)] = (qfloat)(sum + (1LL << (QSHIFT -1))) >> QSHIFT;
                    }
                }
            }
        }
    }
    
}

void DbgPrint_qfloat_matrix(const matrix_qfloat_data *matrix, char *mask){
    if(mask[0] != 0){
        puts(mask);
    }
    for(uint32_t i =0; i<matrix->rows*matrix->cols;i++){
        printf("%.2f\t",qfloat_to_float(matrix->data[i]));
        if(i%matrix->cols==matrix->cols-1) putc('\n',stdout);
    }
}

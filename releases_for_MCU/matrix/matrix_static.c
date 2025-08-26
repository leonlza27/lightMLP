#include "matrix_static.h"

// 定义最大矩阵尺寸，根据实际应用调整
#define MAX_MATRIX_ROWS 32
#define MAX_MATRIX_COLS 32
static qfloat matrix_data_buffer[MAX_MATRIX_ROWS * MAX_MATRIX_COLS];

void matrix_qfloat_init(matrix_qfloat_data *matrix, uint16_t m, uint16_t n, f_q16_16 *data) {
    matrix->cols = n;
    matrix->rows = m;
    matrix->data = matrix_data_buffer;
    
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

void matrix_qfloat_reset(matrix_qfloat_data *matrix, uint16_t m, uint16_t n, f_q16_16 *data) {
    matrix->cols = n;
    matrix->rows = m;
    matrix->data = matrix_data_buffer;
    
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
    register uint32_t i = 0;
    
    // 部分循环展开
    for(; i + 3 < size; i += 4) {
        resu->data[i] = madd1->data[i] + madd2->data[i];
        resu->data[i+1] = madd1->data[i+1] + madd2->data[i+1];
        resu->data[i+2] = madd1->data[i+2] + madd2->data[i+2];
        resu->data[i+3] = madd1->data[i+3] + madd2->data[i+3];
    }
    
    // 处理剩余部分
    for(; i < size; i++) {
        resu->data[i] = madd1->data[i] + madd2->data[i];
    }
}

void matrix_qfloat_mulpty(const matrix_qfloat_data *mmul1, const matrix_qfloat_data *mmul2, matrix_qfloat_data *resu) {
    resu->cols = mmul2->cols;
    resu->rows = mmul1->rows;
    register uint16_t mmul1_cols = mmul1->cols;
    register uint16_t mmul2_cols = mmul2->cols;
    register uint16_t resu_cols = resu->cols;
    
    for(uint16_t i = 0; i < resu->rows; i++) {
        for(uint16_t j = 0; j < resu_cols; j++) {
            register qfloat temp = 0;
            register uint16_t k = 0;
            
            // 部分循环展开
            for(; k + 3 < mmul1_cols; k += 4) {
                temp += qfloat_mul(mmul1->data[i*mmul1_cols + k], mmul2->data[k*mmul2_cols + j]);
                temp += qfloat_mul(mmul1->data[i*mmul1_cols + k+1], mmul2->data[(k+1)*mmul2_cols + j]);
                temp += qfloat_mul(mmul1->data[i*mmul1_cols + k+2], mmul2->data[(k+2)*mmul2_cols + j]);
                temp += qfloat_mul(mmul1->data[i*mmul1_cols + k+3], mmul2->data[(k+3)*mmul2_cols + j]);
            }
            
            // 处理剩余部分
            for(; k < mmul1_cols; k++) {
                temp += qfloat_mul(mmul1->data[i*mmul1_cols + k], mmul2->data[k*mmul2_cols + j]);
            }
            
            resu->data[i*resu_cols + j] = temp;
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

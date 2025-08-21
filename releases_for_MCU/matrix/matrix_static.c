#include "matrix_static.h"

void matrix_qfloat_init(matrix_qfloat_data *matrix, uint16_t m, uint16_t n, f_q16_16 *data){
    matrix->cols = n;
    matrix->rows = m;
    matrix->data = (qfloat*)malloc(sizeof(qfloat)*m*n);
    if(data == 0){
        for(uint32_t i = 0; i < m*n;i++){
            matrix->data[i] = 0;
        }
        return;
    }
    for(uint32_t i = 0; i < m*n;i++){
        matrix->data[i] = data[i];
    }
}

void matrix_qfloat_reset(matrix_qfloat_data *matrix,uint16_t m,uint16_t n,f_q16_16 *data){
    if(matrix->cols * matrix->rows != m * n){
        free(matrix->data);
        matrix->data = (qfloat*)malloc(sizeof(qfloat)*m*n);
    }
    matrix->cols = n;
    matrix->rows = m;    
    if(data == 0){
        for(uint32_t i = 0; i < m*n;i++){
            matrix->data[i] = 0;
        }
        return;
    }
    for(uint32_t i = 0; i < m*n;i++){
        matrix->data[i] = data[i];
    }
}

void matrix_qfloat_add(const matrix_qfloat_data *madd1, const matrix_qfloat_data *madd2, matrix_qfloat_data *resu){
    resu->cols = madd1->cols;
    resu->rows = madd1->rows;
    for(uint32_t i = 0; i < madd1->cols*madd1->rows;i++){
        resu->data[i] = madd1->data[i] + madd2->data[i];
    }
}

void matrix_qfloat_mulpty(const matrix_qfloat_data *mmul1, const matrix_qfloat_data *mmul2, matrix_qfloat_data *resu){
    resu->cols = mmul2->cols;
    resu->rows = mmul1->rows;
    for(uint16_t i =0; i < resu->rows; i++){
        for(uint16_t j =0; j< resu->cols; j++){
            qfloat temp = 0;
            for(uint16_t k =0;k <mmul1->cols;k++){
                temp += qfloat_mul(mmul1->data[i*mmul1->cols+k],mmul2->data[k*mmul2->cols+j]);
            }
            resu->data[i*resu->cols+j] = temp;
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

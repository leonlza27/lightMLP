#include "matrix_static.h"
#include<stdlib.h>

void matrix_Q16_16_init(matrix_Q16_16_data *matrix, uint16_t m, uint16_t n, f_q16_16 *data){
    matrix->cols = m;
    matrix->rows = n;
    matrix->data = (f_q16_16*)malloc(sizeof(f_q16_16)*m*n);
    for(uint32_t i = 0; i < m*n;i++){
        matrix->data[i] = data[i];
    }
}

void matrix_Q16_16_add(const matrix_Q16_16_data *madd1, const matrix_Q16_16_data *madd2, matrix_Q16_16_data *resu){
    resu->cols = madd1->cols;
    resu->rows = madd1->rows;
    for(uint32_t i = 0; i < madd1->cols*madd1->rows;i++){
        resu->data[i] = madd1->data[i] + madd2->data[i];
    }
}

void matrix_Q16_16_mulpty(const matrix_Q16_16_data *mmul1, const matrix_Q16_16_data *mmul2, matrix_Q16_16_data *resu){
    resu->cols = mmul1->rows;
    resu->rows = mmul2->cols;
    for(uint16_t i =0; i <mmul1->rows; i++){
        for(uint16_t j =0; j<mmul2->cols; j++){
            f_q16_16 temp = 0;
            for(uint16_t k =0;k <mmul1->cols;k++){
                temp += q16_16_mul(mmul1->data[i*mmul1->cols+k],mmul2->data[k*mmul2->cols+j]);
            }
            resu->data[i*resu->cols+j] = temp;
        }
    }
}

void DbgPrint_Q16_16_matrix(const matrix_Q16_16_data *matrix, char *mask){
    if(mask[0] != 0){
        puts(mask);
        putc('\n',stdout);
    }
    for(uint32_t i =0; i<matrix->rows*matrix->cols;i++){
        printf("%.2f\t",q16_16_to_float(matrix->data[i]));
        if(i%matrix->cols==matrix->cols-1) putc('\n',stdout);
    }
}

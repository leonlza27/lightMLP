//矩阵库仅用作mlp推理
#ifndef _matrix_static_
#define _matrix_static_

#include "../f_q16_16_operation/f_q16_16_op.h"
#include <stdio.h>
#include <malloc.h>

typedef struct matrix_Q16_16_data{
    uint16_t rows/*行数*/,cols/*列数*/;
    f_q16_16 *data;
}matrix_Q16_16_data;

#define alloc_matrix_Q16_16() (matrix_Q16_16_data*)malloc(sizeof(matrix_Q16_16_data))

#ifdef __cplusplus
extern "C" {
#endif

//初始化:m(列数) * n(行数)
void matrix_Q16_16_init(matrix_Q16_16_data *matrix,uint16_t m,uint16_t n,f_q16_16 *data);

void matrix_Q16_16_add(const matrix_Q16_16_data *madd1, const matrix_Q16_16_data *madd2, matrix_Q16_16_data *resu);

void matrix_Q16_16_mulpty(const matrix_Q16_16_data *mmul1, const matrix_Q16_16_data *mmul2, matrix_Q16_16_data *resu);


void DbgPrint_Q16_16_matrix(const matrix_Q16_16_data *matrix,char* mask);

#ifdef __cplusplus
}
#endif

#endif
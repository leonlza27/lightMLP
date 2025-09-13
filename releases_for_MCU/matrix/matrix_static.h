//矩阵库仅用作mlp推理
#ifndef _matrix_static_
#define _matrix_static_

#include "../qfloat_operation/qfloat_op.h"
#include <stdio.h>
#include <malloc.h>

#define BLOCK_SIZE 16

typedef struct matrix_qfloat_data{
    uint16_t rows/*行数*/,cols/*列数*/;
    qfloat *data;
}matrix_qfloat_data,*p_matrix_qfloat;

#define alloc_matrix_qfloat() (matrix_qfloat_data*)malloc(sizeof(matrix_qfloat_data))

#ifdef __cplusplus
extern "C" {
#endif

//初始化:m(行数) * n(列数)
void matrix_qfloat_init(matrix_qfloat_data *matrix,uint16_t m,uint16_t n,qfloat *data);

void matrix_qfloat_reset(matrix_qfloat_data *matrix,uint16_t m,uint16_t n,qfloat *data);

void matrix_qfloat_add(const matrix_qfloat_data *madd1, const matrix_qfloat_data *madd2, matrix_qfloat_data *resu);

void matrix_qfloat_mulpty(const matrix_qfloat_data *mmul1, const matrix_qfloat_data *mmul2, matrix_qfloat_data *resu);


void DbgPrint_qfloat_matrix(const matrix_qfloat_data *matrix,char* mask);

#ifdef __cplusplus
}
#endif

#endif
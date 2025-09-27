//矩阵库仅用作mlp推理
#ifndef _matrix_static_
#define _matrix_static_

#include "../CustomConf.h"
#include <stdio.h>
#include <malloc.h>

typedef qfix bp;

#define BLOCK_SIZE 16

typedef struct matrix_bp_data{
    bp *data;//行优先展开
    uint16_t rows/*行数*/,cols/*列数*/;
}matrix_bp_data,*p_matrix_bp, *p_matrix_qfix;

#define alloc_matrix_bp() (matrix_bp_data*)malloc(sizeof(matrix_bp_data))

#ifdef __cplusplus
extern "C" {
#endif
//初始化分配
matrix_bp_data *new_matrix_bp();

void matrix_bp_init(matrix_bp_data *matrix, uint16_t m, uint16_t n, bp *data);

//调整:m(行数) * n(列数)
void matrix_bp_set(matrix_bp_data *matrix, uint16_t m, uint16_t n, bp *data);

void matrix_bp_add(const matrix_bp_data *madd1, const matrix_bp_data *madd2, matrix_bp_data *resu);

void matrix_bp_mulpty(const matrix_bp_data *mmul1, const matrix_bp_data *mmul2, matrix_bp_data *resu);

#ifdef ON_DBG

//乘法,用于C误差验证优化
void matrix_bp_mulpty_raw(const matrix_bp_data *mmul1, const matrix_bp_data *mmul2, matrix_bp_data *resu);

void quickcheck_mmul(const matrix_bp_data *mmul1, const matrix_bp_data *mmul2, matrix_bp_data *rawresu, matrix_bp_data *actualresu);
#endif

#ifdef __cplusplus
}
#endif

#endif
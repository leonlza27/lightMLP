//矩阵库仅用作mlp推理
#ifndef _matrix_static_
#define _matrix_static_

#include "../CustomConf.h"
#include "../qfix_ops/qfix_op.h"
#include <stdio.h>
#include <malloc.h>

typedef qfix bp;

#define BLOCK_SIZE 4

#pragma pack(4)
typedef struct matrix_bp_data{
    uint16_t rows/*行数*/,cols/*列数*/;
    bp data[];//行优先展开
}matrix_bp_data,*matrix_bp, *matrix_qfix;

#define alloc_matrix_bp() (matrix_bp_data*)malloc(sizeof(matrix_bp_data))

#ifdef __cplusplus
extern "C" {
#endif
void matrix_bp_add(const matrix_bp_data *madd1, const matrix_bp_data *madd2, matrix_bp_data *resu);

void matrix_bp_mulpty(const matrix_bp_data *mmul1, const matrix_bp_data *mmul2, matrix_bp_data *resu);

void matrix_bp_mulpty_optimized(const matrix_bp_data *mmul1, const matrix_bp_data *mmul2, matrix_bp_data *resu);

#ifdef ON_DBG

//乘法,用于C误差验证优化
void matrix_bp_mulpty_raw(const matrix_bp_data *mmul1, const matrix_bp_data *mmul2, matrix_bp_data *resu);
#endif

#ifdef __cplusplus
}
#endif

#endif
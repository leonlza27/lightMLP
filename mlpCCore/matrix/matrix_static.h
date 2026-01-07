#ifndef _matrix_static
#define _matrix_static

#include <stdint.h>
#include <stdarg.h>
#include "../qfix_ops/qfix_op.h"

typedef qfix bp;

#define alloc_matrix_bp(m, n) (matrix_bp)malloc(sizeof(matrix_bp_data) + m * n * sizeof(qfix))

enum plotFlags{
    shapeOnly = INT32_MAX + 1,
    copyFromExisted
};

#pragma pack(4)
typedef struct matrix_bp_data{
    uint16_t rows/*行数*/,cols/*列数*/;
    bp data[];//行优先展开
}matrix_bp_data,*matrix_bp, *matrix_qfix;
#pragma pack(push)

void matrix_bp_add(const matrix_bp_data *madd1, const matrix_bp_data *madd2, matrix_bp_data *resu);

void matrix_bp_sub(const matrix_bp_data *msrc, const matrix_bp_data *msub, matrix_bp_data *resu);

void matrix_bp_mulpty(const matrix_bp_data *mmul1, const matrix_bp_data *mmul2, matrix_bp_data *resu);

void matrix_bp_mulptyByElem(const matrix_bp_data *mmul1, const matrix_bp_data *mmul2, matrix_bp_data *resu);

void matrix_bp_scale(const matrix_bp_data *msrc, const qfix num, matrix_bp_data *resu);

void matrix_bp_transpose(const matrix_bp_data *source, matrix_bp_data *dest);

/*
更方便的矩阵初始化

使用: [输出目标][行数m][列数n][其他数据]

=> plot_matrix_bp(mat, 3, 2, 1, 2, 3, 4, 5, 6);
或更直观的:
=> plot_matrix_bp(mat, 3, 2,
                    1, 2, 3, 
                    4, 5, 6);


标志(于[其他数据]):
    shapeOnly: 仅标记形状(尺寸) => plot_matrix_bp(mat, 3, 2, shapeOnly);
    copyFromExisted: 从外部拷贝=> plot_matrix_bp(mat, 3, 2, copyFromExisted, array);

注:示例为直接int32输入而非标准qfix, 务必使用float_to_qfix(可传整数与浮点)  

*/
void plot_matrix_bp(matrix_bp tg, uint16_t m, uint16_t n, ...);

#endif
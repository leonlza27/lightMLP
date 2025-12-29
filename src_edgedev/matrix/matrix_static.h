#ifndef _matrix_static
#define _matrix_static

#include <stdint.h>
#include "../qfix_ops/qfix_op.h"

typedef qfix bp;

#pragma pack(4)
typedef struct matrix_bp_data{
    uint16_t rows/*行数*/,cols/*列数*/;
    bp data[];//行优先展开
}matrix_bp_data,*matrix_bp, *matrix_qfix;
#pragma pack(push)

//__cpp_spefic:(强制)启用cpp名称修饰,需要用cpp编译器编译那部分c源码

#if defined(__cplusplus) && !defined(__cpp_spefic)
extern "C" {
#endif

void matrix_bp_add(const matrix_bp_data *madd1, const matrix_bp_data *madd2, matrix_bp_data *resu);

void matrix_bp_sub(const matrix_bp_data *msrc, const matrix_bp_data *msub, matrix_bp_data *resu);

void matrix_bp_mulpty(const matrix_bp_data *mmul1, const matrix_bp_data *mmul2, matrix_bp_data *resu);

void matrix_bp_mulptyByElem(const matrix_bp_data *mmul1, const matrix_bp_data *mmul2, matrix_bp_data *resu);

void matrix_bp_scale(const matrix_bp_data *msrc, const qfix num, matrix_bp_data *resu);

void matrix_bp_transpose(const matrix_bp_data *source, matrix_bp_data *dest);

#if defined(__cplusplus) && !defined(__cpp_spefic)
}
#endif

#endif
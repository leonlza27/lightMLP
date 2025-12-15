#ifndef _matrix_gern
#define _matrix_gern

#include <stdint.h>
#include "../qfix_ops/qfix_op.h"
#include "../tools/mult_proc.h"

typedef qfix bp;

#pragma pack(4)
typedef struct matrix_bp_data{
    uint16_t rows/*行数*/,cols/*列数*/;
    bp data[];//行优先展开
}matrix_bp_data,*matrix_bp, *matrix_qfix;

void matrix_bp_add(const matrix_bp_data *madd1, const matrix_bp_data *madd2, matrix_bp_data *resu);

void matrix_bp_mulpty(const matrix_bp_data *mmul1, const matrix_bp_data *mmul2, matrix_bp_data *resu);

void matrix_bp_transpose(const matrix_bp_data *source, matrix_bp_data *dest);

#endif

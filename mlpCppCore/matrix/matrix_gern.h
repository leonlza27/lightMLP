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
#pragma pack(push)

#ifdef ON_DBG
#include <stdio.h>

#define formatMatrix(mr)    {for(int i = 0; i < mr->rows; i++){\
        for(int j = 0; j < mr->cols; j++){\
            printf("%d\t",mr->data[i * mr->cols + j]);\
        }\
        putc('\n',stdout);\
    }putc('\n',stdout);}\

#define formatMatrix2f(mr)    {for(int i = 0; i < mr->rows; i++){\
        for(int j = 0; j < mr->cols; j++){\
            printf("%f\t",qfix_to_float(mr->data[i * mr->cols + j]));\
        }\
        putc('\n',stdout);\
    }putc('\n',stdout);}\
    
#else
#define formatMatrix(mr)

#define formatMatrix2f(mr)

#endif

void matrix_bp_add(const matrix_bp_data *madd1, const matrix_bp_data *madd2, matrix_bp_data *resu);

void matrix_bp_sub(const matrix_bp_data *msrc, const matrix_bp_data *msub, matrix_bp_data *resu);

void matrix_bp_mulpty(const matrix_bp_data *mmul1, const matrix_bp_data *mmul2, matrix_bp_data *resu);

void matrix_bp_mulptyByElem(const matrix_bp_data *mmul1, const matrix_bp_data *mmul2, matrix_bp_data *resu);

void matrix_bp_scale(const matrix_bp_data *msrc, const qfix num, matrix_bp_data *resu);

void matrix_bp_transpose(const matrix_bp_data *source, matrix_bp_data *dest);

#endif

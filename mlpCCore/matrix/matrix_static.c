#include "matrix_static.h"
#include "../arrop/arrop.h"

matrix_bp alloc_matrix_bp(uint16_t m, uint16_t n){
    matrix_bp ret = (matrix_bp)malloc(sizeof(matrix_bp_data) + m * n * sizeof(qfix));
    ret->rows = m;
    ret->cols = n;
#ifdef MBP_SAFE
    ret->bufcap = m * n;
#endif
    return ret;
}

//旧API接到新API上防止崩溃

void matrix_bp_add(const matrix_bp_data *madd1, const matrix_bp_data *madd2, matrix_bp_data *resu){
    resu->cols = madd1->cols;
    resu->rows = madd1->rows;
    uint32_t size = madd1->cols * madd1->rows;
    arr_add(madd1->data, madd2->data, size, resu->data);
}

void matrix_bp_sub(const matrix_bp_data *msrc, const matrix_bp_data *msub, matrix_bp_data *resu) {
    resu->cols = msrc->cols;
    resu->rows = msrc->rows;
    uint32_t size = msrc->cols * msrc->rows;
    arr_sub(msrc->data, msub->data, size, resu->data);
}

void matrix_bp_scale(const matrix_bp_data *msrc, const qfix num, matrix_bp_data *resu){
    resu->cols = msrc->cols;
    resu->rows = msrc->rows;
    uint32_t size = msrc->cols * msrc->rows;
    arr_scale(msrc->data, num, size, resu->data);
}

void matrix_bp_mulptyByElem(const matrix_bp_data *mmul1, const matrix_bp_data *mmul2, matrix_bp_data *resu){
    resu->cols = mmul1->cols;
    resu->rows = mmul1->rows;
    uint32_t size = mmul1->cols * mmul1->rows;
    arr_emul(mmul1->data, mmul2->data, size, resu->data);
}

void matrix_bp_mulpty(const matrix_bp_data *mmul1, const matrix_bp_data *mmul2, matrix_bp_data *resu) {
    resu->rows = mmul1->rows;
    resu->cols = mmul2->cols;
    arr_mmul(mmul1->data, mmul2->data, mmul1->rows, mmul1->cols, mmul2->cols, resu->data);
}

void matrix_bp_transpose(const matrix_bp_data *source, matrix_bp_data *dest){
    dest->rows = source->cols;
    dest->cols = source->rows;
    arr_mtpose(source->data, source->rows, source->cols, dest->data);
}

void plot_matrix_bp(matrix_bp tg, uint16_t m, uint16_t n, uint8_t flag, ...){
    tg->rows = m;
    tg->cols = n;
    uint32_t size = m * n;
    qfix *dataDst = tg->data;

    va_list elemin;
    va_start(elemin, flag);

    int64_t arg0 = va_arg(elemin, int64_t);

    switch(flag){
        case shapeOnly: goto _ret; break;
        case copyFromExisted: {
            qfix* existed = va_arg(elemin, qfix*);
            for(uint32_t i = 0; i < size; i++){
                dataDst[i] = existed[i];
            }
            goto _ret;
        }
        break;

        default: break;
    }

    for(uint32_t i = 0; i < size; i++){
        dataDst[i] = va_arg(elemin, qfix);
    }

    _ret:
    va_end(elemin);
}

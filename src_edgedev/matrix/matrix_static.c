#include "matrix_static.h"

static inline uint16_t min(uint16_t num1, uint16_t num2){
    return num1 < num2? num1: num2;
}

matrix_bp_data *new_matrix_bp(){
    matrix_bp_data* ret = alloc_matrix_bp();
    ret->data = 0;
    ret->cols = 0;
    ret->rows = 0;
    return ret;
}

void matrix_bp_init(matrix_bp_data *matrix, uint16_t m, uint16_t n, bp *data){
    matrix->cols = n;
    matrix->rows = m;
    
    matrix->data = (bp*)malloc(m * n * sizeof(bp));
    
    if(data == 0) {
        for(uint32_t i = 0; i < m*n; i++) {
            matrix->data[i] = 0;
        }
        return;
    }
    for(uint32_t i = 0; i < m*n; i++) {
        matrix->data[i] = data[i];
    }

}


void matrix_bp_set(matrix_bp_data *matrix, uint16_t m, uint16_t n, bp *data) {    
    if(matrix->data == NULL) {
        matrix->data = (bp*)malloc(m * n * sizeof(bp));
    }else if(m * n != matrix->cols * matrix->rows){
        free(matrix->data);
        matrix->data = (bp*)malloc(m * n * sizeof(bp));
    }
    
    if(matrix->data == NULL) return;
    
    matrix->cols = n;
    matrix->rows = m;
    
    if(data == 0) {
        for(uint32_t i = 0; i < m*n; i++) {
            matrix->data[i] = 0;
        }
        return;
    }
    for(uint32_t i = 0; i < m*n; i++) {
        matrix->data[i] = data[i];
    }
}

void matrix_bp_add(const matrix_bp_data *madd1, const matrix_bp_data *madd2, matrix_bp_data *resu) {
    resu->cols = madd1->cols;
    resu->rows = madd1->rows;
    register uint32_t size = madd1->cols * madd1->rows;
    
    bp *in1_base = madd1->data;
    bp *in2_base = madd2->data;
    bp *out_base = resu->data;
    
    for(uint16_t i = 0; i < size; i++){
        out_base[i] = in1_base[i] + in2_base[i];
    }
}

void matrix_bp_mulpty(const matrix_bp_data *mmul1, const matrix_bp_data *mmul2, matrix_bp_data *resu) {
    resu->cols = mmul2->cols;
    resu->rows = mmul1->rows;
    const uint16_t _k = mmul1->cols,
                    out_rows = mmul1->rows,
                    out_cols = mmul2->cols;
    
    //matrix_qf.data地址源
    bp *in1_base = mmul1->data,
           *in2_base = mmul2->data,
           *out_base = resu->data;
        
    /*
    qfix *A_tiled_start = in1_base + i_blk * _k;
    qfix *C_tiled_start = out_base + i_blk * out_cols;
    */
    
    const uint16_t A_blkrow_offset = BLOCK_SIZE * _k;
    const uint16_t C_blkrow_offset = BLOCK_SIZE * out_cols;
    
    qfix *A_tiled_start = in1_base;
    qfix *C_tiled_start = out_base;
    
    for(uint16_t i_blk = 0; i_blk < out_rows; i_blk += BLOCK_SIZE){
        for(uint16_t j_blk = 0; j_blk < out_cols; j_blk += BLOCK_SIZE){
            for(uint16_t k_blk = 0; k_blk < _k; k_blk += BLOCK_SIZE){
            
                uint16_t i_in_max = min(out_rows - i_blk, BLOCK_SIZE);
                uint16_t j_in_max = min(out_cols - j_blk, BLOCK_SIZE);
                uint16_t k_in_max = min(_k - k_blk, BLOCK_SIZE);
                
                /*
                qfix *A_tiled_buf = A_tiled_start + i * _k + k_blk;
                qfix *C_tiled_buf = C_tiled_start + i * out_cols + j_blk;
                */
                
                qfix *A_tiled_buf_0 = A_tiled_start + k_blk;
                qfix *C_tiled_buf_0 = C_tiled_start + j_blk;
            
                for(uint16_t i = 0; i < i_in_max; i++){
               
                    for(uint16_t j = 0; j < j_in_max; j++){
                    
                        _tmp_larger sum = 0;
                        for(uint16_t k = 0; k < k_in_max; k++){
                            sum += ((_tmp_larger)A_tiled_buf_0[k] * (_tmp_larger)in2_base[(k_blk + k) * out_cols + (j_blk + j)]);
                        }
                        C_tiled_buf_0[j] = (bp)(sum + (1LL << (QSHIFT -1))) >> QSHIFT;
                    }
                    
                    A_tiled_buf_0 += _k;
                    C_tiled_buf_0 += out_cols;
                }
            }
        }
        
        A_tiled_start += A_blkrow_offset;
        C_tiled_start += C_blkrow_offset;
    }
    
}

#ifdef ON_DBG

void matrix_bp_mulpty_raw(const matrix_bp_data *mmul1, const matrix_bp_data *mmul2, matrix_bp_data *resu) {
    resu->cols = mmul2->cols;
    resu->rows = mmul1->rows;
    const uint16_t _k = mmul1->cols,
                    out_rows = mmul1->rows,
                    out_cols = mmul2->cols;

    bp *in1_base = mmul1->data,
           *in2_base = mmul2->data,
           *out_base = resu->data;
           
    for(uint16_t i = 0; i < out_rows; i++){
        for(uint16_t j = 0; j < out_cols; j++){
            _tmp_larger sum = 0;
            for(uint16_t k = 0; k < _k; k++){
                sum += ((_tmp_larger)in1_base[i * _k + k] * in2_base[k * out_cols + j]);
            }
            out_base[i * out_cols + j] = (bp)(sum + (1LL << (QSHIFT -1))) >> QSHIFT;
        }
    }
}

void quickcheck_mmul(const matrix_bp_data *mmul1, const matrix_bp_data *mmul2, matrix_bp_data *rawresu, matrix_bp_data *actualresu){
    matrix_bp_mulpty_raw(mmul1, mmul2, rawresu);
    matrix_bp_mulpty(mmul1, mmul2, actualresu);
}

#endif
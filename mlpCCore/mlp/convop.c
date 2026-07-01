#include "convop.h"
#include "../qfix_ops/qfix_op.h"

void conv_averpool(const qfix* restrict arrin, const uint16_t rows_in, const uint16_t cols_in,
                    const uint16_t pool_rows, const uint16_t pool_cols, const uint16_t step, qfix* restrict arrout){
    const uint16_t rows_out = (rows_in - pool_rows) / step + 1;
    const uint16_t cols_out = (cols_in - pool_cols) / step + 1;

    const uint16_t in_rowstep = step * cols_in;

    const qfix *incur = arrin;
    for(uint16_t ii = 0; ii < pool_rows; ii++){
        incur += cols_in;   //<start> = arrin + ii * rows_out + jj, <0>ii * rows_out
        for(uint16_t jj = 0; jj < pool_cols; jj++){
            incur++;    //<start>, <1>+=1
            qfix *outcur = arrout;
            for (uint16_t i = 0; i < rows_out; i++){
                for (uint16_t j = 0; j < cols_out; j++){
                    outcur[j] += *incur;
                    incur += step;
                    outcur++;
                }
                outcur += cols_out;
                incur += in_rowstep;
            }
        }
    }

    const uint32_t poolsize = pool_rows * pool_cols;
    const uint32_t outsize = rows_out * cols_out;
    for(uint32_t i = 0; i < outsize; i++){
        arrout[i] = qfix_div(arrout[i], poolsize);
    }
}

void conv_maxpool(const qfix* restrict arrin, const uint16_t rows_in, const uint16_t cols_in, const uint16_t pool_rows,
                    const uint16_t pool_cols, const uint16_t step, qfix* restrict arrout, uint16_t* idx_of_maxs){
    const uint16_t rows_out = (rows_in - pool_rows) / step + 1;
    const uint16_t cols_out = (cols_in - pool_cols) / step + 1;

    const uint16_t in_rowstep = step * cols_in;

    const qfix *init_incur = arrin;
    qfix *init_outcur = arrout;
    for (uint16_t i = 0; i < rows_out; i++){
        const qfix *intmp = init_incur;
        for (uint16_t j = 0; j < cols_out; j++){
            init_outcur[j] = *intmp;
            intmp += step;
        }
        init_incur += in_rowstep;
        init_outcur += cols_out;
    }

    const qfix *incur = arrin;
    for(uint16_t ii = 0; ii < pool_rows; ii++){
        incur += cols_in;   //<start> = arrin + ii * rows_out + jj, <0>ii * rows_out
        for(uint16_t jj = 0; jj < pool_cols; jj++){
            incur++;    //<start>, <1>+=1
            qfix *outcur = arrout;
            const uint16_t idxcur = ii * pool_cols + jj;
            uint16_t *idxofmaxcur = idx_of_maxs;
            for (uint16_t i = 0; i < rows_out; i++){
                for (uint16_t j = 0; j < cols_out; j++){
                    if(*incur > outcur[j]){
                        outcur[j] = *incur;
                        idxofmaxcur[j] = idxcur;
                    }
                    incur += step;
                    outcur++;
                }
                outcur += cols_out;
                idxofmaxcur += cols_out;
                incur += in_rowstep;
            }
        }
    }
}

//gxxx/gradxxx = gradxxx

void conv_averpool_backward(const qfix* restrict gradin, const uint16_t rows_gout, const uint16_t cols_gout,
                             const uint16_t pool_rows, const uint16_t pool_cols,
                             const uint16_t step, qfix* restrict gradout){
    const uint16_t cols_out = (cols_gout - pool_cols) / step + 1;
    const uint32_t poolsize = pool_rows * pool_cols;

    //考虑卷积核在滑动过程中出现重合(pool_rows或pool_cols<conv尺寸> > step)
    //传递统一采用梯度累加

    const uint32_t goutsize = rows_gout * cols_gout;
    for(uint32_t i = 0; i < goutsize; i++){
        gradout[i] = 0;
    }

    const qfix *gradin_cur = gradin;
    for (uint16_t i = 0; i < rows_gout; i+=step){
        for (uint16_t j = 0; j < cols_gout; j+=step){
            const qfix gradvalue = qfix_div(*gradin_cur, poolsize);
            for (uint16_t ii = 0; ii < pool_cols; ii++){
                for (uint16_t jj = 0; jj < pool_cols; jj++){
                    gradout[(ii + i) * cols_gout + jj + j] += gradvalue;
                }
            }
            gradin_cur++;
        }
        gradin_cur += cols_out;
    }
}

void conv_maxpool_backward(const qfix* restrict gradin, const uint16_t* idx_of_maxs,
                            const uint16_t rows_gout, const uint16_t cols_gout,
                            const uint16_t pool_rows, const uint16_t pool_cols,
                            const uint16_t step, qfix* restrict gradout){
    const uint16_t cols_out = (cols_gout - pool_cols) / step + 1;
    const uint32_t poolsize = pool_rows * pool_cols;

    //考虑卷积核在滑动过程中出现重合(pool_rows或pool_cols<conv尺寸> > step)
    //传递统一采用梯度累加

    const uint32_t goutsize = rows_gout * cols_gout;
    for(uint32_t i = 0; i < goutsize; i++){
        gradout[i] = 0;
    }

    uint16_t ginidx_cur = 0;
    for (uint16_t i = 0; i < rows_gout; i+=step){
        for (uint16_t j = 0; j < cols_gout; j+=step){
            const qfix gradvalue = qfix_div(gradin[ginidx_cur], poolsize);
            const uint16_t excpidx = idx_of_maxs[ginidx_cur];
            uint16_t curpoolidx = 0;;
            for (uint16_t ii = 0; ii < pool_cols; ii++){
                for (uint16_t jj = 0; jj < pool_cols; jj++){
                    if (curpoolidx == excpidx) gradout[(ii + i) * cols_gout + jj + j] += gradvalue;
                    curpoolidx++;
                }
                curpoolidx += pool_cols;
            }
            ginidx_cur++;
        }
        ginidx_cur += cols_out;
    }
}

void conv_padding(const qfix* restrict arrin, const uint16_t rows_in, const uint16_t cols_in,
                    const uint16_t r_extend, const uint16_t c_extend, qfix* restrict arrout){
    const uint16_t rows_out = rows_in + r_extend * 2;
    const uint16_t cols_out = cols_in + c_extend * 2;
    const uint32_t outsize = rows_out * cols_out;

    for (uint32_t i = 0; i < outsize; i++){
        arrout[i] = 0;
    }

    const qfix *arrin_cur = arrin;
    qfix *arrout_cur = arrout + cols_out * r_extend;

    for (uint16_t i = 0; i < rows_in; i++){
        arrout_cur += c_extend;
        for (uint16_t j = 0; j < cols_in; j++){
            *arrout_cur = *arrin_cur;
            arrin_cur++;
            arrout_cur++;
        }
        arrout_cur += c_extend;
    }
}

void conv_cutedges(const qfix* restrict arrin, const uint16_t rows_in, const uint16_t cols_in,
                    const uint16_t r_sub, const uint16_t c_sub, qfix* restrict arrout){
    const uint16_t rows_out = rows_in - r_sub * 2;
    const uint16_t cols_out = cols_in - c_sub * 2;

    const qfix *arrin_cur = arrin + cols_in * r_sub;
    qfix *arrout_cur = arrout;

    for (uint16_t i = 0; i < rows_out; i++){
        arrin_cur += c_sub;
        for (uint16_t j = 0; j < cols_out; j++){
            *arrout_cur = *arrin_cur;
            arrin_cur++;
            arrout_cur++;
        }
        arrin_cur += c_sub;
    }
}
#include "convop.h"
#include <stdint.h>

#include "../qfix_ops/qfix_op.h"

void conv_averpool(const qfix* restrict arrin, const uint16_t rows_in, const uint16_t cols_in, const uint16_t pool_rows,
                   const uint16_t pool_cols, const uint16_t stepr, const uint16_t stepc, qfix* restrict arrout){
    const uint16_t rows_out = (rows_in - pool_rows) / stepr + 1;
    const uint16_t cols_out = (cols_in - pool_cols) / stepc + 1;

    const uint16_t in_rowstep = stepr * cols_in;
    
    const qfix* in0cur = arrin;

    const qfix *instart = arrin;
    for(uint16_t ii = 0; ii < pool_rows; ii++){
        const qfix* instart_c = instart; 
        for(uint16_t jj = 0; jj < pool_cols; jj++){
            qfix *arrout_cur = arrout;
            //const qfix *arrin_rcur = arrin + cols_in * ii + jj; 
            const qfix *arrin_rcur = instart_c;
            for(uint16_t i = 0; i < rows_out; i++){
                const qfix *arrin_ccur = arrin_rcur;
                for(uint16_t j = 0; j < cols_out; j++){
                    *arrout_cur += *arrin_ccur;
                    arrout_cur++;
                    arrin_ccur+=stepc;
                }
                arrin_rcur+=in_rowstep;
            }
            instart_c++;
        }
        instart += cols_in;
    }

    const uint32_t outsize = rows_out * cols_out, poolsize = pool_cols * pool_rows;
    for(uint32_t i = 0; i < outsize; i++){
        arrout[i] /= poolsize;
        
    }
}

void conv_maxpool(const qfix* restrict arrin, const uint16_t rows_in, const uint16_t cols_in, const uint16_t pool_rows,
                    const uint16_t pool_cols, const uint16_t stepr, const uint16_t stepc, qfix* restrict arrout, uint16_t* idx_of_maxs){
    const uint16_t rows_out = (rows_in - pool_rows) / stepr + 1;
    const uint16_t cols_out = (cols_in - pool_cols) / stepc + 1;

    const uint16_t in_rowstep = stepr * cols_in;

    const qfix *init_incur = arrin;
    qfix *init_outcur = arrout;
    for (uint16_t i = 0; i < rows_out; i++){
        const qfix *intmp = init_incur;
        for (uint16_t j = 0; j < cols_out; j++){
            *init_outcur = *intmp;
            intmp += stepc;
            init_outcur++;
        }
        init_incur += in_rowstep;
    }

    const qfix *instart = arrin;
    uint16_t poolidx = 0;
    for(uint16_t ii = 0; ii < pool_rows; ii++){
        const qfix* instart_c = instart; 
        for(uint16_t jj = 0; jj < pool_cols; jj++){
            qfix *arrout_cur = arrout;
            //const qfix *arrin_rcur = arrin + cols_in * ii + jj; 
            const qfix *arrin_rcur = instart_c;
            for(uint16_t i = 0; i < rows_out; i++){
                const qfix *arrin_ccur = arrin_rcur;
                for(uint16_t j = 0; j < cols_out; j++){
                    if(*arrout_cur < *arrin_ccur){
                        *arrout_cur = *arrin_ccur;
                        idx_of_maxs[i * cols_out + j] = poolidx;
                    }
                    arrout_cur++;
                    arrin_ccur+=stepc;
                }
                arrin_rcur+=in_rowstep;
            }
            instart_c++;
            poolidx++;
        }
        instart += cols_in;
    }
}

//gxxx/gradxxx = gradxxx

void conv_averpool_backward(const qfix* restrict gradin, const uint16_t rows_gout, const uint16_t cols_gout,
                            const uint16_t pool_rows, const uint16_t pool_cols, const uint16_t stepr,
                            const uint16_t stepc, qfix* restrict gradout){
    const uint16_t cols_out = (cols_gout - pool_cols) / stepc + 1;
    const uint16_t rows_out = (rows_gout - pool_rows) / stepr + 1;
    const uint32_t poolsize = pool_rows * pool_cols;

    //考虑卷积核在滑动过程中出现重合(pool_rows或pool_cols<conv尺寸> > step)
    //传递统一采用梯度累加

    const uint32_t goutsize = rows_gout * cols_gout;
    for(uint32_t i = 0; i < goutsize; i++){
        gradout[i] = 0;
    }

    const qfix *gradin_cur = gradin;
    for(uint16_t i = 0; i < rows_out; i++){
        for(uint16_t j = 0; j < cols_out; j++){
            qfix gradvalue = *gradin_cur / poolsize;
            qfix *gradout_rcur = gradout + i * stepr * cols_gout + j * stepc;
            for(uint16_t ii = 0; ii < pool_rows; ii++){
                qfix *gradout_ccur = gradout_rcur;
                for(uint16_t jj = 0; jj < pool_cols; jj++){
                    *gradout_ccur += gradvalue;
                    gradout_ccur++;
                }
                gradout_rcur += cols_gout;
            } 
            gradin_cur++;
        }
    }
}

void conv_maxpool_backward(const qfix* restrict gradin, const uint16_t* idx_of_maxs, const uint16_t rows_gout,
                           const uint16_t cols_gout,
                           const uint16_t pool_rows, const uint16_t pool_cols,
                           const uint16_t stepr, const uint16_t stepc, qfix* restrict gradout){
    const uint16_t cols_out = (cols_gout - pool_cols) / stepc + 1;
    const uint16_t rows_out = (rows_gout - pool_rows) / stepr + 1;
    const uint32_t poolsize = pool_rows * pool_cols;

    //考虑卷积核在滑动过程中出现重合(pool_rows或pool_cols<conv尺寸> > step)
    //传递统一采用梯度累加

    const uint32_t goutsize = rows_gout * cols_gout;
    for(uint32_t i = 0; i < goutsize; i++){
        gradout[i] = 0;
    }

    const qfix *gradin_cur = gradin;
    const uint16_t* idxcur = idx_of_maxs;
    for(uint16_t i = 0; i < rows_out; i++){
        for(uint16_t j = 0; j < cols_out; j++){
            const uint16_t idxr = *idxcur / pool_cols;
            const uint16_t idxc = *idxcur % pool_cols;
            gradout[(i * stepr + idxr) * cols_gout + j * stepc + idxc] += *gradin_cur;
            gradin_cur++;
        }
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

void conv_divelem(const qfix* restrict arrin, const uint16_t rows_in, const uint16_t cols_in, const uint16_t pool_rows,
    const uint16_t pool_cols, const uint16_t expand, const uint16_t stepr, const uint16_t stepc, qfix** restrict arrouts){
    const uint16_t _scale = expand + 1;

    const uint16_t arrsizer = (rows_in - pool_rows * _scale + expand) / stepr + 1;
    const uint16_t arrsizec = (cols_in - pool_cols * _scale + expand) / stepc + 1;

    const uint16_t in_rowstep = _scale * cols_in;

    qfix** outcur_ = arrouts;
    for(uint16_t i = 0; i < arrsizer; i++){
        for (uint16_t j = 0; j < arrsizec; j++){
            const qfix *incur = arrin + i * stepr * cols_in + j * stepc;
            qfix *outcur = *outcur_;
            const qfix *incur_r = incur;
            for (uint16_t ii = 0; ii < pool_rows; ii++){
                const qfix* incur_c = incur_r;
                for(uint16_t jj = 0; jj < pool_cols; jj++){
                    *outcur = *incur_c;
                    incur_c += _scale;
                    outcur++;
                }
                incur_r += in_rowstep;
            }
            outcur_++;
        }
    }
}

void conv_mergeelem(const qfix** restrict arrin, const uint16_t rows_to, const uint16_t cols_to, const uint16_t pool_rows,
    const uint16_t pool_cols, const uint16_t expand, const uint16_t stepr, const uint16_t stepc, qfix* restrict arrout){
    const uint16_t _scale = expand + 1;

    const uint16_t arrsizer = (rows_to - pool_rows * _scale + expand) / stepr + 1;
    const uint16_t arrsizec = (cols_to - pool_cols * _scale + expand) / stepc + 1;

    const uint16_t in_rowstep = _scale * cols_to;

    const qfix** incur_ = arrin;
    for(uint16_t i = 0; i < arrsizer; i++){
        for (uint16_t j = 0; j < arrsizec; j++){
            qfix *outcur = arrout + i * stepr * cols_to + j * stepc;
            const qfix *incur = *incur_;
            qfix *outcur_r = outcur;
            for (uint16_t ii = 0; ii < pool_rows; ii++){
                qfix* outcur_c = outcur_r;
                for(uint16_t jj = 0; jj < pool_cols; jj++){
                    *outcur_c += *incur;
                    outcur_c += _scale;
                    incur++;
                }
                outcur_r += in_rowstep;
            }
            incur_++;
        }
    }
}

void conv_fwdcalc(const qfix* restrict arrin, const qfix* restrict convcore_w, const uint16_t rows_in,
                  const uint16_t cols_in, const uint16_t pool_rows,
                  const uint16_t pool_cols, const uint16_t expand, const uint16_t stepr, const uint16_t stepc, qfix* restrict arrout){
    const uint16_t _scale = expand + 1;

    const uint16_t outsizer = (rows_in - pool_rows * _scale + expand) / stepr + 1;
    const uint16_t outsizec = (cols_in - pool_cols * _scale + expand) / stepc + 1;

    const uint16_t in_rowstep = _scale * cols_in;

    qfix *outcur = arrout;
    for(uint16_t i = 0; i < outsizer; i++){
        for (uint16_t j = 0; j < outsizec; j++){
            const qfix *incur = arrin + i * stepr * cols_in + j * stepc;
            const qfix *incur_r = incur;
            const qfix *w_cur = convcore_w;
            for (uint16_t ii = 0; ii < pool_rows; ii++){
                const qfix* incur_c = incur_r;
                for(uint16_t jj = 0; jj < pool_cols; jj++){
                    *outcur += qfix_mul(*incur_c, *w_cur);
                    incur_c += _scale;
                    w_cur++;
                }
                incur_r += in_rowstep;
            }
            outcur++;
        }
    }
}

#ifndef LIGHTMLP_CONVOP_H
#define LIGHTMLP_CONVOP_H

#include "../CustomConf.h"

void DLLEXPORT conv_averpool(const qfix* restrict arrin, uint16_t rows_in, uint16_t cols_in,
                                uint16_t pool_rows, uint16_t pool_cols, uint16_t step, qfix* restrict arrout);
void DLLEXPORT conv_maxpool(const qfix* restrict arrin, uint16_t rows_in, uint16_t cols_in,
                            uint16_t pool_rows, uint16_t pool_cols, uint16_t step,
                            qfix* restrict arrout, uint16_t* restrict idx_of_maxs);

void DLLEXPORT conv_averpool_backward(const qfix* restrict gradin, uint16_t rows_gout, uint16_t cols_gout,
                                        uint16_t pool_rows, uint16_t pool_cols, uint16_t step, qfix* restrict gradout);
void DLLEXPORT conv_maxpool_backward(const qfix* restrict gradin, const uint16_t *idx_of_maxs,
                                        uint16_t rows_gout, uint16_t cols_gout,
                                        uint16_t pool_rows, uint16_t pool_cols, uint16_t step, qfix* restrict gradout);

void DLLEXPORT conv_padding(const qfix* restrict arrin, uint16_t rows_in, uint16_t cols_in,
                            uint16_t r_extend, uint16_t c_extend, qfix* restrict arrout);
void DLLEXPORT conv_cutedges(const qfix* restrict arrin, uint16_t rows_in, uint16_t cols_in,
                                uint16_t r_sub, uint16_t c_sub, qfix* restrict arrout);

#endif //LIGHTMLP_CONVOP_H

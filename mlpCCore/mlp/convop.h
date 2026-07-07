#ifndef LIGHTMLP_CONVOP_H
#define LIGHTMLP_CONVOP_H

#include "../CustomConf.h"

void DLLEXPORT conv_averpool(const qfix* arrin, const uint16_t rows_in, const uint16_t cols_in,
                                const uint16_t pool_rows, const uint16_t pool_cols,
                                const uint16_t stepr, const uint16_t stepc, qfix* arrout);
void DLLEXPORT conv_maxpool(const qfix* arrin, const uint16_t rows_in, const uint16_t cols_in,
                            const uint16_t pool_rows, const uint16_t pool_cols,
                            const uint16_t stepr, const uint16_t stepc,
                            qfix* arrout, uint16_t* idx_of_maxs);

void DLLEXPORT conv_averpool_backward(const qfix* gradin, const uint16_t rows_gout, const uint16_t cols_gout,
                                        const uint16_t pool_rows, const uint16_t pool_cols, 
                                        const uint16_t stepr, const uint16_t stepc, qfix* gradout);
void DLLEXPORT conv_maxpool_backward(const qfix* gradin, const uint16_t *idx_of_maxs,
                                        const uint16_t rows_gout, const uint16_t cols_gout,
                                        const uint16_t pool_rows, const uint16_t pool_cols, 
                                        const uint16_t stepr, const uint16_t stepc, qfix* gradout);

void DLLEXPORT conv_padding(const qfix* arrin, const uint16_t rows_in, const uint16_t cols_in,
                            const uint16_t r_extend, const uint16_t c_extend, qfix*  arrout);
void DLLEXPORT conv_cutedges(const qfix* arrin, const uint16_t rows_in, const uint16_t cols_in,
                                const uint16_t r_sub, const uint16_t c_sub, qfix* arrout);

void DLLEXPORT conv_divelem(const qfix *arrin, const uint16_t rows_in, const uint16_t cols_in,
                                        const uint16_t pool_rows, const uint16_t pool_cols, const uint16_t expand,
                                        const uint16_t stepr, const uint16_t stepc, qfix **arrouts);

void DLLEXPORT conv_mergeelem(const qfix **arrin, const uint16_t rows_in, const uint16_t cols_in,
                                        const uint16_t pool_rows, const uint16_t pool_cols, const uint16_t expand,
                                        const uint16_t stepr, const uint16_t stepc, qfix *arrout);

void DLLEXPORT conv_fwdcalc(const qfix *arrin, const qfix* convcore_w, const uint16_t rows_in, const uint16_t cols_in,
                                        const uint16_t pool_rows, const uint16_t pool_cols, const uint16_t expand,
                                        const uint16_t stepr, const uint16_t stepc, qfix *arrout);

#endif //LIGHTMLP_CONVOP_H

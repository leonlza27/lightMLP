#include "activator.h"

qfloat exp_qfloat(qfloat x){
    if(x > 10 << QSHIFT) return QF_MAX;
    if(x < -10 << QSHIFT) return 0;

    qf_cur_cast n =  x >> QSHIFT;
    qf_cur_cast r = x & QF_FLOAT_MASK;

    //整数部分

    qf_cur_cast en = 1 << QSHIFT;
    qf_cur_cast base = e_QFBASE;
    qf_cur_cast n_abs = (n < 0)? -n : n;
    while(n_abs > 0){
        if(n_abs & 1) en = qfloat_mul(en,base);
        base = qfloat_mul(base, base);
        n_abs >>= 1; 
    }
    if(n < 0) en = ((_tmp_larger)(1<<(QSHIFT * 2)) / en);

    //小数部分

    qf_cur_cast term1 = r;
    qf_cur_cast term2 = qfloat_mul(r,r) >> 1;
    qf_cur_cast term3 = qfloat_mul(r,qfloat_mul(r,r)) / 6;
    qf_cur_cast er = (1 << QSHIFT) + term1 + term2 + term3;

    return qfloat_mul(en,er);
}
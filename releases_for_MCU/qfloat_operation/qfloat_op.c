#include "qfloat_op.h"

qfloat qfloat_add(qfloat num1, qfloat num2){
    return num1 + num2;
}

qfloat qfloat_sub(qfloat num1, qfloat num2){
    return num1 - num2;
}

qfloat qfloat_mul(qfloat num1, qfloat num2){
    return (qfloat)(((_tmp_larger)num1 * num2) >> QSHIFT);
}

qfloat qfloat_div(qfloat num1, qfloat num2){
    return (qfloat)(((_tmp_larger)num1 << QSHIFT) / num2);
}

qfloat exp_qfloat(qfloat x){
    if(x > 8 << QSHIFT) return QF_MAX;
    if(x < -8 << QSHIFT) return 0;

    //lut + 线性插值

    qf_cur_cast index_raw = ((x + float_to_qfloat(8))*(QF_LUT_LEN - 1)) /16;

    int index = index_raw >> QSHIFT;
    qf_cur_cast insert_y0 = QF_EXP_LUT[index];
    qf_cur_cast insert_y1 = QF_EXP_LUT[index+1];
    
    qf_cur_cast frac = index_raw & QF_FLOAT_MASK;

    return insert_y0 + qfloat_mul(insert_y1 - insert_y0, frac);

    // 保留原有计算作为后备方案
    /*
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
    */
}

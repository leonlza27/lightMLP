#include "qfix_op.h"

qfix qfix_add(qfix num1, qfix num2){
    return num1 + num2;
}

qfix qfix_sub(qfix num1, qfix num2){
    return num1 - num2;
}

qfix qfix_mul(qfix num1, qfix num2){
    return (qfix)(((_tmp_larger)num1 * num2) >> QSHIFT);
}

qfix qfix_div(qfix num1, qfix num2){
    return (qfix)(((_tmp_larger)num1 << QSHIFT) / num2);
}

qfix exp_qfix(qfix x){
    if(x > 8 << QSHIFT) return QF_MAX;
    if(x < -8 << QSHIFT) return 0;

    //lut + 线性插值

    qf_cur_cast index_raw = ((x + float_to_qfix(8))*(QF_LUT_LEN - 1)) /16;

    int index = index_raw >> QSHIFT;
    qf_cur_cast insert_y0 = QF_EXP_LUT[index];
    qf_cur_cast insert_y1 = QF_EXP_LUT[index+1];
    
    qf_cur_cast frac = index_raw & QF_FLOAT_MASK;

    return insert_y0 + qfix_mul(insert_y1 - insert_y0, frac);

    // 保留原有计算作为后备方案
    /*
    qf_cur_cast n =  x >> QSHIFT;
    qf_cur_cast r = x & QF_FLOAT_MASK;

    //整数部分
    qf_cur_cast en = 1 << QSHIFT;
    qf_cur_cast base = e_QFBASE;
    qf_cur_cast n_abs = (n < 0)? -n : n;
    while(n_abs > 0){
        if(n_abs & 1) en = qfix_mul(en,base);
        base = qfix_mul(base, base);
        n_abs >>= 1; 
    }
    if(n < 0) en = ((_tmp_larger)(1<<(QSHIFT * 2)) / en);

    //小数部分
    qf_cur_cast term1 = r;
    qf_cur_cast term2 = qfix_mul(r,r) >> 1;
    qf_cur_cast term3 = qfix_mul(r,qfix_mul(r,r)) / 6;
    qf_cur_cast er = (1 << QSHIFT) + term1 + term2 + term3;

    return qfix_mul(en,er);
    */
}

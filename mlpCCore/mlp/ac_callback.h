#ifndef ac_callbacks
#define ac_callbacks

#include "../CustomConf.h"
#include "../qfix_ops/qfix_op.h"

enum ml_act_type{
    ac_ReLU         ,
    ac_ReLU6        ,
    ac_LeakyReLU    ,
    ac_Sigmoid      ,
    ac_Sigmoid_hard ,
    ac_Tanh         ,
    ac_Tanh_hard    ,
    ac_Sign         ,
    ac_pass         ,
};

typedef qfix (*acCall)(qfix,qfix);


static const qfix Q_0 = float_to_qfix(0.0f);
static const qfix Q_1p0 = float_to_qfix(1.0f);
static const qfix Q_neg1p0 = float_to_qfix(-1.0f);
static const qfix Q_0p5 = float_to_qfix(0.5f);
static const qfix Q_0p2 = float_to_qfix(0.2f);
static const qfix Q_6p0 = float_to_qfix(6.0f);
const static qfix Q_4p0 = float_to_qfix(4.0f);

static qfix ReLU(qfix input, qfix){
    return qfmax(Q_0, input);
}

static qfix ReLu6(qfix input, qfix){
    return qfmin(Q_6p0, qfmax(Q_0, input));
}

static qfix LeakyReLU(qfix input, qfix alpha){
    return qfmax(qfix_mul(input,alpha), input);
}

static qfix SigmoidH(qfix input, qfix){
    qfix val = Q_0p5 + qfix_mul(Q_0p2, input);
    return qfmax(0, qfmin(val, Q_1p0));
}

static qfix Sigmoid(qfix input, qfix){
    return qfix_div(Q_1p0, Q_1p0 + exp_qfix(-input));
}

static qfix TanhH(qfix input, qfix){
    return qfmax(Q_neg1p0, qfmin(input,Q_1p0));
}

static qfix Tanh(qfix input, qfix){
    qfix exp_x = exp_qfix(input);
    qfix exp_negx = exp_qfix(-input);
    return qfix_div(exp_x - exp_negx, exp_x + exp_negx);
}

static qfix Sign(qfix input, qfix){
    return input>0? Q_1p0 : 0;
}

static qfix linear(qfix input, qfix){
    return input;
}

static qfix grad_ReLU(qfix input, qfix){
    return input > 0? input : 0;
}

static qfix grad_ReLu6(qfix input, qfix){
    return (input > 0 || input < Q_6p0)? input : 0;
}

static qfix grad_LeakyReLU(qfix input, qfix alpha){
    return input > 0? input : qfix_mul(alpha, input);
}

static qfix grad_SigmoidH(qfix input, qfix){
    return (input > 0 || input < Q_1p0)? qfix_mul(Q_0p2, input) : 0;
}

static qfix grad_Sigmoid(qfix input, qfix){
    qfix ex = exp_qfix(input);
    qfix ex_pu1 = ex + Q_1p0;
    qfix bottom = qfix_mul(ex_pu1, ex_pu1);
    return qfix_mul(input, qfix_div(ex, bottom));
}

static qfix grad_TanhH(qfix input, qfix){
    return (input > Q_neg1p0 || input < Q_1p0)? input : 0;
}

static qfix grad_Tanh(qfix input, qfix){
    qfix ex = exp_qfix(input);
    qfix ex_sq2 = qfix_mul(ex, ex);
    qfix ex_sq2_pu1 = ex_sq2 + Q_1p0;
    qfix bottom = qfix_mul(ex_sq2_pu1, ex_sq2_pu1);
    return qfix_mul(input, qfix_div(qfix_mul(Q_4p0, ex_sq2), bottom));
}

static qfix grad_Sign(qfix input, qfix){
    return 0;
}

static qfix grad_linear(qfix input, qfix){
    return input;
}

#endif
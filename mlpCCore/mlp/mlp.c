#include "mlp.h"
#include "builtin_ac_callback.h"


void mlptrainer_setup(uint32_t calclyrs, netLyrConf *net, mlpTrainStatus *dest){
    dest->calclyrs = calclyrs;
    dest->modelsrc = net;

    size_t all_size = (calclyrs * 4 + 2) * sizeof(qfix*);

    for(uint32_t i = 0; i < calclyrs; i++){
        const uint16_t indim = net[i].in_dim;
        const uint16_t outdim = net[i].out_dim;
        all_size += (outdim * 2 + indim + outdim * indim) * sizeof(qfix);
    }

    char *fullmem = malloc(all_size);
    
    //dest->fullConnData = (qfix**)malloc(sizeof(qfix*) * (calclyrs + 1));
    //dest->lyrinput_grad = (qfix**)malloc(sizeof(qfix*) * (calclyrs + 1));
    //dest->w_grad = (qfix**)malloc(sizeof(qfix*) * calclyrs);

    dest->fullConnData = (qfix**)fullmem;
    fullmem += sizeof(qfix*) * (calclyrs + 1);
    dest->va_wxb = (qfix**)fullmem;
    fullmem += sizeof(qfix*) * calclyrs;
    dest->lyrinput_grad = (qfix**)fullmem;
    fullmem += sizeof(qfix*) * (calclyrs + 1);
    dest->w_grad = (qfix**)fullmem;
    fullmem += sizeof(qfix*) * calclyrs;

    for(uint32_t i = 0; i < calclyrs; i++){
        const uint16_t indim = net[i].in_dim;
        const uint16_t outdim = net[i].out_dim;
        //dest->fullConnData[i + 1] = (qfix*)malloc(sizeof(qfix) * outdim);
        //dest->lyrinput_grad[i] = (qfix*)malloc(sizeof(qfix) * indim);
        //dest->w_grad[i] = (qfix*)malloc(sizeof(qfix) * indim * outdim);
        dest->fullConnData[i + 1] = (qfix*)fullmem;
        fullmem += sizeof(qfix) * outdim;
        dest->va_wxb[i] = (qfix*)fullmem;
        fullmem += sizeof(qfix) * outdim;
        dest->lyrinput_grad[i] = (qfix*)fullmem;
        fullmem += sizeof(qfix) * indim;
        dest->w_grad[i] = (qfix*)fullmem;
        fullmem += sizeof(qfix) * indim * outdim;
    }
}

/*
void mlptrainer_totalgrads_cap_setup(uint32_t calclyrs ,netLyrConf *net, mlpTrainStatus *dest){
    dest->calclyrs = calclyrs;
    dest->modelsrc = net;
    
    dest->fullConnData = 0;
    dest->lyrinput_grad = (qfix**)malloc(sizeof(qfix*) * (calclyrs + 1));
    dest->w_grad = (qfix**)malloc(sizeof(qfix*) * calclyrs);

    for(uint32_t i = 0; i < calclyrs; i++){
        uint16_t indim = net[i].in_dim;
        uint16_t outdim = net[i].out_dim;
        dest->lyrinput_grad[i] = (qfix*)malloc(sizeof(qfix*) * indim);
        dest->w_grad[i] = (qfix*)malloc(sizeof(qfix) * indim * outdim);
    }
}

void mlptrainer_cleanup(mlpTrainStatus *net){
    uint32_t size = net->calclyrs;
    for(uint32_t i = 0; i < size; i++){
        free(net->lyrinput_grad[i]);
        free(net->w_grad[i]);
    }
    free(net->lyrinput_grad);
    free(net->w_grad);
    if(!net->fullConnData) goto _ret;
    for(uint32_t i = 0; i < size; i++){
        free(net->fullConnData[i + 1]);
    }
    free(net->fullConnData);
_ret:
    net->modelsrc = 0;
}
*/

void mlptrainer_totalgrads_cap_setup(uint32_t calclyrs ,netLyrConf *net, mlpTrainStatus *dest){
    dest->calclyrs = calclyrs;
    dest->modelsrc = net;

    size_t all_size = (calclyrs * 2 + 1) * sizeof(qfix*);

    for(uint32_t i = 0; i < calclyrs; i++){
        const uint16_t indim = net[i].in_dim;
        const uint16_t outdim = net[i].out_dim;
        all_size += (indim + outdim * indim) * sizeof(qfix);
    }

    char *fullmem = malloc(all_size);

    dest->fullConnData = 0;
    dest->va_wxb = 0;
    dest->lyrinput_grad = (qfix**)fullmem;
    fullmem += sizeof(qfix*) * (calclyrs + 1);
    dest->w_grad = (qfix**)fullmem;
    fullmem += sizeof(qfix*) * calclyrs;

    for(uint32_t i = 0; i < calclyrs; i++){
        const uint16_t indim = net[i].in_dim;
        const uint16_t outdim = net[i].out_dim;
        dest->lyrinput_grad[i] = (qfix*)fullmem;
        fullmem += sizeof(qfix) * indim;
        dest->w_grad[i] = (qfix*)fullmem;
        fullmem += sizeof(qfix) * indim * outdim;
    }
}

void mlptrainer_cleanup(mlpTrainStatus *net){
    net->modelsrc = 0;
    net->calclyrs = 0;
    if(net->fullConnData){
        free(net->fullConnData);
        return;
    }
    free(net->lyrinput_grad);
}

/*
void mlpexec_setup(uint32_t calclyrs ,netLyrConf *net, mlpExecStatus *dest){
    dest->calclyrs = calclyrs;
    dest->modelsrc = net;
    uint16_t outdim_max = 0;
    for(uint32_t i = 0; i < calclyrs; i++){
        uint16_t outdim = net[i].out_dim;
        if(outdim > outdim_max) outdim_max = outdim;
    }
    dest->fullConnData_tmp[0] = (qfix*)malloc(sizeof(qfix*) * outdim_max);
    dest->fullConnData_tmp[1] = (qfix*)malloc(sizeof(qfix*) * outdim_max);
}

void mlpexec_cleanup(mlpExecStatus *net){
    free(net->fullConnData_tmp[0]);
    free(net->fullConnData_tmp[1]);
    net->modelsrc = 0;
}
*/
void mlpexec_setup(const uint32_t calclyrs ,const netLyrConf *net, mlpExecStatus *dest){
    dest->calclyrs = calclyrs;
    dest->modelsrc = net;
    uint16_t outdim_max = 0;
    for(uint32_t i = 0; i < calclyrs; i++){
        const uint16_t outdim = net[i].out_dim;
        if(outdim > outdim_max) outdim_max = outdim;
    }
    dest->fullConnData_tmp[0] = (qfix*)malloc(sizeof(qfix*) * outdim_max * 2);
    dest->fullConnData_tmp[1] = dest->fullConnData_tmp[0] + outdim_max;
}

void mlpexec_cleanup(mlpExecStatus *net){
    free(net->fullConnData_tmp[0]);
    net->modelsrc = 0;
    net->calclyrs = 0;
}

void mlptrainer_totalgrads_savegrads(const mlpTrainStatus *net_or_gradcap, const mlpTrainStatus *gradscap_dest){
    const uint32_t size = net_or_gradcap->calclyrs;
    const netLyrConf *nsrc = net_or_gradcap->modelsrc;
    for(uint32_t i = 0; i < size; i++){
        const uint16_t indi = nsrc->in_dim;
        const uint32_t wsize = nsrc->out_dim * indi;
        const qfix *bgrad_src = net_or_gradcap->lyrinput_grad[i];
        const qfix *wgrad_src = net_or_gradcap->w_grad[i];
        qfix *bgrad_dst = gradscap_dest->lyrinput_grad[i];
        qfix *wgrad_dst = gradscap_dest->w_grad[i];

        for(uint16_t i_ = 0; i_ < indi; i_++) bgrad_dst[i_] += bgrad_src[i_];
        for(uint32_t i_ = 0; i_ < wsize; i_++) wgrad_dst[i_] += wgrad_src[i_];

        nsrc++;
    }
}


void mlptrainer_backward(const mlpTrainStatus *model, const qfix *grad_inital, const qfix lr){
    const uint32_t size = model->calclyrs;
    const netLyrConf *nsrc = &model->modelsrc[size - 1];
    model->lyrinput_grad[size] = grad_inital;
    qfix **grads = model->lyrinput_grad;
    acCall gradfn;
    for(uint32_t l = size; l > 0; l--){
        const uint32_t curidx = l - 1;
        const qfix *grad_in = grads[l];
        qfix *grads_dst = grads[curidx];
        qfix *wgrad_cur = model->w_grad[curidx];
        qfix *w_this = nsrc->weights;
        qfix *bcur = nsrc->bias;
        const qfix dext = nsrc->dExtra;

        const uint16_t indi = nsrc->in_dim;
        const uint16_t outdi = nsrc->out_dim;
        switch(nsrc->acTp){
            case ac_ReLU: gradfn = grad_ReLU; break;
            case ac_ReLU6: gradfn = grad_ReLu6; break;;
            case ac_LeakyReLU: gradfn = grad_LeakyReLU; break;
            case ac_Sigmoid: gradfn = grad_Sigmoid; break;
            case ac_Sigmoid_hard: gradfn = grad_SigmoidH; break;
            case ac_Tanh: gradfn = grad_Tanh; break;
            case ac_Tanh_hard: gradfn = grad_TanhH; break;
            case ac_TripleLine: gradfn = grad_TripleLine; break;
            case ac_pass:
            default: gradfn = grad_linear; break;
        }

        const qfix *gi_ref = grad_in;
        qfix *gw_cur = wgrad_cur;
        const qfix *x_last = model->fullConnData[curidx];
        const qfix *x_ac = model->va_wxb[curidx];
        qfix *wcur = w_this;
        for(uint16_t i = 0; i < outdi; i++){
            const qfix grad_cur = qfix_mul(*gi_ref ,gradfn(*x_ac, dext));
            for(uint16_t j = 0; j < indi; j++){
                *gw_cur = qfix_mul(grad_cur, x_last[j]);
                grads_dst[j] += qfix_mul(grad_cur, *wcur);
                wcur++;
                gw_cur++;
            }
            bcur[i] += qfix_mul(lr, grad_cur);
            gi_ref++;
            x_ac++;
        }

        const uint32_t wsize = outdi * indi;
        for(uint32_t i = 0; i < wsize; i++) wcur[i] += qfix_mul(lr, wgrad_cur[i]);

        nsrc--;
    }
}

void mlptrainer_totalgrads_backward(const mlpTrainStatus *model, const mlpTrainStatus *gradscap, const qfix lr){
    const uint32_t size = model->calclyrs;
    const netLyrConf *nsrc = model->modelsrc;
    for(uint32_t i = 0; i < size; i++){
        const uint16_t outdi = nsrc->out_dim;
        const uint32_t wsize = nsrc->in_dim * outdi;
        const qfix *bgrad_src = gradscap->lyrinput_grad[i];
        const qfix *wgrad_src = gradscap->w_grad[i];
        qfix *b_dst = nsrc->bias;
        qfix *w_dst = nsrc->weights;

        for(uint16_t i_ = 0; i_ < outdi; i_++) b_dst[i_] += qfix_mul(bgrad_src[i_], lr);
        for(uint32_t i_ = 0; i_ < wsize; i_++) w_dst[i_] += qfix_mul(wgrad_src[i_], lr);

        nsrc++;
    }
}
/*
 * conv(via mlptrainer):局部conv_to_mmul展开
 * calclyrs = 1
 * W<conv_core> * [in] + b -> out
 *                       ^
 *                       =0
 *  [indim = n<conv_size> , outdim = 1]
 */

void mlptrainer_convcalc_zerobia(const mlpTrainStatus* model){
    model->modelsrc->bias[0] = 0;
}

void mlptrainer_execute(const mlpTrainStatus *model, const qfix *arrin){
    const uint32_t size = model->calclyrs;
    model->fullConnData[0] = arrin;
    const netLyrConf *nsrc = model->modelsrc;
    qfix** fcd_export = model->fullConnData;
    acCall acfn;
    for(uint32_t i = 0; i < size; i++){
        const qfix *in = fcd_export[i];
        qfix *out = fcd_export[i + 1];
        qfix *wcur = nsrc->weights;
        const qfix *bcur = nsrc->bias;
        const qfix dext = nsrc->dExtra;

        uint16_t indi = nsrc->in_dim;
        uint16_t outdi = nsrc->out_dim;

        switch(nsrc->acTp){
            case ac_ReLU: acfn = ReLU; break;
            case ac_ReLU6: acfn = ReLu6; break;;
            case ac_LeakyReLU: acfn = LeakyReLU; break;
            case ac_Sigmoid: acfn = Sigmoid; break;
            case ac_Sigmoid_hard: acfn = SigmoidH; break;
            case ac_Tanh: acfn = Tanh; break;
            case ac_Tanh_hard: acfn = TanhH; break;
            case ac_TripleLine: acfn = TripleLine; break;
            case ac_pass:
            default: acfn = linear; break;
        }

        const qfix *wcurr = wcur;
        qfix *oriout_cur = model->va_wxb[i];
        for(uint16_t ii = 0; ii < outdi; ii++){
            _tmp_larger sumw = 0;
            for(uint16_t j = 0; j < indi; j++){
                sumw += (_tmp_larger)(*wcurr) * *in;
                wcurr++;
                in++;
            }
            qfix cur = (sumw >> QSHIFT) + *bcur;
            *oriout_cur = cur;
            *out = acfn(cur, dext);
            bcur++;
            out++;
            oriout_cur++;
            in -= indi;
        }

        nsrc++;
    }
}

void mlpexec_execute(const mlpExecStatus *model, const qfix *arrin){
    const uint32_t size = model->calclyrs;
    const netLyrConf *nsrc = model->modelsrc;
    qfix** fcd_export = model->fullConnData_tmp;
    acCall acfn;
    uint32_t i = 0;
    qfix *in, *out;
    goto _lyr0;
    for(; i < size; i++){
        goto _lyroth;
    _lyr0:
        in = arrin;
        goto _norm_exec;
    _lyroth:
        in = fcd_export[i%2];
    _norm_exec:
        out = fcd_export[(i + 1)%2];
        const qfix *wcur = nsrc->weights;
        const qfix *bcur = nsrc->bias;
        const qfix dext = nsrc->dExtra;

        uint16_t indi = nsrc->in_dim;
        uint16_t outdi = nsrc->out_dim;

        switch(nsrc->acTp){
            case ac_ReLU: acfn = ReLU; break;
            case ac_ReLU6: acfn = ReLu6; break;;
            case ac_LeakyReLU: acfn = LeakyReLU; break;
            case ac_Sigmoid: acfn = Sigmoid; break;
            case ac_Sigmoid_hard: acfn = SigmoidH; break;
            case ac_Tanh: acfn = Tanh; break;
            case ac_Tanh_hard: acfn = TanhH; break;
            case ac_TripleLine: acfn = TripleLine; break;
            case ac_pass:
            default: acfn = linear; break;
        }

        const qfix *wcurr = wcur;
        for(uint16_t ii = 0; ii < outdi; ii++){
            _tmp_larger sumw = 0;
            for(uint16_t j = 0; j < indi; j++){
                sumw += (_tmp_larger)(*wcurr) * *in;
                wcurr++;
                in++;
            }
            *out = acfn((sumw >> QSHIFT) + *bcur, dext);
            bcur++;
            out++;
            in -= indi;
        }

        nsrc++;
    }
}

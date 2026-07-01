#include "mlp.h"

void mlptrainer_setup(uint32_t calclyrs, netLyrConf *net, mlpTrainStatus *dest){
    dest->calclyrs = calclyrs;
    dest->modelsrc = net;

    size_t all_size = (calclyrs * 3 + 2) * sizeof(qfix*);

    for(uint32_t i = 0; i < calclyrs; i++){
        uint16_t indim = net[i].in_dim;
        uint16_t outdim = net[i].out_dim;
        all_size += (outdim + indim + outdim * indim) * sizeof(qfix);
    }

    char *fullmem = malloc(all_size);
    
    //dest->fullConnData = (qfix**)malloc(sizeof(qfix*) * (calclyrs + 1));
    //dest->lyrinput_grad = (qfix**)malloc(sizeof(qfix*) * (calclyrs + 1));
    //dest->w_grad = (qfix**)malloc(sizeof(qfix*) * calclyrs);

    dest->fullConnData = (qfix**)fullmem;
    fullmem += sizeof(qfix*) * (calclyrs + 1);
    dest->lyrinput_grad = (qfix**)fullmem;
    fullmem += sizeof(qfix*) * (calclyrs + 1);
    dest->w_grad = (qfix**)fullmem;
    fullmem += sizeof(qfix*) * calclyrs;

    for(uint32_t i = 0; i < calclyrs; i++){
        uint16_t indim = net[i].in_dim;
        uint16_t outdim = net[i].out_dim;
        //dest->fullConnData[i + 1] = (qfix*)malloc(sizeof(qfix) * outdim);
        //dest->lyrinput_grad[i] = (qfix*)malloc(sizeof(qfix) * indim);
        //dest->w_grad[i] = (qfix*)malloc(sizeof(qfix) * indim * outdim);
        dest->fullConnData[i + 1] = (qfix*)fullmem;
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
        uint16_t indim = net[i].in_dim;
        uint16_t outdim = net[i].out_dim;
        all_size += (indim + outdim * indim) * sizeof(qfix);
    }

    char *fullmem = malloc(all_size);

    dest->fullConnData = 0;
    dest->lyrinput_grad = (qfix**)fullmem;
    fullmem += sizeof(qfix*) * (calclyrs + 1);
    dest->w_grad = (qfix**)fullmem;
    fullmem += sizeof(qfix*) * calclyrs;

    for(uint32_t i = 0; i < calclyrs; i++){
        uint16_t indim = net[i].in_dim;
        uint16_t outdim = net[i].out_dim;
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
    return;
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
void mlpexec_setup(uint32_t calclyrs ,netLyrConf *net, mlpExecStatus *dest){
    dest->calclyrs = calclyrs;
    dest->modelsrc = net;
    uint16_t outdim_max = 0;
    for(uint32_t i = 0; i < calclyrs; i++){
        uint16_t outdim = net[i].out_dim;
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

void mlptrainer_totalgrads_savegrads(mlpTrainStatus *net_or_gradcap, mlpTrainStatus *gradscap_dest){
    uint32_t size = net_or_gradcap->calclyrs;
    netLyrConf *nsrc = net_or_gradcap->modelsrc;
    for(uint32_t i = 0; i < size; i++){
        uint16_t indi = nsrc->in_dim;
        uint32_t wsize = nsrc->out_dim * indi;
        qfix *bgrad_src = net_or_gradcap->lyrinput_grad[i];
        qfix *wgrad_src = net_or_gradcap->w_grad[i];
        qfix *bgrad_dst = gradscap_dest->lyrinput_grad[i];
        qfix *wgrad_dst = gradscap_dest->w_grad[i];

        for(uint16_t i_ = 0; i_ < indi; i_++) bgrad_dst[i_] += bgrad_src[i_];
        for(uint32_t i_ = 0; i_ < wsize; i_++) wgrad_dst[i_] += wgrad_src[i_];

        nsrc++;
    }
}


void mlptrainer_backward(mlpTrainStatus *model, qfix *grad_inital, qfix lr){
    uint32_t size = model->calclyrs;
    netLyrConf *nsrc = &model->modelsrc[size - 1];
    model->lyrinput_grad[size] = grad_inital;
    qfix** fcd_export = model->fullConnData;
    qfix **grads = model->lyrinput_grad;
    acCall gradfn;
    for(uint32_t l = size; l > 0; l--){
        uint32_t curidx = l - 1;
        qfix *grad_in = grads[l];
        qfix *grads_dst = grads[curidx];
        qfix *wgrad_cur = model->w_grad[curidx];
        qfix *wcur = nsrc->weights;
        qfix *bcur = nsrc->bias;
        qfix dext = nsrc->dExtra;

        uint16_t indi = nsrc->in_dim;
        uint16_t outdi = nsrc->out_dim;
        switch(nsrc->acTp){
            case ac_ReLU: gradfn = grad_ReLU; break;
            case ac_ReLU6: gradfn = grad_ReLu6; break;;
            case ac_LeakyReLU: gradfn = grad_LeakyReLU; break;
            case ac_Sigmoid: gradfn = grad_Sigmoid; break;
            case ac_Sigmoid_hard: gradfn = grad_SigmoidH; break;
            case ac_Tanh: gradfn = grad_Tanh; break;
            case ac_Tanh_hard: gradfn = grad_TanhH; break;
            case ac_Sign: gradfn = grad_Sign; break;
            case ac_pass:
            default: gradfn = grad_linear; break;
        }

        qfix *gi_ref = grad_in;
        qfix *gw_cur = wgrad_cur;
        qfix *x_last = model->fullConnData[curidx];
        uint16_t widx = 0;
        for(uint16_t i = 0; i < outdi; i++){
            qfix grad_cur = gradfn(*gi_ref, dext);
            for(uint16_t j = 0; j < indi; j++){
                *gw_cur = qfix_mul(grad_cur, x_last[j]);
                grads_dst[j] += qfix_mul(grad_cur, wcur[widx]);
                widx++;
                gw_cur++;
            }
            gi_ref++;
        }
        
        for(uint16_t i = 0; i < outdi; i++) bcur[i] += qfix_mul(lr, gradfn(grad_in[i], dext));
        uint32_t wsize = outdi * indi;
        for(uint32_t i = 0; i < wsize; i++) wcur[i] += qfix_mul(lr, wgrad_cur[i]);

        nsrc--;
    }
}

void mlptrainer_totalgrads_backward(mlpTrainStatus *model, mlpTrainStatus *gradscap, qfix lr){
    uint32_t size = model->calclyrs;
    netLyrConf *nsrc = model->modelsrc;
    for(uint32_t i = 0; i < size; i++){
        uint16_t outdi = nsrc->out_dim;
        uint32_t wsize = nsrc->in_dim * outdi;
        qfix *bgrad_src = gradscap->lyrinput_grad[i];
        qfix *wgrad_src = gradscap->w_grad[i];
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

void mlptrainer_convcalc_zerobia(mlpTrainStatus* model){
    model->modelsrc->bias[0] = 0;
}

void mlptrainer_execute(mlpTrainStatus *model, qfix *arrin){
    uint32_t size = model->calclyrs;
    model->fullConnData[0] = arrin;
    netLyrConf *nsrc = model->modelsrc;
    qfix** fcd_export = model->fullConnData;
    acCall acfn;
    for(uint32_t i = 0; i < size; i++){
        qfix *in = fcd_export[i];
        qfix *out = fcd_export[i + 1];
        qfix *wcur = nsrc->weights;
        qfix *bcur = nsrc->bias;
        qfix dext = nsrc->dExtra;

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
            case ac_Sign: acfn = Sign; break;
            case ac_pass:
            default: acfn = linear; break;
        }

        qfix *wcurr = wcur;
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

void mlpexec_execute(mlpExecStatus *model, qfix *arrin){
    uint32_t size = model->calclyrs;
    netLyrConf *nsrc = model->modelsrc;
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
        qfix *wcur = nsrc->weights;
        qfix *bcur = nsrc->bias;
        qfix dext = nsrc->dExtra;

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
            case ac_Sign: acfn = Sign; break;
            case ac_pass:
            default: acfn = linear; break;
        }

        qfix *wcurr = wcur;
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

#include "mlp.h"
#include "../threadpool/threadpool.h"

/*DOC
多线程优化指导
对于mlp运算的链式特性,多线程优化将采用规约方式计算并用线程池整合

如:

        W        *      x   +   B
| 1, 2, 3, 4, 5, 6 |  | 1 |   | 3 |
| 1, 2, 3, 4, 5, 6 |  | 2 |   | 6 |
| 1, 2, 3, 4, 5, 6 |  | 3 |   | 9 |
| 1, 2, 3, 4, 5, 6 |  | 4 |   | 2 |
| 1, 2, 3, 4, 5, 6 |  | 5 |   | 4 |
| 1, 2, 3, 4, 5, 6 |  | 6 |   | 6 |

<mlpxxx_exec_mt>对于一般前向,其基本流程可分解为
sub1: sum_part_wx(w,x,r,c, delta_c) = sum(i = c, delta_c + c - 1, w[r, i] * x[i])
sub2: resu = call sub1 * ceil(w_c / delta_c) ; resu += B[i]; ac(resu)
sub3: sub2(i) * out_dim

<mlptrainer_backward_mt>对于一般反向,权重,偏置更新不影响反传,可以:
grad -> grad1 -> ... -> grad(n-1) -> gradn
     |        |                   |
     v        v                   v
     w,b      w,b                 w,b


<mlptrainer_totalgrads_xxx_mt>采用一般多线程模式即可

验证: 一般计算,指定 W,x,b 值并让其计算, 设置 actp = pass, 检测多线程结果是否符合预期；
    totalgrad的设置两组,第一组trainer_backward(lr任意),第二组trainer_backward(lr = 0)&savegrads&total_backward(lr与第一组一致),比较两组再次trainer_execute输出结果,两组输出相同则通过
*/

/*
 * 实现说明(基于 ../threadpool 的常驻线程池, 用法与 arrop.c 保持一致)
 *   - 线程池在进程内懒加载一次, 之后所有 mlp_*_mt 调用复用同一批工作线程;
 *     由于共享同一线程池, 各 mlp_*_mt 调用应串行使用(上一批任务完成后才可提交下一批)
 *   - 每批任务按连续区间切分, 每个任务只写自己负责的输出区间, 任务间无数据竞争
 *   - 工作量不足以抵消并发调度开销时自动退化为单线程, 且与单线程实现逐位等价
 *   - 前向按 DOC 的 sub1/sub2/sub3 分解: 行数足够时按行切分(无需规约);
 *     行数不足(如 out_dim = 1 的卷积核)时按列分块计算原始部分和, 再规约出结果
 *   - 反向按 DOC 的链式结构逐层推进(grad -> grad1 -> ... -> gradn),
 *     层内分两阶段并行: 阶段一按行(激活梯度/权重梯度/偏置), 阶段二按列(输入梯度/权重更新)
 *   - totalgrads 采用一般的多线程模式(按层并行)
 */

#define MLP_MT_MAX_TASKS 32u    /* 单批任务数上限, 避免超多核平台上的调度开销 */
#define MLP_MT_MIN_WORK 4096u   /* 单个任务至少承担的乘加/元素数 */
#define MLP_MT_GC_STACK 256u    /* 反向激活梯度缓存: 不超过该值时使用栈内存 */

/* 任务单元: 处理 [start, end) 连续区间, arg 指向各算子私有的参数结构体 */
typedef void (*mlp_mt_range_fn)(uint32_t start, uint32_t end, void *arg);

typedef struct mlp_mt_task{
    mlp_mt_range_fn fn;
    void *arg;
    uint32_t start;
    uint32_t end;
} mlp_mt_task;

/* 线程池回调: 从参数区取出一个任务并执行 */
static void mlp_mt_task_run(void *param){
    const mlp_mt_task *task = (const mlp_mt_task *)param;
    task->fn(task->start, task->end, task->arg);
}

/* 进程内共享的线程池(懒加载, 供所有 mlp_*_mt 调用复用) */
static threadpool g_mlp_mt_pool;
static int g_mlp_mt_pool_ready = 0;

/* 按工作量估算任务数: 工作量不足返回1, 由调用方在当前线程内联执行 */
static uint32_t mlp_mt_task_count(uint64_t work, uint32_t units){
    uint32_t cnt;
    if(units == 0) return 0;
    if(work < MLP_MT_MIN_WORK) return 1;
    cnt = (uint32_t)(work / MLP_MT_MIN_WORK);
    if(cnt > units) cnt = units;
    if(cnt > MLP_MT_MAX_TASKS) cnt = MLP_MT_MAX_TASKS;
    return cnt;
}

/* 把 [0,total) 切成 count 段连续区间任务提交线程池, 并等待本批任务全部完成 */
static void mlp_mt_parallel(uint32_t total, uint32_t count,
                            mlp_mt_range_fn fn, void *arg){
    mlp_mt_task tasks[MLP_MT_MAX_TASKS];
    uint32_t per, rem, start;

    if(total == 0) return;
    if(count == 0) count = 1;
    if(count > total) count = total;
    if(count > MLP_MT_MAX_TASKS) count = MLP_MT_MAX_TASKS;
    if(count <= 1){ /* 单任务: 直接在当前线程执行, 不经过线程池 */
        fn(0, total, arg);
        return;
    }

    per = total / count;
    rem = total % count;
    start = 0;
    for(uint32_t i = 0; i < count; i++){
        const uint32_t end = start + per + ((i < rem) ? 1u : 0u);
        tasks[i].fn = fn;
        tasks[i].arg = arg;
        tasks[i].start = start;
        tasks[i].end = end;
        start = end;
    }

    if(!g_mlp_mt_pool_ready){
        /* 首次调用创建线程池, 同时提交第一批任务 */
        init_pool(&g_mlp_mt_pool, mlp_mt_task_run, tasks, count, sizeof(mlp_mt_task));
        g_mlp_mt_pool_ready = 1;
    }else{
        pool_set_newtask(&g_mlp_mt_pool, tasks, count);
    }

    /* pool_checkTaskProcessing 返回0表示本批任务已全部执行完毕 */
    while(pool_checkTaskProcessing(&g_mlp_mt_pool)){
    }
}

/* 激活函数分发(与单线程实现完全一致) */
static acCall mlp_mt_acfn(uint8_t acTp){
    switch(acTp){
        case ac_ReLU: return ReLU;
        case ac_ReLU6: return ReLu6;
        case ac_LeakyReLU: return LeakyReLU;
        case ac_Sigmoid: return Sigmoid;
        case ac_Sigmoid_hard: return SigmoidH;
        case ac_Tanh: return Tanh;
        case ac_Tanh_hard: return TanhH;
        case ac_TripleLine: return TripleLine;
        case ac_pass:
        default: return linear;
    }
}

/* 激活函数导数分发(与单线程实现完全一致) */
static acCall mlp_mt_gradfn(uint8_t acTp){
    switch(acTp){
        case ac_ReLU: return grad_ReLU;
        case ac_ReLU6: return grad_ReLu6;
        case ac_LeakyReLU: return grad_LeakyReLU;
        case ac_Sigmoid: return grad_Sigmoid;
        case ac_Sigmoid_hard: return grad_SigmoidH;
        case ac_Tanh: return grad_Tanh;
        case ac_Tanh_hard: return grad_TanhH;
        case ac_TripleLine: return grad_TripleLine;
        case ac_pass:
        default: return grad_linear;
    }
}

/* --------------------------------------------------------------------------
 * 前向: DOC sub1 / sub2 / sub3
 *   sub1: sum_part_wx(w,x,r,c,delta_c) = sum(i = c .. c + delta_c - 1) w[r,i]*x[i]
 *   sub2: resu = sub1 累加 ceil(w_c / delta_c) 次, 再加偏置并激活
 *   sub3: 对 out_dim 行重复 sub2
 * 行切分: 每个任务完整负责若干行(每行只调用一次 sub1, 取全部列)
 * 列切分: 任务单元为(行, 列块), 只算 sub1 的原始部分和, 再由 sub2 规约
 * ------------------------------------------------------------------------ */

/* sub1: 第 r 行从第 c 列起 delta_c 个元素的原始部分和(不移位, 便于多线程分块后精确相加) */
static _tmp_larger mlp_mt_sum_part_wx(const qfix *w, const qfix *x,
                                      uint32_t r, uint32_t indi,
                                      uint32_t c, uint32_t delta_c){
    const qfix *wcur = w + (size_t)r * indi + c;
    const qfix *xcur = x + c;
    _tmp_larger sum = 0;
    for(uint32_t i = 0; i < delta_c; i++){
        sum += (_tmp_larger)(*wcur) * (*xcur);
        wcur++;
        xcur++;
    }
    return sum;
}

typedef struct mlp_mt_fwd_arg{
    const qfix *w;         /* 权重, 行优先 [outdi][indi] */
    const qfix *x;         /* 输入向量 [indi] */
    const qfix *bias;      /* 偏置 [outdi] */
    qfix *raw;             /* 激活前结果(va_wxb), 可为 NULL */
    qfix *out;             /* 激活结果 [outdi] */
    acCall acfn;
    qfix dext;
    uint32_t indi;
    uint32_t outdi;
    uint32_t nchunks;      /* 列分块数(规约模式); 1 表示行切分模式 */
    uint32_t chunk;        /* 每个列块的列数 */
    _tmp_larger *partial;  /* 规约模式的部分和 [outdi][nchunks] */
} mlp_mt_fwd_arg;

/* sub2 / sub3: 汇总第 r 行的部分和, 移位/加偏置/激活后写入 raw 与 out */
static void mlp_mt_fwd_emit(const mlp_mt_fwd_arg *a, uint32_t r){
    _tmp_larger sumw = 0;

    if(a->nchunks <= 1){
        sumw = mlp_mt_sum_part_wx(a->w, a->x, r, a->indi, 0, a->indi);
    }else{
        const _tmp_larger *prow = a->partial + (size_t)r * a->nchunks;
        for(uint32_t k = 0; k < a->nchunks; k++) sumw += prow[k];
    }

    const qfix cur = (qfix)((sumw >> QSHIFT) + a->bias[r]);
    if(a->raw) a->raw[r] = cur;
    a->out[r] = a->acfn(cur, a->dext);
}

/* 行切分任务: 每个任务完整负责 [start, end) 行 */
static void mlp_mt_fwd_rows(uint32_t start, uint32_t end, void *arg){
    const mlp_mt_fwd_arg *a = (const mlp_mt_fwd_arg *)arg;
    for(uint32_t r = start; r < end; r++) mlp_mt_fwd_emit(a, r);
}

/* 列分块任务: 任务单元 t = 行 r * nchunks + 列块 k, 只计算 sub1 的原始部分和 */
static void mlp_mt_fwd_parts(uint32_t start, uint32_t end, void *arg){
    const mlp_mt_fwd_arg *a = (const mlp_mt_fwd_arg *)arg;

    for(uint32_t t = start; t < end; t++){
        const uint32_t r = t / a->nchunks;
        const uint32_t k = t % a->nchunks;
        const uint32_t c = k * a->chunk;
        uint32_t delta = a->indi - c;
        if(delta > a->chunk) delta = a->chunk;

        a->partial[(size_t)r * a->nchunks + k] =
            mlp_mt_sum_part_wx(a->w, a->x, r, a->indi, c, delta);
    }
}

/* 单层前向: 依工作量与行数选择 行切分 或 列分块规约 */
static void mlp_mt_forward_lyr(const qfix *in, qfix *raw, qfix *out, const netLyrConf *lyr){
    mlp_mt_fwd_arg a;
    _tmp_larger partial[MLP_MT_MAX_TASKS];
    uint64_t work;
    uint32_t want, rows;

    if(lyr->in_dim == 0 || lyr->out_dim == 0) return;

    a.w = lyr->weights;
    a.x = in;
    a.bias = lyr->bias;
    a.raw = raw;
    a.out = out;
    a.acfn = mlp_mt_acfn(lyr->acTp);
    a.dext = lyr->dExtra;
    a.indi = lyr->in_dim;
    a.outdi = lyr->out_dim;
    a.nchunks = 1;
    a.chunk = lyr->in_dim;
    a.partial = partial;

    work = (uint64_t)lyr->in_dim * (uint64_t)lyr->out_dim;
    want = mlp_mt_task_count(work, MLP_MT_MAX_TASKS);
    rows = (want < a.outdi) ? want : a.outdi;   /* 行切分可用的任务数 */

    if(want >= 2 && rows < want){
        /* 行数不足(如 out_dim = 1 的卷积核): 行切分 + 列分块规约
         * 此处 rows == outdi, 故 nchunks >= 2, 且 outdi * nchunks <= want <= 上限 */
        const uint32_t nchunks = want / rows;
        if(nchunks >= 2){
            a.nchunks = nchunks;
            a.chunk = (a.indi + nchunks - 1) / nchunks;
            mlp_mt_parallel(a.outdi * nchunks, a.outdi * nchunks, mlp_mt_fwd_parts, &a);
            for(uint32_t r = 0; r < a.outdi; r++) mlp_mt_fwd_emit(&a, r);
            return;
        }
    }

    mlp_mt_parallel(a.outdi, rows, mlp_mt_fwd_rows, &a);
}

/* --------------------------------------------------------------------------
 * 反向: DOC 的链式结构 grad -> grad1 -> ... -> gradn
 *   权重/偏置更新不影响反传, 因此每层内部可并行:
 *   阶段一(按行): gc = grad_in * act'(x_ac), 写权重梯度行, 更新偏置
 *   阶段二(按列): 用更新前的权重规约出输入梯度, 再用权重梯度更新权重
 *   (每个任务只读写自己负责的列/行, 无数据竞争; 结果与单线程逐位一致)
 * ------------------------------------------------------------------------ */

typedef struct mlp_mt_bwd_arg{
    const qfix *grad_in;   /* 回传梯度 [outdi] */
    const qfix *x_last;    /* 本层输入 [indi] */
    const qfix *x_ac;      /* 激活前原始值 [outdi] */
    qfix *gc;              /* 激活梯度 [outdi] */
    qfix *wgrad;           /* 权重梯度 [outdi][indi] */
    qfix *w;               /* 权重 [outdi][indi] */
    qfix *xgrad;           /* 输入梯度累加器 [indi] */
    qfix *bias;            /* 偏置 [outdi] */
    acCall gradfn;
    qfix dext;
    qfix lr;
    uint32_t indi;
    uint32_t outdi;
} mlp_mt_bwd_arg;

/* 阶段一(按行切分) */
static void mlp_mt_bwd_rows(uint32_t start, uint32_t end, void *arg){
    const mlp_mt_bwd_arg *a = (const mlp_mt_bwd_arg *)arg;

    for(uint32_t i = start; i < end; i++){
        const qfix grad_cur = qfix_mul(a->grad_in[i], a->gradfn(a->x_ac[i], a->dext));
        qfix *gw = a->wgrad + (size_t)i * a->indi;

        a->gc[i] = grad_cur;
        for(uint32_t j = 0; j < a->indi; j++){
            gw[j] = qfix_mul(grad_cur, a->x_last[j]);
        }
        a->bias[i] += qfix_mul(a->lr, grad_cur);
    }
}

/* 阶段二(按列切分) */
static void mlp_mt_bwd_cols(uint32_t start, uint32_t end, void *arg){
    const mlp_mt_bwd_arg *a = (const mlp_mt_bwd_arg *)arg;

    for(uint32_t j = start; j < end; j++){
        qfix acc = a->xgrad[j];
        for(uint32_t i = 0; i < a->outdi; i++){
            const size_t wpos = (size_t)i * a->indi + j;
            /* 先用更新前的权重累加输入梯度, 再更新该权重(同一单元格, 无竞争) */
            acc += qfix_mul(a->gc[i], a->w[wpos]);
            a->w[wpos] += qfix_mul(a->lr, a->wgrad[wpos]);
        }
        a->xgrad[j] = acc;
    }
}

/* 单层反向: 阶段一按行并行, 阶段二按列并行 */
static void mlp_mt_backward_lyr(const netLyrConf *lyr, const qfix *grad_in,
                                qfix *grads_dst, qfix *wgrad_cur,
                                const qfix *x_last, const qfix *x_ac,
                                qfix *gc, const qfix lr){
    mlp_mt_bwd_arg a;
    const uint64_t work = (uint64_t)lyr->in_dim * (uint64_t)lyr->out_dim;

    a.grad_in = grad_in;
    a.x_last = x_last;
    a.x_ac = x_ac;
    a.gc = gc;
    a.wgrad = wgrad_cur;
    a.w = lyr->weights;
    a.xgrad = grads_dst;
    a.bias = lyr->bias;
    a.gradfn = mlp_mt_gradfn(lyr->acTp);
    a.dext = lyr->dExtra;
    a.lr = lr;
    a.indi = lyr->in_dim;
    a.outdi = lyr->out_dim;

    mlp_mt_parallel(a.outdi, mlp_mt_task_count(work, a.outdi), mlp_mt_bwd_rows, &a);
    mlp_mt_parallel(a.indi, mlp_mt_task_count(work, a.indi), mlp_mt_bwd_cols, &a);
}

void mlptrainer_backward_mt(const mlpTrainStatus *model, const qfix *grad_inital, const qfix lr){
    const uint32_t size = model->calclyrs;
    qfix gc_stack[MLP_MT_GC_STACK];
    qfix *gc = gc_stack;
    uint32_t maxoutdi = 0;

    if(size == 0) return;

    model->lyrinput_grad[size] = grad_inital;

    /* 激活梯度缓存: 小网络用栈内存, 大网络堆分配 */
    for(uint32_t i = 0; i < size; i++){
        const uint16_t outdi = model->modelsrc[i].out_dim;
        if(outdi > maxoutdi) maxoutdi = outdi;
    }
    if(maxoutdi > MLP_MT_GC_STACK){
        gc = (qfix*)malloc(sizeof(qfix) * maxoutdi);
        if(gc == NULL){ /* 内存不足: 退化为单线程实现(结果一致) */
            mlptrainer_backward(model, grad_inital, lr);
            return;
        }
    }

    {
        qfix **grads = model->lyrinput_grad;
        const netLyrConf *nsrc = &model->modelsrc[size - 1];
        for(uint32_t l = size; l > 0; l--){
            const uint32_t curidx = l - 1;
            mlp_mt_backward_lyr(nsrc, grads[l], grads[curidx], model->w_grad[curidx],
                                model->fullConnData[curidx], model->va_wxb[curidx],
                                gc, lr);
            nsrc--;
        }
    }

    if(gc != gc_stack) free(gc);
}

/* --------------------------------------------------------------------------
 * totalgrads: 采用一般的多线程模式(按层并行)
 *   任务单元 = 层索引, 每层只读写自己的权重/偏置与梯度缓冲, 无数据竞争
 * ------------------------------------------------------------------------ */

typedef struct mlp_mt_tgb_arg{
    const mlpTrainStatus *model;
    const mlpTrainStatus *gradscap;
    qfix lr;
} mlp_mt_tgb_arg;

static void mlp_mt_tgb_layers(uint32_t start, uint32_t end, void *arg){
    const mlp_mt_tgb_arg *a = (const mlp_mt_tgb_arg *)arg;

    for(uint32_t i = start; i < end; i++){
        const netLyrConf *nsrc = &a->model->modelsrc[i];
        const uint16_t outdi = nsrc->out_dim;
        const uint32_t wsize = (uint32_t)nsrc->in_dim * outdi;
        const qfix *bgrad_src = a->gradscap->lyrinput_grad[i];
        const qfix *wgrad_src = a->gradscap->w_grad[i];
        qfix *b_dst = nsrc->bias;
        qfix *w_dst = nsrc->weights;

        for(uint16_t i_ = 0; i_ < outdi; i_++) b_dst[i_] += qfix_mul(bgrad_src[i_], a->lr);
        for(uint32_t i_ = 0; i_ < wsize; i_++) w_dst[i_] += qfix_mul(wgrad_src[i_], a->lr);
    }
}

void mlptrainer_totalgrads_backward_mt(const mlpTrainStatus *model, const mlpTrainStatus *gradscap, const qfix lr){
    const uint32_t size = model->calclyrs;
    mlp_mt_tgb_arg a;
    uint64_t work = 0;

    if(size == 0) return;

    for(uint32_t i = 0; i < size; i++){
        const netLyrConf *nsrc = &model->modelsrc[i];
        work += (uint64_t)nsrc->out_dim + (uint64_t)nsrc->in_dim * nsrc->out_dim;
    }

    a.model = model;
    a.gradscap = gradscap;
    a.lr = lr;
    mlp_mt_parallel(size, mlp_mt_task_count(work, size), mlp_mt_tgb_layers, &a);
}

typedef struct mlp_mt_sg_arg{
    const mlpTrainStatus *src;
    const mlpTrainStatus *dst;
} mlp_mt_sg_arg;

static void mlp_mt_sg_layers(uint32_t start, uint32_t end, void *arg){
    const mlp_mt_sg_arg *a = (const mlp_mt_sg_arg *)arg;

    for(uint32_t i = start; i < end; i++){
        const netLyrConf *nsrc = &a->src->modelsrc[i];
        const uint16_t indi = nsrc->in_dim;
        const uint32_t wsize = (uint32_t)nsrc->out_dim * indi;
        const qfix *bgrad_src = a->src->lyrinput_grad[i];
        const qfix *wgrad_src = a->src->w_grad[i];
        qfix *bgrad_dst = a->dst->lyrinput_grad[i];
        qfix *wgrad_dst = a->dst->w_grad[i];

        for(uint16_t i_ = 0; i_ < indi; i_++) bgrad_dst[i_] += bgrad_src[i_];
        for(uint32_t i_ = 0; i_ < wsize; i_++) wgrad_dst[i_] += wgrad_src[i_];
    }
}

void mlptrainer_totalgrads_savegrads_mt(const mlpTrainStatus *net_or_gradcap, const mlpTrainStatus *gradscap_dest){
    const uint32_t size = net_or_gradcap->calclyrs;
    mlp_mt_sg_arg a;
    uint64_t work = 0;

    if(size == 0) return;

    for(uint32_t i = 0; i < size; i++){
        const netLyrConf *nsrc = &net_or_gradcap->modelsrc[i];
        work += (uint64_t)nsrc->in_dim + (uint64_t)nsrc->out_dim * nsrc->in_dim;
    }

    a.src = net_or_gradcap;
    a.dst = gradscap_dest;
    mlp_mt_parallel(size, mlp_mt_task_count(work, size), mlp_mt_sg_layers, &a);
}

/* --------------------------------------------------------------------------
 * 执行入口(mlpxxx_exec_mt): 逐层前向, 层内并行
 * ------------------------------------------------------------------------ */

void mlptrainer_execute_mt(const mlpTrainStatus *model, const qfix *arrin){
    const uint32_t size = model->calclyrs;
    const netLyrConf *nsrc = model->modelsrc;
    qfix **fcd_export = model->fullConnData;

    if(size == 0) return;

    model->fullConnData[0] = arrin;
    for(uint32_t i = 0; i < size; i++){
        mlp_mt_forward_lyr(fcd_export[i], model->va_wxb[i], fcd_export[i + 1], nsrc);
        nsrc++;
    }
}

void mlpexec_execute_mt(const mlpExecStatus *model, const qfix *arrin){
    const uint32_t size = model->calclyrs;
    const netLyrConf *nsrc = model->modelsrc;
    qfix **fcd_export = model->fullConnData_tmp;
    const qfix *in = arrin;

    for(uint32_t i = 0; i < size; i++){
        qfix *out = fcd_export[(i + 1) % 2];
        mlp_mt_forward_lyr(in, NULL, out, nsrc);
        in = out;   /* 下一层的输入即本层输出(与原双缓冲写法等价) */
        nsrc++;
    }
}

#include "mlp.h"

//GUIDE:多线程化使用这两个(threadpool.h + sharedpool.h)
#include "../threadpool/threadpool.h"
#include "../threadpool/sharedpool.h"

/*GUIDE:
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
行内求和sub1: sum_part_wx(w,x,r,c, delta_c) = sum(i = c, delta_c + c - 1, w[r, i] * x[i])       processor
汇总单行结果sub2: resu = call sub1 * ceil(w_c / delta_c) ; resu += B[i]; ac(resu)               checker
派发任务<mlpxxx,按层>: sub2(i) * out_dim                                                        checker

其可表示为
lyr --派发--> sub2 --派发--> sub1
    <--汇总--      <--汇总--  

执行时出现以下情况:
sub1 -- sub2<sub1任务没完成> -- ... -- caller<sub2没完成>
那么为保持线程池资源不被长期占用,派发者检测到子任务未完成,直接退出,等待下一次被轮询到并重复以上

例:
设立完成标志fin,运行标志start,初始为0
sub2派发sub1,退出,sub1设置fin = 0, start = 1
sub2被轮询到,检查到fin==0,退出
sub2被轮询到,检查到fin==1,则做后续处理

对于checker,若processor/子checker对应的完成标志为0且pubpools_checkprocessing返回0,则把同一批processor/子checker任务再提交一次,checker需要记录对应processor/子checker的运行状态

<mlptrain_backward_mt>对于一般反向,权重,偏置更新不影响反传,可以:
grad -> grad1 -> ... -> grad(n-1) -> gradn
     |        |                   |
     v        v                   v
     w,b      w,b                 w,b


<mlptrainer_collector_xxx_mt>采用一般多线程模式即可

验证: 一般计算,指定 W,x,b 值并让其计算, 设置 actp = pass, 检测多线程结果是否符合预期,使用原实现(./mlp.c)作为参考;
    collector的设置两组,第一组trainer_backward(lr任意),第二组trainer_backward(lr = 0)&savegrads&total_upgrade(lr与第一组一致),比较两组再次train_execute输出结果,两组输出相同则通过
*/

/* =========================================================================
 * 多线程基础设施
 * -------------------------------------------------------------------------
 * 与 arrop.c 的多线程实现共用同一套约定:
 *   - 经 sharedpool 的进程级共享线程池表申请/提交/归还线程池, 复用常驻工作线程;
 *   - 每个任务只写自己负责的输出区间, 任务之间无数据竞争, 无需额外加锁;
 *   - sharedpool 复用线程池时沿用首次 init_pool 的参数步长(threadpool.c 的
 *     pool_set_newtask 不更新 param_size), 故本文件所有批次共用同一个任务结构体
 *     与同一个池回调; 任务结构布局亦与 arrop 的 arr_mt_task 一致(两者可能复用同一池);
 *   - 规模不足以抵消调度开销时本线程内联执行; 长时间无空闲线程池时退化为本线程
 *     串行执行, 保证结果与单线程实现(./mlp.c)逐位一致且不与池内任务互相等待.
 * ========================================================================= */

#define MLP_MT_MAX_TASKS        128u  /* 单批任务数上限 */
#define MLP_MT_TARGET_TASKS     64u   /* 期望的并行工作项数 */
#define MLP_MT_MIN_OPS_PER_TASK 64u   /* 单个任务最少的乘加次数 */
#define MLP_MT_INLINE_OPS       128u  /* 低于该规模直接本线程计算(小层内联) */
#define MLP_MT_WAIT_SPIN        512u  /* 等待本批完成的自旋轮数, 超过后小睡让出CPU */
#define MLP_MT_DISPATCH_RETRY   32u   /* 无空闲线程池时的退避重试轮数 */
#define MLP_MT_RESUBMIT_ROUNDS  4u    /* 完成标志未全部就绪时, 重复提交同一批的轮数 */

/*
 * 任务单元(布局与 arrop 的 arr_mt_task 一致: 函数指针 + 参数 + 区间)
 * 约定 fn(arg, start, end) 处理 [start, end) 号工作项
 */
typedef struct mlpmt_task{
    void (*fn)(void *arg, uint32_t start, uint32_t end);
    void *arg;
    uint32_t start;
    uint32_t end;
} mlpmt_task;

/* 线程池回调: 取出一个任务并执行(所有批次共用, 满足复用池时回调统一的约束) */
static void mlpmt_task_run(void *param){
    const mlpmt_task *task = (const mlpmt_task *)param;
    task->fn(task->arg, task->start, task->end);
}

/* GUIDE中的"完成标志": 全部为1表示这一批processor/子checker已全部完成 */
typedef struct mlpmt_verify{
    const volatile int32_t *flags;
    uint32_t count;
} mlpmt_verify;

static int mlpmt_verify_incomplete(void *arg){
    const mlpmt_verify *v = (const mlpmt_verify *)arg;
    for(uint32_t i = 0; i < v->count; i++){
        if(v->flags[i] == 0) return 1;
    }
    return 0;
}

/* 按工作量决定任务数, 并把 [0,total) 切成连续区间写入任务数组, 返回任务数 */
static uint32_t mlpmt_split(mlpmt_task *tasks, uint32_t total, uint32_t min_per_task,
                            void (*fn)(void*,uint32_t,uint32_t), void *arg){
    uint32_t ntasks, per, rem, start, i;
    if(total == 0) return 0;
    if(min_per_task == 0) min_per_task = 1;
    ntasks = total / min_per_task;
    if(ntasks == 0) ntasks = 1;
    if(ntasks > total) ntasks = total;
    if(ntasks > MLP_MT_MAX_TASKS) ntasks = MLP_MT_MAX_TASKS;
    per = total / ntasks;
    rem = total % ntasks;
    start = 0;
    for(i = 0; i < ntasks; i++){
        const uint32_t end = start + per + ((i < rem) ? 1u : 0u);
        tasks[i].fn = fn;
        tasks[i].arg = arg;
        tasks[i].start = start;
        tasks[i].end = end;
        start = end;
    }
    return ntasks;
}

/*
 * 派发一批任务到共享线程池并等待完成.
 * GUIDE: 派发者检测到子任务未完成则直接退出, 等下一次被轮询到再处理; 对checker而言,
 * 若完成标志未全部置位而线程池已空闲, 则把同一批任务再提交一次.
 * 长时间没有可用线程池时本线程串行执行整批任务, 结果等价且不会与池内任务互相等待.
 */
static void mlpmt_run_batch(mlpmt_task *tasks, uint32_t ntasks, mlpmt_verify *verify){
    uint32_t i;
    uint8_t round;
    if(ntasks == 0) return;
    for(round = 0; round < MLP_MT_RESUBMIT_ROUNDS; round++){
        int32_t slot = -1;
        for(i = 0; i < MLP_MT_DISPATCH_RETRY; i++){
            slot = pubpools_append_task(mlpmt_task_run, tasks, ntasks, sizeof(mlpmt_task));
            if(slot >= 0) break;
            tp_sleep_ms(1);
        }
        if(slot < 0){        /* 暂时没有空闲线程池: 本线程串行执行, 结果一致 */
            for(i = 0; i < ntasks; i++) mlpmt_task_run(&tasks[i]);
            return;
        }
        /* 等待本批任务全部完成(pubpools_checkprocessing返回0表示已完成) */
        for(i = 0; pubpools_checkprocessing((uint16_t)slot); i++){
            if(i >= MLP_MT_WAIT_SPIN) tp_sleep_ms(1);
        }
        pubpools_release((uint16_t)slot);
        if(verify == 0 || !mlpmt_verify_incomplete(verify)) return;  /* 完成标志已全部置位 */
    }
}

/* 激活函数/激活梯度查表(与 mlp.c 的 switch 保持一致) */
static acCall mlpmt_acfn_of(uint8_t acTp){
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

static acCall mlpmt_gradfn_of(uint8_t acTp){
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

/* =========================================================================
 * 一般前向: <mlpxxx_exec_mt>
 * -------------------------------------------------------------------------
 * 分层执行, 层内按GUIDE分解为
 *   sub1(processor): 行内求和 sum_part_wx(w,x,r,c,delta_c)
 *   sub2(checker)  : 汇总单行分块和 = call sub1 * ceil(w_c/delta_c); += B[r]; ac(...)
 * 层派发者(mlpmt_forward_layer)按层派发checker; checker依赖的processor完成标志由
 * processor置位, checker被轮询时若processor未完成则记录待处理状态并退出, 由层派发者再次轮询.
 * ========================================================================= */

typedef struct mlpmt_fwd_ctx{
    const qfix *w;             /* 权重(行优先) */
    const qfix *x;             /* 层输入 */
    const qfix *bias;          /* 偏置 */
    qfix *out;                 /* 层输出 */
    qfix *raw;                 /* 激活前原始值(mlpexec无需, 可为NULL) */
    acCall acfn;
    qfix dext;
    uint16_t indi;
    uint16_t outdi;
    uint16_t nchunk;           /* 每行分块数 ceil(indi/delta) */
    uint16_t delta;            /* 每块元素数 */
    _tmp_larger *parts;        /* outdi*nchunk 个分块和 */
    volatile int32_t *fin;     /* processor完成标志(GUIDE: sub1设置fin) */
    volatile int32_t *done;    /* checker完成标志 */
} mlpmt_fwd_ctx;

/* 单行完整计算(小层内联与兜底路径使用, 与 mlp.c 的实现逐位一致) */
static void mlpmt_forward_row(const qfix *wrow, const qfix *x, uint16_t indi,
                              const qfix *bias, qfix *raw, qfix *out, acCall acfn, qfix dext){
    _tmp_larger sumw = 0;
    for(uint16_t k = 0; k < indi; k++) sumw += (_tmp_larger)wrow[k] * x[k];
    const qfix cur = (qfix)(sumw >> QSHIFT) + *bias;
    if(raw) *raw = cur;
    *out = acfn(cur, dext);
}

/* sub1 processor: 求工作项[job = 行*nchunk + 块]的块内和, 并置完成标志 */
static void mlpmt_fwd_processor(void *arg, uint32_t start, uint32_t end){
    const mlpmt_fwd_ctx *c = (const mlpmt_fwd_ctx *)arg;
    for(uint32_t job = start; job < end; job++){
        const uint32_t r = job / c->nchunk;
        const uint32_t blk = job % c->nchunk;
        uint32_t off = blk * c->delta;
        uint32_t len = c->indi - off;
        if(len > c->delta) len = c->delta;
        const qfix *wp = c->w + (size_t)r * c->indi + off;
        const qfix *xp = c->x + off;
        _tmp_larger s = 0;
        for(uint32_t k = 0; k < len; k++) s += (_tmp_larger)wp[k] * xp[k];
        c->parts[job] = s;
        c->fin[job] = 1;       /* GUIDE: sub1置自身完成标志 */
    }
}

/* sub2 checker: 汇总 [start,end) 行的分块和, 加偏置并过激活 */
static void mlpmt_fwd_checker(void *arg, uint32_t start, uint32_t end){
    const mlpmt_fwd_ctx *c = (const mlpmt_fwd_ctx *)arg;
    for(uint32_t r = start; r < end; r++){
        const uint32_t base = r * c->nchunk;
        uint32_t k;
        for(k = 0; k < c->nchunk; k++){
            if(c->fin[base + k] == 0) break;
        }
        if(k < c->nchunk){     /* GUIDE: 子任务未完成, 记录状态并退出, 等待再次被轮询 */
            c->done[r] = 0;
            continue;
        }
        _tmp_larger sumw = 0;
        for(k = 0; k < c->nchunk; k++) sumw += c->parts[base + k];
        const qfix cur = (qfix)(sumw >> QSHIFT) + c->bias[r];
        if(c->raw) c->raw[r] = cur;
        c->out[r] = c->acfn(cur, c->dext);
        c->done[r] = 1;
    }
}

/* 层派发者: 决定分块 -> 派发sub1(processor) -> 派发sub2(checker) -> 复查未完成行 */
static void mlpmt_forward_layer(const qfix *x, qfix *out, qfix *raw,
                                const qfix *w, const qfix *bias,
                                uint16_t indi, uint16_t outdi, acCall acfn, qfix dext){
    mlpmt_fwd_ctx c;
    mlpmt_task tasks[MLP_MT_MAX_TASKS];
    mlpmt_verify vf;
    _tmp_larger *parts;
    volatile int32_t *fin, *done;
    uint32_t njobs, nchunk, delta, ops, r, ntasks;
    uint8_t round, pending;

    ops = (uint32_t)indi * outdi;
    if(ops <= MLP_MT_INLINE_OPS){    /* 小层内联: 免去调度与分配开销 */
        for(r = 0; r < outdi; r++)
            mlpmt_forward_row(w + (size_t)r * indi, x, indi, bias + r,
                              raw ? raw + r : 0, out + r, acfn, dext);
        return;
    }

    /* out_dim大时按行并行; 行数少时把行内再分块以凑足并行度 */
    if(outdi >= MLP_MT_TARGET_TASKS){
        nchunk = 1;
    }else{
        nchunk = (MLP_MT_TARGET_TASKS + outdi - 1) / outdi;
        if(nchunk > indi) nchunk = indi;
    }
    delta = (indi + nchunk - 1) / nchunk;
    if(delta == 0) delta = 1;
    nchunk = (indi + delta - 1) / delta;
    njobs = (uint32_t)outdi * nchunk;

    parts = (_tmp_larger *)malloc(sizeof(_tmp_larger) * njobs);
    fin   = (volatile int32_t *)calloc(njobs, sizeof(int32_t));
    done  = (volatile int32_t *)calloc(outdi, sizeof(int32_t));
    if(parts == NULL || fin == NULL || done == NULL){    /* 分配失败: 回退本线程串行 */
        free(parts); free((void *)fin); free((void *)done);
        for(r = 0; r < outdi; r++)
            mlpmt_forward_row(w + (size_t)r * indi, x, indi, bias + r,
                              raw ? raw + r : 0, out + r, acfn, dext);
        return;
    }

    c.w = w; c.x = x; c.bias = bias; c.out = out; c.raw = raw;
    c.acfn = acfn; c.dext = dext;
    c.indi = indi; c.outdi = outdi;
    c.nchunk = (uint16_t)nchunk; c.delta = (uint16_t)delta;
    c.parts = parts; c.fin = fin; c.done = done;

    /* sub1: 行内分块求和(processor) */
    ntasks = mlpmt_split(tasks, njobs, MLP_MT_MIN_OPS_PER_TASK / (delta ? delta : 1),
                         mlpmt_fwd_processor, &c);
    vf.flags = fin; vf.count = njobs;
    mlpmt_run_batch(tasks, ntasks, &vf);

    /* sub2: 汇总单行结果(checker); 未完成的行由层派发者再次轮询 */
    for(round = 0; round < MLP_MT_RESUBMIT_ROUNDS; round++){
        pending = 0;
        for(r = 0; r < outdi; r++){
            if(done[r] == 0){ pending = 1; break; }
        }
        if(!pending) break;
        ntasks = mlpmt_split(tasks, outdi, 1, mlpmt_fwd_checker, &c);
        vf.flags = done; vf.count = outdi;
        mlpmt_run_batch(tasks, ntasks, &vf);
    }

    /* 兜底: 极端情况下(checker始终拿不到processor结果)本线程直接收尾 */
    for(r = 0; r < outdi; r++){
        if(done[r] == 0)
            mlpmt_forward_row(w + (size_t)r * indi, x, indi, bias + r,
                              raw ? raw + r : 0, out + r, acfn, dext);
    }

    free(parts); free((void *)fin); free((void *)done);
}

/* 一般前向(训练用: 记录激活前原始值供反向使用) */
void mlptrain_execute_mt(const mlpTrainStatus *model, const qfix *arrin){
    const uint32_t size = model->calclyrs;
    model->fullConnData[0] = (qfix *)arrin;
    const netLyrConf *nsrc = model->modelsrc;
    qfix **fcd_export = model->fullConnData;
    for(uint32_t i = 0; i < size; i++){
        mlpmt_forward_layer(fcd_export[i], fcd_export[i + 1], model->va_wxb[i],
                            nsrc->weights, nsrc->bias,
                            nsrc->in_dim, nsrc->out_dim, mlpmt_acfn_of(nsrc->acTp), nsrc->dExtra);
        nsrc++;
    }
}

/* 一般前向(推理用: 双缓冲, 不记录激活前原始值) */
void mlpexec_execute_mt(const mlpExecStatus *model, const qfix *arrin){
    const uint32_t size = model->calclyrs;
    const netLyrConf *nsrc = model->modelsrc;
    for(uint32_t i = 0; i < size; i++){
        const qfix *in = (i == 0) ? arrin : model->fullConnData_tmp[i % 2];
        qfix *out = model->fullConnData_tmp[(i + 1) % 2];
        mlpmt_forward_layer(in, out, 0, nsrc->weights, nsrc->bias,
                            nsrc->in_dim, nsrc->out_dim, mlpmt_acfn_of(nsrc->acTp), nsrc->dExtra);
        nsrc++;
    }
}

/* =========================================================================
 * 一般反向: <mlptrain_backward_mt>
 * -------------------------------------------------------------------------
 * 权重/偏置更新不影响反传, 故逐层执行:
 * grad -> grad1 -> ... -> grad(n-1) -> gradn
 *      |        |                   |
 *      v        v                   v
 *      w,b      w,b                 w,b
 * 为与单线程实现逐位一致, 单层内分三个阶段(加法顺序、所读权重旧值均保持一致):
 *   阶段A 逐行: grad_cur = grad_in[i]*gradfn(x_ac[i]); w_grad[i][*]; bias[i] += lr*grad_cur
 *   阶段B 逐列: grads_dst[j] += sum_i grad_cur[i]*w[i][j]  (读权重旧值)
 *   阶段C 逐元素: weights[*] += lr*w_grad[*]               (必须晚于阶段B)
 * ========================================================================= */

typedef struct mlpmt_bwd_ctx{
    const qfix *grad_in;
    qfix *grads_dst;
    qfix *bgrad_dst;
    qfix *wgrad;
    qfix *w;
    qfix *bias;
    const qfix *x_last;
    const qfix *x_ac;
    qfix *gc;              /* 各输出行的梯度 grad_cur */
    acCall gradfn;
    qfix dext;
    qfix lr;
    uint16_t indi;
    uint16_t outdi;
} mlpmt_bwd_ctx;

/* 阶段A processor: 逐行求输出梯度, 并更新权梯度/偏置 */
static void mlpmt_bwd_gradrows(void *arg, uint32_t start, uint32_t end){
    const mlpmt_bwd_ctx *c = (const mlpmt_bwd_ctx *)arg;
    for(uint32_t i = start; i < end; i++){
        const qfix grad_cur = qfix_mul(c->grad_in[i], c->gradfn(c->x_ac[i], c->dext));
        qfix *gw = c->wgrad + (size_t)i * c->indi;
        c->gc[i] = grad_cur;
        for(uint16_t j = 0; j < c->indi; j++) gw[j] = qfix_mul(grad_cur, c->x_last[j]);
        c->bias[i] += qfix_mul(c->lr, grad_cur);
        c->bgrad_dst[i] = grad_cur;
    }
}

/* 阶段B processor: 逐列回传输入梯度(同一列的加法顺序与单线程一致) */
static void mlpmt_bwd_inrows(void *arg, uint32_t start, uint32_t end){
    const mlpmt_bwd_ctx *c = (const mlpmt_bwd_ctx *)arg;
    for(uint32_t j = start; j < end; j++){
        qfix acc = 0;   /* 该缓冲在 setup 阶段未初始化, 不能作为累加基值(与单线程版一致) */
        for(uint16_t i = 0; i < c->outdi; i++)
            acc += qfix_mul(c->gc[i], c->w[(size_t)i * c->indi + j]);
        c->grads_dst[j] = acc;
    }
}

/* 阶段C processor: 按权梯度就地更新权重(逐元素独立) */
static void mlpmt_bwd_wupdate(void *arg, uint32_t start, uint32_t end){
    const mlpmt_bwd_ctx *c = (const mlpmt_bwd_ctx *)arg;
    for(uint32_t idx = start; idx < end; idx++)
        c->w[idx] += qfix_mul(c->lr, c->wgrad[idx]);
}

/* 单层反向: 小层内联, 否则按阶段A/B/C各派发一批任务 */
static void mlpmt_backward_layer(const qfix *grad_in, qfix *grads_dst, qfix *bgrad_dst,
                                 qfix *wgrad, qfix *w, qfix *bias,
                                 const qfix *x_last, const qfix *x_ac,
                                 acCall gradfn, qfix dext, qfix lr,
                                 uint16_t indi, uint16_t outdi){
    mlpmt_task tasks[MLP_MT_MAX_TASKS];
    mlpmt_bwd_ctx c;
    const uint32_t wsize = (uint32_t)outdi * indi;
    uint32_t ntasks, minper;
    qfix *gc;

    c.grad_in = grad_in; c.grads_dst = grads_dst; c.bgrad_dst = bgrad_dst;
    c.wgrad = wgrad; c.w = w; c.bias = bias;
    c.x_last = x_last; c.x_ac = x_ac; c.gc = 0;
    c.gradfn = gradfn; c.dext = dext; c.lr = lr;
    c.indi = indi; c.outdi = outdi;

    gc = (qfix *)malloc(sizeof(qfix) * outdi);
    if(gc == NULL || wsize <= MLP_MT_INLINE_OPS){    /* 小层内联/分配失败: 本线程串行 */
        c.gc = gc;
        mlpmt_bwd_gradrows(&c, 0, outdi);
        mlpmt_bwd_inrows(&c, 0, indi);
        mlpmt_bwd_wupdate(&c, 0, wsize);
        free(gc);
        return;
    }
    c.gc = gc;

    /* 阶段A: 逐行求梯度 */
    minper = MLP_MT_MIN_OPS_PER_TASK / (indi ? indi : 1);
    ntasks = mlpmt_split(tasks, outdi, minper, mlpmt_bwd_gradrows, &c);
    mlpmt_run_batch(tasks, ntasks, 0);

    /* 阶段B: 逐列回传输入梯度(需在阶段C之前读到权重旧值) */
    minper = MLP_MT_MIN_OPS_PER_TASK / (outdi ? outdi : 1);
    ntasks = mlpmt_split(tasks, indi, minper, mlpmt_bwd_inrows, &c);
    mlpmt_run_batch(tasks, ntasks, 0);

    /* 阶段C: 更新权重 */
    ntasks = mlpmt_split(tasks, wsize, MLP_MT_MIN_OPS_PER_TASK, mlpmt_bwd_wupdate, &c);
    mlpmt_run_batch(tasks, ntasks, 0);

    free(gc);
}

void mlptrain_backward_mt(const mlpTrainStatus *model, const qfix *grad_inital, const qfix lr){
    const uint32_t size = model->calclyrs;
    const netLyrConf *nsrc = &model->modelsrc[size - 1];
    model->lyrinput_grad[size] = (qfix *)grad_inital;
    qfix **grads = model->lyrinput_grad;
    for(uint32_t l = size; l > 0; l--){
        const uint32_t curidx = l - 1;
        mlpmt_backward_layer(grads[l], grads[curidx], model->b_grad[curidx],
                             model->w_grad[curidx], nsrc->weights, nsrc->bias,
                             model->fullConnData[curidx], model->va_wxb[curidx],
                             mlpmt_gradfn_of(nsrc->acTp), nsrc->dExtra, lr,
                             nsrc->in_dim, nsrc->out_dim);
        nsrc--;
    }
}

/* =========================================================================
 * collector / 参数更新: <mlptrainer_collector_xxx_mt>
 * 采用一般多线程模式(逐元素切分派发)即可.
 * 注: 单线程版 mlptrain_update_from_collector 中 "modelsrc[l].bias += ..." /
 *     "modelsrc[l].weights += ..." 是对指针自增(疑似笔误), 多线程版按元素累加实现,
 *     与 mlptrain_backward_mt 的就地 w,b 更新语义一致, 也更适合多线程.
 * ========================================================================= */

typedef struct mlpmt_acc_ctx{
    const qfix *src;
    qfix *dst;
    qfix scale;      /* 缩放比例(collect用lr_cur, update为直接累加) */
} mlpmt_acc_ctx;

static void mlpmt_acc_scaled(void *arg, uint32_t start, uint32_t end){
    const mlpmt_acc_ctx *c = (const mlpmt_acc_ctx *)arg;
    for(uint32_t i = start; i < end; i++) c->dst[i] += qfix_mul(c->scale, c->src[i]);
}

static void mlpmt_acc_plain(void *arg, uint32_t start, uint32_t end){
    const mlpmt_acc_ctx *c = (const mlpmt_acc_ctx *)arg;
    for(uint32_t i = start; i < end; i++) c->dst[i] += c->src[i];
}

/* 一般多线程模式: 逐元素累加, 规模小时内联 */
static void mlpmt_accum_range(const qfix *src, qfix *dst, uint32_t len,
                              void (*fn)(void*,uint32_t,uint32_t), mlpmt_acc_ctx *c){
    mlpmt_task tasks[MLP_MT_MAX_TASKS];
    uint32_t ntasks;
    if(len == 0) return;
    c->src = src; c->dst = dst;
    if(len <= MLP_MT_INLINE_OPS){
        fn(c, 0, len);
        return;
    }
    ntasks = mlpmt_split(tasks, len, MLP_MT_MIN_OPS_PER_TASK, fn, c);
    mlpmt_run_batch(tasks, ntasks, 0);
}

void DLLEXPORT mlptrain_collector_collect_mt(const mlpTrainStatus* net_or_gradcap,
                                            const mlpTrainStatus* gradscap_dest, const qfix lr_cur){
    const uint32_t netlen = gradscap_dest->calclyrs;
    for(uint32_t l = 0; l < netlen; l++){
        const uint16_t indi = gradscap_dest->modelsrc[l].in_dim;
        const uint16_t outdi = gradscap_dest->modelsrc[l].out_dim;
        mlpmt_acc_ctx c;
        c.scale = lr_cur;
        mlpmt_accum_range(net_or_gradcap->b_grad[l], gradscap_dest->b_grad[l],
                          outdi, mlpmt_acc_scaled, &c);
        mlpmt_accum_range(net_or_gradcap->w_grad[l], gradscap_dest->w_grad[l],
                          (uint32_t)indi * outdi, mlpmt_acc_scaled, &c);
    }
}

void DLLEXPORT mlptrain_update_from_collector_mt(const mlpTrainStatus *model, const mlpTrainStatus *gradscap){
    const uint32_t netlen = model->calclyrs;
    netLyrConf *modelsrc = model->modelsrc;
    for(uint32_t l = 0; l < netlen; l++){
        const uint16_t indi = modelsrc[l].in_dim;
        const uint16_t outdi = modelsrc[l].out_dim;
        mlpmt_acc_ctx c;
        c.scale = Q_1p0;
        mlpmt_accum_range(gradscap->b_grad[l], modelsrc[l].bias, outdi, mlpmt_acc_plain, &c);
        mlpmt_accum_range(gradscap->w_grad[l], modelsrc[l].weights,
                          (uint32_t)indi * outdi, mlpmt_acc_plain, &c);
    }
}

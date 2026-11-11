#include "arrop.h"
#include "../qfix_ops/qfix_op.h"
#include "../CustomConf.h"

DLLEXPORT void arr_add(const qfix *in1, const qfix* in2, const uint32_t len, qfix* out){
    for(uint32_t i = 0; i < len; i++){
        out[i] = in1[i] + in2[i];
    }
}

DLLEXPORT void arr_sub(const qfix *in1, const qfix* in2, const uint32_t len, qfix* out){
    for(uint32_t i = 0; i < len; i++){
        out[i] = in1[i] - in2[i];
    }
}

DLLEXPORT void arr_scale(const qfix *in1, const qfix scale, const uint32_t len, qfix* out){
    for(uint32_t i = 0; i < len; i++){
        out[i] = qfix_mul(in1[i], scale);
    }
}

DLLEXPORT void arr_emul(const qfix *in1, const qfix* in2, const uint32_t len, qfix* out){
    for(uint32_t i = 0; i < len; i++){
        out[i] = qfix_mul(in1[i], in2[i]);
    }
}

DLLEXPORT void arr_mmul(const qfix *in1, const qfix* in2, const uint16_t rows1, const uint16_t c1_r2, const uint16_t cols2, qfix* out){
    uint32_t a_row_offset = 0;
    for (uint16_t i = 0; i < rows1; i++){
        for (uint16_t j = 0; j < cols2; j++){
            _tmp_larger sum = 0;
            // 内循环使用指针运算
            const qfix *a_ptr = &in1[a_row_offset];
            const qfix *b_ptr = &in2[j]; // B的第j列起始位置
            for (uint16_t kk = 0; kk < c1_r2; kk++){
                sum += ((_tmp_larger)(*a_ptr) * (*b_ptr));
                a_ptr++;
                b_ptr += cols2; // 跳到B的下一行（列优先访问）
            }
            out[i * cols2 + j] = (qfix)(sum  >> QSHIFT);
        }
        a_row_offset += c1_r2;
    }
}

DLLEXPORT void arr_mtpose(const qfix *in1, const uint16_t rows1, const uint16_t cols1, qfix* out){
    for (uint16_t i = 0; i < rows1; i++){
        const qfix *start0 = in1 + (size_t)i * cols1;
        qfix *destBlock0 = out + i;
        for (uint16_t j = 0; j < cols1; j++){
            *destBlock0 = *start0;
            start0++;
            destBlock0 += rows1;
        }
    }
}

#ifdef MULTI_PROCESS
//<Promot for AI>注:多线程函数借助 ../threadpool 的 sharedpool 共享线程池实现, 支持win(winapi)/apple/linux(pthread)

/*
 * 多线程实现(基于 ../threadpool 的 sharedpool 共享线程池)
 *
 * 说明:
 *   - sharedpool 维护一张进程内共享的受管线程池表(动态库装载时自动初始化),
 *     每次 arr_*_mt 调用向表中申请一个空闲线程池并提交本批任务, 等全部完成后归还;
 *     这样既复用了常驻工作线程, 又允许多个调用方并发使用不同的池(互不阻塞)
 *   - 每批任务按连续区间切分, 每个任务只写自己负责的输出区间, 任务间无数据竞争, 无需额外加锁
 *   - sharedpool 复用线程池时沿用首次 init_pool 的参数步长(pool_set_newtask 不更新 param_size),
 *     因此所有算子必须共用同一个任务结构体(arr_mt_task)与同一个池回调(arr_mt_task_run)
 *   - 数据量不足以抵消并发调度开销、或暂时没有空闲线程池时,
 *     自动退化为本线程串行执行(结果与单线程版函数完全等价, 也避免与池内任务互相等待)
 */

#include "../threadpool/sharedpool.h"
#include "../threadpool/threadpool.h"  /* 仅取 tp_sleep_ms 等平台辅助, 池的申请/归还走 sharedpool */

#define ARR_MT_MAX_TASKS 32u            //任务数上限, 避免超多核平台上的调度开销
#define ARR_MT_MIN_ELEMS_PER_TASK 4096u //单个任务至少处理的元素数
#define ARR_MT_WAIT_SPIN_ROUNDS 256u    //等待本批完成时先自旋的轮数, 超过后小睡让出CPU

//任务单元: 处理 [start, end) 区间, arg 指向各算子私有的参数结构体
typedef void (*arr_mt_range_fn)(uint32_t start, uint32_t end, void *arg);

typedef struct arr_mt_task{
    arr_mt_range_fn fn;
    void *arg;
    uint32_t start;
    uint32_t end;
} arr_mt_task;

//线程池回调: 从参数区取出一个任务并执行(所有算子共用, 满足复用池时回调/步长一致的约束)
static void arr_mt_task_run(void *param){
    const arr_mt_task *task = (const arr_mt_task *)param;
    task->fn(task->start, task->end, task->arg);
}

//根据总任务量与单任务最小元素数决定任务数(线程池按任务动态分发到工作线程)
static uint32_t arr_mt_task_count(uint32_t total, uint32_t min_per_task){
    uint32_t by_work;

    if(total == 0) return 0;
    if(min_per_task == 0) min_per_task = 1;

    by_work = total / min_per_task;
    if(by_work == 0) by_work = 1;
    if(by_work > total) by_work = total;
    if(by_work > ARR_MT_MAX_TASKS) by_work = ARR_MT_MAX_TASKS;
    return by_work;
}

//通用并行for: 将 [0, total) 切分为多个连续区间任务提交给共享线程池, 并等待本批任务完成
static void arr_mt_parallel_for(uint32_t total, uint32_t min_per_task,
                                arr_mt_range_fn fn, void *arg){
    uint32_t ntasks, per, rem, start, i;
    arr_mt_task tasks[ARR_MT_MAX_TASKS];
    int32_t slot;

    if(total == 0) return;

    ntasks = arr_mt_task_count(total, min_per_task);
    if(ntasks <= 1){
        fn(0, total, arg);
        return;
    }

    per = total / ntasks;
    rem = total % ntasks;
    start = 0;

    for(i = 0; i < ntasks; i++){
        uint32_t end = start + per + ((i < rem) ? 1u : 0u);
        if(end > total) end = total;

        tasks[i].fn = fn;
        tasks[i].arg = arg;
        tasks[i].start = start;
        tasks[i].end = end;

        start = end;
    }

    //向共享线程池表申请一个空闲池并提交本批任务(槽位空闲则复用已有池, 否则按需创建)
    slot = pubpools_append_task(arr_mt_task_run, tasks, ntasks, sizeof(arr_mt_task));
    if(slot < 0){
        //暂时没有空闲线程池: 退化为本线程串行执行, 保证结果正确
        for(i = 0; i < ntasks; i++){
            tasks[i].fn(tasks[i].start, tasks[i].end, tasks[i].arg);
        }
        return;
    }

    //等待本批任务全部完成(pubpools_checkprocessing 返回0表示已完成), 期间 tasks 与 arg 始终有效
    for(i = 0; pubpools_checkprocessing((uint16_t)slot); i++){
        if(i >= ARR_MT_WAIT_SPIN_ROUNDS) tp_sleep_ms(1); //长任务让出CPU, 避免与工作线程争抢
    }

    //归还线程池: 该池进入冷却期, 冷却期内可被其它 arr_*_mt 调用复用
    pubpools_release((uint16_t)slot);
}

typedef struct arr_mt_binop_arg{
    const qfix *in1;
    const qfix *in2;
    qfix *out;
} arr_mt_binop_arg;

typedef struct arr_mt_scale_arg{
    const qfix *in1;
    qfix *out;
    qfix scale;
} arr_mt_scale_arg;

typedef struct arr_mt_mmul_arg{
    const qfix *in1;
    const qfix *in2;
    qfix *out;
    uint16_t c1_r2;
    uint16_t cols2;
} arr_mt_mmul_arg;

typedef struct arr_mt_transpose_arg{
    const qfix *in1;
    qfix *out;
    uint16_t rows1;
    uint16_t cols1;
} arr_mt_transpose_arg;

static void arr_add_range(uint32_t start, uint32_t end, void *arg){
    const arr_mt_binop_arg *a = (const arr_mt_binop_arg *)arg;
    for(uint32_t i = start; i < end; i++){
        a->out[i] = a->in1[i] + a->in2[i];
    }
}

static void arr_sub_range(uint32_t start, uint32_t end, void *arg){
    const arr_mt_binop_arg *a = (const arr_mt_binop_arg *)arg;
    for(uint32_t i = start; i < end; i++){
        a->out[i] = a->in1[i] - a->in2[i];
    }
}

static void arr_scale_range(uint32_t start, uint32_t end, void *arg){
    const arr_mt_scale_arg *a = (const arr_mt_scale_arg *)arg;
    for(uint32_t i = start; i < end; i++){
        a->out[i] = qfix_mul(a->in1[i], a->scale);
    }
}

static void arr_emul_range(uint32_t start, uint32_t end, void *arg){
    const arr_mt_binop_arg *a = (const arr_mt_binop_arg *)arg;
    for(uint32_t i = start; i < end; i++){
        a->out[i] = qfix_mul(a->in1[i], a->in2[i]);
    }
}

//按行切分矩阵乘法, 每个线程负责 [start, end) 行
static void arr_mmul_range(uint32_t start, uint32_t end, void *arg){
    const arr_mt_mmul_arg *a = (const arr_mt_mmul_arg *)arg;

    for(uint32_t i = start; i < end; i++){
        const size_t a_row_offset = (size_t)i * a->c1_r2;
        qfix *out_row = a->out + (size_t)i * a->cols2;

        for(uint16_t j = 0; j < a->cols2; j++){
            _tmp_larger sum = 0;
            //内循环使用指针运算
            const qfix *a_ptr = a->in1 + a_row_offset;
            const qfix *b_ptr = a->in2 + j; //B的第j列起始位置
            for(uint16_t kk = 0; kk < a->c1_r2; kk++){
                sum += ((_tmp_larger)(*a_ptr) * (*b_ptr));
                a_ptr++;
                b_ptr += a->cols2; //跳到B的下一行(列优先访问)
            }
            out_row[j] = (qfix)(sum >> QSHIFT);
        }
    }
}

//按行切分转置, 每个线程负责 [start, end) 行
static void arr_mtpose_range(uint32_t start, uint32_t end, void *arg){
    const arr_mt_transpose_arg *a = (const arr_mt_transpose_arg *)arg;

    for(uint32_t i = start; i < end; i++){
        const qfix *src = a->in1 + (size_t)i * a->cols1;
        qfix *dst = a->out + i;
        for(uint16_t j = 0; j < a->cols1; j++){
            *dst = *src;
            src++;
            dst += a->rows1;
        }
    }
}

DLLEXPORT void arr_add_mt(const qfix *in1, const qfix* in2, uint32_t len, qfix* out){
    arr_mt_binop_arg arg;
    arg.in1 = in1;
    arg.in2 = in2;
    arg.out = out;
    arr_mt_parallel_for(len, ARR_MT_MIN_ELEMS_PER_TASK, arr_add_range, &arg);
}

DLLEXPORT void arr_sub_mt(const qfix *in1, const qfix* in2, const uint32_t len, qfix* out){
    arr_mt_binop_arg arg;
    arg.in1 = in1;
    arg.in2 = in2;
    arg.out = out;
    arr_mt_parallel_for(len, ARR_MT_MIN_ELEMS_PER_TASK, arr_sub_range, &arg);
}

DLLEXPORT void arr_scale_mt(const qfix *in1, const qfix scale, const uint32_t len, qfix* out){
    arr_mt_scale_arg arg;
    arg.in1 = in1;
    arg.out = out;
    arg.scale = scale;
    arr_mt_parallel_for(len, ARR_MT_MIN_ELEMS_PER_TASK, arr_scale_range, &arg);
}

DLLEXPORT void arr_emul_mt(const qfix *in1, const qfix* in2, const uint32_t len, qfix* out){
    arr_mt_binop_arg arg;
    arg.in1 = in1;
    arg.in2 = in2;
    arg.out = out;
    arr_mt_parallel_for(len, ARR_MT_MIN_ELEMS_PER_TASK, arr_emul_range, &arg);
}

DLLEXPORT void arr_mmul_mt(const qfix *in1, const qfix* in2, const uint16_t rows1, const uint16_t c1_r2, const uint16_t cols2, qfix* out){
    arr_mt_mmul_arg arg;
    uint64_t work_per_row;
    uint32_t min_rows;

    if(rows1 == 0 || c1_r2 == 0 || cols2 == 0) return;

    arg.in1 = in1;
    arg.in2 = in2;
    arg.out = out;
    arg.c1_r2 = c1_r2;
    arg.cols2 = cols2;

    //每行计算量为 c1_r2 * cols2 次乘加, 折算成每个线程至少负责的行数
    work_per_row = (uint64_t)c1_r2 * (uint64_t)cols2;
    min_rows = (uint32_t)((ARR_MT_MIN_ELEMS_PER_TASK + work_per_row - 1) / work_per_row);
    if(min_rows == 0) min_rows = 1;

    arr_mt_parallel_for(rows1, min_rows, arr_mmul_range, &arg);
}

DLLEXPORT void arr_mtpose_mt(const qfix *in1, const uint16_t rows1, const uint16_t cols1, qfix* out){
    arr_mt_transpose_arg arg;
    uint32_t min_rows;

    if(rows1 == 0 || cols1 == 0) return;

    arg.in1 = in1;
    arg.out = out;
    arg.rows1 = rows1;
    arg.cols1 = cols1;

    //每行搬运 cols1 个元素
    min_rows = ARR_MT_MIN_ELEMS_PER_TASK / cols1;
    if(min_rows == 0) min_rows = 1;

    arr_mt_parallel_for(rows1, min_rows, arr_mtpose_range, &arg);
}

#endif
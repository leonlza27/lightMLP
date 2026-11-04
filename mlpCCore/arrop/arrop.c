#include "arrop.h"
#include "../qfix_ops/qfix_op.h"
#include "../CustomConf.h"

//多线程版本前向声明
#ifdef MULTI_PROCESS
DLLEXPORT void arr_add_mt(const qfix *in1, const  qfix* in2, uint32_t len, qfix* out);
DLLEXPORT void arr_sub_mt(const qfix *in1, const qfix* in2, const uint32_t len, qfix* out);
DLLEXPORT void arr_scale_mt(const qfix *in1, const qfix scale, const uint32_t len, qfix* out);
DLLEXPORT void arr_emul_mt(const qfix *in1, const qfix* in2, const uint32_t len, qfix* out);
DLLEXPORT void arr_mmul_mt(const qfix *in1, const qfix* in2, const uint16_t rows1, const uint16_t c1_r2, const uint16_t cols2, qfix* out);
DLLEXPORT void arr_mtpose_mt(const qfix *in1, const uint16_t rows1, const uint16_t cols1, qfix* out);
#endif

#ifndef MULTI_PROCESS

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

#else

DLLEXPORT void arr_add(const qfix *in1, const qfix* in2, const uint32_t len, qfix* out){
    arr_add_mt(in1, in2, len, out);
}

DLLEXPORT void arr_sub(const qfix *in1, const qfix* in2, const uint32_t len, qfix* out){
    arr_sub_mt(in1, in2, len, out);
}

DLLEXPORT void arr_scale(const qfix *in1, const qfix scale, const uint32_t len, qfix* out){
    arr_scale_mt(in1, scale, len, out);
}

DLLEXPORT void arr_emul(const qfix *in1, const qfix* in2, const uint32_t len, qfix* out){
    arr_emul_mt(in1, in2, len, out);
}

DLLEXPORT void arr_mmul(const qfix *in1, const qfix* in2, const uint16_t rows1, const uint16_t c1_r2, const uint16_t cols2, qfix* out){
    arr_mmul_mt(in1, in2, rows1, c1_r2, cols2, out);
}

DLLEXPORT void arr_mtpose(const qfix *in1, const uint16_t rows1, const uint16_t cols1, qfix* out){
    arr_mtpose_mt(in1, rows1, cols1, out);
}

//<Promot for AI>注:把多线程函数写在这,支持win(winapi)/apple(pthread)/linux(pthread)

/*
 * 多线程实现
 *
 * 平台:
 *   Windows        -> WinAPI CreateThread
 *   Apple / Linux  -> pthread
 *
 * 说明:
 *   - 任务按连续区间切分, 每个线程只写自己负责的输出区间, 线程间无数据竞争, 无需加锁
 *   - 每次调用临时创建线程并在返回前全部join, 不保留全局线程状态, 可重入
 *   - 数据量不足以抵消线程创建开销时自动退化为单线程(与原函数完全等价)
 */

#if defined(_WIN32)
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

#define ARR_MT_MAX_THREADS 32u            //线程数上限, 避免超多核平台上的调度开销
#define ARR_MT_MIN_ELEMS_PER_THREAD 4096u //单个线程至少处理的元素数

#if defined(_WIN32)
typedef HANDLE arr_mt_thread;
#else
typedef pthread_t arr_mt_thread;
#endif

//区间任务: 处理 [start, end)
typedef void (*arr_mt_worker_fn)(uint32_t start, uint32_t end, void *arg);

typedef struct arr_mt_job{
    arr_mt_worker_fn fn;
    void *arg;
    uint32_t start;
    uint32_t end;
} arr_mt_job;

//获取可用逻辑核心数(获取失败时按1处理)
static uint32_t arr_mt_cpu_count(void){
    static uint32_t cached = 0; //并发写入相同值, 无需加锁
    if(cached != 0) return cached;
#if defined(_WIN32)
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    cached = (uint32_t)info.dwNumberOfProcessors;
#else
    long cores = sysconf(_SC_NPROCESSORS_ONLN);
    cached = (cores > 0) ? (uint32_t)cores : 1u;
#endif
    if(cached == 0) cached = 1u;
    return cached;
}

#if defined(_WIN32)
static DWORD WINAPI arr_mt_entry(LPVOID param){
#else
static void *arr_mt_entry(void *param){
#endif
    arr_mt_job *job = (arr_mt_job *)param;
    job->fn(job->start, job->end, job->arg);
#if defined(_WIN32)
    return 0;
#else
    return NULL;
#endif
}

static int arr_mt_thread_create(arr_mt_thread *thread, arr_mt_job *job){
#if defined(_WIN32)
    *thread = CreateThread(NULL, 0, arr_mt_entry, job, 0, NULL);
    return (*thread != NULL);
#else
    return (pthread_create(thread, NULL, arr_mt_entry, job) == 0);
#endif
}

static void arr_mt_thread_join(arr_mt_thread thread){
#if defined(_WIN32)
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
#else
    pthread_join(thread, NULL);
#endif
}

//根据总任务量与单线程最小任务量决定线程数
static uint32_t arr_mt_worker_count(uint32_t total, uint32_t min_per_thread){
    uint32_t cpus, by_work;

    if(total == 0) return 0;
    if(min_per_thread == 0) min_per_thread = 1;

    by_work = total / min_per_thread;
    if(by_work == 0) by_work = 1;
    if(by_work > total) by_work = total;

    cpus = arr_mt_cpu_count();
    if(cpus > ARR_MT_MAX_THREADS) cpus = ARR_MT_MAX_THREADS;
    if(cpus > by_work) cpus = by_work;
    if(cpus == 0) cpus = 1;
    return cpus;
}

//通用并行for: 将 [0, total) 切分给多个线程, 主线程也参与计算
static void arr_mt_parallel_for(uint32_t total, uint32_t min_per_thread,
                                arr_mt_worker_fn fn, void *arg){
    uint32_t nthreads, per, rem, start, first_end, created, i;
    arr_mt_thread threads[ARR_MT_MAX_THREADS];
    arr_mt_job jobs[ARR_MT_MAX_THREADS];

    if(total == 0) return;

    nthreads = arr_mt_worker_count(total, min_per_thread);
    if(nthreads <= 1){
        fn(0, total, arg);
        return;
    }

    per = total / nthreads;
    rem = total % nthreads;

    //主线程处理第一段, 其余段交给子线程
    first_end = per + ((rem > 0) ? 1u : 0u);
    start = first_end;
    created = 0;

    for(i = 1; i < nthreads; i++){
        uint32_t count = per + ((i < rem) ? 1u : 0u);
        uint32_t end = start + count;

        if(start >= total) break;
        if(end > total) end = total;

        jobs[created].fn = fn;
        jobs[created].arg = arg;
        jobs[created].start = start;
        jobs[created].end = end;

        if(arr_mt_thread_create(&threads[created], &jobs[created])){
            created++;
        }else{
            //线程创建失败时由当前线程直接完成该段, 保证结果正确
            fn(start, end, arg);
        }
        start = end;
    }

    fn(0, first_end, arg);

    for(i = 0; i < created; i++){
        arr_mt_thread_join(threads[i]);
    }
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
    arr_mt_parallel_for(len, ARR_MT_MIN_ELEMS_PER_THREAD, arr_add_range, &arg);
}

DLLEXPORT void arr_sub_mt(const qfix *in1, const qfix* in2, const uint32_t len, qfix* out){
    arr_mt_binop_arg arg;
    arg.in1 = in1;
    arg.in2 = in2;
    arg.out = out;
    arr_mt_parallel_for(len, ARR_MT_MIN_ELEMS_PER_THREAD, arr_sub_range, &arg);
}

DLLEXPORT void arr_scale_mt(const qfix *in1, const qfix scale, const uint32_t len, qfix* out){
    arr_mt_scale_arg arg;
    arg.in1 = in1;
    arg.out = out;
    arg.scale = scale;
    arr_mt_parallel_for(len, ARR_MT_MIN_ELEMS_PER_THREAD, arr_scale_range, &arg);
}

DLLEXPORT void arr_emul_mt(const qfix *in1, const qfix* in2, const uint32_t len, qfix* out){
    arr_mt_binop_arg arg;
    arg.in1 = in1;
    arg.in2 = in2;
    arg.out = out;
    arr_mt_parallel_for(len, ARR_MT_MIN_ELEMS_PER_THREAD, arr_emul_range, &arg);
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
    min_rows = (uint32_t)((ARR_MT_MIN_ELEMS_PER_THREAD + work_per_row - 1) / work_per_row);
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
    min_rows = ARR_MT_MIN_ELEMS_PER_THREAD / cols1;
    if(min_rows == 0) min_rows = 1;

    arr_mt_parallel_for(rows1, min_rows, arr_mtpose_range, &arg);
}

#endif
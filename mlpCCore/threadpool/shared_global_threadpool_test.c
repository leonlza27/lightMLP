/*
 * 全局资源管理 + threadpool.c 线程池对象 —— 基础逻辑验证
 *
 * 目标
 * ----
 * 复用同目录 threadpool.c / threadpool.h 提供的线程池(init_pool / pool_set_newtask /
 * pool_checkTaskProcessing / pool_clear)作为"被管理的资源", 并移植
 * thread_own_global_resource_verify.c 里的全局资源分配逻辑, 搭出一套
 * "全局线程池对象管理器":
 *      - 固定 GLOB_POOL_SLOTS 个受管槽位, 每槽位最多持有一个 threadpool 对象句柄;
 *      - 用 ishold 位表示占用, free_cntdwn 位域作为"归还后的冷却倒计时";
 *      - 申请(acquire)时环形轮询查找空闲槽位, 命中后惰性创建或复用该线程池对象;
 *      - 归还(release)时启动冷却倒计时, 由后台回收线程到期后 pool_clear + free
 *        掉该线程池对象(惰性销毁 / 延迟回收);
 *      - 多个"客户线程"共享这套管理器, 各自申请线程池跑一批任务再归还.
 *
 * 与 threadpool.c 的契合点
 * ------------------------
 *   - 首次占用槽位: init_pool(...) 创建并启动该线程池, 处理首批任务;
 *   - 冷却期内再次占用同一槽位: pool_set_newtask(...) 复用已有线程池(省去重建开销),
 *     即 threadpool.c 注释里"复用线程池"的用法;
 *   - 冷却期结束仍空闲: pool_clear(...) 停止并回收线程, 然后 free 掉该对象.
 *   - 注意: threadpool.c 的回调只在 init_pool 时绑定, 因此同一个受管线程池被复用时
 *     必须沿用相同的回调; 本文件所有批次统一使用 batch_job_callback.
 *   - 同理, 复用要求批次参数步长一致(pool_set_newtask 沿用 init_pool 时的 param_size),
 *     故所有批次统一使用 sizeof(job_param).
 *
 * 与 thread_own_global_resource_verify.c 的差异(必要加固)
 * ------------------------------------------------------
 *   - 资源表访问加互斥量(原程序在 128 线程下存在数据竞争);
 *   - 回收只处理"空闲"槽位, 且把销毁动作放到锁外执行(pool_clear 内部会 join,
 *     持锁执行会阻塞其它申请);
 *   - 申请槽位时清空冷却倒计时, 使其在占用期间不参与回收.
 *
 * 编译与运行(Windows / MinGW; *nix 需 -pthread, 且需一并编译 threadpool.c):
 *   gcc -O2 -Wall -o shared_global_threadpool_test shared_global_threadpool_test.c threadpool.c
 *   ./shared_global_threadpool_test
 *
 * 本文件仅作独立验证, 只依赖同目录 threadpool.c / threadpool.h, 不改动它们.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include "threadpool.h"   /* threadpool 类型与 init_pool/pool_set_newtask/... 声明 */

/* ========================= 平台辅助 ========================= */
/* 线程句柄 tp_thread / 互斥量 tp_mutex 已由 threadpool.h 依平台定义 */

/* 单调毫秒时钟(用于统计耗时) */
static uint64_t tp_now_ms(void){
#if defined(_WIN32)
    return (uint64_t)GetTickCount();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000u + (uint64_t)(ts.tv_nsec / 1000000L);
#endif
}

/* ========================= 配置 ========================= */

//genral configs

#define GLOB_POOL_SLOTS          4    /* 受管槽位数(每槽位最多持有一个 threadpool 对象) */
#define GLOB_POOL_COOLDOWN       8    /* 归还后经过多少个回收周期才真正销毁线程池对象 */
#define RECLAIM_INTERVAL_MS      5    /* 后台回收线程的周期(毫秒) */

//debug configs

#define CLIENT_THREADS           8    /* 共享该管理器的客户线程数 */
#define CLIENT_ATTEMPTS          6    /* 每个客户申请线程池的次数 */
#define CLIENT_BATCH_JOBS        4    /* 每次申请后提交给线程池的任务数 */
#define JOB_HOLD_MAX_MS          60   /* 单个任务的随机耗时上限(毫秒) */
#define MISS_BACKOFF_MAX_MS      10   /* 无空闲槽位时的退避上限(毫秒) */

#define VERBOSE_LOG              0    /* 1: 打印每次申请/完成; 0: 仅汇总 */

/* ========================= 全局线程池对象管理器 ========================= */
/* 资源分配逻辑取自 thread_own_global_resource_verify.c, 槽位负载改为 threadpool 对象 */

typedef struct managed_slot{
    threadpool *pool;          /* 受管线程池对象(惰性创建 / 延迟销毁), NULL 表示尚未创建 */
    uint8_t     ishold     : 1;/* 是否被占用 */
    uint8_t     free_cntdwn: 4;/* 归还后的冷却倒计时 */
} managed_slot;

static managed_slot g_slots[GLOB_POOL_SLOTS];
static uint16_t     g_slot_cursor;
static tp_mutex     g_slot_lock;

/* 统计量(均在 g_slot_lock 保护下更新) */
static size_t g_acquire_cnt;     /* 申请成功次数 */
static size_t g_miss_cnt;        /* 申请失败(无空闲槽位)次数 */
static size_t g_pool_create_cnt; /* threadpool 对象创建次数(init_pool) */
static size_t g_pool_reuse_cnt;  /* threadpool 对象复用次数(pool_set_newtask) */
static size_t g_pool_destroy_cnt;/* threadpool 对象销毁次数(pool_clear + free) */

static void managed_pool_init(void){
    uint16_t i;
    for(i = 0; i < GLOB_POOL_SLOTS; i++){
        g_slots[i].pool = NULL;
        g_slots[i].ishold = 0;
        g_slots[i].free_cntdwn = 0;
    }
    g_slot_cursor = 0;
    g_acquire_cnt = 0;
    g_miss_cnt = 0;
    g_pool_create_cnt = 0;
    g_pool_reuse_cnt = 0;
    g_pool_destroy_cnt = 0;
    tp_mutex_init(&g_slot_lock);
}

/*
 * 申请一个受管线程池:
 *   - 环形轮询查找空闲槽位(最多两圈), 命中后置占用位、清冷却倒计时;
 *   - 若该槽位尚无对象, 则锁外 malloc + init_pool 惰性创建;
 *     否则锁外 pool_set_newtask 复用已有对象;
 *   - 成功返回槽位下标, 无空闲返回 -1.
 * 说明: 回调在对象首次创建时经 init_pool 绑定, 后续复用时保持不变(须为同一回调).
 */
static int32_t managed_pool_acquire(void (*callback)(void*), void *params, size_t count, size_t param_size){
    uint8_t cycles = 2;
    int32_t got = -1;
    threadpool *pool = NULL;
    int is_new = 0;

    tp_mutex_lock(&g_slot_lock);
    while(cycles){
        managed_slot *slot = &g_slots[g_slot_cursor];
        if(!slot->ishold){
            slot->ishold = 1;
            slot->free_cntdwn = 0;      /* 占用期间不参与回收 */
            pool = slot->pool;
            is_new = (pool == NULL);
            got = (int32_t)g_slot_cursor;
            g_acquire_cnt++;
            break;
        }
        g_slot_cursor++;
        if(g_slot_cursor >= GLOB_POOL_SLOTS){
            g_slot_cursor = 0;
            cycles--;
        }
    }
    if(got < 0) g_miss_cnt++;
    tp_mutex_unlock(&g_slot_lock);

    if(got < 0) return -1;

    /* 锁外创建/复用(init_pool 会创建多个工作线程, 不宜持锁执行) */
    if(is_new){
        pool = (threadpool*)malloc(sizeof(threadpool));
        if(pool == NULL){
            tp_mutex_lock(&g_slot_lock);
            g_slots[got].ishold = 0;    /* 回滚占用 */
            tp_mutex_unlock(&g_slot_lock);
            return -1;
        }
        init_pool(pool, callback, params, count, param_size);
        tp_mutex_lock(&g_slot_lock);
        g_slots[got].pool = pool;       /* 记录对象指针 */
        g_pool_create_cnt++;
        tp_mutex_unlock(&g_slot_lock);
    }else{
        pool_set_newtask(pool, callback, params, count);
        tp_mutex_lock(&g_slot_lock);
        g_pool_reuse_cnt++;
        tp_mutex_unlock(&g_slot_lock);
    }
    return got;
}

/* 轮询指定受管线程池当前批次是否处理完毕; 返回 0 表示已完成 */
static uint8_t managed_pool_processing(uint16_t idx){
    threadpool *pool;
    uint8_t processing = 0;
    tp_mutex_lock(&g_slot_lock);
    pool = (idx < GLOB_POOL_SLOTS) ? g_slots[idx].pool : NULL;
    tp_mutex_unlock(&g_slot_lock);
    if(pool != NULL) processing = pool_checkTaskProcessing(pool);
    return processing;
}

/* 归还受管线程池: 清占用位并启动冷却倒计时(对象暂不销毁) */
static void managed_pool_release(uint16_t idx){
    tp_mutex_lock(&g_slot_lock);
    if(idx < GLOB_POOL_SLOTS){
        g_slots[idx].ishold = 0;
        g_slots[idx].free_cntdwn = GLOB_POOL_COOLDOWN;
    }
    tp_mutex_unlock(&g_slot_lock);
}

/* 单步回收: 递减空闲槽位的冷却倒计时, 到期后在锁外 pool_clear + free 掉对象 */
static void managed_pool_reclaim_step(void){
    threadpool *doomed[GLOB_POOL_SLOTS];
    size_t doomed_cnt = 0;
    uint16_t i;

    tp_mutex_lock(&g_slot_lock);
    for(i = 0; i < GLOB_POOL_SLOTS; i++){
        managed_slot *slot = &g_slots[i];
        if(!slot->ishold && slot->free_cntdwn){   /* 仅空闲槽位参与冷却 */
            slot->free_cntdwn--;
        }
        if(!slot->ishold && slot->free_cntdwn == 0 && slot->pool != NULL){
            doomed[doomed_cnt++] = slot->pool;
            slot->pool = NULL;   /* 先摘除: 之后若被申请将重新惰性创建 */
        }
    }
    tp_mutex_unlock(&g_slot_lock);

    for(i = 0; i < doomed_cnt; i++){
        pool_clear(doomed[i]);   /* 停线程并回收(pool_clear 内部 join), 放锁外 */
        free(doomed[i]);
        tp_mutex_lock(&g_slot_lock);
        g_pool_destroy_cnt++;
        tp_mutex_unlock(&g_slot_lock);
    }
}

/* 停止后台回收后, 把仍在冷却中的空闲线程池一次性销毁(干净退出) */
static void managed_pool_flush_idle(void){
    int round;
    for(round = 0; round <= GLOB_POOL_COOLDOWN; round++){
        managed_pool_reclaim_step();
    }
}

/* ========================= 后台回收线程 ========================= */

static tp_thread g_reclaimer_thread;
static volatile uint8_t g_reclaimer_onrun;

static TP_WORKER_ENTRY reclaimer_routine(void *param){
    (void)param;
    while(g_reclaimer_onrun){
        managed_pool_reclaim_step();
        tp_sleep_ms(RECLAIM_INTERVAL_MS);
    }
    return TP_WORKER_RETVAL;
}

/* ========================= 客户线程(共享管理器) ========================= */

typedef struct job_param{
    uint16_t client_no;
    uint16_t job_no;
    uint32_t hold_ms;
} job_param;

/*
 * 线程池任务回调: 所有受管线程池统一使用该回调
 * (threadpool.c 的回调只在 init_pool 时绑定, 复用同一对象时不能更换)
 */
static void batch_job_callback(void *arg){
    job_param *job = (job_param*)arg;
    tp_sleep_ms(job->hold_ms);   /* 模拟一次可并行的处理 */
}

typedef struct client_context{
    uint16_t client_no;
} client_context;

/* 线程内独立 PRNG(xorshift32; rand() 非线程安全, 故不使用) */
static uint32_t client_rand(uint32_t *state){
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/* 客户线程: 反复"申请受管线程池 -> 跑一批任务 -> 等待 -> 归还" */
static TP_WORKER_ENTRY client_routine(void *arg){
    client_context *ctx = (client_context*)arg;
    uint32_t rng = (uint32_t)time(NULL) ^ ((uint32_t)ctx->client_no * 2654435761u);
    job_param jobs[CLIENT_BATCH_JOBS];
    int attempt;
    if(rng == 0) rng = 0x9e3779b9u;

    for(attempt = 0; attempt < CLIENT_ATTEMPTS; attempt++){
        int32_t slot;
        int j;

        /* 先填好本批次参数: 线程池会异步读取, 必须在申请前准备完毕 */
        for(j = 0; j < CLIENT_BATCH_JOBS; j++){
            jobs[j].client_no = ctx->client_no;
            jobs[j].job_no = (uint16_t)j;
            jobs[j].hold_ms = client_rand(&rng) % JOB_HOLD_MAX_MS;
        }

        slot = managed_pool_acquire(batch_job_callback, jobs, CLIENT_BATCH_JOBS, sizeof(job_param));
        if(slot < 0){
            unsigned int backoff = 1u + client_rand(&rng) % MISS_BACKOFF_MAX_MS;
#if VERBOSE_LOG
            printf("客户 %2u <第 %d/%d 批>: 无空闲受管线程池, 退避 %u ms\n",
                   (unsigned)ctx->client_no, attempt + 1, CLIENT_ATTEMPTS, backoff);
#endif
            tp_sleep_ms(backoff);
            continue;
        }
#if VERBOSE_LOG
        printf("客户 %2u <第 %d/%d 批>: 获得受管线程池槽位 %d\n",
               (unsigned)ctx->client_no, attempt + 1, CLIENT_ATTEMPTS, (int)slot);
#endif
        while(managed_pool_processing((uint16_t)slot)){
            tp_sleep_ms(1);   /* 等待本批任务执行完毕 */
        }
        managed_pool_release((uint16_t)slot);
    }
    return TP_WORKER_RETVAL;
}

/* ========================= 测试入口 ========================= */

int main(void){
    size_t i;
    size_t started = 0;
    uint64_t t0, t1;
    client_context *clients;
    tp_thread *client_threads;

    printf("==== 全局资源管理 + threadpool.c 线程池对象 验证 ====\n");
    //printf("逻辑核心数: %zu (每个受管线程池的工作线程数)\n", tp_cpu_count());
    printf("受管槽位: %d 个, 冷却周期: %d (每 %d ms 递减一次)\n",
           GLOB_POOL_SLOTS, GLOB_POOL_COOLDOWN, RECLAIM_INTERVAL_MS);
    printf("客户线程: %d 个, 每个申请 %d 批, 每批 %d 个任务(单任务<=%d ms)\n\n",
           CLIENT_THREADS, CLIENT_ATTEMPTS, CLIENT_BATCH_JOBS, JOB_HOLD_MAX_MS);

    /* 1. 初始化管理器并启动后台回收线程 */
    managed_pool_init();
    g_reclaimer_onrun = 1;
    if(!tp_thread_create(&g_reclaimer_thread, reclaimer_routine, NULL)){
        printf("后台回收线程创建失败\n");
        return 1;
    }

    /* 2. 启动共享该管理器的客户线程 */
    clients = (client_context*)malloc(sizeof(client_context) * CLIENT_THREADS);
    client_threads = (tp_thread*)malloc(sizeof(tp_thread) * CLIENT_THREADS);
    if(clients == NULL || client_threads == NULL){
        printf("客户线程资源分配失败\n");
        g_reclaimer_onrun = 0;
        tp_thread_join(g_reclaimer_thread);
        free(clients);
        free(client_threads);
        return 1;
    }
    t0 = tp_now_ms();
    for(i = 0; i < CLIENT_THREADS; i++){
        clients[i].client_no = (uint16_t)i;
        if(tp_thread_create(&client_threads[i], client_routine, &clients[i])) started++;
    }
    for(i = 0; i < started; i++){
        tp_thread_join(client_threads[i]);
    }
    t1 = tp_now_ms();

    /* 3. 停止回收线程并销毁残余线程池对象 */
    g_reclaimer_onrun = 0;
    tp_thread_join(g_reclaimer_thread);
    managed_pool_flush_idle();

    /* 4. 汇总 */
    printf("---- 汇总 ----\n");
    printf("客户线程: %zu/%d\n", started, CLIENT_THREADS);
    printf("受管线程池申请成功: %zu 次\n", g_acquire_cnt);
    printf("受管线程池申请失败: %zu 次\n", g_miss_cnt);
    printf("线程池对象创建: %zu 次, 复用: %zu 次, 销毁: %zu 次\n",
           g_pool_create_cnt, g_pool_reuse_cnt, g_pool_destroy_cnt);
    printf("剩余未销毁线程池对象: %zu 个\n",
           (g_pool_create_cnt >= g_pool_destroy_cnt) ? (g_pool_create_cnt - g_pool_destroy_cnt) : 0);
    printf("总耗时: %llu ms\n", (unsigned long long)(t1 - t0));

    tp_mutex_destroy(&g_slot_lock);
    free(clients);
    free(client_threads);
    return 0;
}

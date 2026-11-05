#include "threadpool.h"
#include <stdlib.h>
#include <time.h>

/*
 * 线程池实现
 *
 * 平台:
 *   Windows                        -> WinAPI (CreateThread / Mutex)
 *   Apple / Linux / Android native -> pthread
 *
 * 说明:
 *   - init_pool 创建与逻辑核心数相同的常驻工作线程
 *   - 工作线程通过互斥锁按索引领取任务, 保证每个任务只被执行一次
 *   - pool_set_newtask 提交一批新任务, pool_checkTaskProcessing 返回0表示该批任务全部执行完毕
 *   - pool_clear 通知线程退出并等待所有线程结束
 *   - 复用线程池时, 需等上一批任务全部完成后再调用 pool_set_newtask
 */

#if defined(_WIN32)
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

//线程句柄/互斥量类型与 threadpool 结构体定义已移至 threadpool.h,
//以便调用方能在栈或静态区直接分配线程池对象

static void tp_mutex_init(tp_mutex *mutex){
#if defined(_WIN32)
    *mutex = CreateMutexA(NULL, FALSE, NULL);
#else
    pthread_mutex_init(mutex, NULL);
#endif
}

static void tp_mutex_lock(tp_mutex *mutex){
#if defined(_WIN32)
    WaitForSingleObject(*mutex, INFINITE);
#else
    pthread_mutex_lock(mutex);
#endif
}

static void tp_mutex_unlock(tp_mutex *mutex){
#if defined(_WIN32)
    ReleaseMutex(*mutex);
#else
    pthread_mutex_unlock(mutex);
#endif
}

static void tp_mutex_destroy(tp_mutex *mutex){
#if defined(_WIN32)
    CloseHandle(*mutex);
#else
    pthread_mutex_destroy(mutex);
#endif
}

//休眠指定毫秒数(线程池空闲轮询用)
static void tp_sleep_ms(unsigned int ms){
#if defined(_WIN32)
    Sleep(ms);
#else
    struct timespec req;
    req.tv_sec = (time_t)(ms / 1000u);
    req.tv_nsec = (long)(ms % 1000u) * 1000000L;
    nanosleep(&req, NULL);
#endif
}

//获取可用逻辑核心数(获取失败时按1处理)
static size_t tp_cpu_count(void){
    static size_t cached = 0; //并发写入相同值, 无需加锁
    if(cached != 0) return cached;
#if defined(_WIN32)
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    cached = (size_t)info.dwNumberOfProcessors;
#else
    long cores = sysconf(_SC_NPROCESSORS_ONLN);
    cached = (cores > 0) ? (size_t)cores : 1u;
#endif
    if(cached == 0) cached = 1u;
    return cached;
}

#if defined(_WIN32)
static DWORD WINAPI subthread_capactor(void *init_params)
#else
static void *subthread_capactor(void *init_params)
#endif
{
    threadpool *pool = (threadpool*)init_params;
    void (*callback)(void*) = pool->callback;
    while(pool->onrun){
        tp_mutex_lock(&pool->param_lock);
        if(pool->curidx < pool->param_capacity){
            size_t idxcur = pool->curidx++;
            //在锁内捕获参数区指针与步长, 避免与 pool_set_newtask 复用批次时竞争
            char *params = pool->params;
            size_t param_size = pool->param_size;
            tp_mutex_unlock(&pool->param_lock);
            callback(&params[idxcur * param_size]);
            tp_mutex_lock(&pool->param_lock);
            pool->finished_cnt++;
            tp_mutex_unlock(&pool->param_lock);
        }else{
            tp_mutex_unlock(&pool->param_lock);
            tp_sleep_ms(10);
        }
    }
#if defined(_WIN32)
    return 0;
#else
    return NULL;
#endif
}

//创建单个工作线程(成功返回非0)
static int tp_thread_create(tp_thread *thread, threadpool *pool){
#if defined(_WIN32)
    *thread = CreateThread(NULL, 0, subthread_capactor, pool, 0, NULL);
    return (*thread != NULL);
#else
    return (pthread_create(thread, NULL, subthread_capactor, pool) == 0);
#endif
}

//等待并回收单个工作线程
static void tp_thread_join(tp_thread thread){
#if defined(_WIN32)
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
#else
    pthread_join(thread, NULL);
#endif
}

//无可用工作线程时退化为调用线程同步执行整批任务
static void tp_run_sync(threadpool *pool, void *constructed_params, size_t tasks_count, size_t param_size){
    size_t i;
    char *params = (char*)constructed_params;
    for(i = 0; i < tasks_count; i++){
        pool->callback(&params[i * param_size]);
    }
    tp_mutex_lock(&pool->param_lock);
    pool->curidx = tasks_count;
    pool->finished_cnt = tasks_count;
    tp_mutex_unlock(&pool->param_lock);
}

void init_pool(threadpool *pool, void (*callback)(void*), void* constructed_params, size_t tasks_count, size_t param_size){
    size_t i, created = 0;
    pool->callback = callback;
    pool->onrun = 1;
    pool->params = (char*)constructed_params;
    pool->param_capacity = tasks_count;
    pool->param_size = param_size;
    pool->curidx = 0;
    pool->finished_cnt = 0;
    pool->available_subtask_cnt = tp_cpu_count();

    //初始持锁形成"就绪门": 工作线程创建后会先阻塞, 直到参数与状态完全就绪
    tp_mutex_init(&pool->param_lock);
    tp_mutex_lock(&pool->param_lock);

    pool->subthreads = (tp_thread*)malloc(sizeof(tp_thread) * pool->available_subtask_cnt);
    if(pool->subthreads == NULL){
        pool->available_subtask_cnt = 0;
        tp_mutex_unlock(&pool->param_lock);
        tp_run_sync(pool, constructed_params, tasks_count, param_size);
        return;
    }

    for(i = 0; i < pool->available_subtask_cnt; i++){
        if(tp_thread_create(&pool->subthreads[i], pool)) created++;
        else break;
    }
    pool->available_subtask_cnt = created; //仅记录实际创建成功的线程数, 避免回收无效线程句柄
    tp_mutex_unlock(&pool->param_lock);

    if(created == 0) tp_run_sync(pool, constructed_params, tasks_count, param_size);
}

void pool_set_newtask(threadpool *pool, void* constructed_params, size_t tasks_count){
    size_t workers, param_size;
    tp_mutex_lock(&pool->param_lock);
    pool->curidx = 0;
    pool->finished_cnt = 0;
    pool->params = (char*)constructed_params;
    pool->param_capacity = tasks_count;
    workers = pool->available_subtask_cnt;
    param_size = pool->param_size;
    tp_mutex_unlock(&pool->param_lock);

    if(workers == 0) tp_run_sync(pool, constructed_params, tasks_count, param_size);
}

uint8_t pool_checkTaskProcessing(threadpool *pool){
    uint8_t processing;
    //在锁内读取完成计数: 与工作线程写入形成happens-before,
    //保证回调对params的写入在ARM等弱内存模型平台上对调用线程可见
    tp_mutex_lock(&pool->param_lock);
    processing = (pool->finished_cnt < pool->param_capacity);
    tp_mutex_unlock(&pool->param_lock);
    return processing;
}

void pool_clear(threadpool *pool){
    size_t i;
    pool->onrun = 0;
    for(i = 0; i < pool->available_subtask_cnt; i++){
        tp_thread_join(pool->subthreads[i]);
    }
    free(pool->subthreads);
    pool->subthreads = NULL;
    tp_mutex_destroy(&pool->param_lock);
}

//用于安卓侧的多核处理

#ifndef mobile_threadpool
#define mobile_threadpool

#include <stddef.h>
#include <stdint.h>

/*
 * 线程句柄、互斥量类型与 threadpool 结构体在头文件中给出完整定义:
 * init_pool / pool_clear 等接口由调用方提供 threadpool 对象(栈/静态/堆均可),
 * 因此调用方需要看到其完整布局才能完成分配。
 */
#if defined(_WIN32)
#include <windows.h>
typedef HANDLE tp_thread;
typedef HANDLE tp_mutex;
#else
#include <pthread.h>
typedef pthread_t tp_thread;
typedef pthread_mutex_t tp_mutex;
#endif

struct threadpool{
    void (*callback)(void*);
    tp_thread *subthreads;
    tp_mutex param_lock;
    size_t available_subtask_cnt;
    volatile size_t curidx;        //已领取的任务数(用于任务分发)
    volatile size_t finished_cnt;  //已执行完成的任务数(用于完成判定)
    volatile size_t param_capacity;
    size_t param_size;
    char *params;

    volatile uint8_t onrun : 1;
};

typedef struct threadpool threadpool;

#if defined(_WIN32)
#define TP_WORKER_ENTRY   DWORD WINAPI
#define TP_WORKER_RETVAL  0
typedef LPTHREAD_START_ROUTINE tp_thread_fn;
#else
#define TP_WORKER_ENTRY   void*
#define TP_WORKER_RETVAL  NULL
typedef void *(*tp_thread_fn)(void*);
#endif

void tp_mutex_init(tp_mutex *mutex);
void tp_mutex_lock(tp_mutex *mutex);
void tp_mutex_unlock(tp_mutex *mutex);
void tp_mutex_destroy(tp_mutex *mutex);
void tp_sleep_ms(unsigned int ms);
int tp_thread_create(tp_thread *thread, tp_thread_fn fn, void *arg);
void tp_thread_join(tp_thread thread);

void init_pool(threadpool *pool, void (*callback)(void*), void* constructed_params, size_t tasks_count, size_t param_size);
void pool_set_newtask(threadpool *pool, void (*new_callback)(void*), void* constructed_params, size_t tasks_count); //需等上一批任务全部完成后再调用
uint8_t pool_checkTaskProcessing(threadpool *pool); //返回0表示当前批次任务已全部执行完毕
void pool_clear(threadpool *pool);
#endif
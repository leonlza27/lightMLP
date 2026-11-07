#include "sharedpool.h"
#include "threadpool.h"
#include <stdint.h>
#include <stdlib.h>   /* malloc / free */

typedef struct shared_pool_node{
    threadpool *pool;
    uint8_t free_cntdown;
    uint8_t ishold;
}shared_pool_node;

static shared_pool_node *pub_threadpools;
static uint16_t pools_cnt;
static uint16_t _cursor;
static tp_mutex poollst_onoperate_lock;

static uint16_t flush_delay;
static uint8_t flush_hold_cycles;

static uint16_t backflush_onrun; 

tp_thread back_flush_proc;

static void flush_op(){
    uint16_t unused_cnt = 0;
    threadpool *unused[pools_cnt];
    tp_mutex_lock(&poollst_onoperate_lock);
    for(uint16_t i = 0; i < pools_cnt; i++){
        if(pub_threadpools[i].ishold) continue;
        if(pub_threadpools[i].free_cntdown) pub_threadpools[i].free_cntdown--;
        else if(pub_threadpools[i].pool){
            unused[unused_cnt] = pub_threadpools[i].pool;
            unused_cnt++;
            pub_threadpools[i].pool = 0;
        }
    }
    tp_mutex_unlock(&poollst_onoperate_lock);
    for(uint16_t i = 0; i < unused_cnt; i++){
        pool_clear(unused[i]);
        free(unused[i]);
    }
}

static TP_WORKER_ENTRY flush_reclaimer_routine(void *param){
    (void)param;
    while(backflush_onrun){
        flush_op();
        tp_sleep_ms(flush_delay);
    }
    return TP_WORKER_RETVAL;
}

uint8_t public_pool_reconfig(uint16_t mananged_count, uint16_t delay_per_flush, uint8_t flush_cycles_to_free){
    tp_mutex_lock(&poollst_onoperate_lock);
    flush_delay = delay_per_flush;
    flush_hold_cycles = flush_cycles_to_free;
    shared_pool_node *pools_new = (shared_pool_node*)malloc(sizeof(shared_pool_node) * mananged_count);
    if(!pools_new){
        tp_mutex_unlock(&poollst_onoperate_lock);
        return -1;
    }
    for(uint16_t i = 0; i < mananged_count; i++){
        pools_new[i].pool = 0;
        pools_new[i].free_cntdown = 0;
        pools_new[i].ishold=0;
    }
    for(uint16_t i = 0; i < pools_cnt; i++){
        pools_new[i].pool = pub_threadpools[i].pool;
        pools_new[i].free_cntdown = pub_threadpools[i].free_cntdown;
        pools_new[i].ishold = pub_threadpools[i].ishold;
    }
    free(pub_threadpools);
    pub_threadpools = pools_new;
    pools_cnt = mananged_count;
    _cursor = 0;
    tp_mutex_unlock(&poollst_onoperate_lock);
    return 0;
}

int32_t pubpools_append_task(void (*callback)(void*), void *params, size_t count, size_t param_size){
    uint8_t cycles = 2;
    int32_t got = -1;
    threadpool *pool = NULL;
    int is_new = 0;

    tp_mutex_lock(&poollst_onoperate_lock);
    while(cycles){
        shared_pool_node *slot = &pub_threadpools[_cursor];
        if(!slot->ishold){
            slot->ishold = 1;
            slot->free_cntdown = 0;      /* 占用期间不参与回收 */
            pool = slot->pool;
            is_new = (pool == NULL);
            got = (int32_t)_cursor;
            break;
        }
        _cursor++;
        if(_cursor >= pools_cnt){
            _cursor = 0;
            cycles--;
        }
    }
    tp_mutex_unlock(&poollst_onoperate_lock);

    if(got < 0) return -1;

    /* 锁外创建/复用(init_pool 会创建多个工作线程, 不宜持锁执行) */
    if(is_new){
        pool = (threadpool*)malloc(sizeof(threadpool));
        if(pool == NULL){
            tp_mutex_lock(&poollst_onoperate_lock);
            pub_threadpools[got].ishold = 0;    /* 回滚占用 */
            tp_mutex_unlock(&poollst_onoperate_lock);
            return -1;
        }
        init_pool(pool, callback, params, count, param_size);
        tp_mutex_lock(&poollst_onoperate_lock);
        pub_threadpools[got].pool = pool;       /* 记录对象指针 */
        tp_mutex_unlock(&poollst_onoperate_lock);
    }else{
        pool_set_newtask(pool, callback, params, count);
        tp_mutex_lock(&poollst_onoperate_lock);
        tp_mutex_unlock(&poollst_onoperate_lock);
    }
    return got;
}

threadpool *pubpools_getpool(uint16_t idx){
    if(idx > pools_cnt) return 0;
    return pub_threadpools[idx].pool;
}

uint8_t pubpools_checkprocessing(uint16_t idx){
    threadpool *pool;
    uint8_t processing = 0;
    tp_mutex_lock(&poollst_onoperate_lock);
    pool = (idx < pools_cnt) ? pub_threadpools[idx].pool : NULL;
    tp_mutex_unlock(&poollst_onoperate_lock);
    if(pool != NULL) processing = pool_checkTaskProcessing(pool);
    return processing;
}

void pubpools_release(uint16_t idx){
    tp_mutex_lock(&poollst_onoperate_lock);
    if(idx < pools_cnt){
        pub_threadpools[idx].ishold = 0;
        pub_threadpools[idx].free_cntdown = flush_hold_cycles;
    }
    tp_mutex_unlock(&poollst_onoperate_lock);
}

void pubpools_init_default(){
    pools_cnt = 4;
    flush_delay = 10;
    flush_hold_cycles = 5;
    tp_mutex_init(&poollst_onoperate_lock);   /* 资源表锁随初始化建立: 后台回收与卸载钩子都要持有它 */
    pub_threadpools = (shared_pool_node*)malloc(sizeof(shared_pool_node) * pools_cnt);
    for(uint16_t i = 0; i < pools_cnt; i++){
        pub_threadpools[i].pool = 0;
        pub_threadpools[i].free_cntdown = 0;
        pub_threadpools[i].ishold=0;
    }
    backflush_onrun = 1;
    tp_thread_create(&back_flush_proc, flush_reclaimer_routine, 0);
}

void pubpools_flush_onexit(){
    for(uint8_t round = 0; round <= flush_hold_cycles; round++){
        flush_op();
    }
    backflush_onrun = 0;
    tp_thread_join(&back_flush_proc);
}

/* ============================================================
 * 动态库生命周期自动挂接
 * ------------------------------------------------------------
 * 目标: 本文件被编译进动态库(shared library / DLL)时,
 *       库装载后自动执行 pubpools_init_default(),
 *       库卸载(或进程退出)时自动执行 pubpools_flush_onexit(),
 *       调用方无需再显式初始化与回收全局线程池表.
 *
 * 实现:
 *   - Windows(MSVC / MinGW): 由动态库入口 DllMain 的
 *     DLL_PROCESS_ATTACH / DLL_PROCESS_DETACH 通知触发;
 *   - *nix / Apple(GCC / Clang): 由 constructor / destructor 属性触发.
 *
 * 说明:
 *   - 两个入口都带一次性保护: 应用若已手动调用过初始化, 自动初始化不会重复分配资源表;
 *   - DllMain 只对动态库有意义: 若本文件被静态链接进可执行程序,
 *     Windows 上不会自动挂钩(此时可手动调用上述两个函数), *nix 上仍由属性自动挂钩;
 *   - Windows 显式 FreeLibrary 卸载时不做回收(原因见 DllMain 注释),
 *     需在卸载前自行调用 pubpools_flush_onexit().
 * ============================================================ */

static uint8_t pubpools_auto_ready;

//装载/进程附着: 自动初始化默认线程池表
static void pubpools_auto_enter(void){
    if(pubpools_auto_ready) return;
    pubpools_auto_ready = 1;
    pubpools_init_default();
}

//卸载/进程分离: 自动回收线程池表
static void pubpools_auto_exit(void){
    if(!pubpools_auto_ready) return;
    pubpools_auto_ready = 0;
    pubpools_flush_onexit();
}

#if defined(_WIN32)
/*
 * 动态库入口:
 *   DLL_PROCESS_ATTACH(装载) -> 自动初始化默认线程池表;
 *   DLL_PROCESS_DETACH(分离) -> 进程退出(lpvReserved != NULL)时自动回收.
 *
 * 为什么显式 FreeLibrary(lpvReserved == NULL)不做回收:
 *   该回调运行在加载器锁内, 而工作线程退出也要求取加载器锁
 *   (ntdll!LdrShutdownThread), 在此执行 pool_clear 的 join 会与线程退出互相等待
 *   而死锁; 进程退出时其它线程已被系统结束, join 可立即返回, 因此只在此时回收.
 *   需要在 FreeLibrary 前回收的话, 调用方应在普通线程中调用 pubpools_flush_onexit().
 */
BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved){
    (void)hinstDLL;
    if(fdwReason == DLL_PROCESS_ATTACH) pubpools_auto_enter();
    else if(fdwReason == DLL_PROCESS_DETACH && lpvReserved != NULL) pubpools_auto_exit();
    return TRUE;
}
#elif defined(__GNUC__) || defined(__clang__)
//共享库加载完成后自动初始化
__attribute__((constructor)) static void pubpools_lib_onload(void){
    pubpools_auto_enter();
}

//共享库卸载/进程退出时自动回收
__attribute__((destructor)) static void pubpools_lib_onunload(void){
    pubpools_auto_exit();
}
#endif
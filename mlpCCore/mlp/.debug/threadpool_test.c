//草稿,验证可行性

#include <stdint.h>
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

#include <time.h>

typedef struct threadpool{
    HANDLE subthreads[4];
    HANDLE param_lock;
    volatile size_t paras_len;   /* 主线程写 / 子线程读,需 volatile 防止被优化掉 */
    volatile size_t curidx;
    volatile uint8_t onrun;
    void **paras;
}threadpool;

typedef struct init_param{
    uint8_t number;
    threadpool *pool;
    void (*callback)(void*);
}init_param;

/* CreateThread 要求线程入口使用 WINAPI(__stdcall) 调用约定,否则(x86 下)线程返回时
   调用方与被调用方对栈的清理方式不一致,极易崩溃。 */
static DWORD WINAPI subthread(void* param){
    init_param para = *(init_param*)param;
    threadpool *pool = para.pool;
    uint8_t order = para.number;
    void (*callback)(void*) = para.callback;
    (void)order;
    free(param);              /* 参数是本线程独占的堆内存,拷贝完成后立即释放 */
    while(pool->onrun){
        WaitForSingleObject(pool->param_lock, INFINITE);
        if(pool->curidx < pool->paras_len){
            size_t idxcur = pool->curidx++;
            ReleaseMutex(pool->param_lock);
            callback(pool->paras[idxcur]);
        }else{
            ReleaseMutex(pool->param_lock);
            Sleep(10);
        }
    }
    return 0;
}

void init_tpool(threadpool *pool, void (*callback)(void*)){
    pool->curidx = 0;
    pool->paras_len = 0;
    pool->onrun = 1;
    /* bInitialOwner = TRUE:主线程先持有锁,子线程起来后会在 WaitForSingleObject 处阻塞,
       直到主线程把 paras/paras_len 准备好再 ReleaseMutex,天然形成"就绪门"。 */
    pool->param_lock = CreateMutexA(0, 1, 0);
    for(uint8_t i = 0; i < 4; i++){
        /* 关键修复:参数必须堆分配。若用局部数组 init_param params[4] 并传 &params[i],
           init_tpool 返回后该栈内存失效,子线程读取时即 SIGSEGV。 */
        init_param *p = (init_param*)malloc(sizeof(init_param));
        p->number = i;
        p->pool = pool;
        p->callback = callback;
        pool->subthreads[i] = CreateThread(0, 0, subthread, p, 0, 0);
    }
}

void test_callback(void* param){
    int *num = (int*)param;
    *num = *num * 2;
}


int main(){
    threadpool testpool;
    init_tpool(&testpool, test_callback);
    int *nums = (int*)malloc(sizeof(int) * 20000);
    int **cap = (int**)malloc(sizeof(int*) * 20000);

    for(int i = 0; i < 20000; i++) {nums[i] = i; cap[i] = &nums[i];}
    time_t norm_start = time(0);
    for(int i = 0; i < 20000; i++) {nums[i] *= 2;}
    time_t norm_end = time(0);
    testpool.paras = (void**)cap;
    testpool.paras_len = 20000;          /* 修复:之前从未赋值,工作队列始终为空 */
    time_t mt_start = time(0);
    ReleaseMutex(testpool.param_lock);  /* 放行子线程开始消费任务 */
    while(testpool.paras_len > testpool.curidx);
    testpool.onrun = 0;
    WaitForMultipleObjects(4, testpool.subthreads, 1, INFINITE);
    time_t mt_end = time(0);
    free(nums);
    free(cap);
    printf("norm %d, mt %d\n",norm_end - norm_start, mt_end - mt_start);
    for(int i = 0; i < 4; i++) CloseHandle(testpool.subthreads[i]);
    CloseHandle(testpool.param_lock);
    return 0;
}
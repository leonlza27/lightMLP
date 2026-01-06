#ifndef _mult_proc_c
#define _mult_proc_c
#include <stdint.h>
#include <stdlib.h>

/*
说明:
为做到全平台循环加速,尤为非完整OpenMP支持(或完全不支持)编译器平台下的for加速
循环逻辑由子线程负责

若在嵌入式或非完整STL支持下保留完整推理-训练框架应将多线程回退至普通循环
你最好在子线程再次手动循环展开
或自定义multi_process函数接入平台线程调度器对应函数
*/

typedef void *(*SubFn)(void*);

#define BLOCK_SIZE 512
#define BIG_BLOCK_SIZE BLOCK_SIZE * 5

//工作区间: [start, end)
/*
使用
1.定义子线程函数(registerTask),start, end与data将在函数体内自动展开,直接用(data指向自定义数据转发的结构体)
2.在需要做for多线程加速的父线程准备数据到自定义结构体变量
3.multi_process_c(总任务数, 你定义的函数, 准备的数据)

*/

typedef struct Rate{
    size_t start;
    size_t end;
    void* data;
}Rate;

#define registerTask(mark, fn)\
void (mark)(void* param){\
    Rate *para = (Rate*)param;\
    size_t start = para->start;\
    size_t end = para->end;\
    void* data = para->data;\
    fn;\
}\

#define _max(a,b) a > b? a : b
#define _min(a,b) a < b? a : b

#ifndef forced_sigle_thread
#ifdef _WIN32

#else

#include <pthread.h>
#include <unistd.h>
#include <sys/sysinfo.h>

#define multi_process_c(total_tasks, workerFunc, dataIn){\
    long avaCpuCount = sysconf(_SC_NPROCESSORS_ONLN);\
    size_t threadnum =  (size_t)((total_tasks * (total_tasks + 4)) / (total_tasks * 4 + 8));\
    threadnum = _max(_min(threadnum,avaCpuCount),1);\
    pthread_t *subthreads = (pthread_t*)(malloc(sizeof(pthread_t)*threadnum));\
    Rate *subpara = (Rate*)(malloc(sizeof(Rate)*threadnum));\
\
    size_t tasks_per_proc = (size_t)(total_tasks/threadnum) + 2;\
\
    for(size_t i = 1; i< threadnum; i++){\
        Rate *loc = &subpara[i];\
        size_t ratestart = i * tasks_per_proc;\
        loc->start = ratestart;\
        loc->end = _min(ratestart + tasks_per_proc,total_tasks);\
        loc->data = &dataIn;\
        pthread_create(subthreads + i, 0, workerFunc, loc);\
    }\
\
    Rate *locm = subpara;\
    locm->start = 0;\
    locm->end = _min(tasks_per_proc,total_tasks);\
    locm->data = &dataIn;\
    (workerFunc)(locm);\
\
    for(size_t i = 1; i< threadnum; i++){\
        pthread_join(subthreads[i], 0);\
    }\
    free(subthreads);\
    free(subpara);\
}\

#endif
#else
#define multi_process_c(total_tasks, workerFunc, dataIn){\
    Rate locm;\
    locm.start = 0;\
    locm.end = min(tasks_per_proc,total_tasks);\
    locm.data = &dataIn;\
    (workerFunc)(&locm);\
}\

#endif


#endif
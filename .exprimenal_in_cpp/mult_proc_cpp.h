#ifndef _multproc_
#define _multproc_
#include <cstddef>
#include <thread>
#include <algorithm>
#include <stdlib.h>
#include <math.h>

/*
说明:
为做到全平台循环加速,尤为非完整OpenMP支持或完全不支持编译器平台下的
循环逻辑由子线程负责

若在嵌入式或非完整STL支持下保留完整推理-训练框架应将多线程回退至普通循环
你最好在子线程再次手动循环展开
或自定义multi_process函数接入平台线程调度器对应函数
*/

/*
用法:
total_tasks:总任务数
workerFunc:逻辑处理用的子函数,可以是一般函数或lambda及其他callables

子函数定义:
- 必须接受两个参数用作任务区间接收, 范围[start, end)
- 返回: void
*/

#define wrapper_custom_start_end (size_t start,size_t end)
#define wrapper_inner_loop(vaname) for(size_t vaname = start;vaname < end;vaname++)

#define BLOCK_SIZE 512
#define BIG_BLOCK_SIZE BLOCK_SIZE * 5

#ifdef forced_sigle_thread

template <typename SubFn>
inline void multi_process(size_t total_tasks, SubFn &&workerFunc){
    workerFunc(0,total_tasks);
}

#else

template <typename SubFn>
inline void multi_process(size_t total_tasks, SubFn &&workerFunc){
    unsigned int maxCPUCores = std::thread::hardware_concurrency();
    //size_t threadnum = (size_t)(total_tasks/tasks_per_proc) + 1;
    size_t threadnum = static_cast<size_t>(sqrt(total_tasks/BLOCK_SIZE) + total_tasks / BIG_BLOCK_SIZE - 1);
    threadnum = std::max<size_t>(std::min<size_t>(threadnum,maxCPUCores),1);
    std::thread *subthreads = static_cast<std::thread*>(malloc(sizeof(std::thread)*threadnum));

    size_t tasks_per_proc = static_cast<size_t>(total_tasks/threadnum) + 2;

    for(size_t i = 1;i < threadnum;i++){
        size_t ratestart = i * tasks_per_proc;
        size_t rateend = std::min(ratestart + tasks_per_proc,total_tasks);
        
        //subthreads.emplace_back(std::forward<SubFn>(workerFunc),ratestart,rateend,std::ref(mtx));
        new (subthreads + i - 1) std::thread(std::forward<SubFn>(workerFunc),ratestart,rateend);
    }

    size_t rateend = std::min(tasks_per_proc,total_tasks);
    workerFunc(0,rateend);

    for(size_t i = 0;i < threadnum - 1;i++) 
        if(subthreads[i].joinable())
            subthreads[i].join();

    free(subthreads);
}

#endif

#endif
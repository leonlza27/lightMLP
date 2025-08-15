#define __FN_WRAPPER
#include "wrapper.h"
template <typename SubFn>
inline void multi_process(size_t total_tasks, SubFn &&workerFunc){
    unsigned int maxCPUCores = std::thread::hardware_concurrency();
    //size_t threadnum = (size_t)(total_tasks/tasks_per_proc) + 1;
    size_t threadnum = static_cast<size_t>(sqrt(total_tasks/BLOCK_SIZE) + total_tasks / BIG_BLOCK_SIZE);
    threadnum = std::max<size_t>(std::min<size_t>(threadnum,maxCPUCores),1);
    std::thread *subthreads = static_cast<std::thread*>(malloc(sizeof(std::thread)*threadnum));
    std::mutex mtx;

    size_t tasks_per_proc = static_cast<size_t>(total_tasks/threadnum) + 2;

    for(size_t i = 0;i < threadnum;i++){
        size_t ratestart = i * tasks_per_proc;
        size_t rateend = std::min(ratestart + tasks_per_proc,total_tasks);
        
        //subthreads.emplace_back(std::forward<SubFn>(workerFunc),ratestart,rateend,std::ref(mtx));
        new (subthreads + i) std::thread(std::forward<SubFn>(workerFunc),ratestart,rateend,std::ref(mtx));
    }

    for(size_t i = 0;i < threadnum;i++) 
        if(subthreads[i].joinable())
            subthreads[i].join();

    free(subthreads);
}
#include <cstddef>
template <typename SubFn>
inline void multi_process(size_t total_tasks, size_t tasks_per_proc, SubFn &&workerFunc){
    size_t threadnum = (size_t)(total_tasks/tasks_per_proc) + 1;
    std::thread *subthreads = static_cast<std::thread*>(malloc(sizeof(std::thread)*threadnum));
    std::mutex mtx;

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
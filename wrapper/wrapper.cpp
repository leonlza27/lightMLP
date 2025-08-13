#include "wrapper.h"

template <typename SubFn>
inline void multi_process(size_t total_tasks, size_t tasks_per_proc, SubFn workerFunc){
    size_t threadnum = (size_t)(total_tasks/tasks_per_proc) + 1;
}
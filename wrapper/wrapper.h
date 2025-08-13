#include <cstddef>
#ifndef _wrapper_
#define _wrapper_

template<typename SubFn>
void multi_process(size_t total_tasks,size_t tasks_per_proc,SubFn workerFunc);

#endif
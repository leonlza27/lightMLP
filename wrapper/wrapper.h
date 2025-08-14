#ifndef _wrapper_
#define _wrapper_
#include <cstddef>
#include <mutex>
#include <thread>
#include <algorithm>
#include <stdlib.h>

#define wrapper_custom_start_end_mtx (size_t start,size_t end,std::mutex &mtx)

template<typename SubFn>
void multi_process(size_t total_tasks,size_t tasks_per_proc,SubFn &&workerFunc);

#include "wrapper.tpp"

#endif
#ifndef _wrapper_
#define _wrapper_
#include <cstddef>
#include <mutex>
#include <thread>
#include <algorithm>
#include <stdlib.h>

#define wrapper_custom_start_end_mtx (size_t start,size_t end,std::mutex &mtx)
#define wrapper_inner_loop(vaname) for(size_t vaname = start;vaname < end;vaname++)

template<typename SubFn>
void multi_process(size_t total_tasks,size_t tasks_per_proc,SubFn &&workerFunc);

#include "wrapper.tpp"

#endif
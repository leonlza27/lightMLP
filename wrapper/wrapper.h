#ifndef _wrapper_
#define _wrapper_
#include <cstddef>
#include <mutex>
#include <thread>
#include <algorithm>
#include <stdlib.h>
#include <math.h>

#define wrapper_custom_start_end_mtx (size_t start,size_t end,std::mutex &mtx)
#define wrapper_inner_loop(vaname) for(size_t vaname = start;vaname < end;vaname++)

#define BLOCK_SIZE 65536
#define BIG_BLOCK_SIZE BLOCK_SIZE * 5

#ifndef __FN_WRAPPER
template<typename SubFn>
void multi_process(size_t total_tasks,SubFn &&workerFunc);
#endif

#include "wrapper.tpp"

#endif
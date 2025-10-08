//softpool:用于pc/服务器/移动平台低碎片微GC中小对象+mmap存大对象
#ifndef _softpool_pc_
#define _softpool_pc_
#include <malloc.h>
#include <mutex>
#include <string.h>
#include <forward_list>

#define _8KB_PAGE_UNIT 8 * 1024

#ifdef _WINDOWS_    //windows页操作
#include <memoryapi.h>
#define alloc_8kbpages(n) VirtualAlloc(0, n * _8KB_PAGE_UNIT, MEM_COMMIT | MEM_RELEASE, PAGE_READWRITE)
#define freepage(ptr, size) VirtualFree(ptr, 0, MEM_RELEASE)
#else               //linux/macos页操作
#include <sys/mman.h>
#define alloc_8kbpages(n) mmap(0, n * _8KB_PAGE_UNIT, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
#define freepage(ptr, size) munmap(ptr, size)
#endif

#include "DynPool.h"

#include "../softmalloc.h"

class SoftPool{
private:
    std::forward_list<NormPool*> NormPools;


public:
    SoftPool();
    void *Allocate(size_t size);

    
};

#endif
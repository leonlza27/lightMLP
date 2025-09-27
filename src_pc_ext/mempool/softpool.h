//softpool:用于pc/服务器/移动平台低碎片微GC中小对象+mmap存大对象
#ifndef _softpool_pc_
#define _softpool_pc_
#include <malloc.h>
#include <mutex>
#include <unordered_map>
#include <string.h>

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

class NormPool{
private:
    char *bitmap;   //位图索引 
    void *poolmem;  //8kb * n 总pool内存
    size_t memLength;   // **记录大小可用于mmap释放*

    inline size_t matchBitmapFreePart(size_t objsize);
    inline void changeBitmapMark(size_t start, size_t end, char bitmark);
public:
    NormPool(int units);
    ~NormPool();

    void *poolmalloc(size_t size);
    void poolfree(void *ptr, size_t free_size);
};

class SoftPool{
private:

public:
    SoftPool(size_t initalsize);
    
};

#endif
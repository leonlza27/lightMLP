#include <stdio.h>
#include <sys/mman.h>
#include <vector>

#include "pooleg.h"

class plswitcher{
private:
    uint8_t HotPoolUsableIdx;
    uint8_t UsableIdx;
    spin_lock globAllocLock;
    mempool *hotpools[10];
    mempool *pools[53];
public:
    plswitcher(): HotPoolUsableIdx(1),UsableIdx(1){ 
        for (auto &p : hotpools) {
            p = 0;
        }
        pools[0] = CreateMempool();
        hotpools[0] = pools[0];
    }

    ~plswitcher(){
        for(auto &p : pools){
            munmap(p, 8192);
        }
    }

    void *alllocate(size_t size){
        while(globAllocLock.is_locked()){
            std::this_thread::yield();
        }
        void* retAddr;
        for(auto &p : hotpools){
            if(p == 0) break;
            if(!p->isOperating()){
                retAddr = p->pAlloc(size);
                if(retAddr) return retAddr;
            }
        }

        //热池未满:追加
        if(HotPoolUsableIdx < 10){
            printf("pool: extenction occupyed\n");
            std::lock_guard<spin_lock> appendLock(globAllocLock);
            mempool *pNew = CreateMempool();
            pools[UsableIdx] = pNew;
            hotpools[HotPoolUsableIdx] = pNew;
            UsableIdx++;
            HotPoolUsableIdx++;
            return pNew->pAlloc(size);
        }


        //扫描&更改


        printf("pool: full\n");

        return 0;
    }

    void free(void* ptr){
        char* freePtr = (char*)ptr - 8;
        DataNode *freeNode = (DataNode*)(freePtr);
        mempool *parentPool = (mempool*)(freePtr - freeNode->thisStart - 3);
        parentPool->pFree(ptr);
    }

};

plswitcher *switcher;

void testHandler(){
    size_t _size = rand()%225 + 1;
    char *testmem = (char*)switcher->alllocate(_size);
    if(testmem == 0) {
        printf("thread: got a null addr\n");
        return;
    }
    int __handleTime = rand()%9999 + 1;
    std::this_thread::sleep_for(std::chrono::milliseconds(__handleTime));
    switcher->free(testmem);
}

int main(){ 
    switcher = (plswitcher*)mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    switcher = new(switcher) plswitcher;

    std::vector<std::thread> workers;
    workers.reserve(1024);
    for(int i = 0; i< 1024; i++)
        workers.emplace_back(testHandler);

    for(auto &t : workers) t.join();    

    return 0;
}

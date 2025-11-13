#include <stdio.h>
#include <sys/mman.h>

#include "pooleg.h"

class plswitcher{
private:
    uint8_t HotPoolUsableIdx;
    uint8_t UsableIdx;
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
            mempool *pNew = CreateMempool();
            pools[UsableIdx] = pNew;
            hotpools[HotPoolUsableIdx] = pNew;
            UsableIdx++;
            HotPoolUsableIdx++;
            return pNew->pAlloc(size);
        }

        //扫描&更改




        return 0;
    }

    void free(void* ptr){
        char* freePtr = (char*)ptr - 8;
        DataNode *freeNode = (DataNode*)(freePtr);
        DataNode *freeNext = (DataNode*)(freePtr + freeNode->fullsize + freeNode->startNext - freeNode->end);
        mempool *parentPool = (mempool*)(freePtr - freeNext->prev - 3);
        parentPool->pFree(ptr);
    }

};

plswitcher switcher;

int main(){ 
    

    return 0;
}

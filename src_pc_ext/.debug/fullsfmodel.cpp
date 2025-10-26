#include <stdio.h>
#include <sys/mman.h>
#include <stdint.h>

struct PoolHead{
    char* ActualData0;
    uint16_t fullsize;
    uint16_t padding;
};

template<short PByte>
struct PoolPadding{
    char _FIX_[PByte];
};

using ppad4 = PoolPadding<4>;
using ppad2 = PoolPadding<2>;

//元数据头节点 空闲[end, startNext)
struct DataNode{
    uint16_t end;
    uint16_t startNext;
    uint16_t prev;
};

class mempool{
private:
    char* mem;
public:
    mempool(){
        mem = (char*)mmap(0, 8192, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        PoolHead *head = (PoolHead*)mem;
        head->fullsize = 8192;
        head->padding = 4;
        head->ActualData0 = mem + sizeof(PoolHead) + (4 - sizeof(PoolHead)%4)*(sizeof(PoolHead)%4 != 0);
        DataNode *NodeHead = (DataNode*)head->ActualData0;
        NodeHead->startNext = -1;
        NodeHead->end = sizeof(DataNode);
        NodeHead->prev = -1;
    }
    
    ~mempool(){
        munmap(mem, 8192);
    }

    void *pAlloc(uint16_t size){
        uint16_t total_size = size + sizeof(DataNode);
        uint16_t blocks = total_size/4 + (total_size%4 != 0);
        ppad4 *idx0 = (ppad4*)((PoolHead*)mem)->ActualData0;
        ppad4 *cur = idx0;
        DataNode *curNode;
        for(curNode = (DataNode*)cur; curNode->startNext != -1;cur = idx0 + curNode->startNext){
            if(curNode->startNext - curNode->end >= size) break;
        }
        
        
        return 0;
    }

    void pFree(void* ptr){
        
    }
};

int main(){
    mempool pool;
    printf("%lu\n",sizeof(DataNode));

    return 0;
}

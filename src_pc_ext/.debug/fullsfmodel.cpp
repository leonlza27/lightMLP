#include <stdio.h>
#include <sys/mman.h>
#include <stdint.h>
#include <string.h>

struct PoolHead{
    uint16_t fullsize;
    uint16_t padding;
};

//元数据头节点 空闲[end, startNext)
struct DataNode{
    int16_t end;
    int16_t startNext;
    int16_t prev;
};

class mempool{
private:
    char *mem;
    char *dataStart;
public:
    mempool(){
        mem = (char*)mmap(0, 8192, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        memset(mem, 0, 8192);
        PoolHead *head = (PoolHead*)mem;
        head->fullsize = 8192;
        head->padding = 4;
        
        dataStart = mem + 4;
        DataNode *lkhead = (DataNode*)dataStart;
        DataNode *lktail = (DataNode*)(dataStart + 8180);
        
        lkhead->end = 8;
        lkhead->startNext = 8180;
        lkhead->prev = -1;

        lktail->startNext = -1;
        lktail->end = 8188;
        lktail->prev = 0;

        //total 8172B
    }
    
    ~mempool(){
        dataStart = 0;
        munmap(mem, 8192);
    }

    void *pAlloc(uint16_t size){
        size_t total_size = size + 4;
        size_t _needed = (total_size + 3) & (~3); 
        
        printf("input:%d needed:%lu\n",size,_needed);

        DataNode *curnode;
        int16_t _idxnxt = 0;
        int16_t _idxcur = 0;
        int16_t ava0 = 0;        //可用起始偏移
        while (_idxcur != -1){
            curnode = (DataNode*)(dataStart + _idxcur);
            _idxnxt = curnode->startNext;
            ava0 = curnode->end;
            if(_idxnxt - ava0 >= _needed) break;
            printf("idxcur:%d\n",_idxcur);
            _idxcur = _idxnxt;
        }
        printf("final idxcur:%d ava start:%d\n",_idxcur, ava0);
        if(_idxnxt == -1) return 0;
        
        DataNode *nodeNew = (DataNode*)(dataStart + ava0);
        nodeNew->end = ava0 + _needed;
        nodeNew->prev = _idxcur;
        nodeNew->startNext = _idxnxt;

        DataNode *nnewNext = (DataNode*)(dataStart + _idxnxt);

        curnode->startNext = ava0;
        nnewNext->prev = ava0;
        
        return dataStart + ava0 + 6;
    }

    void pFree(void* ptr){
        if(ptr == 0) return;
        size_t delidx = ((size_t)ptr) - ((size_t)dataStart) - 6;
        if(delidx >= 8172 || delidx < 8) return;

        DataNode *delNode = (DataNode*)(dataStart + delidx);
        if(delNode->end == 0) return;
        DataNode *prev_of_del = (DataNode*)(dataStart + delNode->prev);
        DataNode *next_of_del = (DataNode*)(dataStart + delNode->startNext);
        
        next_of_del->prev = delNode->prev;
        prev_of_del->startNext = delNode->startNext;
        delNode->end = 0;
    }
};

int main(){
    mempool pool;
    
    printf("alloc1\n");
    int *test = (int*)pool.pAlloc(sizeof(int)*5);
    printf("addr:%lu\n",(size_t)test);
    printf("\nalloc2\n");
    int *test2 = (int*)pool.pAlloc(sizeof(int)*5);
    printf("addr:%lu\n",(size_t)test2);    
    pool.pFree(test);
    
    return 0;
}

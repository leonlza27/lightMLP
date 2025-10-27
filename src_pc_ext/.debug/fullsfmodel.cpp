#include <stdio.h>
#include <sys/mman.h>
#include <stdint.h>

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
        PoolHead *head = (PoolHead*)mem;
        head->fullsize = 8192;
        head->padding = 4;
        
        dataStart = mem + sizeof(PoolHead);
        ((DataNode*)dataStart)->end = 8;
        ((DataNode*)dataStart)->prev = -1;
        ((DataNode*)dataStart)->startNext = -1;
    }
    
    ~mempool(){
        dataStart = 0;
        munmap(mem, 8192);
    }

    void *pAlloc(uint16_t size){
        size_t total_size = size + sizeof(DataNode);
        size_t _needed = total_size + (4 - total_size%4)*(total_size%4 != 0);
        DataNode *curnode;
        int8_t _idxnxt = 0;
        int8_t _idxcur = 0;
        int8_t ava0 = 0;
        do{
            curnode = (DataNode*)(dataStart + _idxnxt);
            ava0 = curnode->end;
            _idxnxt = curnode->startNext;

            if(_idxnxt - ava0 >= _needed) break;

            _idxcur = _idxnxt;

        }while (_idxnxt != -1);
        
        //头 4b
        if(ava0 + _needed > 8188) return 0;
        
        DataNode *_insert = (DataNode*)(dataStart + ava0);
        _insert->startNext = _idxnxt;
        _insert->end = ava0 + _needed;
        _insert->prev = _idxcur;
        

        return dataStart + ava0 + 6;
    }

    void pFree(void* ptr){
        int8_t delidx = ((size_t)ptr) - ((size_t)dataStart) - 6;
    }
};

int main(){
    mempool pool;
    printf("%lu\n",sizeof(DataNode));

    return 0;
}

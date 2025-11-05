#include <stdio.h>
#include <sys/mman.h>
#include "fullmempleg.h"

class plDequeNode;

struct plDequeNodeLayout{
    plDequeNode *ndnext;
    alignas(8) uint16_t usableDStart;
    mempool *pools[510];
};

class plDequeNode{
private:
    plDequeNodeLayout *_node;
public:
    plDequeNode(){
        _node = (plDequeNodeLayout*)mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        _node->ndnext = 0;
        _node->usableDStart = 0;
    }

    ~plDequeNode(){
        if(_node->ndnext) delete _node->ndnext;
        munmap(_node, 4096);
    }

    bool AddPool(){
        uint16_t avaidx = _node->usableDStart;
        if(avaidx < 510){            
            _node->pools[avaidx] = new mempool;
            avaidx++;
            _node->usableDStart = avaidx;
            return 1;
        }
        return 0;
    }

};

namespace GlobPoolSwitcher{
    mempool *hotpool[10];

}


int main(){
    printf("%lu\n", sizeof(std::mutex));

    return 0;
}
#include <stdio.h>
#include <thread>
#include <vector>
#include <string>
#include <sstream>

#include "spin_lock.h"

struct _wrapeg{
    std::vector<std::string> globlst;
    spin_lock lock;
} wrapeg;

int main(){
    std::thread workers[10];
    for(int i = 0; i < 10 ;i++) workers[i] = std::thread([i](){
        wrapeg.lock.lock();
        printf("thread %d working\n", i);
        for(int j = 0; j < 50; j++){
            std::stringstream topt;
            topt << "from thread " << i << " : " << j;
            wrapeg.globlst.push_back(topt.str());
        }
        printf("thread %d quit\n", i);
        wrapeg.lock.unlock();
    });
    for (auto &t : workers) t.join();

    
    for(auto s : wrapeg.globlst) printf("%s\n", s.c_str());

}

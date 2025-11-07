#include <stdio.h>
#include <thread>
#include <atomic>
#include <vector>
#include <string>
#include <sstream>

#include "spin_lock.h"

std::vector<std::string> globlst;

spin_lock lock;

int main(){
    std::thread workers[10];
    for(int i = 0; i < 10 ;i++) workers[i] = std::thread([i](){
        lock.lock();
        printf("thread %d working\n", i);
        for(int j = 0; j < 50; j++){
            std::stringstream topt;
            topt << "from thread " << i << " : " << j;
            globlst.push_back(topt.str());
        }
        printf("thread %d quit\n", i);
        lock.unlock();
    });
    for (auto &t : workers) t.join();

    
    for(auto s : globlst) printf("%s\n", s.c_str());

}

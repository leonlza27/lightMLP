#ifndef __spin_lock
#define __spin_lock

#include <atomic>
#include <thread>

class spin_lock{
private:
    std::atomic<char> flag;
public:
    spin_lock(){
        flag.store(0);
    }

    void lock() {
        while (flag.exchange(1,std::memory_order_acquire) == 1)
            std::this_thread::yield();
    }

    void unlock() {
        flag.store(0,std::memory_order_release);
    }

    char try_lock(){
        return flag.exchange(1,std::memory_order_acquire);
    }

    char is_locked(){
        return flag.load(std::memory_order_relaxed);
    }
};

#endif

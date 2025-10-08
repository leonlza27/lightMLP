#include <iostream>
#include <cstdint>
#include <cstddef>
#include <string.h>
#include <stdlib.h>


//查找位图连续0个数(位图,位图尺寸(总位数),连续0数)->bitmap符合条件的第一个bit偏移量,未找到返回-1
int findConsecutiveZeros(const uint8_t* bitmap, size_t size_bytes, size_t NumOfZeros) {
    if (NumOfZeros == 0) return 0;
    if (size_bytes < NumOfZeros) return -1;
    
    size_t left = 0;   // 窗口左边界
    size_t right = 0;  // 窗口右边界
    
    // 滑动窗口查找连续的0
    while (right < size_bytes) {
        // 如果right位置是1，则重置窗口
        size_t right_byte_index = right / 8;
        size_t right_bit_index = right % 8;
        if ((bitmap[right_byte_index] & (1 << right_bit_index)) != 0) {
            // 遇到1，重置窗口
            left = right + 1;
        } else {
            // right位置是0
            // 检查窗口大小是否达到要求
            if (right - left + 1 == NumOfZeros) {
                return left;  // 返回起始位置
            }
        }
        right++;  // 扩展右边界
    }
    
    return -1;  // 未找到
}

// 测试函数
int main(int argc, char** argv) {
    // 创建一个测试位图
    const size_t bitmap_size = 16; // 128位
    uint8_t bitmap[bitmap_size];
    memset(bitmap, 0xFF, bitmap_size); // 初始全部设置为1（占用）
    
    // 设置一些空闲区域
    //  bit位次   hi    low
    //            v      v
    bitmap[1] = 0b00000001; 
    bitmap[2] = 0b00010000; 
    bitmap[3] = 0b11110000; 
    
    std::cout << "测试位图匹配函数:\n";

    size_t nums = atoll(argv[1]);
    
    // 测试1: 查找4个连续0位（应该在字节1中找到）
    size_t result = findConsecutiveZeros(bitmap, bitmap_size * 8, nums);
    std::cout << (int)result << " (期望: 8)\n";
    
    return 0;
}

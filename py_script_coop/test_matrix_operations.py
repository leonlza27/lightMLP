#!/usr/bin/env python3
"""
测试脚本用于验证 matrix_qfloat_add 和 matrix_qfloat_mulpty 函数
使用已编译的 libmatrixtest.so 库和numpy进行结果验证
"""

import cffi
import numpy as np
import os
import sys

ffi = cffi.FFI()

gcs = []

ffi.cdef("""
typedef struct matrix_bp_data{
    int32_t *data;
    uint16_t rows/*行数*/,cols/*列数*/;
}matrix_bp_data;

void quickcheck_mmul(const matrix_bp_data *mmul1, const matrix_bp_data *mmul2, matrix_bp_data *rawresu, matrix_bp_data *actualresu);

void matrix_bp_mulpty(const matrix_bp_data *mmul1, const matrix_bp_data *mmul2, matrix_bp_data *resu);

void *malloc(size_t size);
void free(void* ptr);

""")

# 加载共享库
lib_path = os.path.join(os.path.dirname(__file__), '..', "build",'src_edgedev', 'libpymatrixtest.so')
try:
    lib = ffi.dlopen(lib_path)
except Exception as e:
    print(f"无法加载库 {lib_path}: {e}")
    sys.exit(1)

# 定义 qfloat 类型 - 根据 CustomConf.h 配置
# 默认使用 32位 q16.16 定点数
QSHIFT = 16

# 辅助函数：浮点数转qfloat（定点数）
def float_to_qfix(f):
    """将浮点数转换为q16.16定点数"""
    return int(f * (1 << QSHIFT))

# 辅助函数：qfloat转浮点数
def qfix_to_float(q):
    """将q16.16定点数转换为浮点数"""
    return q / (1 << QSHIFT)

def ConstructNumpyToMatrixBP(rows :int, cols :int, rawNumpy):
    if rawNumpy is not None:
        m_bp16 = np.vectorize(float_to_qfix)(rawNumpy).flatten("C")

        cbuf = m_bp16
    else:
        cbuf = np.array([0 for _ in range(rows*cols)],dtype = np.int32)

    structret = ffi.new("matrix_bp_data*")
    structret.rows = rows
    structret.cols = cols
    cbuf_cast = ffi.cast("int32_t*",lib.malloc(ffi.sizeof("int32_t") * rows * cols))
    if cbuf_cast == ffi.NULL:
        raise MemoryError("malloc fail")
    for i in range(rows * cols):
        cbuf_cast[i] = cbuf[i]
    
    structret.data = cbuf_cast
    
    gcs.append(ffi.gc(cbuf_cast,lib.free))
    return structret
    
def check(tg, raw, excepted):
    print("diffs(data from c - numpy exception)")
    resucast = []
    rawresucast = []
    exceptcast = excepted.flatten("C")
    for i in range(len(exceptcast)):     
        resucast.append(qfix_to_float(tg[i]))
        rawresucast.append(qfix_to_float(raw[i]))
   
    resu = np.array(resucast)
    rawresu = np.array(rawresucast)
    diff_sys = rawresu - exceptcast
    diff = resu - exceptcast
    
    print(f"[sys_diff] aver:{np.std(diff_sys)}\nmax:{np.max(diff_sys)}\nmin:{np.min(diff_sys)}\nvar:{np.var(diff_sys)}")
    print(f"[actual_diff] aver:{np.std(diff)}\nmax:{np.max(diff)}\nmin:{np.min(diff)}\nvar:{np.var(diff)}")


def test_add():
    m,k,n=10,30,5

    testA = np.random.rand(m, k)
    testB = np.random.rand(k, n)

    exceptedC = np.dot(testA ,testB)
    
    cmatA = ConstructNumpyToMatrixBP(m,k,testA)
    cmatB = ConstructNumpyToMatrixBP(k,n,testB)
    cmatC = ConstructNumpyToMatrixBP(m,n,None)
    cmatC_raw = ConstructNumpyToMatrixBP(m,n,None)
    
    print(cmatC.data,cmatC_raw.data)
    
    print("main test fun")
    lib.quickcheck_mmul(cmatA, cmatB, cmatC_raw, cmatC)

    check(cmatC.data, cmatC_raw.data, exceptedC)
    pass

if __name__ == "__main__" :
    test_add()
    pass
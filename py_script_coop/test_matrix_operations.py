#!/usr/bin/env python3
"""
测试脚本用于验证 matrix_qfloat_add 和 matrix_qfloat_mulpty 函数
使用已编译的 libmatrixtest.so 库和numpy进行结果验证
"""

import ctypes
import numpy as np
import os
import sys

# 加载共享库
lib_path = os.path.join(os.path.dirname(__file__), '..', "build",'releases_for_MCU', 'libpymatrixtest.so')
try:
    lib = ctypes.CDLL(lib_path)
except Exception as e:
    print(f"无法加载库 {lib_path}: {e}")
    sys.exit(1)

# 定义 qfloat 类型 - 根据 CustomConf.h 配置
# 默认使用 32位 q16.16 定点数
QSHIFT = 16
qfloat = ctypes.c_int32

# 定义 matrix_qfloat_data 结构 - 与C代码中的结构匹配
class MatrixQFloatData(ctypes.Structure):
    _fields_ = [
        ("rows", ctypes.c_uint16),    # 行数
        ("cols", ctypes.c_uint16),    # 列数
        ("data", ctypes.POINTER(qfloat))  # 数据指针
    ]

# 定义函数原型 - 确保参数类型正确
lib.matrix_qfloat_add.argtypes = [
    ctypes.POINTER(MatrixQFloatData),  # madd1 - 指向矩阵结构的指针
    ctypes.POINTER(MatrixQFloatData),  # madd2 - 指向矩阵结构的指针
    ctypes.POINTER(MatrixQFloatData)   # resu - 指向结果矩阵结构的指针
]
lib.matrix_qfloat_add.restype = None

lib.matrix_qfloat_mulpty.argtypes = [
    ctypes.POINTER(MatrixQFloatData),  # mmul1 - 指向矩阵结构的指针
    ctypes.POINTER(MatrixQFloatData),  # mmul2 - 指向矩阵结构的指针
    ctypes.POINTER(MatrixQFloatData)   # resu - 指向结果矩阵结构的指针
]
lib.matrix_qfloat_mulpty.restype = None

lib.matrix_qfloat_init.argtypes = [
    ctypes.POINTER(MatrixQFloatData),  # matrix - 指向矩阵结构的指针
    ctypes.c_uint16,                   # m - 行数
    ctypes.c_uint16,                   # n - 列数
    ctypes.POINTER(qfloat)             # data - 指向qfloat数据的指针
]
lib.matrix_qfloat_init.restype = None

# 辅助函数：浮点数转qfloat（定点数）
def float_to_qfloat(f):
    """将浮点数转换为q16.16定点数"""
    return int(round(f * (1 << QSHIFT)))

# 辅助函数：qfloat转浮点数
def qfloat_to_float(q):
    """将q16.16定点数转换为浮点数"""
    return q / (1 << QSHIFT)

def ConstructNumpyToMatrixQF(rows :int, cols :int, rawNumpy):
    if rawNumpy is not None:
        m_bp16 = np.vectorize(float_to_qfloat)(rawNumpy).flatten("C").tolist()

        cbuf = (qfloat * (rows * cols))(*m_bp16)
    else:
        cbuf = (qfloat * (rows * cols))(*[0 for i in range(rows*cols)])

    structret = MatrixQFloatData()
    structret.rows = ctypes.c_uint16(rows)
    structret.cols = ctypes.c_uint16(cols)
    structret.data = cbuf
    structret._data = cbuf
    return structret
    
def check(tg, excepted):
    resucast = []
    exceptcast = excepted.flatten("C")
    for i in range(len(exceptcast)):     
        resucast.append(qfloat_to_float(tg[i]))
   
    resu = np.array(resucast)
    diff = np.abs(resu - exceptcast)
    
    print(f"aver:{np.std(diff)}\nmax:{np.max(diff)}\nmin:{np.min(diff)}")
      
    
        
        
    

def test_add():
    m,k,n=10,30,5

    testA = np.random.rand(m, k)
    testB = np.random.rand(k, n)

    exceptedC = np.dot(testA ,testB)


    cmatA = ConstructNumpyToMatrixQF(m, k, testA)
    cmatB = ConstructNumpyToMatrixQF(k, n, testB)
    cmatC = ConstructNumpyToMatrixQF(m, n, None)
    
    lib.matrix_qfloat_mulpty(cmatA, cmatB, cmatC)

    check(cmatC.data,exceptedC)
    pass

if __name__ == "__main__" :
    test_add()
    pass
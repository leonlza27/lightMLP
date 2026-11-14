"""``core.arrop.mt``：多线程数组算子（仅 ``-DMULTI_PROCESS`` 构建存在）。

参数、返回值与容量要求与 :mod:`core.arrop.norm` 完全一致，区别仅在核心
用多线程池执行。所有函数都不校验数组容量，返回传入的 ``ret`` 本身。
"""

from . import arr as arr

__all__ = ["vadd", "vsub", "vscale", "vemul", "mmul", "mtpose"]


def vadd(in1: arr, in2: arr, ret: arr, /) -> arr:
    """逐元素相加，元素个数取 ``len(in1)``；需 ``len(in2)``、
    ``len(ret)`` 均不小于 ``len(in1)``（不校验）。返回 ``ret``。"""
    ...


def vsub(in1: arr, in2: arr, ret: arr, /) -> arr:
    """逐元素相减，元素个数取 ``len(in1)``；需 ``len(in2)``、
    ``len(ret)`` 均不小于 ``len(in1)``（不校验）。返回 ``ret``。"""
    ...


def vscale(in1: arr, scaler: float, ret: arr, /) -> arr:
    """逐元素缩放，``scaler`` 为 float；需 ``len(ret) >= len(in1)``
    （不校验）。返回 ``ret``。"""
    ...


def vemul(in1: arr, in2: arr, ret: arr, /) -> arr:
    """逐元素相乘，元素个数取 ``len(in1)``；需 ``len(in2)``、
    ``len(ret)`` 均不小于 ``len(in1)``（不校验）。返回 ``ret``。"""
    ...


def mmul(in1: arr, in2: arr, m: int, k: int, n: int, ret: arr, /) -> arr:
    """矩阵相乘 ``(m x k) @ (k x n) -> ret(m x n)``；``m/k/n`` 为 uint16。

    需 ``len(in1) >= m*k``、``len(in2) >= k*n``、``len(ret) >= m*n``
    （均不校验）。返回 ``ret``。
    """
    ...


def mtpose(in1: arr, m: int, n: int, ret: arr, /) -> arr:
    """矩阵转置 ``m x n -> n x m``；``m/n`` 为 uint16。

    需 ``len(in1) >= m*n``、``len(ret) >= m*n``（不校验）。返回 ``ret``。
    """
    ...

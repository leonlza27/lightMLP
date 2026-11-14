"""``core.arrop.norm``：单线程数组算子。

所有函数都以 ``in1``/``in2`` 的**裸数据指针**直接计算，绑定层不校验任何
数组容量，越界即堆越界读写；调用方必须保证 ``ret``（以及 ``in2``）足够大。
函数返回传入的 ``ret`` 本身（C 侧 ``Py_INCREF(ret)`` 后返回）。
``m``/``k``/``n`` 用 C 格式 ``H`` 接收，超出 0..65535 会按 2**16 取模且
不报错。
"""

from . import arr as arr

__all__ = ["vadd", "vsub", "vscale", "vemul", "mmul", "mtpose"]


def vadd(in1: arr, in2: arr, ret: arr, /) -> arr:
    """逐元素相加：``ret[i] = in1[i] + in2[i]``，元素个数取 ``len(in1)``。

    要求 ``len(in2) >= len(in1)`` 且 ``len(ret) >= len(in1)``（不校验）。
    返回 ``ret``。
    """
    ...


def vsub(in1: arr, in2: arr, ret: arr, /) -> arr:
    """逐元素相减：``ret[i] = in1[i] - in2[i]``，元素个数取 ``len(in1)``。

    要求 ``len(in2) >= len(in1)`` 且 ``len(ret) >= len(in1)``（不校验）。
    返回 ``ret``。
    """
    ...


def vscale(in1: arr, scaler: float, ret: arr, /) -> arr:
    """逐元素缩放：``ret[i] = in1[i] * scaler``，元素个数取 ``len(in1)``。

    ``scaler`` 按 C 格式 ``"d"`` 解析（int/float/实现了 ``__float__`` 的
    对象均可），并量化为定点数。要求 ``len(ret) >= len(in1)``（不校验）。
    返回 ``ret``。
    """
    ...


def vemul(in1: arr, in2: arr, ret: arr, /) -> arr:
    """逐元素相乘：``ret[i] = in1[i] * in2[i]``，元素个数取 ``len(in1)``。

    要求 ``len(in2) >= len(in1)`` 且 ``len(ret) >= len(in1)``（不校验）。
    返回 ``ret``。
    """
    ...


def mmul(in1: arr, in2: arr, m: int, k: int, n: int, ret: arr, /) -> arr:
    """矩阵相乘：``(m x k) @ (k x n) -> (m x n)``。

    Args:
        m, k, n: 均为 ``uint16``（0..65535），分别表示 in1 行数、公共维、
            in2 列数。

    要求 ``len(in1) >= m*k``、``len(in2) >= k*n``、``len(ret) >= m*n``
    （均不校验）。返回 ``ret``。
    """
    ...


def mtpose(in1: arr, m: int, n: int, ret: arr, /) -> arr:
    """矩阵转置：把 ``in1`` 的 ``m x n`` 转成 ``ret`` 的 ``n x m``。

    Args:
        m, n: 均为 ``uint16``（0..65535），in1 的行数与列数。

    要求 ``len(in1) >= m*n``、``len(ret) >= m*n``（不校验）。返回 ``ret``。
    """
    ...

"""``core.arrop``：定点数数组容器与按元素/矩阵算子。

``arr`` 是绑定层自定义的 PyTypeObject；``norm``/``mt`` 是分别挂在
``core.arrop`` 下的子模块，均需通过点号导入访问::

    import core
    a = core.arrop.arr(8)
    out = core.arrop.arr(8)
    import core.arrop.norm
    core.arrop.norm.vadd(a, a, out)
"""

from collections.abc import Iterator
from typing import overload

from . import mt as mt
from . import norm as norm

__all__ = ["arr", "norm", "mt"]


class arr:
    """定长定点数数组（元素类型由构建选项决定：Q16.16 或 Q8.8）。

    支持序列/映射协议::

        len(arr)          -> 容量
        arr[i]            -> float（支持负索引）
        arr[i] = x        -> 写入（x 为 int/float 或可被 float() 转换）
        arr[a:b:c]        -> list[float]（新列表，不是视图）
        for v in arr      -> 按索引迭代到 IndexError 为止

    数组容量在构造时固定；``core.arrop.norm``/``core.arrop.mt`` 的算子不会
    检查任何数组容量，``core.convops`` 则只检查容量、不检查形状，调用方
    必须自行保证（见各算子说明）。

    整数索引/写入值与 ``fromlist`` 元素都用 CPython 的 ``PyLong_Check``
    判断：``bool`` 可当 int，其它整数类型（如 numpy 整数）不保证被接受。
    """

    def __init__(self, size: int, /) -> None:
        """分配 ``size`` 个定点数元素（初值未定义，读取前应先写入）。

        Args:
            size: C 格式 ``"I"``，即 ``unsigned int``。按无符号掩码转换：
                负数与超大整数会按 2**32 取模（不报 ``OverflowError``），
                因此正常使用应传 1..2**32-1。

        Raises:
            SystemError: ``size == 0``（C 侧直接返回 NULL 且未设置异常），
                或内部 ``malloc`` 失败（同样未设置异常）。
            TypeError: ``size`` 不是整数（或未实现 ``__index__``）。
            MemoryError: CPython 对象分配失败。
        """
        ...

    def __len__(self) -> int:
        """返回数组容量（构造时的 ``size``，与已写入元素个数无关）。"""
        ...

    def __iter__(self) -> Iterator[float]:
        """按 ``arr[0]``、``arr[1]``... 迭代，越界时由 ``IndexError`` 终止。"""
        ...

    @overload
    def __getitem__(self, key: int, /) -> float:
        """读取单个元素；支持负索引，越界抛 ``IndexError``。"""
        ...

    @overload
    def __getitem__(self, key: slice, /) -> list[float]:
        """读取切片，返回 ``list[float]``（新列表）；步长可为负。

        ``key`` 既不是 int 也不是 slice 时抛 ``TypeError``。
        """
        ...

    def __setitem__(self, key: int, value: float, /) -> None:
        """写入单个元素。

        Args:
            key: 只接受 int（支持负索引）；越界抛 ``IndexError``，非 int
                抛 ``TypeError``。
            value: float/int 或任何可被 ``float()`` 转换的对象；无法转换时
                抛 ``TypeError``。写入值会量化为定点数。

        Raises:
            TypeError: 不支持切片赋值，也不支持删除（``del arr[i]``）。
        """
        ...

    def fromlist(self, values: list[float | int], /) -> None:
        """把列表元素写入数组开头。

        C 侧要求恰好一个 ``list`` 对象（``"O!"`` + ``PyList_Type``），
        传入 tuple 或其它序列会抛 ``TypeError``。

        Args:
            values: 元素必须是 int 或 float（bool 按 int 处理）；其它类型
                抛 ``TypeError``。

        Raises:
            ValueError: ``len(values) > len(self)``。

        Note:
            只覆盖前 ``len(values)`` 个元素，其余容量保持原值；整数元素经
            ``PyLong_AsLong`` 转换，需能放入 C ``long``。
        """
        ...

    def tolist(self, /) -> list[float]:
        """返回容量大小的新 ``list[float]``（包含未赋值部分）。"""
        ...

"""``core.mlp.mt``：多线程前向、反向与梯度采集。

仅 ``-DMULTI_PROCESS`` 构建存在。参数、返回值、异常与
:mod:`core.mlp.norm` 完全一致，区别只在核心使用线程池执行。
"""

from typing import overload

from ..arrop import arr
from . import mlpexec
from . import mlptrain

__all__ = ["execute", "backward", "collect_grad", "upgrade_from_collected"]


@overload
def execute(model: mlpexec, arrin: arr, arrout: arr = ..., /) -> arr:
    """多线程执行一次前向计算。

    ``model`` 必须是精确的 ``mlpexec`` 或 ``mlptrain``（采集器不可用）；
    ``arrin`` 容量需 >= 第一层 ``in_dim``；``arrout`` 可选，容量需
    >= 最后一层 ``out_dim``。返回输出数组（传入的数组或新建数组）。

    Raises:
        TypeError: ``model`` 类型不符。
        ValueError: 输入/输出数组过小，或 ``model`` 是采集器。
    """
    ...


@overload
def execute(model: mlptrain, arrin: arr, arrout: arr = ..., /) -> arr:
    """训练态前向（采集器不可用）；参数、返回与异常同 ``mlpexec`` 版本。"""
    ...


def backward(model: mlptrain, arrin: arr, lr: float, /) -> None:
    """多线程反向传播；``arrin`` 容量需 >= 最后一层 ``out_dim``。

    会直接更新 ``model`` 所引用 ``modeldef`` 的 weights/bias。

    Raises:
        ValueError: ``model`` 是采集器，或 ``arrin`` 容量不足。
    """
    ...


def collect_grad(src: mlptrain, dest: mlptrain, lr: float, /) -> None:
    """多线程版梯度采集：``dest.grad += lr * src.grad``。

    ``dest`` 必须是由 ``mlptrain(netdef, True)`` 创建的采集器，否则
    ``ValueError``；``src`` 与 ``dest`` 必须描述同一网络结构。
    """
    ...


def upgrade_from_collected(src: mlptrain, dest: mlptrain, /) -> None:
    """多线程版梯度回写：``dest.modeldef.weights += src.grad``。

    ``src`` 必须是采集器、``dest`` 必须是普通模型，否则 ``ValueError``；
    两者必须描述同一网络结构。
    """
    ...

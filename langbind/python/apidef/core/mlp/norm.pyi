"""``core.mlp.norm``：单线程前向、反向与梯度采集。

模块级函数与类型对象分离：``execute``/``backward`` 等是模块函数，
``mlptrain.finalgrads`` 等实例方法在 :mod:`core.mlp` 中声明。
"""

from typing import overload

from ..arrop import arr
from . import mlpexec
from . import mlptrain

__all__ = ["execute", "backward", "collect_grad", "upgrade_from_collected"]


@overload
def execute(model: mlpexec, arrin: arr, arrout: arr = ..., /) -> arr:
    """执行一次前向计算。

    Args:
        model: 必须是**精确**的 ``core.mlp.mlpexec`` 或
            ``core.mlp.mlptrain``（子类也会 ``TypeError``）；采集器
            （``iscollector()`` 为 True）不可用。
        arrin: 输入数组，容量需 >= 第一层的 ``in_dim``。
        arrout: 可选输出数组。省略时按最后一层 ``out_dim`` 新建；
            提供时容量需 >= ``out_dim``，只覆盖前 ``out_dim`` 个元素，
            其余保持不变。

    Returns:
        输出数组（传入的 ``arrout`` 本身，或新建的数组）。

    Raises:
        TypeError: ``model`` 不是上述两种精确类型。
        ValueError: 输入数组过小、``model`` 是采集器，或输出数组过小。

    Note:
        输出长度取最后一层的 ``out_dim``；本函数不检查数组形状，只按
        层配置的维度读写。
    """
    ...


@overload
def execute(model: mlptrain, arrin: arr, arrout: arr = ..., /) -> arr:
    """训练态前向（采集器不可用）；参数、返回与异常同 ``mlpexec`` 版本。"""
    ...


def backward(model: mlptrain, arrin: arr, lr: float, /) -> None:
    """执行一次反向传播并更新权重。

    Args:
        model: ``core.mlp.mlptrain``；采集器会抛 ``ValueError``。
        arrin: 最后一层输出的梯度，容量需 >= 最后一层的 ``out_dim``。
        lr: 学习率，按 C 格式 ``"d"`` 解析后量化为定点数。

    Raises:
        ValueError: ``model`` 是采集器，或 ``arrin`` 容量不足。

    Note:
        反向会直接累加修改 ``model`` 所引用的 ``modeldef`` 的 weights/bias。
    """
    ...


def collect_grad(
    src: mlptrain,
    dest: mlptrain,
    lr: float,
    /,
) -> None:
    """把一个模型的梯度按学习率累加到梯度采集器。

    计算形式为 ``dest.grad += lr * src.grad``（逐层、逐元素）。

    Args:
        src: 梯度来源，通常是对同结构模型执行过 ``backward`` 的
            ``mlptrain``。
        dest: 目标采集器，必须由 ``mlptrain(netdef, True)`` 创建；否则
            ``ValueError``。
        lr: 学习率（float），与 ``src`` 已有梯度相乘后累加。

    Raises:
        ValueError: ``dest`` 不是采集器。

    Note:
        绑定层不核对 ``src`` 与 ``dest`` 的层数/维度是否一致，二者必须
        描述同一网络结构，否则会越界访问。
    """
    ...


def upgrade_from_collected(src: mlptrain, dest: mlptrain, /) -> None:
    """把采集器累计的梯度一次性加到目标模型的权重上。

    计算形式为 ``dest.modeldef.weights += src.grad``（bias 同理）。

    Args:
        src: 梯度采集器（``iscollector()`` 为 True）。
        dest: 普通可训练模型（非采集器）。

    Raises:
        ValueError: ``src`` 不是采集器或 ``dest`` 是采集器。

    Note:
        绑定层不核对两者层数/维度是否一致；``src`` 中累计的梯度已包含
        ``collect_grad`` 时传入的学习率。
    """
    ...

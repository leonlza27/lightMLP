"""``core.mlp``：MLP 模型定义、训练/推理状态与算子。

模块内容::

    core.mlp.modeldef   层配置容器（运行时 tp_name 为 core.mlp.netdef）
    core.mlp.mlptrain   可训练状态（或梯度采集器）
    core.mlp.mlpexec    纯推理状态
    core.mlp.ioops      模型构造/权重写入/序列化
    core.mlp.norm       单线程 execute/backward/collect_grad/upgrade_from_collected
    core.mlp.mt         多线程版本（仅 -DMULTI_PROCESS 构建存在）

生命周期要求（重要）:
    ``mlptrain``/``mlpexec`` 只保存 ``modeldef`` 内部 ``netLyrConf`` 数组的
    裸指针，既**不复制**层配置/权重，也不持有 ``modeldef`` 的 Python 引用。
    因此 ``modeldef`` 必须在这两个对象存活期间一直存活；训练会原地修改
    ``modeldef`` 里的 weights/bias，所有共享同一 ``modeldef`` 的实例都会
    看到这些修改。
"""

from ..arrop import arr
from . import ioops as ioops
from . import mt as mt
from . import norm as norm

__all__ = ["modeldef", "mlptrain", "mlpexec", "ioops", "norm", "mt"]


class modeldef:
    """网络层配置容器。

    应通过 :func:`core.mlp.ioops.constructmodel` 或
    :func:`core.mlp.ioops.loadmodel` 创建并填充，再用
    :func:`core.mlp.ioops.set_wb` 写入权重/偏置。

    Note:
        直接 ``modeldef()`` 只能得到一个空模型（层数为 0），把空模型交给
        ``mlptrain``/``mlpexec`` 会在执行时访问空指针；C 侧构造函数不解析
        任何参数，位置参数与关键字参数都会被静默忽略。
    """

    def __init__(self, /) -> None:
        """创建空模型（不接受参数；运行时传参也会被忽略）。"""
        ...


class mlptrain:
    """可训练模型状态；或作为梯度采集器（collector）使用。

    Args:
        netdef: 层配置，必须来自 ``core.mlp.modeldef``。对象只引用其内部
            C 数组，调用方需保证 ``netdef`` 比本对象活得久。
        ascollector: 可选，仅接受 ``bool`` 或 ``int``（其它类型
            ``TypeError``）。``True`` 或非 0 整数表示创建梯度采集器：
            它不能执行 ``execute``/``backward``/``finalgrads``，只能作为
            ``collect_grad`` 的目标累加梯度。

    Raises:
        TypeError: 首参不是 ``core.mlp.modeldef``，或 ``ascollector``
            既不是 bool 也不是 int。
        SystemError: 内部 ``malloc`` 失败（C 侧未设置异常）。

    Note:
        构造函数只解析位置参数，关键字参数会被静默忽略：例如
        ``mlptrain(m, ascollector=True)`` 实际创建的是普通模型且不报错。
        请始终按位置传参。
    """

    def __init__(
        self,
        netdef: modeldef,
        ascollector: bool | int = ...,
        /,
    ) -> None:
        ...

    def iscollector(self, /) -> bool:
        """是否为梯度采集器（``ascollector`` 为 True/非 0）。"""
        ...

    def finalgrads(self, dest: arr = ..., /) -> arr:
        """取出最近一次 ``backward`` 计算出的输入层梯度。

        梯度长度等于第一层的 ``in_dim``；必须在 ``backward`` 之后调用，
        之前内容未定义。

        Args:
            dest: 可选目标数组；省略时新建一个长度为 ``in_dim`` 的 ``arr``。
                提供时容量需 >= ``in_dim``（多余部分不变）。

        Returns:
            写入梯度后的数组（``dest`` 本身或新建的数组）。

        Raises:
            ValueError: 本对象是梯度采集器，或 ``dest`` 容量不足。
            SystemError: 省略 ``dest`` 时内部 ``malloc`` 失败（C 侧未设置
                异常）。

        Note:
            调用 ``collect_grad``/``upgrade_from_collected`` 时使用的模型
            相关约束见 :mod:`core.mlp.norm` 对应函数。
        """
        ...


class mlpexec:
    """纯推理模型状态（由 ``modeldef`` 建立，只支持 ``execute``）。

    Args:
        netdef: 层配置，必须来自 ``core.mlp.modeldef``；本对象只引用其内部
            C 数组，调用方需保证 ``netdef`` 比本对象活得久。

    Raises:
        TypeError: ``netdef`` 不是 ``core.mlp.modeldef``。
        SystemError: 内部 ``malloc`` 失败（C 侧未设置异常）。

    Note:
        构造函数只解析位置参数；``netdef`` 必须按位置传入，写成关键字会
        因位置参数缺失而 ``TypeError``。
    """

    def __init__(self, netdef: modeldef, /) -> None:
        ...

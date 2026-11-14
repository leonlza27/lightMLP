"""lightMLP 的 CPython 扩展模块 ``core`` 的类型声明。

模块层次与 ``bind_src/pyinclude_entry.c`` 的 ``PyInit_core`` 一一对应::

    core
    |-- arrop                容器子模块
    |   |-- arr              PyTypeObject arr_handle_def
    |   |-- norm             单线程算子模块
    |   `-- mt               多线程算子模块（仅 -DMULTI_PROCESS 构建存在）
    |-- convops              卷积/池化算子模块
    `-- mlp                  容器子模块
        |-- modeldef         PyTypeObject modeldef_handle_def
        |-- mlptrain         PyTypeObject mlptrain_pyhandle_def
        |-- mlpexec          PyTypeObject mlpexec_pyhandle_def
        |-- ioops            模型定义与读写
        |-- norm             单线程前向/反向
        `-- mt               多线程前向/反向（仅 -DMULTI_PROCESS 构建存在）

约定（由 C 侧实现决定，调用方必须遵守）:

* ``core`` 运行时是扩展模块（``core.pyd``/``core.so``），没有 ``__path__``；
  本目录只是供类型检查器使用的 stub 包，不参与运行时导入。
* 绝大多数函数、方法与类型构造都由 ``METH_VARARGS``/``tp_new`` +
  ``PyArg_ParseTuple`` 解析，**只读取位置参数**；因此 stub 中全部参数
  都标记为仅限位置参数（``/``）。普通函数用关键字调用会 ``TypeError``；
  少数 ``tp_new`` 实现会静默忽略关键字参数（见各类说明）。
* C 格式 ``b`` 对应 ``unsigned char``（0..255，越界抛 ``OverflowError``）；
  ``H``/``I`` 对应 ``uint16_t``/``unsigned int``，采用无符号掩码截断：
  负数或超范围值会按 2**16/2**32 取模且**不报错**，调用方需自行限制范围。
* 字符串格式 ``s`` 接受 ``str``（按 UTF-8 传给 C）或只读 bytes-like
  （如 ``bytes``），不接受 ``None``。
* 数组元素在写入时按定点格式量化（默认 Q16.16；``BP_Q8p8_Acquire``
  构建为 Q8.8），写入 ``float`` 会有精度损失。
"""

from . import arrop as arrop
from . import convops as convops
from . import mlp as mlp

__all__ = ["arrop", "convops", "mlp", "threadpool_recofig"]


def threadpool_recofig(
    new_tpools_count: int,
    new_flush_delay_ms: int,
    new_cooldown_cycles: int,
    /,
) -> None:
    """热重配置全局线程池（仅多线程构建 ``-DMULTI_PROCESS`` 存在）。

    参数按 C 格式 ``"HHb"`` 解析:

    Args:
        new_tpools_count: 线程池数量，``uint16``（名义范围 0..65535；
            超出范围按 2**16 取模）。
        new_flush_delay_ms: 每次刷新的延迟毫秒数，``uint16``（同上）。
        new_cooldown_cycles: 空闲线程回收周期，C 格式 ``"b"`` 即
            ``unsigned char``（0..255），越界抛 ``OverflowError``。

    Raises:
        TypeError: 参数个数/类型不符。
        OverflowError: ``new_cooldown_cycles`` 超出 0..255。
        AttributeError: 单线程构建中该函数不存在。

    Note:
        导出名沿用历史拼写 ``recofig``（不是 ``reconfig``），实现文件名为
        ``mt_tpool_reconfig.c``；C 侧忽略底层返回值，总是返回 ``None``。
    """
    ...

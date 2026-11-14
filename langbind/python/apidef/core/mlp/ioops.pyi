"""``core.mlp.ioops``：模型定义、权重写入与序列化。

这些函数是创建 ``core.mlp.modeldef`` 的唯一正常途径。绑定层用
``PyArg_ParseTuple`` 解析参数，不支持关键字参数。
"""

from . import modeldef

__all__ = ["constructmodel", "set_wb", "dumpmodel", "loadmodel"]


def constructmodel(layers: list[tuple[int, int, int, float]], /) -> modeldef:
    """按层配置新建并分配权重/偏置的模型定义。

    Args:
        layers: 非空列表，每个元素必须是**精确**的 ``tuple``（子类会被
            拒绝），格式为 ``(in_dim, out_dim, actp, dexa)``:

            * ``in_dim`` / ``out_dim``: ``uint16``，层的输入/输出维度；
              正常应 >= 1（绑定层只检查层间连续性，不检查 0）。
              第 2 层起 ``in_dim`` 必须等于上一层的 ``out_dim``。
            * ``actp``: 激活函数编号，``uint8``（0..255），含义见
              ``mlpCCore/mlp/builtin_ac_callback.h``。
            * ``dexa``: 激活函数附加参数（float），按定点数量化。

    Returns:
        新的 ``modeldef``；每层的 ``weights`` 分配 ``in_dim * out_dim`` 个
        元素、``bias`` 分配 ``out_dim`` 个元素，**均未初始化**，需用
        :func:`set_wb` 写入。

    Raises:
        TypeError: 首参不是 list、列表元素不是 tuple，或元组元素类型不符。
        OverflowError: ``actp`` 超出 ``unsigned char`` 的 0..255
            （``in_dim``/``out_dim`` 用 ``H`` 掩码转换，超范围会取模）。
        ValueError: 列表为空、层间维度不连续，或内存分配失败
            （C 侧统一报 "bad layer config or out of memory"）。
    """
    ...


def set_wb(
    model: modeldef,
    wb: list[tuple[list[float | int], list[float | int]]],
    /,
) -> None:
    """把权重和偏置写入模型定义。

    Args:
        model: 目标 ``core.mlp.modeldef``。
        wb: 列表，第 i 项对应第 i 层，元素必须是**精确**的 ``tuple``
            ``(weights, bias)``（tuple 子类会被拒绝），两项都必须是
            ``list``（list 子类可用），元素为 int 或 float（bool 按 int
            处理），写入时量化为定点数。

    Note:
        * 只处理 ``min(len(wb), 模型的层数)`` 层；多余的项被忽略。
        * 每层的 ``weights``/``bias`` 只取 ``min(len(list), in_dim*out_dim)``
          与 ``min(len(list), out_dim)`` 个元素；列表过短时只更新前缀，
          其余保持原值。
        * 这是原地修改：已创建的 ``mlptrain``/``mlpexec`` 共享同一份
          层配置，会立即看到新权重。

    Raises:
        TypeError: 首参不是 ``modeldef``，或 ``wb``/层项/权重列表的类型
            不符合 ``PyArg_ParseTuple`` 的格式要求。
        SystemError: C 侧某些分支直接返回 NULL 而未设置异常
            （如层项不是 tuple、元素既不是 int 也不是 float）。
    """
    ...


def dumpmodel(
    model: modeldef,
    path: str | bytes,
    dump_font: str | bytes = ...,
    dump_name: str | bytes = ...,
    /,
) -> None:
    """把模型保存到文件。

    两种模式（由可选参数组合决定）:

    * 省略 ``dump_font``：二进制格式，写入 ``path``。
    * 提供 ``dump_font``：必须同时提供 ``dump_name``，且
      ``dump_font == "csrc"``，把模型导出为 C 头文件 ``path``，
      数组/符号名使用 ``dump_name``。

    Args:
        model: 要保存的 ``core.mlp.modeldef``。
        path: 目标文件路径；``str`` 按 UTF-8 传给 C，也接受只读
            bytes-like（如 ``bytes``）。
        dump_font: 导出格式，目前仅支持 ``"csrc"``。
        dump_name: 导出为 C 头文件时的模型符号名。

    Raises:
        TypeError: 参数类型不符（路径/格式/名称不接受 ``None``）。
        SystemError: 只给了 ``dump_font`` 没给 ``dump_name``，或底层
            写文件失败（这些分支返回 NULL 但未设置 Python 异常）。

    Note:
        同时给了 ``dump_font`` 和 ``dump_name`` 但 ``dump_font`` 不是
        ``"csrc"`` 时，绑定层会静默返回 ``None``（不写文件）。
    """
    ...


def loadmodel(binpath: str | bytes, /) -> modeldef:
    """从二进制模型文件读取模型定义。

    Args:
        binpath: 文件路径；``str`` 按 UTF-8 传给 C，也接受只读 bytes-like。

    Returns:
        已填充层配置和权重的 ``modeldef``。

    Raises:
        TypeError: ``binpath`` 不是 ``str``/只读 bytes-like。
        SystemError: 文件不存在、格式非法或读取失败——C 侧返回 NULL
            但未设置 Python 异常，CPython 会报 ``SystemError``。
    """
    ...

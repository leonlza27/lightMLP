"""``core.convops``：卷积/池化的前向与反向算子。

绑定层只做参数解析和必要的安全检查；所有数组都以「裸指针 + 尺寸」传给
核心，``arr`` 本身不携带形状信息，因此尺寸必须由调用方通过
``(rows, cols)`` 二元组显式给出。所有函数返回 ``None``。

通用约束:

* 尺寸二元组必须是 ``tuple`` 且恰好两个整数元素；步长、``expand`` 等 C 侧
  按 ``uint16``（格式 ``H``）接收，超出 0..65535 会按 2**16 取模且不报错。
* ``pool_size``、步长必须 >= 1，否则 ``ValueError``（核心会除零）。
* 除特别说明外，绑定层会核对输入/输出 ``arr`` 的**容量**，不足时
  ``ValueError``；但不会检查同一数组在多个角色间的别名问题。
* 直接作为参数的 ``arr`` 允许子类；而 ``list`` 中的数组要求**精确**为
  ``core.arrop.arr``（子类会 ``TypeError``）。
"""

from .arrop import arr as arr

__all__ = [
    "averpool",
    "maxpool",
    "averpool_grad",
    "maxpool_grad",
    "padding",
    "cutedges",
    "divelem",
    "mergeelem",
]


def averpool(
    input: arr,
    input_size: tuple[int, int],
    pool_size: tuple[int, int],
    step_rows: int,
    step_cols: int,
    out: arr,
    /,
) -> None:
    """平均池化前向。

    Args:
        input: 输入数据，容量需 >= ``rows_in * cols_in``。
        input_size: ``(rows_in, cols_in)``。
        pool_size: ``(rows_pool, cols_pool)``，均需 >= 1 且不得超过输入尺寸。
        step_rows, step_cols: 步长，需 >= 1。
        out: 输出，容量需 >= ``rows_out * cols_out``，其中
            ``rows_out = (rows_in - rows_pool) // step_rows + 1``，
            ``cols_out`` 同理；核心要求输出区进入时已清零，绑定层会先清零。

    Raises:
        ValueError: 池核或步长为 0、池核大于输入、数组容量不足。
    """
    ...


def maxpool(
    input: arr,
    input_size: tuple[int, int],
    pool_size: tuple[int, int],
    step_rows: int,
    step_cols: int,
    out: arr,
    idx_max: arr,
    /,
) -> None:
    """最大池化前向，同时输出每个池窗最大值在核内的线性下标。

    Args:
        input: 输入数据，容量需 >= ``rows_in * cols_in``。
        input_size: ``(rows_in, cols_in)``。
        pool_size: ``(rows_pool, cols_pool)``，均 >= 1 且不超过输入尺寸。
        step_rows, step_cols: 步长，均 >= 1。
        out: 池化结果，容量需 >= ``rows_out * cols_out``。
        idx_max: 最大值的核内线性下标（以定点数保存的整数），容量需
            >= ``rows_out * cols_out``。

    Raises:
        ValueError: 池核/步长非法、池核大于输入、数组容量不足。
    """
    ...


def averpool_grad(
    input: arr,
    input_size: tuple[int, int],
    pool_size: tuple[int, int],
    step_rows: int,
    step_cols: int,
    out: arr,
    /,
) -> None:
    """平均池化反向（把梯度平摊回原输入）。

    Args:
        input: 前向池化输出的梯度，容量需 >= ``rows_out * cols_out``
            （``rows_out``/``cols_out`` 按前向公式计算）。
        input_size: 前向时的原输入尺寸 ``(rows_in, cols_in)``。
        pool_size: 前向时的 ``(rows_pool, cols_pool)``，均 >= 1。
        step_rows, step_cols: 前向步长，均 >= 1。
        out: 原输入尺寸的梯度，容量需 >= ``rows_in * cols_in``；
            核心会先清零该缓冲区。

    Raises:
        ValueError: 池核/步长非法、池核大于输入、数组容量不足。
    """
    ...


def maxpool_grad(
    input: arr,
    idx_max: arr,
    output_size: tuple[int, int],
    pool_size: tuple[int, int],
    step_rows: int,
    step_cols: int,
    out: arr,
    /,
) -> None:
    """最大池化反向（按前向记录的核内下标回填梯度）。

    Args:
        input: 前向池化输出尺寸的梯度，容量需 >= ``rows_out * cols_out``。
        idx_max: 前向 ``maxpool`` 写出的核内线性下标，容量需
            >= ``rows_out * cols_out``；每个值必须满足
            ``0 <= int(value) < rows_pool * cols_pool``。
        output_size: 前向时的原输入尺寸 ``(rows_in, cols_in)``。
        pool_size: 前向时的 ``(rows_pool, cols_pool)``，均 >= 1。
        step_rows, step_cols: 前向步长，均 >= 1。
        out: 原输入尺寸的梯度，容量需 >= ``rows_in * cols_in``。

    Raises:
        ValueError: 池核/步长非法、池核大于输入、``idx_max`` 越界、
            数组容量不足。
    """
    ...


def padding(
    input: arr,
    input_size: tuple[int, int],
    r_extend: int,
    c_extend: int,
    out: arr,
    /,
) -> None:
    """零填充：在输入四周各扩展 ``r_extend`` 行、``c_extend`` 列。

    Args:
        input: 输入数据，容量需 >= ``rows_in * cols_in``。
        input_size: ``(rows_in, cols_in)``。
        r_extend, c_extend: 上/下、左/右的扩展量，``uint16``。
        out: 输出，容量需 >= ``rows_out * cols_out``，其中
            ``rows_out = rows_in + 2 * r_extend``、
            ``cols_out = cols_in + 2 * c_extend``（均按 uint16 计算）。

    Note:
        ``rows_in + 2 * r_extend`` 在 C 侧以 ``uint16_t`` 保存，绑定层
        未检查溢出；极端参数下尺寸会回绕，调用方需自行保证不溢出。

    Raises:
        ValueError: 数组容量不足。
    """
    ...


def cutedges(
    input: arr,
    input_size: tuple[int, int],
    r_cut: int,
    c_cut: int,
    out: arr,
    /,
) -> None:
    """裁掉输入上下各 ``r_cut`` 行、左右各 ``c_cut`` 列。

    Args:
        input: 输入数据，容量需 >= ``rows_in * cols_in``。
        input_size: ``(rows_in, cols_in)``。
        r_cut, c_cut: 单侧裁剪量，``uint16``；要求
            ``2 * r_cut <= rows_in`` 且 ``2 * c_cut <= cols_in``。
        out: 输出，容量需 >= ``rows_out * cols_out``，其中
            ``rows_out = rows_in - 2 * r_cut``、``cols_out`` 同理。

    Raises:
        ValueError: 裁剪量超过输入的一半、数组容量不足。
    """
    ...


def divelem(
    input: arr,
    in_size: tuple[int, int],
    pool_size: tuple[int, int],
    step_rows: int,
    step_cols: int,
    expand: int,
    outs: list[arr],
    /,
) -> None:
    """按滑窗把输入切块，分别写入 ``outs`` 中的数组。

    Args:
        input: 输入数据，容量需 >= ``rows_in * cols_in``。
        in_size: ``(rows_in, cols_in)``。
        pool_size: ``(rows_pool, cols_pool)``，均 >= 1。
        step_rows, step_cols: 滑窗步长，均 >= 1。
        expand: 扩张量，``uint16``；``scale = expand + 1``，
            需保证 ``rows_pool * scale <= rows_in + expand``、
            ``cols_pool * scale <= cols_in + expand``。
        outs: 输出块列表，长度需 >= ``arr_size``（块数按核心公式计算）；
            前 ``arr_size`` 个元素必须是**精确**的 ``core.arrop.arr``，
            且每个容量 >= ``rows_pool * cols_pool``。多余元素被忽略。

    Raises:
        ValueError: 池核/步长非法、窗口覆盖超过输入、输入容量不足、
            列表过短或块数组容量不足。
        TypeError: 列表元素不是精确的 ``core.arrop.arr``。
    """
    ...


def mergeelem(
    ins: list[arr],
    out_size: tuple[int, int],
    pool_size: tuple[int, int],
    step_rows: int,
    step_cols: int,
    expand: int,
    out: arr,
    /,
) -> None:
    """按滑窗把多个块累加合并到输出（``divelem`` 的逆过程）。

    Args:
        ins: 输入块列表，长度需 >= ``arr_size``（块数按核心公式计算）；
            前 ``arr_size`` 个元素必须是**精确**的 ``core.arrop.arr``，
            且每个容量 >= ``rows_pool * cols_pool``。多余元素被忽略。
        out_size: ``(rows_to, cols_to)``。
        pool_size: ``(rows_pool, cols_pool)``，均 >= 1。
        step_rows, step_cols: 滑窗步长，均 >= 1。
        expand: 扩张量，``uint16``；需保证
            ``rows_pool * scale <= rows_to + expand``、
            ``cols_pool * scale <= cols_to + expand``（``scale = expand + 1``）。
        out: 输出，容量需 >= ``rows_to * cols_to``；核心是累加写入，
            绑定层会先把整个输出区清零。

    Raises:
        ValueError: 池核/步长非法、窗口覆盖超过输出、列表过短、
            块数组或输出容量不足。
        TypeError: 列表元素不是精确的 ``core.arrop.arr``。
    """
    ...

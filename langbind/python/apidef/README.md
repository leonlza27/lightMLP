# `core` Python 类型声明（stub）

本目录存放 `langbind/python/bind_src` 中 CPython 绑定的 `.pyi` 声明，供
Pylance/pyright、mypy 等类型检查器使用；**不参与运行时导入**。运行时仍使用
构建产物 `core.pyd`/`core.so`。

## 目录结构

```
core/
├── __init__.pyi          PyInit_core 顶层模块（arrop / convops / mlp）
├── convops.pyi           convop_topy.c
├── arrop/
│   ├── __init__.pyi      arr_handle_def（arr 类型）
│   ├── norm.pyi          normal/arrop_topy.c 单线程算子
│   └── mt.pyi            multi_processed/arrop_topy.c 多线程算子
└── mlp/
    ├── __init__.pyi      modeldef / mlptrain / mlpexec 类型
    ├── ioops.pyi         normal/mlp_modelop.c
    ├── norm.pyi          normal/mlp_calc_topy.c
    └── mt.pyi            multi_processed/mlp_calc_topy.c
```

## 使用方式

- Pylance/pyright：把 `langbind/python/apidef` 加入 `python.analysis.extraPaths`。
- mypy：`MYPYPATH=langbind/python/apidef mypy ...`。

## 构建变体

- `core.arrop.mt`、`core.mlp.mt` 与 `core.threadpool_recofig` 仅在
  `-DMULTI_PROCESS` 构建中存在；stub 为保持接口完整始终声明它们。
- `threadpool_recofig` 的拼写就是 C 侧导出名（历史拼写，不是 `reconfig`）。

## C 侧约定摘要

- 绑定函数只按位置解析（`PyArg_ParseTuple`），stub 全部参数使用 `/`
  标记仅限位置参数；少数 `tp_new`（如 `modeldef`）会静默忽略参数/关键字。
- `b` = `unsigned char`（0..255，越界 `OverflowError`）；`H`/`I` 按无符号
  掩码截断（超范围取模，不报错）；`s` 接受 `str` 或只读 bytes-like。
- `arr` 是定长容量；`core.arrop.norm`/`core.arrop.mt` 完全不校验容量，
  `core.convops` 只校验容量不校验形状；形状由调用方以 `(rows, cols)`
  二元组显式传入。
- `mlptrain`/`mlpexec` 不复制层配置、不持有 `modeldef` 的 Python 引用：
  `modeldef` 必须比它们活得久；训练会原地修改其 weights/bias。

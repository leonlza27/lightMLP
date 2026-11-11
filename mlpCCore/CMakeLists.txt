#config is for rich-hardware environment and with full function of lightMLP core
#if you need to use for the embdded or have moditifed one, just use the source code
#and use the cmake script carefully

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/pc_release/core)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/pc_release/core)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/pc_release/core)

#模型文件导出库(filedump.c 经由 mlp.h 间接引用 builtin_ac_callback.h 中的 Sigmoid/Tanh 等,
#这些 static 函数体内调用 exp_qfix, 故必须一并编译 qfix_ops/qfix_op.c)
add_library(lmlp_fioop SHARED mlp/filedump.c qfix_ops/qfix_op.c)

#基本运算
add_library(lmlpcore_basic SHARED mlp/mlp.c qfix_ops/qfix_op.c mlp/convop.c)
add_library(arrop_basic SHARED arrop/arrop.c)

#多线程版
add_library(shared_tpool SHARED threadpool/sharedpool.c threadpool/threadpool.c)
add_library(arrop_mt SHARED arrop/arrop.c)
find_package(Threads REQUIRED)

add_library(lmlpcore_mt SHARED mlp/mlp.c mlp/convop.c mlp/mlp_process_mt.c qfix_ops/qfix_op.c)
target_compile_definitions(arrop_mt PRIVATE MULTI_PROCESS)
target_link_libraries(lmlpcore_mt PRIVATE shared_tpool)
#arrop_mt 的 MULTI_PROCESS 分支使用了 shared_tpool 提供的 pubpools_*/tp_sleep_ms
target_link_libraries(arrop_mt PRIVATE shared_tpool)
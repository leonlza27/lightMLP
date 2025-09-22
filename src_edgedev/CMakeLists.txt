project(release_objs_for_MCU LANGUAGES C CXX)

# 建立测试库
add_library(pymatrixtest SHARED matrix/matrix_static.c qfix_ops/qfix_op.c)

add_definitions(-DON_DBG)

add_compile_options(-g)
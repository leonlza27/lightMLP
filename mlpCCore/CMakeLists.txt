#config is for rich-hardware environment and with full function of lightMLP core
#if you need to use for the embdded or have moditifed one, just use the source code
#and use the cmake script carefully

add_library(lmlpcore  SHARED mlp/mlp.c qfix_ops/qfix_op.c)
add_library(lmlp_fioop SHARED mlp/filedump.c)
add_library(matrix_static SHARED matrix/matrix_static.c qfix_ops/qfix_op.c)

target_link_libraries(lmlp_fioop PRIVATE matrix_static)
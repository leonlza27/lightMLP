#config is for rich-hardware environment and with full function of lightMLP core
#if you need to use for the embdded or have moditifed one, just use the source code
#and use the cmake script carefully

enable_language(C)

add_library(matrix_static SHARED matrix_static.c ../qfix_ops/qfix_op.c)

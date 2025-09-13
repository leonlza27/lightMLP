project(release_objs_for_MCU LANGUAGES C CXX)

# Build moduledebug executable
add_library(pymatrixtest SHARED matrix/matrix_static.c qfloat_operation/qfloat_op.c)

add_compile_options(-g)
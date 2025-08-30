project(release_objs_for_MCU LANGUAGES C CXX)


find_package(jsoncpp REQUIRED)
# Build moduledebug executable
add_executable(moduledebug 
    moduledebug.cpp
    matrix/matrix_static.c
    qfloat_operation/qfloat_op.c
    mlp/activator.cpp)

target_link_libraries(moduledebug PRIVATE jsoncpp)

add_compile_options(-g)
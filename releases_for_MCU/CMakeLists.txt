cmake_minimum_required(VERSION 3.10)
project(arfnet)

if(GENTYPE STREQUAL "spectest")
    set(SOURCES f_q16_16_operation/f_q16_16_op.c moudledebug.cpp)
    add_compile_definitions(-D__ON_DBG)
    add_compile_options(-g)
else()
    set(SOURCES matrix/matrix.cpp mlp/mlp.cpp matrix_test.cpp)
    
endif()

add_executable(test ${SOURCES})
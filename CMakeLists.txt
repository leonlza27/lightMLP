#config is for rich-hardware environment and with full function of lightMLP core
#if you need to use for the embdded or have moditifed one, just use the source code
#and use the cmake script carefully

cmake_minimum_required(VERSION 3.30.5)

project(lightMLP)

#源码均为UTF-8(含中文注释), MSVC需显式指定编码, 否则在非UTF-8代码页下会误解析
if(MSVC)
    add_compile_options(/utf-8)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/__out__)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/__out__)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/__out__)

ADD_SUBDIRECTORY(mlpCCore)
ADD_SUBDIRECTORY(langbind)

#for embdded platform source export
file(COPY mlpCCore/arrop DESTINATION ${CMAKE_BINARY_DIR}/embdded_include)
file(COPY mlpCCore/mlp DESTINATION ${CMAKE_BINARY_DIR}/embdded_include)
file(COPY mlpCCore/qfix_ops DESTINATION ${CMAKE_BINARY_DIR}/embdded_include)
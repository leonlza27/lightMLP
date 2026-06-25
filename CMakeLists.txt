#config is for rich-hardware environment and with full function of lightMLP core
#if you need to use for the embdded or have moditifed one, just use the source code
#and use the cmake script carefully

cmake_minimum_required(VERSION 3.30.5)

project(lightMLP)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/__out__)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/__out__)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/__out__)

ADD_SUBDIRECTORY(mlpCCore)
ADD_SUBDIRECTORY(langbind)

file(COPY install.py DESTINATION ${CMAKE_BINARY_DIR}/__out__)
find_program(Python_EXE python NO_CMAKE_FIND_ROOT_PATH)
set(Python_ARGS ${CMAKE_BINARY_DIR}/__out__/install.py)
execute_process(COMMAND ${python_EXE} ${Python_ARGS})

#for embdded platform source export
file(COPY mlpCCore/matrix DESTINATION ${CMAKE_BINARY_DIR}/__out__/lmlp)
file(COPY mlpCCore/mlp DESTINATION ${CMAKE_BINARY_DIR}/__out__/lmlp)
file(COPY mlpCCore/qfix_ops DESTINATION ${CMAKE_BINARY_DIR}/__out__/lmlp)
file(COPY mlpCCore/tools DESTINATION ${CMAKE_BINARY_DIR}/__out__/lmlp)

#for pc/phone(native) include
file(COPY mlpCCore/matrix/matrix_static.h DESTINATION ${CMAKE_BINARY_DIR}/__out__/include/matrix)
file(COPY mlpCCore/mlp/mlp.h DESTINATION ${CMAKE_BINARY_DIR}/__out__/include/mlp)
file(COPY mlpCCore/mlp/ac_callback.h DESTINATION ${CMAKE_BINARY_DIR}/__out__/include/mlp)
file(COPY mlpCCore/mlp/filedump.h DESTINATION ${CMAKE_BINARY_DIR}/__out__/include/mlp)
file(COPY mlpCCore/qfix_ops/qfix_op.h DESTINATION ${CMAKE_BINARY_DIR}/__out__/include/qfix_ops)
file(COPY mlpCCore/tools/mult_proc.h DESTINATION ${CMAKE_BINARY_DIR}/__out__/include/tools)
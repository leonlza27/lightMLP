#config is for rich-hardware environment and with full function of lightMLP core
#if you need to use for the embdded or have moditifed one, just use the source code
#and use the cmake script carefully


find_package(Python 3.12 COMPONENTS Development REQUIRED)


link_directories(${PROJECT_BINARY_DIR}/mlpCCore)

add_library(mbpbuffer SHARED python/mbpbuffer.c)
add_library(libcorepy MODULE python/open2py.c)
add_library(libmbp16d MODULE python/matrixbp2py.c python/mbpbuffer.c)

target_include_directories(libcorepy PRIVATE ${PROJECT_SOURCE_DIR})
target_include_directories(libmbp16d PRIVATE ${PROJECT_SOURCE_DIR})
target_include_directories(mbpbuffer PRIVATE ${PROJECT_SOURCE_DIR})

target_link_libraries(mbpbuffer PRIVATE matrix_static Python::Python)
target_link_libraries(libmbp16d PRIVATE mbpbuffer matrix_static Python::Python)
target_link_libraries(libcorepy PRIVATE mbpbuffer matrix_static lmlpcore lmlp_fioop Python::Python)

set_target_properties(libcorepy PROPERTIES SUFFIX ".pyd")
set_target_properties(libmbp16d PROPERTIES SUFFIX ".pyd")

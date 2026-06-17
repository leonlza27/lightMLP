find_package(Python 3.12 COMPONENTS Development REQUIRED)

link_directories(${PROJECT_BINARY_DIR}/mlpCCore)

add_library(mbpbuffer SHARED mbpbuffer.c)
add_library(libcorepy MODULE open2py.c)
add_library(libmbp16d MODULE matrixbp2py.c mbpbuffer.c)

target_include_directories(libcorepy PRIVATE ${PROJECT_SOURCE_DIR})
target_include_directories(libmbp16d PRIVATE ${PROJECT_SOURCE_DIR})
target_include_directories(mbpbuffer PRIVATE ${PROJECT_SOURCE_DIR})

target_link_libraries(mbpbuffer PRIVATE matrix_static Python::Python)
target_link_libraries(libmbp16d PRIVATE mbpbuffer matrix_static Python::Python)
target_link_libraries(libcorepy PRIVATE mbpbuffer matrix_static lmlpcore lmlp_fioop Python::Python)


IF(WIN32)
    set_target_properties(libcorepy PROPERTIES SUFFIX ".pyd")
    set_target_properties(libmbp16d PROPERTIES SUFFIX ".pyd")
ENDIF()

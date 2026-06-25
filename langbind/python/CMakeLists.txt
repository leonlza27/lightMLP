find_package(Python 3.12 COMPONENTS Development REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/__out__/py_bind)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/__out__/py_bind)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/__out__/py_bind)

link_directories(${PROJECT_BINARY_DIR}/__out__/core)

add_library(mbpbuffer SHARED mbpbuffer.c)
add_library(corepy MODULE open2py.c)
add_library(mbp16dpy MODULE matrixbp2py.c mbpbuffer.c)

target_include_directories(corepy PRIVATE ${PROJECT_SOURCE_DIR})
target_include_directories(mbp16dpy PRIVATE ${PROJECT_SOURCE_DIR})
target_include_directories(mbpbuffer PRIVATE ${PROJECT_SOURCE_DIR})

target_link_libraries(mbpbuffer PRIVATE matrix_static Python::Python)
target_link_libraries(mbp16dpy PRIVATE mbpbuffer matrix_static Python::Python)
target_link_libraries(corepy PRIVATE mbpbuffer matrix_static lmlpcore lmlp_fioop Python::Python)


IF(WIN32)
    set_target_properties(corepy PROPERTIES SUFFIX ".pyd")
    set_target_properties(mbp16dpy PROPERTIES SUFFIX ".pyd")
ENDIF()

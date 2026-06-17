# 基础Java运行时检测
find_package(Java 1.7 REQUIRED)

# 完整开发环境检测（含编译器和文档工具）
find_package(Java COMPONENTS Development JarSigner REQUIRED)

find_package(JNI COMPONENTS AWT JVM REQUIRED)

# 验证JNI环境变量
message(STATUS "JNI头文件路径: ${JNI_INCLUDE_DIRS}")
message(STATUS "JNI库文件: ${JNI_LIBRARIES}")

set(JAVA_INCLUDE $ENV{JAVA_HOME}/include)

IF (WIN32)
    set(JAVA_INCLUDE_OS $ENV{JAVA_HOME}/include/win32)
ELSEIF (UNIX)
    set(JAVA_INCLUDE_OS $ENV{JAVA_HOME}/include/linux)
ELSEIF (APPLE)
    set(JAVA_INCLUDE_OS $ENV{JAVA_HOME}/include/darwin)
ENDIF ()

add_library(mbp_tojava SHARED mbp/mbpops_tojava.c mbp/mbpstruct_tojava.c)
add_library(lmlpcore_tojava SHARED core/mlpcore_tojava.c core/netdef_tojava.c)

target_include_directories(mbp_tojava PRIVATE ${PROJECT_SOURCE_DIR} ${JAVA_INCLUDE} ${JAVA_INCLUDE_OS})
target_include_directories(lmlpcore_tojava PRIVATE ${PROJECT_SOURCE_DIR} ${JAVA_INCLUDE} ${JAVA_INCLUDE_OS})

target_link_libraries(mbp_tojava PRIVATE matrix_static)
target_link_libraries(lmlpcore_tojava PRIVATE matrix_static lmlpcore lmlp_fioop)
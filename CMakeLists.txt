#config is for rich-hardware environment and with full function of lightMLP core
#if you need to use for the embdded or have moditifed one, just use the source code
#and use the cmake script carefully
project(test)

include_directories("D:\\LibericaJDK-17\\include\\win32" "D:\\LibericaJDK-17\\include")

set(CMAKE_C_CLANG_TIDY clang-tidy;
    -ID:\\LibericaJDK-17\\include\\win32,
        -ID:\\LibericaJDK-17\\include;
)

ADD_SUBDIRECTORY(mlpCCore)
ADD_SUBDIRECTORY(langbind)

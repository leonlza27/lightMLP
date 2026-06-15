project(lmlpjava)

include_directories("D:\\LibericaJDK-17\\include\\win32" "D:\\LibericaJDK-17\\include")

add_library(lmlpcore_tojava SHARED "core/netdef_tojava.c")
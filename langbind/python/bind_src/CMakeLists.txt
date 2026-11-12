#配置得分两版
#单线程
add_library(core SHARED pyinclude_entry.c normal/arrop_topy.c)
target_link_libraries(core PRIVATE lmlpcore_basic arrop_basic)

#多线程,得开-MULTI_PROCESS宏
add_library(core SHARED pyinclude_entry.c normal/arrop_topy.c multi_processed/arrop_topy.c)
target_compile_definitions(core PRIVATE MULTI_PROCESS)
target_link_libraries(core PRIVATE lmlpcore_mt arrop_mt)
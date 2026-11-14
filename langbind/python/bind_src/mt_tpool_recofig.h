#ifndef mt_tpool_recofig_h
#define mt_tpool_recofig_h

#include "PythonBind_CustomConf.h"

//core 的线程池重配置方法表: 仅多线程版(-DMULTI_PROCESS)挂到 core 模块上,
//由 pyinclude_entry.c 构造 core_moduledef 时引用; 实现见 mt_tpool_reconfig.c
//(注: 头文件名沿用历史拼写 recofig, 实现文件名为 reconfig)
extern PyMethodDef pool_reconfig[];

#endif

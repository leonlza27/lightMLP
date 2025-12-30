#ifndef _matrix_static_ref_cpp
#define _matrix_static_ref_cpp
//  源/src_edgedev/matrix/matrix_static.h, 本项目用于在cpp的转发(对照测试用)
namespace mbp_static{
//__cpp_spefic:(强制)启用cpp名称修饰,需要用cpp编译器编译那部分c源码
#define __cpp_spefic
#include "../../JustInferCore/matrix/matrix_static.h"

}

//常用数据类型展开

using matrix_bp = mbp_static::matrix_bp;
using qfix = mbp_static::qfix;

#endif
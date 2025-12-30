#ifndef _lightmlp_core_infer_
#define _lightmlp_core_infer_

//这里就是纯嵌入式推理引擎,满足嵌入式(裸机)编译器最小cpp支持
//为了编译通过,你最好这样做:modelRef = (lightmlp_ref::NetLyrConf*)model
//其中 model 为 lightmlp::NetLyrConf*类型
//这样做只是让类型检查通过,其实两个命名空间所包含的NetLyrConf结构完全相同 :)

namespace lightmlp_inf{
    #include "JustInferCore/mlp/mlp.h"
}

#endif
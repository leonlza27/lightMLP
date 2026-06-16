#include "core_mlpexec.h"
#include "../jnibind_customconf.h"
#include "mlpCCore/matrix/matrix_static.h"
#include "mlpCCore/mlp/mlp.h"

//mlptrain

JNIEXPORT void JNICALL Java_core_mlptrain_setuptrainer(JNIEnv *env, jobject _this, jobject netdef_java){
    jclass mlptrain_jdef = idx_javaclass(javaclasspath_mlpexec);
    jclass netdef_jdef = idx_javaclass(javaclasspath_netdef);

    jfieldID mlptrain_obnetdef = jnienvcall->GetFieldID(env, mlptrain_jdef, "netsrc", "Lcore/netdef;");
    jfieldID mlptrain_exechandle = jnienvcall->GetFieldID(env, mlptrain_jdef, "exec_exahandle", "J");

    jfieldID netdef_netlyrdef = jnienvcall->GetFieldID(env, mlptrain_jdef, "_addr", "J");
    jfieldID netdef_netlen = jnienvcall->GetFieldID(env, netdef_jdef, "netlen", "I");

    int netlen = jnienvcall->GetIntField(env, netdef_java, netdef_netlen);
    jlong _modelsrc = jnienvcall->GetLongField(env, netdef_java, netdef_netlyrdef);
    jnienvcall->SetObjectField(env, _this, mlptrain_obnetdef, netdef_java);
    mlpTrainStatus *statnew;
    mlptrainer_setup(netlen, _reinterpt(netLyrConf*, _modelsrc), statnew);

    jnienvcall->SetLongField(env, _this, mlptrain_exechandle, _reinterpt(jlong, statnew));
}

JNIEXPORT void JNICALL Java_core_mlptrain_setuptrainer_1asGradCollector(JNIEnv *env, jobject _this, jobject netdef_java){
    jclass mlptrain_jdef = idx_javaclass(javaclasspath_mlpexec);
    jclass netdef_jdef = idx_javaclass(javaclasspath_netdef);

    jfieldID mlptrain_obnetdef = jnienvcall->GetFieldID(env, mlptrain_jdef, "netsrc", "Lcore/netdef;");
    jfieldID mlptrain_exechandle = jnienvcall->GetFieldID(env, mlptrain_jdef, "exec_exahandle", "J");

    jfieldID netdef_netlyrdef = jnienvcall->GetFieldID(env, mlptrain_jdef, "_addr", "J");
    jfieldID netdef_netlen = jnienvcall->GetFieldID(env, netdef_jdef, "netlen", "I");

    int netlen = jnienvcall->GetIntField(env, netdef_java, netdef_netlen);
    jlong _modelsrc = jnienvcall->GetLongField(env, netdef_java, netdef_netlyrdef);
    jnienvcall->SetObjectField(env, _this, mlptrain_obnetdef, netdef_java);
    mlpTrainStatus *statnew;
    mlptrainer_totalgrads_cap_setup(netlen, _reinterpt(netLyrConf*, _modelsrc), statnew);

    jnienvcall->SetLongField(env, _this, mlptrain_exechandle, _reinterpt(jlong, statnew));
}

JNIEXPORT void JNICALL Java_core_mlptrain_dealloctrainer(JNIEnv *env, jobject _this){
    jclass mlptrain_jdef = idx_javaclass(javaclasspath_mlpexec);

    jfieldID mlptrain_exechandle = jnienvcall->GetFieldID(env, mlptrain_jdef, "exec_exahandle", "J");

    jlong _net = jnienvcall->GetLongField(env, _this, mlptrain_exechandle);
    mlpTrainStatus *net = _reinterpt(mlpTrainStatus*, _net);

    mlptrainer_cleanup(net);

    jnienvcall->SetLongField(env, _this, mlptrain_exechandle, 0);
}


JNIEXPORT void JNICALL Java_core_mlptrain_execute(JNIEnv *env, jobject _this, jobject vecino, jobject vecouto){
    jclass mlptrain_jdef = idx_javaclass(javaclasspath_mlpexec);
    jclass mbp_jdef = idx_javaclass(javaclasspath_mbp);

    jfieldID mlptrain_exechandle = jnienvcall->GetFieldID(env, mlptrain_jdef, "exec_exahandle", "J");
    jfieldID mbp_data_java_slot = jnienvcall->GetFieldID(env, mbp_jdef, "data", "[B");

    jlong _net = jnienvcall->GetLongField(env, _this, mlptrain_exechandle);
    mlpTrainStatus *net = _reinterpt(mlpTrainStatus*, _net);
    jboolean oncopy = 0;

    if(!net->fullConnData){
        jnienvcall->ThrowNew(env, idx_javaclass("java/lang/IllegalStateException"), "try to let gradcollector mlptrain run model, check via isGradSaver()\n");
        return;
    }

    matrix_bp vecin = (matrix_bp)jnienvcall->GetByteArrayElements(env, vecino, &oncopy);
    matrix_bp vecout = (matrix_bp)jnienvcall->GetByteArrayElements(env, vecouto, &oncopy);
    uint16_t outdim = net->modelsrc[net->calclyrs - 1].out_dim;

    mlptrainer_execute(net, vecin->data);
    qfix *resu = net->fullConnData[net->calclyrs], *dest = vecout->data;
    for(uint16_t i = 0; i < outdim; i++) dest[i] = resu[i];

    jnienvcall->ReleaseByteArrayElements(env, vecino, (jbyte*)vecin, 0);
    jnienvcall->ReleaseByteArrayElements(env, vecouto, (jbyte*)vecout, 0);
}

JNIEXPORT void JNICALL Java_core_mlptrain_backward(JNIEnv *env, jobject _this, jobject vecino, jfloat lr){
    jclass mlptrain_jdef = idx_javaclass(javaclasspath_mlpexec);
    jclass mbp_jdef = idx_javaclass(javaclasspath_mbp);

    jfieldID mlptrain_exechandle = jnienvcall->GetFieldID(env, mlptrain_jdef, "exec_exahandle", "J");
    jfieldID mbp_data_java_slot = jnienvcall->GetFieldID(env, mbp_jdef, "data", "[B");

    jlong _net = jnienvcall->GetLongField(env, _this, mlptrain_exechandle);
    mlpTrainStatus *net = _reinterpt(mlpTrainStatus*, _net);
    jboolean oncopy = 0;

    if(!net->fullConnData){
        jnienvcall->ThrowNew(env, idx_javaclass("java/lang/IllegalStateException"), "try to let gradcollector mlptrain update model, check via isGradSaver()\n");
        return;
    }

    matrix_bp vecin = (matrix_bp)jnienvcall->GetByteArrayElements(env, vecino, &oncopy);
    uint16_t outdim = net->modelsrc[net->calclyrs - 1].out_dim;

    mlptrainer_backward(net, vecin->data, float_to_qfix(lr));

    jnienvcall->ReleaseByteArrayElements(env, vecino, (jbyte*)vecin, 0);
}

JNIEXPORT void JNICALL Java_core_mlptrain_savegrads_1to(JNIEnv *env, jobject _this, jobject dest_mlptrain){
    jclass mlptrain_jdef = idx_javaclass(javaclasspath_mlpexec);

    jfieldID mlptrain_exechandle = jnienvcall->GetFieldID(env, mlptrain_jdef, "exec_exahandle", "J");

    jlong _net = jnienvcall->GetLongField(env, _this, mlptrain_exechandle);
    jlong _netto = jnienvcall->GetLongField(env, dest_mlptrain, mlptrain_exechandle);
    mlpTrainStatus *net = _reinterpt(mlpTrainStatus*, _net);
    mlpTrainStatus *netto = _reinterpt(mlpTrainStatus*, _netto);

    if(netto->fullConnData){
        jnienvcall->ThrowNew(env, idx_javaclass("java/lang/IllegalStateException"), "<arg gradscap>try to let mlptrain save total grads, check via isGradSaver()\n");
        return;
    }

    mlptrainer_totalgrads_savegrads(net, netto);
}

JNIEXPORT void JNICALL Java_core_mlptrain_backward_1from_1totalgrads(JNIEnv *env, jobject _this, jobject allgradso, jfloat lr){
    jclass mlptrain_jdef = idx_javaclass(javaclasspath_mlpexec);

    jfieldID mlptrain_exechandle = jnienvcall->GetFieldID(env, mlptrain_jdef, "exec_exahandle", "J");

    jlong _net = jnienvcall->GetLongField(env, _this, mlptrain_exechandle);
    jlong _netto = jnienvcall->GetLongField(env, allgradso, mlptrain_exechandle);
    mlpTrainStatus *net = _reinterpt(mlpTrainStatus*, _net);
    mlpTrainStatus *allgrads = _reinterpt(mlpTrainStatus*, _netto);

    if(!net->fullConnData){
        jnienvcall->ThrowNew(env, idx_javaclass("java/lang/IllegalStateException"), "try to let gradcollector mlptrain update model, check via isGradSaver()\n");
        return;
    }

    if(allgrads->fullConnData){
        jnienvcall->ThrowNew(env, idx_javaclass("java/lang/IllegalStateException"), "<arg gradscap>try to update mlptrain from a mlptrain, check via isGradSaver()\n");
        return;
    }

    mlptrainer_totalgrads_backward(net, allgrads, float_to_qfix(lr));
}

JNIEXPORT void JNICALL Java_core_mlptrain_get_1grad2last(JNIEnv *env, jobject _this, jobject vecouto){
    jclass mlptrain_jdef = idx_javaclass(javaclasspath_mlpexec);
    jclass mbp_jdef = idx_javaclass(javaclasspath_mbp);

    jfieldID mlptrain_exechandle = jnienvcall->GetFieldID(env, mlptrain_jdef, "exec_exahandle", "J");
    jfieldID mbp_data_java_slot = jnienvcall->GetFieldID(env, mbp_jdef, "data", "[B");

    jlong _net = jnienvcall->GetLongField(env, _this, mlptrain_exechandle);
    mlpTrainStatus *net = _reinterpt(mlpTrainStatus*, _net);
    jboolean oncopy = 0;

    uint32_t indim0 = net->modelsrc->in_dim;

    matrix_bp vecout = (matrix_bp)jnienvcall->GetByteArrayElements(env, vecouto, &oncopy);

    qfix *resu = net->lyrinput_grad[0], *dest = vecout->data;
    for(uint16_t i = 0; i < indim0; i++) dest[i] = resu[i];

    jnienvcall->ReleaseByteArrayElements(env, vecouto, (jbyte*)vecout, 0);
}

JNIEXPORT jboolean JNICALL Java_core_mlptrain_isGradSaver(JNIEnv *env, jobject _this){
    jclass mlptrain_jdef = idx_javaclass(javaclasspath_mlpexec);

    jfieldID mlptrain_exechandle = jnienvcall->GetFieldID(env, mlptrain_jdef, "exec_exahandle", "J");

    jlong _net = jnienvcall->GetLongField(env, _this, mlptrain_exechandle);
    mlpTrainStatus *net = _reinterpt(mlpTrainStatus*, _net);

    return (net->fullConnData)? 1 : 0;
}

//exec

JNIEXPORT void JNICALL Java_core_mlpexec_deallocexec(JNIEnv *env, jobject _this){
    jclass mlpexec_jdef = idx_javaclass(javaclasspath_mlpexec);

    jfieldID mlpexec_exechandle = jnienvcall->GetFieldID(env, mlpexec_jdef, "exec_exahandle", "J");
    jlong _net = jnienvcall->GetLongField(env, _this, mlpexec_exechandle);

    mlpexec_cleanup(_reinterpt(mlpExecStatus*, _net));

    jnienvcall->SetLongField(env, _this, mlpexec_exechandle, 0);
}

JNIEXPORT void JNICALL Java_core_mlpexec_setupexec(JNIEnv *env, jobject _this, jobject netdef_java){
    jclass mlpexec_jdef = idx_javaclass(javaclasspath_mlpexec);
    jclass netdef_jdef = idx_javaclass(javaclasspath_netdef);

    jfieldID mlpexec_obnetdef = jnienvcall->GetFieldID(env, mlpexec_jdef, "netsrc", "Lcore/netdef;");
    jfieldID mlpexec_exechandle = jnienvcall->GetFieldID(env, mlpexec_jdef, "exec_exahandle", "J");

    jfieldID netdef_netlyrdef = jnienvcall->GetFieldID(env, mlpexec_jdef, "_addr", "J");
    jfieldID netdef_netlen = jnienvcall->GetFieldID(env, netdef_jdef, "netlen", "I");

    int netlen = jnienvcall->GetIntField(env, netdef_java, netdef_netlen);
    jlong _modelsrc = jnienvcall->GetLongField(env, netdef_java, netdef_netlyrdef);
    jnienvcall->SetObjectField(env, _this, mlpexec_obnetdef, netdef_java);
    mlpExecStatus *statnew;
    mlpexec_setup(netlen, _reinterpt(netLyrConf*, _modelsrc), statnew);

    jnienvcall->SetLongField(env, _this, mlpexec_exechandle, _reinterpt(jlong, statnew));
}


JNIEXPORT void JNICALL Java_core_mlpexec_execute(JNIEnv *env, jobject _this, jobject vecino, jobject vecouto){
    jclass mlpexec_jdef = idx_javaclass(javaclasspath_mlpexec);
    jclass mbp_jdef = idx_javaclass(javaclasspath_mbp);

    jfieldID mlpexec_exechandle = jnienvcall->GetFieldID(env, mlpexec_jdef, "exec_exahandle", "J");
    jfieldID mbp_data_java_slot = jnienvcall->GetFieldID(env, mbp_jdef, "data", "[B");

    jlong _net = jnienvcall->GetLongField(env, _this, mlpexec_exechandle);
    mlpExecStatus *net = _reinterpt(mlpExecStatus*, _net);
    jboolean oncopy = 0;

    matrix_bp vecin = (matrix_bp)jnienvcall->GetByteArrayElements(env, vecino, &oncopy);
    matrix_bp vecout = (matrix_bp)jnienvcall->GetByteArrayElements(env, vecouto, &oncopy);
    uint16_t outdim = net->modelsrc[net->calclyrs - 1].out_dim;

    mlpexec_execute(net, vecin->data);
    qfix *resu = net->fullConnData_tmp[net->calclyrs % 2], *dest = vecout->data;
    for(uint16_t i = 0; i < outdim; i++) dest[i] = resu[i];

    jnienvcall->ReleaseByteArrayElements(env, vecino, (jbyte*)vecin, 0);
    jnienvcall->ReleaseByteArrayElements(env, vecouto, (jbyte*)vecout, 0);
}
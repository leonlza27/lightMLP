#include "core_netdef.h"
#include "jni.h"
#include "jni_md.h"
#include "mlpCCore/matrix/matrix_static.h"
#include "mlpCCore/mlp/mlp.h"
#include "mlpCCore/mlp/filedump.h"
#include <malloc.h>
#include <stdint.h>

#define jnienvcall (*env)
#define idx_javaclass(logpath) jnienvcall->FindClass(env, logpath)

#define _reinterpt(type, value) *(type*)&value

JNIEXPORT void JNICALL Java_core_netdef_setup_1ndef(JNIEnv *env, jobject _this, jobjectArray confin){
    jclass netdefjava = idx_javaclass("core/netdef");
    jclass netlyrjava = idx_javaclass("core/netlyr");

    uint32_t netlen = jnienvcall->GetArrayLength(env, confin);

    jfieldID netlyr_indi_slot = jnienvcall->GetFieldID(env, netlyrjava,  "indim", "I");
    jfieldID netlyr_outdi_slot = jnienvcall->GetFieldID(env, netlyrjava, "outdim", "I");
    jfieldID netlyr_actp_slot = jnienvcall->GetFieldID(env, netlyrjava, "_actp", "B");
    jfieldID netlyr_dexa_slot = jnienvcall->GetFieldID(env, netlyrjava, "dexa", "F");
    
    jfieldID netlen_j = jnienvcall->GetFieldID(env, netdefjava, "netlen", "I");
    jfieldID addr_of_ndef = jnienvcall->GetFieldID(env, netdefjava, "_addr", "J");

    netLyrConf *ret_addr = (netLyrConf*)malloc(sizeof(netLyrConf) * netlen);
    if(!ret_addr){
        jnienvcall->ThrowNew(env, idx_javaclass("java/lang/NullPointerException"), "can not allocte a memory for netdef at native");
        return;
    }
    uint32_t i_cut = 0;

    for(uint32_t i = 0; i < netlen; i++){
        jobject cur = jnienvcall->GetObjectArrayElement(env, confin, i);
        uint16_t indim = jnienvcall->GetIntField(env, cur, netlyr_indi_slot);
        uint16_t outdim = jnienvcall->GetIntField(env, cur, netlyr_outdi_slot);
        qfix *w_new = (qfix*)malloc(sizeof(qfix) * indim * outdim);
        if(!w_new){
            jnienvcall->ThrowNew(env, idx_javaclass("java/lang/NullPointerException"), "can not allocte a memory for weight of some layer at native");
            i_cut = i;
            goto _err_ret;
        }
        qfix *b_new = (qfix*)malloc(sizeof(qfix) * outdim);
        if(!b_new){
            jnienvcall->ThrowNew(env, idx_javaclass("java/lang/NullPointerException"), "can not allocte a memory for bias of some layer at native");
            i_cut = i;
            goto _err_ret;
        }
        ret_addr[i].in_dim = indim;
        ret_addr[i].out_dim = outdim;
        ret_addr[i].acTp = jnienvcall->GetByteField(env, cur, netlyr_actp_slot);
        ret_addr[i].dExtra = float_to_qfix(jnienvcall->GetFloatField(env, cur, netlyr_dexa_slot));
        ret_addr[i].weights = w_new;
        ret_addr[i].bias = b_new;
    }

    jnienvcall->SetIntField(env, _this, netlen_j, *((jint*)&netlen));
    jnienvcall->SetLongField(env, _this, addr_of_ndef, *((jlong*)&ret_addr));
    
    return;

_err_ret:
    for(uint32_t i = 0; i < i_cut; i++){
        free(ret_addr[i].weights);
        free(ret_addr[i].weights);
    }
    free(ret_addr);
    return;
}

JNIEXPORT void JNICALL Java_core_netdef_setup_1fromfile(JNIEnv *env, jobject _this, jstring modelpath){
    jclass netdefjava = idx_javaclass("core/netdef");
    const char *path = jnienvcall->GetStringUTFChars(env, modelpath, 0);
    if(!path){
        goto _model_cannot_load;
    }

    netLyrConf *model_imported;
    int netlen = loadmodel(path, &model_imported);
    if(netlen < 0){
        _model_cannot_load:
        jnienvcall->ThrowNew(env, idx_javaclass("java/lang/NullPointerException"), "can not get string or open the file");
        return;
    }

    jfieldID netlen_j = jnienvcall->GetFieldID(env, netdefjava, "netlen", "I");
    jfieldID addr_of_ndef = jnienvcall->GetFieldID(env, netdefjava, "_addr", "J");

    jnienvcall->SetIntField(env, _this, netlen_j, *((jint*)&netlen));
    jnienvcall->SetLongField(env, _this, addr_of_ndef, *((jlong*)&model_imported));
}

JNIEXPORT void JNICALL Java_core_netdef_freemodel(JNIEnv *env, jobject _this){
    jclass netdefjava = idx_javaclass("core/netdef");
    jfieldID addr_of_ndef = jnienvcall->GetFieldID(env, netdefjava, "_addr", "J");
    jlong addr_java = jnienvcall->GetLongField(env, _this, addr_of_ndef);
    netLyrConf *modelfree = *(netLyrConf**)&addr_java;
    jfieldID netlen_j = jnienvcall->GetFieldID(env, netdefjava, "netlen", "I");
    int netlen = jnienvcall->GetIntField(env, _this, netlen_j);
    for(int i = 0; i < netlen; i++){
        netLyrConf *lyrthis = &modelfree[i];
        free(lyrthis->bias);
        free(lyrthis->weights);
    }
    free(modelfree);
}

JNIEXPORT void JNICALL Java_core_netdef_savemodel(JNIEnv *env, jobject _this, jstring modelpath){
    jclass netdefjava = idx_javaclass("core/netdef");
    const char *path = jnienvcall->GetStringUTFChars(env, modelpath, 0);
    if(!path){
        goto _model_cannot_save;
    }
    jfieldID addr_of_ndef = jnienvcall->GetFieldID(env, netdefjava, "_addr", "J");
    jlong addr_java = jnienvcall->GetLongField(env, _this, addr_of_ndef);
    netLyrConf *modelsave = _reinterpt(netLyrConf*, addr_java);
    jfieldID netlen_j = jnienvcall->GetFieldID(env, netdefjava, "netlen", "I");
    int netlen = jnienvcall->GetIntField(env, _this, netlen_j);
    if(!savemodel(path, netlen, modelsave)){
        _model_cannot_save:
        jnienvcall->ThrowNew(env, idx_javaclass("java/lang/NullPointerException"), "can not get string or open&write to the file");
        return;
    }
}

#include "core_mlptrain.h"

JNIEXPORT void JNICALL Java_core_mlptrain_setuptrainer(JNIEnv *env, jobject _this, jobject netdef_java){
    jclass mlptrain_jdef = idx_javaclass("core/mlptrain");
    jclass netdef_jdef = idx_javaclass("core/netdef");

    jfieldID mlptrain_obnetdef = jnienvcall->GetFieldID(env, mlptrain_jdef, "netsrc", "Lcore/netdef;");
    jfieldID mlptrain_trainhandle = jnienvcall->GetFieldID(env, mlptrain_jdef, "train_exahandle", "J");

    jfieldID netdef_netlyrdef = jnienvcall->GetFieldID(env, mlptrain_jdef, "_addr", "J");
    jfieldID netdef_netlen = jnienvcall->GetFieldID(env, netdef_jdef, "netlen", "I");

    int netlen = jnienvcall->GetIntField(env, netdef_java, netdef_netlen);
    jlong _modelsrc = jnienvcall->GetLongField(env, netdef_java, netdef_netlyrdef);
    jnienvcall->SetObjectField(env, _this, mlptrain_obnetdef, netdef_java);
    mlpTrainStatus *statnew;
    mlptrainer_setup(netlen, _reinterpt(netLyrConf*, _modelsrc), statnew);

    jnienvcall->SetLongField(env, _this, mlptrain_trainhandle, _reinterpt(jlong, statnew));
}

JNIEXPORT void JNICALL Java_core_mlptrain_execute(JNIEnv *env, jobject _this, jobject vecino, jobject vecouto){
    jclass mlptrain_jdef = idx_javaclass("core/mlptrain");
    jclass mbp_jdef = idx_javaclass("mlib/matrixbp");

    jfieldID mlptrain_trainhandle = jnienvcall->GetFieldID(env, mlptrain_jdef, "train_exahandle", "J");
    jfieldID mbp_data_java_slot = jnienvcall->GetFieldID(env, mbp_jdef, "data", "[B");

    jlong _net = jnienvcall->GetLongField(env, _this, mlptrain_trainhandle);
    mlpTrainStatus *net = _reinterpt(mlpTrainStatus*, _net);
    jboolean oncopy = 0;

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
    jclass mlptrain_jdef = idx_javaclass("core/mlptrain");
    jclass mbp_jdef = idx_javaclass("mlib/matrixbp");

    jfieldID mlptrain_trainhandle = jnienvcall->GetFieldID(env, mlptrain_jdef, "train_exahandle", "J");
    jfieldID mbp_data_java_slot = jnienvcall->GetFieldID(env, mbp_jdef, "data", "[B");
    jlong _net = jnienvcall->GetLongField(env, _this, mlptrain_trainhandle);
    mlpTrainStatus *net = _reinterpt(mlpTrainStatus*, _net);
    jboolean oncopy = 0;

    matrix_bp vecin = (matrix_bp)jnienvcall->GetByteArrayElements(env, vecino, &oncopy);

    mlptrainer_backward(net, vecin->data, float_to_qfix(lr));

    jnienvcall->ReleaseByteArrayElements(env, vecino, (jbyte*)vecin, 0);
}

JNIEXPORT void JNICALL Java_core_mlptrain_dealloctrainer(JNIEnv *env, jobject _this){
    jclass mlptrain_jdef = idx_javaclass("core/mlptrain");

    jfieldID mlptrain_trainhandle = jnienvcall->GetFieldID(env, mlptrain_jdef, "train_exahandle", "J");
    jlong _net = jnienvcall->GetLongField(env, _this, mlptrain_trainhandle);

    mlptrainer_cleanup(_reinterpt(mlpTrainStatus*, _net));

    jnienvcall->SetLongField(env, _this, mlptrain_trainhandle, 0);
}

JNIEXPORT void JNICALL Java_core_mlptrain_get_1grad2last(JNIEnv *env, jobject _this, jobject vecouto){
    jclass mlptrain_jdef = idx_javaclass("core/mlptrain");
    jclass mbp_jdef = idx_javaclass("mlib/matrixbp");

    jfieldID mlptrain_trainhandle = jnienvcall->GetFieldID(env, mlptrain_jdef, "train_exahandle", "J");
    jfieldID mbp_data_java_slot = jnienvcall->GetFieldID(env, mbp_jdef, "data", "[B");

    jboolean oncopy = 0;
    matrix_bp vecout = (matrix_bp)jnienvcall->GetByteArrayElements(env, vecouto, &oncopy);
    jlong _net = jnienvcall->GetLongField(env, _this, mlptrain_trainhandle);
    mlpTrainStatus *net = _reinterpt(mlpTrainStatus*, _net);
    
    qfix *dest = vecout->data, *source = net->lyrinput_grad[0];
    uint16_t indim = net->modelsrc[0].in_dim;

    for(uint16_t i = 0; i < indim; i++) dest[i] = source[i];

    jnienvcall->ReleaseByteArrayElements(env, vecouto, (jbyte*)vecout, 0); 
}

#include "core_mlpexec.h"

JNIEXPORT void JNICALL Java_core_mlpexec_setupexec(JNIEnv *env, jobject _this, jobject netdef_java){
    jclass mlpexec_jdef = idx_javaclass("core/mlpexec");
    jclass netdef_jdef = idx_javaclass("core/netdef");

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
    jclass mlpexec_jdef = idx_javaclass("core/mlpexec");
    jclass mbp_jdef = idx_javaclass("mlib/matrixbp");

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

JNIEXPORT void JNICALL Java_core_mlpexec_deallocexec(JNIEnv *env, jobject _this){
    jclass mlpexec_jdef = idx_javaclass("core/mlpexec");

    jfieldID mlpexec_exechandle = jnienvcall->GetFieldID(env, mlpexec_jdef, "exec_exahandle", "J");
    jlong _net = jnienvcall->GetLongField(env, _this, mlpexec_exechandle);

    mlpexec_cleanup(_reinterpt(mlpExecStatus*, _net));

    jnienvcall->SetLongField(env, _this, mlpexec_exechandle, 0);
}
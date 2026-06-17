#include "core_netdef.h"
#include "mlpCCore/mlp/mlp.h"
#include "mlpCCore/mlp/filedump.h"
#include <malloc.h>
#include <stdint.h>
#include "../jnibind_customconf.h"

JNIEXPORT void JNICALL Java_core_netdef_setup_1ndef(JNIEnv *env, jobject _this, jobjectArray confin){
    jclass netdefjava = idx_javaclass(javaclasspath_netdef);
    jclass netlyrjava = idx_javaclass(javaclasspath_netlyr);

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
    //printf("addr malloced %p\n",ret_addr);
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

    jnienvcall->SetIntField(env, _this, netlen_j, _reinterpt(jint, netlen));
    jnienvcall->SetLongField(env, _this, addr_of_ndef, _reinterpt(jlong, ret_addr));
    //printf("netdef::_addr %p\n", addr_of_ndef);
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
    jclass netdefjava = idx_javaclass(javaclasspath_netdef);
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
    jclass netdefjava = idx_javaclass(javaclasspath_netdef);
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
    jclass netdefjava = idx_javaclass(javaclasspath_netdef);
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
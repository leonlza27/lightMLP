#include "core_netdef.h"
#include "mlpCCore/mlp/mlp.h"
#include <malloc.h>

#define jnienvcall (*env)
#define idx_javaclass(logpath) jnienvcall->FindClass(env, logpath)

JNIEXPORT void JNICALL Java_core_netdef_setup_1ndef(JNIEnv *env, jobject _this, jobjectArray confin){
    jclass netdefjava = idx_javaclass("core/netdef");
    jclass netlyrjava = idx_javaclass("core/netlyr");

    uint32_t netlen = jnienvcall->GetArrayLength(env, confin);

    jfieldID netlyr_indi_slot = jnienvcall->GetFieldID(env, netlyrjava,  "indim", "I");
    jfieldID netlyr_outdi_slot = jnienvcall->GetFieldID(env, netlyrjava, "outdim", "I");
    jfieldID netlyr_actp_slot = jnienvcall->GetFieldID(env, netlyrjava, "_actp", "B");
    jfieldID netlyr_dexa_slot = jnienvcall->GetFieldID(env, netlyrjava, "dexa", "F");
    
    jfieldID netlen_j = jnienvcall->GetFieldID(env, netdefjava, "netlen", "I");
    jfieldID addr_of_ndef = jnienvcall->GetFieldID(env, netdefjava, "addr", "J");

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


#include "mlib_matrixbp.h"
#include "mlpCCore/matrix/matrix_static.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define jnienvcall (*env)

JNIEXPORT void JNICALL Java_mlib_matrixbp_setupmbp(JNIEnv *env, jobject _this, jint rows, jint cols){
    jclass mbp_tpdef_java = jnienvcall->FindClass(env,"mlib/matrixbp");
    jfieldID mbp_data_java_slot = jnienvcall->GetFieldID(env, mbp_tpdef_java, "data", "[B");

    jbyteArray mbp_data_java = jnienvcall->NewByteArray(env, sizeof(matrix_bp_data) + sizeof(qfix) * rows * cols);

    jboolean oncopy = 0;
    matrix_bp dnew = (matrix_bp)jnienvcall->GetByteArrayElements(env, mbp_data_java, &oncopy);
    dnew->rows = rows;
    dnew->cols = cols;

    jnienvcall->ReleaseByteArrayElements(env, mbp_data_java, (jbyte*)dnew, 0);
    jnienvcall->SetObjectField(env, _this, mbp_data_java_slot, mbp_data_java);
}

JNIEXPORT jstring JNICALL Java_mlib_matrixbp_printmatrix(JNIEnv *env, jobject _this){
    jclass mbp_tpdef_java = jnienvcall->FindClass(env,"mlib/matrixbp");
    jfieldID mbp_data_java_slot = jnienvcall->GetFieldID(env, mbp_tpdef_java, "data", "[B");
    jbyteArray mbp_data_java = jnienvcall->GetObjectField(env, _this, mbp_data_java_slot);
    jboolean oncopy = 0;
    matrix_bp dpnt = (matrix_bp)jnienvcall->GetByteArrayElements(env, mbp_data_java, &oncopy);

    printf("[matrixbp at %d rows * %d cols]\n", dpnt->rows, dpnt->cols);

    for(uint16_t i = 0; i < dpnt->rows; i++){
        qfix *lncur = dpnt->data + i * dpnt->cols;
        putc('|', stdout);
        for(uint16_t j = 0; j < dpnt->cols; j++){
            printf(" %.4f,", qfix_to_float(lncur[j]));
        }
        putc('|', stdout);
        putc('\n', stdout);
    }

    return jnienvcall->NewStringUTF(env, "\n");
}

JNIEXPORT void JNICALL Java_mlib_matrixbp_fromlist(JNIEnv *env, jobject _this, jfloatArray lstin){
    jclass mbp_tpdef_java = jnienvcall->FindClass(env,"mlib/matrixbp");
    jfieldID mbp_data_java_slot = jnienvcall->GetFieldID(env, mbp_tpdef_java, "data", "[B");
    jbyteArray mbp_data_java = jnienvcall->GetObjectField(env, _this, mbp_data_java_slot);
    jboolean oncopy = 0;
    matrix_bp mdata = (matrix_bp)jnienvcall->GetByteArrayElements(env, mbp_data_java, &oncopy);
    
    uint32_t size = mdata->rows * mdata->cols;
    uint32_t size_ofin = jnienvcall->GetArrayLength(env, lstin);
    if(size_ofin < size){
        jnienvcall->ThrowNew(env,jnienvcall->FindClass(env, "java/lang/IllegalArgumentException"),"len of input list is less than this matrixbp required");
        return;
    }

    float *arrin = jnienvcall->GetFloatArrayElements(env, lstin, &oncopy);

    qfix *dst = mdata->data;
    for(uint32_t i = 0; i < size; i++) dst[i] = float_to_qfix(arrin[i]);

    jnienvcall->ReleaseByteArrayElements(env, mbp_data_java, (jbyte*)mdata, 0);
    jnienvcall->SetObjectField(env, _this, mbp_data_java_slot, mbp_data_java);
}

JNIEXPORT void JNICALL Java_mlib_matrixbp_fromrand(JNIEnv *env, jobject _this){
    jclass mbp_tpdef_java = jnienvcall->FindClass(env,"mlib/matrixbp");
    jfieldID mbp_data_java_slot = jnienvcall->GetFieldID(env, mbp_tpdef_java, "data", "[B");
    jbyteArray mbp_data_java = jnienvcall->GetObjectField(env, _this, mbp_data_java_slot);
    jboolean oncopy = 0;
    matrix_bp mdata = (matrix_bp)jnienvcall->GetByteArrayElements(env, mbp_data_java, &oncopy);
    
    srand(time(0));

    uint32_t size = mdata->rows * mdata->cols;
    qfix *dst = mdata->data;
    for(uint32_t i = 0; i < size; i++) dst[i] = rand() % UINT32_MAX;

    jnienvcall->ReleaseByteArrayElements(env, mbp_data_java, (jbyte*)mdata, 0);
    jnienvcall->SetObjectField(env, _this, mbp_data_java_slot, mbp_data_java);

}



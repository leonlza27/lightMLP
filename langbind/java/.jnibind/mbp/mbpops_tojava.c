#include "mlib_ops.h"
#include "mlpCCore/matrix/matrix_static.h"
#include "../jnibind_customconf.h"

JNIEXPORT void JNICALL Java_mlib_ops_madd(JNIEnv *env, jclass _nspace, jobject m1o, jobject m2o, jobject mro){
    jclass mbp_tpdef_java = jnienvcall->FindClass(env,javaclasspath_mbp);
    jfieldID mbp_data_java_slot = jnienvcall->GetFieldID(env, mbp_tpdef_java, "data", "[B");
    
    jboolean oncopy = 0;

    jbyteArray m1_jobj = jnienvcall->GetObjectField(env, m1o, mbp_data_java_slot);
    jbyteArray m2_jobj = jnienvcall->GetObjectField(env, m2o, mbp_data_java_slot);
    jbyteArray mr_jobj = jnienvcall->GetObjectField(env, mro, mbp_data_java_slot);

    matrix_bp m1 = (matrix_bp)jnienvcall->GetByteArrayElements(env, m1_jobj, &oncopy);
    matrix_bp m2 = (matrix_bp)jnienvcall->GetByteArrayElements(env, m2_jobj, &oncopy);
    matrix_bp mr = (matrix_bp)jnienvcall->GetByteArrayElements(env, mr_jobj, &oncopy);
    
    matrix_bp_add(m1, m2, mr);

    jnienvcall->ReleaseByteArrayElements(env, m1_jobj, (jbyte*)m1, 0);
    jnienvcall->ReleaseByteArrayElements(env, m2_jobj, (jbyte*)m2, 0);
    jnienvcall->ReleaseByteArrayElements(env, mr_jobj, (jbyte*)mr, 0);
}

JNIEXPORT void JNICALL Java_mlib_ops_msub(JNIEnv *env, jclass _nspace, jobject m1o, jobject m2o, jobject mro){
    jclass mbp_tpdef_java = jnienvcall->FindClass(env,javaclasspath_mbp);
    jfieldID mbp_data_java_slot = jnienvcall->GetFieldID(env, mbp_tpdef_java, "data", "[B");
    
    jboolean oncopy = 0;

    jbyteArray m1_jobj = jnienvcall->GetObjectField(env, m1o, mbp_data_java_slot);
    jbyteArray m2_jobj = jnienvcall->GetObjectField(env, m2o, mbp_data_java_slot);
    jbyteArray mr_jobj = jnienvcall->GetObjectField(env, mro, mbp_data_java_slot);

    matrix_bp m1 = (matrix_bp)jnienvcall->GetByteArrayElements(env, m1_jobj, &oncopy);
    matrix_bp m2 = (matrix_bp)jnienvcall->GetByteArrayElements(env, m2_jobj, &oncopy);
    matrix_bp mr = (matrix_bp)jnienvcall->GetByteArrayElements(env, mr_jobj, &oncopy);
    
    matrix_bp_sub(m1, m2, mr);

    jnienvcall->ReleaseByteArrayElements(env, m1_jobj, (jbyte*)m1, 0);
    jnienvcall->ReleaseByteArrayElements(env, m2_jobj, (jbyte*)m2, 0);
    jnienvcall->ReleaseByteArrayElements(env, mr_jobj, (jbyte*)mr, 0);
}

JNIEXPORT void JNICALL Java_mlib_ops_mscale(JNIEnv *env, jclass _nspace, jobject m1o, jfloat num, jobject mro){
    jclass mbp_tpdef_java = jnienvcall->FindClass(env,javaclasspath_mbp);
    jfieldID mbp_data_java_slot = jnienvcall->GetFieldID(env, mbp_tpdef_java, "data", "[B");
    
    jboolean oncopy = 0;

    jbyteArray m1_jobj = jnienvcall->GetObjectField(env, m1o, mbp_data_java_slot);
    jbyteArray mr_jobj = jnienvcall->GetObjectField(env, mro, mbp_data_java_slot);

    matrix_bp m1 = (matrix_bp)jnienvcall->GetByteArrayElements(env, m1_jobj, &oncopy);
    matrix_bp mr = (matrix_bp)jnienvcall->GetByteArrayElements(env, mr_jobj, &oncopy);
    
    matrix_bp_scale(m1, float_to_qfix(num), mr);

    jnienvcall->ReleaseByteArrayElements(env, m1_jobj, (jbyte*)m1, 0);
    jnienvcall->ReleaseByteArrayElements(env, mr_jobj, (jbyte*)mr, 0);
}

JNIEXPORT void JNICALL Java_mlib_ops_mmul(JNIEnv *env, jclass _nspace, jobject m1o, jobject m2o, jobject mro){
    jclass mbp_tpdef_java = jnienvcall->FindClass(env,javaclasspath_mbp);
    jfieldID mbp_data_java_slot = jnienvcall->GetFieldID(env, mbp_tpdef_java, "data", "[B");
    
    jboolean oncopy = 0;

    jbyteArray m1_jobj = jnienvcall->GetObjectField(env, m1o, mbp_data_java_slot);
    jbyteArray m2_jobj = jnienvcall->GetObjectField(env, m2o, mbp_data_java_slot);
    jbyteArray mr_jobj = jnienvcall->GetObjectField(env, mro, mbp_data_java_slot);

    matrix_bp m1 = (matrix_bp)jnienvcall->GetByteArrayElements(env, m1_jobj, &oncopy);
    matrix_bp m2 = (matrix_bp)jnienvcall->GetByteArrayElements(env, m2_jobj, &oncopy);
    matrix_bp mr = (matrix_bp)jnienvcall->GetByteArrayElements(env, mr_jobj, &oncopy);
    
    matrix_bp_mulpty(m1, m2, mr);

    jnienvcall->ReleaseByteArrayElements(env, m1_jobj, (jbyte*)m1, 0);
    jnienvcall->ReleaseByteArrayElements(env, m2_jobj, (jbyte*)m2, 0);
    jnienvcall->ReleaseByteArrayElements(env, mr_jobj, (jbyte*)mr, 0);
}

JNIEXPORT void JNICALL Java_mlib_ops_mmul_1byelem(JNIEnv *env, jclass _nspace, jobject m1o, jobject m2o, jobject mro){
    jclass mbp_tpdef_java = jnienvcall->FindClass(env,javaclasspath_mbp);
    jfieldID mbp_data_java_slot = jnienvcall->GetFieldID(env, mbp_tpdef_java, "data", "[B");
    
    jboolean oncopy = 0;

    jbyteArray m1_jobj = jnienvcall->GetObjectField(env, m1o, mbp_data_java_slot);
    jbyteArray m2_jobj = jnienvcall->GetObjectField(env, m2o, mbp_data_java_slot);
    jbyteArray mr_jobj = jnienvcall->GetObjectField(env, mro, mbp_data_java_slot);

    matrix_bp m1 = (matrix_bp)jnienvcall->GetByteArrayElements(env, m1_jobj, &oncopy);
    matrix_bp m2 = (matrix_bp)jnienvcall->GetByteArrayElements(env, m2_jobj, &oncopy);
    matrix_bp mr = (matrix_bp)jnienvcall->GetByteArrayElements(env, mr_jobj, &oncopy);
    
    matrix_bp_mulptyByElem(m1, m2, mr);

    jnienvcall->ReleaseByteArrayElements(env, m1_jobj, (jbyte*)m1, 0);
    jnienvcall->ReleaseByteArrayElements(env, m2_jobj, (jbyte*)m2, 0);
    jnienvcall->ReleaseByteArrayElements(env, mr_jobj, (jbyte*)mr, 0);
}

JNIEXPORT void JNICALL Java_mlib_ops_mtpose(JNIEnv *env, jclass _nspace, jobject m1o, jobject mro){
    jclass mbp_tpdef_java = jnienvcall->FindClass(env,javaclasspath_mbp);
    jfieldID mbp_data_java_slot = jnienvcall->GetFieldID(env, mbp_tpdef_java, "data", "[B");
    
    jboolean oncopy = 0;

    jbyteArray m1_jobj = jnienvcall->GetObjectField(env, m1o, mbp_data_java_slot);
    jbyteArray mr_jobj = jnienvcall->GetObjectField(env, mro, mbp_data_java_slot);

    matrix_bp m1 = (matrix_bp)jnienvcall->GetByteArrayElements(env, m1_jobj, &oncopy);
    matrix_bp mr = (matrix_bp)jnienvcall->GetByteArrayElements(env, mr_jobj, &oncopy);
    
    matrix_bp_transpose(m1, mr);

    jnienvcall->ReleaseByteArrayElements(env, m1_jobj, (jbyte*)m1, 0);
    jnienvcall->ReleaseByteArrayElements(env, mr_jobj, (jbyte*)mr, 0);
}

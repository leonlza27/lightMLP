#include "core_convop.h"
#include "../jnibind_customconf.h"
#include "jni.h"
#include "jni_md.h"
#include "mlpCCore/matrix/matrix_static.h"
#include "mlpCCore/mlp/convop.h"

JNIEXPORT void JNICALL Java_core_convop_maxpool(JNIEnv *env, jclass _nspace, jobject arrino, jshort pool_rows, jshort pool_cols, jshort stepr, jshort stepc, jobject arrouto, jshortArray idx_of_maxso){
    jclass mbp_tpdef_java = idx_javaclass_ob(arrino);
    jfieldID mbp_data_java_slot = jnienvcall->GetFieldID(env, mbp_tpdef_java, "data", "[B");

    jboolean oncopy = 0;    
    jbyteArray min_jobj = jnienvcall->GetObjectField(env, arrino, mbp_data_java_slot);
    jbyteArray mout_jobj = jnienvcall->GetObjectField(env, arrouto, mbp_data_java_slot);

    matrix_bp m_in = (matrix_bp)jnienvcall->GetByteArrayElements(env, min_jobj, &oncopy);
    matrix_bp m_out = (matrix_bp)jnienvcall->GetByteArrayElements(env, mout_jobj, &oncopy);

    uint16_t* idx_of_maxs = (uint16_t*)jnienvcall->GetShortArrayElements(env, idx_of_maxso, &oncopy);

    conv_maxpool(m_in->data, m_in->rows, m_in->cols, pool_rows, pool_cols, stepr, stepc, m_out->data, idx_of_maxs);

    jnienvcall->ReleaseByteArrayElements(env, min_jobj, (jbyte*)m_in, 0);
    jnienvcall->ReleaseByteArrayElements(env, mout_jobj, (jbyte*)m_out, 0);
    jnienvcall->ReleaseShortArrayElements(env, idx_of_maxso, (jshort*)idx_of_maxs, 0);
}

JNIEXPORT void JNICALL Java_core_convop_averpool(JNIEnv *env, jclass _nspace, jobject arrino, jshort pool_rows, jshort pool_cols, jshort stepr, jshort stepc, jobject arrouto){
    jclass mbp_tpdef_java = idx_javaclass_ob(arrino);
    jfieldID mbp_data_java_slot = jnienvcall->GetFieldID(env, mbp_tpdef_java, "data", "[B");

    jboolean oncopy = 0;    
    jbyteArray min_jobj = jnienvcall->GetObjectField(env, arrino, mbp_data_java_slot);
    jbyteArray mout_jobj = jnienvcall->GetObjectField(env, arrouto, mbp_data_java_slot);

    matrix_bp m_in = (matrix_bp)jnienvcall->GetByteArrayElements(env, min_jobj, &oncopy);
    matrix_bp m_out = (matrix_bp)jnienvcall->GetByteArrayElements(env, mout_jobj, &oncopy);

    conv_averpool(m_in->data, m_in->rows, m_in->cols, pool_rows, pool_cols, stepr, stepc, m_out->data);

    jnienvcall->ReleaseByteArrayElements(env, min_jobj, (jbyte*)m_in, 0);
    jnienvcall->ReleaseByteArrayElements(env, mout_jobj, (jbyte*)m_out, 0);
}

JNIEXPORT void JNICALL Java_core_convop_maxpool_1backward(JNIEnv *env, jclass _nspace, jobject arrino, jshortArray idx_of_maxso, jshort pool_rows, jshort pool_cols, jshort stepr, jshort stepc, jobject arrouto){
    jclass mbp_tpdef_java = idx_javaclass_ob(arrino);
    jfieldID mbp_data_java_slot = jnienvcall->GetFieldID(env, mbp_tpdef_java, "data", "[B");

    jboolean oncopy = 0;    
    jbyteArray min_jobj = jnienvcall->GetObjectField(env, arrino, mbp_data_java_slot);
    jbyteArray mout_jobj = jnienvcall->GetObjectField(env, arrouto, mbp_data_java_slot);

    matrix_bp m_in = (matrix_bp)jnienvcall->GetByteArrayElements(env, min_jobj, &oncopy);
    matrix_bp m_out = (matrix_bp)jnienvcall->GetByteArrayElements(env, mout_jobj, &oncopy);

    uint16_t* idx_of_maxs = (uint16_t*)jnienvcall->GetShortArrayElements(env, idx_of_maxso, &oncopy);

    conv_maxpool_backward(m_in->data, idx_of_maxs, m_out->rows, m_out->cols, pool_rows, pool_cols, stepr, stepc, m_out->data);

    jnienvcall->ReleaseByteArrayElements(env, min_jobj, (jbyte*)m_in, 0);
    jnienvcall->ReleaseByteArrayElements(env, mout_jobj, (jbyte*)m_out, 0);
    jnienvcall->ReleaseShortArrayElements(env, idx_of_maxso, (jshort*)idx_of_maxs, 0);
}

JNIEXPORT void JNICALL Java_core_convop_averpool_1backward(JNIEnv *env, jclass _nspace, jobject arrino, jshort pool_rows, jshort pool_cols, jshort stepr, jshort stepc, jobject arrouto){
    jclass mbp_tpdef_java = idx_javaclass_ob(arrino);
    jfieldID mbp_data_java_slot = jnienvcall->GetFieldID(env, mbp_tpdef_java, "data", "[B");

    jboolean oncopy = 0;    
    jbyteArray min_jobj = jnienvcall->GetObjectField(env, arrino, mbp_data_java_slot);
    jbyteArray mout_jobj = jnienvcall->GetObjectField(env, arrouto, mbp_data_java_slot);

    matrix_bp m_in = (matrix_bp)jnienvcall->GetByteArrayElements(env, min_jobj, &oncopy);
    matrix_bp m_out = (matrix_bp)jnienvcall->GetByteArrayElements(env, mout_jobj, &oncopy);

    conv_averpool_backward(m_in->data, m_out->rows, m_out->cols, pool_rows, pool_cols, stepr, stepc, m_out->data);

    jnienvcall->ReleaseByteArrayElements(env, min_jobj, (jbyte*)m_in, 0);
    jnienvcall->ReleaseByteArrayElements(env, mout_jobj, (jbyte*)m_out, 0);
}

JNIEXPORT void JNICALL Java_core_convop_padding(JNIEnv *env, jclass _nspace, jobject arrino, jshort rext, jshort cext, jobject arrouto){
    jclass mbp_tpdef_java = idx_javaclass_ob(arrino);
    jfieldID mbp_data_java_slot = jnienvcall->GetFieldID(env, mbp_tpdef_java, "data", "[B");

    jboolean oncopy = 0;    
    jbyteArray min_jobj = jnienvcall->GetObjectField(env, arrino, mbp_data_java_slot);
    jbyteArray mout_jobj = jnienvcall->GetObjectField(env, arrouto, mbp_data_java_slot);

    matrix_bp m_in = (matrix_bp)jnienvcall->GetByteArrayElements(env, min_jobj, &oncopy);
    matrix_bp m_out = (matrix_bp)jnienvcall->GetByteArrayElements(env, mout_jobj, &oncopy);

    conv_padding(m_in->data, m_in->rows, m_in->cols, rext, cext, m_out->data);

    jnienvcall->ReleaseByteArrayElements(env, min_jobj, (jbyte*)m_in, 0);
    jnienvcall->ReleaseByteArrayElements(env, mout_jobj, (jbyte*)m_out, 0);
}

JNIEXPORT void JNICALL Java_core_convop_curedges(JNIEnv *env, jclass _nspace, jobject arrino, jshort rsub, jshort csub, jobject arrouto){
    jclass mbp_tpdef_java = idx_javaclass_ob(arrino);
    jfieldID mbp_data_java_slot = jnienvcall->GetFieldID(env, mbp_tpdef_java, "data", "[B");

    jboolean oncopy = 0;    
    jbyteArray min_jobj = jnienvcall->GetObjectField(env, arrino, mbp_data_java_slot);
    jbyteArray mout_jobj = jnienvcall->GetObjectField(env, arrouto, mbp_data_java_slot);

    matrix_bp m_in = (matrix_bp)jnienvcall->GetByteArrayElements(env, min_jobj, &oncopy);
    matrix_bp m_out = (matrix_bp)jnienvcall->GetByteArrayElements(env, mout_jobj, &oncopy);

    conv_cutedges(m_in->data, m_in->rows, m_in->cols, rsub, csub, m_out->data);

    jnienvcall->ReleaseByteArrayElements(env, min_jobj, (jbyte*)m_in, 0);
    jnienvcall->ReleaseByteArrayElements(env, mout_jobj, (jbyte*)m_out, 0);
}

JNIEXPORT void JNICALL Java_core_convop_divelem(JNIEnv *env, jclass _nspace, jobject arrino, jshort pool_rows, jshort pool_cols, jshort expand, jshort stepr, jshort stepc, jobjectArray arroutos){
    jclass mbp_tpdef_java = idx_javaclass_ob(arrino);
    jfieldID mbp_data_java_slot = jnienvcall->GetFieldID(env, mbp_tpdef_java, "data", "[B");

    jboolean oncopy = 0;    
    jbyteArray min_jobj = jnienvcall->GetObjectField(env, arrino, mbp_data_java_slot);
    matrix_bp m_in = (matrix_bp)jnienvcall->GetByteArrayElements(env, min_jobj, &oncopy);

    const uint16_t _scale = expand + 1;

    const uint16_t rows_in = m_in->rows, cols_in = m_in->cols;
    const qfix* arrin = m_in->data;

    const uint16_t arrsizer = (rows_in - pool_rows * _scale + expand) / stepr + 1;
    const uint16_t arrsizec = (cols_in - pool_cols * _scale + expand) / stepc + 1;

    const uint16_t in_rowstep = _scale * cols_in;

    uint32_t idx = 0;
    for(uint16_t i = 0; i < arrsizer; i++){
        for (uint16_t j = 0; j < arrsizec; j++){
            const qfix *incur = arrin + i * stepr * cols_in + j * stepc;
            jobject arrouto_cur = jnienvcall->GetObjectArrayElement(env, arroutos,idx);
            jbyteArray mouto_cur = jnienvcall->GetObjectField(env, arrouto_cur, mbp_data_java_slot);
            matrix_bp mout_cur = (matrix_bp)jnienvcall->GetByteArrayElements(env, mouto_cur, &oncopy);
            qfix *outcur = mout_cur->data;
            const qfix *incur_r = incur;
            for (uint16_t ii = 0; ii < pool_rows; ii++){
                const qfix* incur_c = incur_r;
                for(uint16_t jj = 0; jj < pool_cols; jj++){
                    *outcur = *incur_c;
                    incur_c += _scale;
                    outcur++;
                }
                incur_r += in_rowstep;
            }
            idx++;
            jnienvcall->ReleaseByteArrayElements(env, mouto_cur, (jbyte*)mout_cur, 0);
        }
    }
}

JNIEXPORT void JNICALL Java_core_convop_mergeelem(JNIEnv *env, jclass _nspace, jobjectArray arrinos, jshort expand, jshort stepr, jshort stepc, jobject arrouto){
    jclass mbp_tpdef_java = idx_javaclass_ob(arrouto);
    jfieldID mbp_data_java_slot = jnienvcall->GetFieldID(env, mbp_tpdef_java, "data", "[B");

    jboolean oncopy = 0;    
    jbyteArray mout_jobj = jnienvcall->GetObjectField(env, arrouto, mbp_data_java_slot);
    matrix_bp m_out = (matrix_bp)jnienvcall->GetByteArrayElements(env, mout_jobj, &oncopy);

    jobject _fromcap_obj = jnienvcall->GetObjectArrayElement(env, arrinos,0);
    jbyteArray _mfrom_obj = jnienvcall->GetObjectField(env, _fromcap_obj, mbp_data_java_slot);
    matrix_bp _mfrom = (matrix_bp)jnienvcall->GetByteArrayElements(env, _mfrom_obj, &oncopy);
    const uint16_t pool_rows = _mfrom->rows, pool_cols = _mfrom->cols; 
    jnienvcall->ReleaseByteArrayElements(env, _mfrom_obj, (jbyte*)_mfrom, 0);

    const uint16_t rows_to = m_out->rows, cols_to = m_out->cols;
    qfix* arrout = m_out->data;

    const uint16_t _scale = expand + 1;

    const uint16_t arrsizer = (rows_to - pool_rows * _scale + expand) / stepr + 1;
    const uint16_t arrsizec = (cols_to - pool_cols * _scale + expand) / stepc + 1;

    const uint16_t in_rowstep = _scale * cols_to;
    uint32_t idx = 0;
    for(uint16_t i = 0; i < arrsizer; i++){
        for (uint16_t j = 0; j < arrsizec; j++){
            qfix *outcur = arrout + i * stepr * cols_to + j * stepc;
            jobject arrino_cur = jnienvcall->GetObjectArrayElement(env, arrinos, idx);
            jbyteArray mino_cur = jnienvcall->GetObjectField(env, arrino_cur, mbp_data_java_slot);
            matrix_bp min_cur = (matrix_bp)jnienvcall->GetByteArrayElements(env, mino_cur, &oncopy);
            const qfix *incur = min_cur->data;
            qfix *outcur_r = outcur;
            for (uint16_t ii = 0; ii < pool_rows; ii++){
                qfix* outcur_c = outcur_r;
                for(uint16_t jj = 0; jj < pool_cols; jj++){
                    *outcur_c += *incur;
                    outcur_c += _scale;
                    incur++;
                }
                outcur_r += in_rowstep;
            }
            idx++;
        }
    }
}

JNIEXPORT void JNICALL Java_core_convop_conv(JNIEnv *env, jclass _nspace, jobject arrino, jobject convwo, jshort expand, jshort stepr, jshort stepc, jobject arrouto){
    jclass mbp_tpdef_java = idx_javaclass_ob(arrouto);
    jfieldID mbp_data_java_slot = jnienvcall->GetFieldID(env, mbp_tpdef_java, "data", "[B");

    jboolean oncopy = 0;    
    jbyteArray min_jobj = jnienvcall->GetObjectField(env, arrino, mbp_data_java_slot);
    jbyteArray mout_jobj = jnienvcall->GetObjectField(env, arrouto, mbp_data_java_slot);
    jbyteArray mw_jobj = jnienvcall->GetObjectField(env, convwo, mbp_data_java_slot);

    matrix_bp m_in = (matrix_bp)jnienvcall->GetByteArrayElements(env, min_jobj, &oncopy);
    matrix_bp m_out = (matrix_bp)jnienvcall->GetByteArrayElements(env, mout_jobj, &oncopy);
    matrix_bp m_convw = (matrix_bp)jnienvcall->GetByteArrayElements(env, mw_jobj, &oncopy);

    conv_fwdcalc(m_in->data, m_convw->data, m_in->rows, m_in->cols, m_convw->rows, m_convw->cols, expand, stepr, stepc, m_out->data);

    jnienvcall->ReleaseByteArrayElements(env, min_jobj, (jbyte*)m_in, 0);
    jnienvcall->ReleaseByteArrayElements(env, mout_jobj, (jbyte*)m_out, 0);
    jnienvcall->ReleaseByteArrayElements(env, mw_jobj, (jbyte*)m_convw, 0);
}
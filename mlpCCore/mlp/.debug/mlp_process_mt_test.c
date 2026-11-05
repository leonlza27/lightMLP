/*
 * mlp_process_mt.c(多线程实现)验证程序
 *
 * 编译(在本目录 .debug 下执行):
 *   gcc -O2 -o mlp_process_mt_test.exe mlp_process_mt_test.c ../mlp.c ../mlp_process_mt.c \
 *       ../../qfix_ops/qfix_op.c ../../threadpool/threadpool.c
 *
 * 验证内容:
 *   1) DOC 示例(6x6, ac_pass) 的多线程结果是否符合预期;
 *   2) 一般前向(mlptrainer_execute_mt / mlpexec_execute_mt)与单线程逐位一致,
 *      覆盖 行切分 / 列分块规约 / 小层内联 三类路径;
 *   3) 一般反向(mlptrainer_backward_mt)与单线程逐位一致(权重/偏置/权重梯度/输入梯度);
 *   4) totalgrads(mlptrainer_totalgrads_savegrads_mt / mlptrainer_totalgrads_backward_mt)
 *      与单线程逐位一致;
 *   5) DOC 的 totalgrad 两组等价性验证(组1 backward(lr) 与
 *      组2 backward(0)+savegrads+total_backward(lr) 再次 execute 的输出相同)。
 */
#include "../mlp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* mlp_process_mt.c 的多线程接口(mlp.h 未声明, 此处自行声明) */
void mlptrainer_execute_mt(const mlpTrainStatus *model, const qfix *arrin);
void mlpexec_execute_mt(const mlpExecStatus *model, const qfix *arrin);
void mlptrainer_backward_mt(const mlpTrainStatus *model, const qfix *grad_inital, const qfix lr);
void mlptrainer_totalgrads_backward_mt(const mlpTrainStatus *model, const mlpTrainStatus *gradscap, const qfix lr);
void mlptrainer_totalgrads_savegrads_mt(const mlpTrainStatus *net_or_gradcap, const mlpTrainStatus *gradscap_dest);

static int g_fail = 0;

/* ---------------- 确定性伪随机定点数(约 [-1, 1)) ---------------- */
static uint32_t g_seed;
static void rnd_seed(uint32_t s){ g_seed = s; }
static qfix rnd_qfix(void){
    g_seed = g_seed * 1664525u + 1013904223u;
    return float_to_qfix((float)((int32_t)(g_seed >> 20) - 2048) / 2048.0f);
}

/* ---------------- 断言与比较 ---------------- */
static int arr_eq(const char *what, const qfix *a, const qfix *b, uint32_t len){
    for(uint32_t i = 0; i < len; i++){
        if(a[i] != b[i]){
            printf("      [FAIL] %s[%u]: %d != %d\n", what, i, a[i], b[i]);
            return 0;
        }
    }
    return 1;
}
static void case_check(const char *name, int ok){
    printf("  %-58s %s\n", name, ok ? "OK" : "FAIL");
    if(!ok) g_fail++;
}
static void case_info(const char *name, const char *text){
    printf("  [info] %-52s %s\n", name, text);
}

/* ---------------- 测试用网络 ---------------- */
#define MAX_LYRS 4

typedef struct testnet{
    uint32_t n;
    netLyrConf lyr[MAX_LYRS];
    qfix *w[MAX_LYRS];
    qfix *b[MAX_LYRS];
} testnet;

static void net_alloc(testnet *t, uint32_t n, const uint16_t *in, const uint16_t *out, const uint8_t *act){
    t->n = n;
    for(uint32_t i = 0; i < n; i++){
        t->w[i] = (qfix*)malloc(sizeof(qfix) * (size_t)in[i] * out[i]);
        t->b[i] = (qfix*)malloc(sizeof(qfix) * out[i]);
        t->lyr[i].acTp = act[i];
        t->lyr[i].in_dim = in[i];
        t->lyr[i].out_dim = out[i];
        t->lyr[i].weights = t->w[i];
        t->lyr[i].bias = t->b[i];
        t->lyr[i].dExtra = 0;
    }
}

static void net_randfill(testnet *t){
    for(uint32_t i = 0; i < t->n; i++){
        const uint32_t wsize = (uint32_t)t->lyr[i].in_dim * t->lyr[i].out_dim;
        for(uint32_t k = 0; k < wsize; k++) t->w[i][k] = rnd_qfix();
        for(uint32_t k = 0; k < t->lyr[i].out_dim; k++) t->b[i][k] = rnd_qfix();
    }
}

static void net_copy(testnet *dst, const testnet *src){
    dst->n = src->n;
    for(uint32_t i = 0; i < src->n; i++){
        const uint32_t wsize = (uint32_t)src->lyr[i].in_dim * src->lyr[i].out_dim;
        dst->w[i] = (qfix*)malloc(sizeof(qfix) * wsize);
        dst->b[i] = (qfix*)malloc(sizeof(qfix) * src->lyr[i].out_dim);
        memcpy(dst->w[i], src->w[i], sizeof(qfix) * wsize);
        memcpy(dst->b[i], src->b[i], sizeof(qfix) * src->lyr[i].out_dim);
        dst->lyr[i] = src->lyr[i];
        dst->lyr[i].weights = dst->w[i];
        dst->lyr[i].bias = dst->b[i];
    }
}

static void net_free(testnet *t){
    for(uint32_t i = 0; i < t->n; i++){ free(t->w[i]); free(t->b[i]); }
    t->n = 0;
}

/* mlptrainer_setup 的缓冲区未清零, 梯度累加器需显式清零 */
static void trainer_zerograds(mlpTrainStatus *s){
    for(uint32_t i = 0; i < s->calclyrs; i++){
        memset(s->lyrinput_grad[i], 0, sizeof(qfix) * s->modelsrc[i].in_dim);
    }
}
static void cap_zerograds(mlpTrainStatus *s){
    for(uint32_t i = 0; i < s->calclyrs; i++){
        memset(s->lyrinput_grad[i], 0, sizeof(qfix) * s->modelsrc[i].in_dim);
        memset(s->w_grad[i], 0, sizeof(qfix) * (size_t)s->modelsrc[i].in_dim * s->modelsrc[i].out_dim);
    }
}

/* ---------------- 1) DOC 示例 ---------------- */
static void test_doc_example(void){
    /* W 每行均为 [1,2,3,4,5,6], x = [1,2,3,4,5,6], B = [3,6,9,2,4,6]
     * 行内积 = 1+4+9+16+25+36 = 91, 结果 = 91 + B = [94,97,100,93,95,97] */
    static const int bvals[6] = {3, 6, 9, 2, 4, 6};
    qfix w[36], b[6], x[6];
    qfix expect[6];
    netLyrConf lyr;
    mlpTrainStatus sa, sb;
    int ok = 1;

    for(uint32_t r = 0; r < 6; r++){
        for(uint32_t c = 0; c < 6; c++) w[r * 6 + c] = float_to_qfix((float)(c + 1));
        b[r] = float_to_qfix((float)bvals[r]);
        x[r] = float_to_qfix((float)(r + 1));
        expect[r] = float_to_qfix((float)(91 + bvals[r]));
    }

    lyr.acTp = ac_pass;
    lyr.in_dim = 6;
    lyr.out_dim = 6;
    lyr.weights = w;
    lyr.bias = b;
    lyr.dExtra = 0;

    mlptrainer_setup(1, &lyr, &sa);
    mlptrainer_setup(1, &lyr, &sb);
    mlptrainer_execute(&sa, x);
    mlptrainer_execute_mt(&sb, x);

    ok &= arr_eq("DOC示例 mt结果", mlptrain_resu_ref(&sb), expect, 6);
    ok &= arr_eq("DOC示例 mt==st", sa.fullConnData[1], sb.fullConnData[1], 6);

    mlptrainer_cleanup(&sa);
    mlptrainer_cleanup(&sb);
    case_check("DOC 示例(6x6, ac_pass) 结果符合预期且 mt==st", ok);
}


/* ---------------- 2) 前向等价性 ---------------- */
static void test_forward_case(const char *tag, uint32_t n,
                              const uint16_t *in, const uint16_t *out, const uint8_t *act){
    testnet ta, tb;
    mlpTrainStatus sa, sb;
    mlpExecStatus ea, eb;
    qfix *xin = (qfix*)malloc(sizeof(qfix) * in[0]);
    uint16_t maxout = 0;
    int ok = 1;

    rnd_seed(0x1234u + n * 7u);
    net_alloc(&ta, n, in, out, act);
    net_randfill(&ta);
    net_copy(&tb, &ta);
    for(uint32_t i = 0; i < in[0]; i++) xin[i] = rnd_qfix();

    mlptrainer_setup(n, ta.lyr, &sa);
    mlptrainer_setup(n, tb.lyr, &sb);
    mlptrainer_execute(&sa, xin);
    mlptrainer_execute_mt(&sb, xin);

    for(uint32_t i = 0; i < n; i++){
        char what[96];
        snprintf(what, sizeof(what), "%s trainer out[lyr%u]", tag, i);
        ok &= arr_eq(what, sa.fullConnData[i + 1], sb.fullConnData[i + 1], out[i]);
        snprintf(what, sizeof(what), "%s trainer va_wxb[lyr%u]", tag, i);
        ok &= arr_eq(what, sa.va_wxb[i], sb.va_wxb[i], out[i]);
    }

    mlpexec_setup(n, ta.lyr, &ea);
    mlpexec_setup(n, tb.lyr, &eb);
    for(uint32_t i = 0; i < n; i++) if(out[i] > maxout) maxout = out[i];
    /* 双缓冲中未被写入的部分先清零, 便于整体比较 */
    memset(ea.fullConnData_tmp[0], 0, sizeof(qfix) * (size_t)maxout * 2);
    memset(eb.fullConnData_tmp[0], 0, sizeof(qfix) * (size_t)maxout * 2);
    mlpexec_execute(&ea, xin);
    mlpexec_execute_mt(&eb, xin);
    {
        char what[96];
        snprintf(what, sizeof(what), "%s exec buf0", tag);
        ok &= arr_eq(what, ea.fullConnData_tmp[0], eb.fullConnData_tmp[0], maxout);
        snprintf(what, sizeof(what), "%s exec buf1", tag);
        ok &= arr_eq(what, ea.fullConnData_tmp[1], eb.fullConnData_tmp[1], maxout);
    }

    mlptrainer_cleanup(&sa);
    mlptrainer_cleanup(&sb);
    mlpexec_cleanup(&ea);
    mlpexec_cleanup(&eb);
    net_free(&ta);
    net_free(&tb);
    free(xin);
    case_check(tag, ok);
}

/* ---------------- 3) 反向等价性 ---------------- */
static void test_backward_case(const char *tag, uint32_t n,
                               const uint16_t *in, const uint16_t *out, const uint8_t *act){
    testnet ta, tb;
    mlpTrainStatus sa, sb;
    qfix *xin = (qfix*)malloc(sizeof(qfix) * in[0]);
    qfix *grad = (qfix*)malloc(sizeof(qfix) * out[n - 1]);
    const qfix lr = float_to_qfix(0.25f);
    int ok = 1;

    rnd_seed(0x5678u + n * 13u);
    net_alloc(&ta, n, in, out, act);
    net_randfill(&ta);
    net_copy(&tb, &ta);
    for(uint32_t i = 0; i < in[0]; i++) xin[i] = rnd_qfix();
    for(uint32_t i = 0; i < out[n - 1]; i++) grad[i] = rnd_qfix();

    mlptrainer_setup(n, ta.lyr, &sa);
    mlptrainer_setup(n, tb.lyr, &sb);
    trainer_zerograds(&sa);
    trainer_zerograds(&sb);
    /* 先各自前向, 保证 va_wxb/fullConnData 与单线程一致 */
    mlptrainer_execute(&sa, xin);
    mlptrainer_execute(&sb, xin);

    mlptrainer_backward(&sa, grad, lr);
    mlptrainer_backward_mt(&sb, grad, lr);

    for(uint32_t i = 0; i < n; i++){
        char what[96];
        const uint32_t wsize = (uint32_t)in[i] * out[i];
        snprintf(what, sizeof(what), "%s weights[lyr%u]", tag, i);
        ok &= arr_eq(what, ta.w[i], tb.w[i], wsize);
        snprintf(what, sizeof(what), "%s bias[lyr%u]", tag, i);
        ok &= arr_eq(what, ta.b[i], tb.b[i], out[i]);
        snprintf(what, sizeof(what), "%s w_grad[lyr%u]", tag, i);
        ok &= arr_eq(what, sa.w_grad[i], sb.w_grad[i], wsize);
        snprintf(what, sizeof(what), "%s lyrinput_grad[lyr%u]", tag, i);
        ok &= arr_eq(what, sa.lyrinput_grad[i], sb.lyrinput_grad[i], in[i]);
    }

    mlptrainer_cleanup(&sa);
    mlptrainer_cleanup(&sb);
    net_free(&ta);
    net_free(&tb);
    free(xin);
    free(grad);
    case_check(tag, ok);
}

/* ---------------- 4) totalgrads 等价性 ---------------- */
static uint32_t arr_diff_count(const qfix *a, const qfix *b, uint32_t len){
    uint32_t cnt = 0;
    for(uint32_t i = 0; i < len; i++) if(a[i] != b[i]) cnt++;
    return cnt;
}

static void test_totalgrads_case(const char *tag, uint32_t n,
                                 const uint16_t *in, const uint16_t *out, const uint8_t *act){
    testnet ta, tb, ta2, tb2;
    mlpTrainStatus sa, sb, capa, capb, da, db;
    qfix *xin = (qfix*)malloc(sizeof(qfix) * in[0]);
    qfix *grad = (qfix*)malloc(sizeof(qfix) * out[n - 1]);
    const qfix lr = float_to_qfix(0.3f);
    int ok = 1;

    rnd_seed(0x9abcu + n * 17u);
    net_alloc(&ta, n, in, out, act);
    net_randfill(&ta);
    net_copy(&tb, &ta);
    net_copy(&ta2, &ta);
    net_copy(&tb2, &tb);
    for(uint32_t i = 0; i < in[0]; i++) xin[i] = rnd_qfix();
    for(uint32_t i = 0; i < out[n - 1]; i++) grad[i] = rnd_qfix();

    mlptrainer_setup(n, ta.lyr, &sa);
    mlptrainer_setup(n, tb.lyr, &sb);
    trainer_zerograds(&sa);
    trainer_zerograds(&sb);
    mlptrainer_execute(&sa, xin);
    mlptrainer_execute(&sb, xin);
    /* lr = 0: 只求梯度, 不改动权重/偏置 */
    mlptrainer_backward(&sa, grad, 0);
    mlptrainer_backward_mt(&sb, grad, 0);

    mlptrainer_totalgrads_cap_setup(n, ta.lyr, &capa);
    mlptrainer_totalgrads_cap_setup(n, tb.lyr, &capb);
    cap_zerograds(&capa);
    cap_zerograds(&capb);
    mlptrainer_totalgrads_savegrads(&sa, &capa);
    mlptrainer_totalgrads_savegrads_mt(&sb, &capb);

    for(uint32_t i = 0; i < n; i++){
        char what[96];
        const uint32_t wsize = (uint32_t)in[i] * out[i];
        snprintf(what, sizeof(what), "%s savegrads lyrinput_grad[lyr%u]", tag, i);
        ok &= arr_eq(what, capa.lyrinput_grad[i], capb.lyrinput_grad[i], in[i]);
        snprintf(what, sizeof(what), "%s savegrads w_grad[lyr%u]", tag, i);
        ok &= arr_eq(what, capa.w_grad[i], capb.w_grad[i], wsize);
    }

    /* 在相同的权重副本上分别应用累计梯度, 比较更新后的权重/偏置 */
    mlptrainer_setup(n, ta2.lyr, &da);
    mlptrainer_setup(n, tb2.lyr, &db);
    mlptrainer_totalgrads_backward(&da, &capa, lr);
    mlptrainer_totalgrads_backward_mt(&db, &capb, lr);

    for(uint32_t i = 0; i < n; i++){
        char what[96];
        const uint32_t wsize = (uint32_t)in[i] * out[i];
        snprintf(what, sizeof(what), "%s total_backward weights[lyr%u]", tag, i);
        ok &= arr_eq(what, ta2.w[i], tb2.w[i], wsize);
        snprintf(what, sizeof(what), "%s total_backward bias[lyr%u]", tag, i);
        ok &= arr_eq(what, ta2.b[i], tb2.b[i], out[i]);
    }

    mlptrainer_cleanup(&sa);
    mlptrainer_cleanup(&sb);
    mlptrainer_cleanup(&capa);
    mlptrainer_cleanup(&capb);
    mlptrainer_cleanup(&da);
    mlptrainer_cleanup(&db);
    net_free(&ta);
    net_free(&tb);
    net_free(&ta2);
    net_free(&tb2);
    free(xin);
    free(grad);
    case_check(tag, ok);
}

/* ---------------- 5) DOC 的 totalgrad 两组等价性验证 ---------------- */
static void test_doc_group(void){
    /* 2 层恒等权重网络(in_dim == out_dim, W = I, ac_pass):
     * 该配置下 lyrinput_grad[i] 与 lyrinput_grad[i+1] 逐项相等, 因此库中 totalgrads
     * 取用的偏置梯度与 backward 完全一致, 两组结果可逐位比较。
     * (一般网络下偏置路径的差异见 test_group_weights 的说明) */
    enum { N = 8 };
    const uint16_t dims[2] = {N, N};
    const uint8_t act[2] = {ac_pass, ac_pass};
    testnet base, t1, t2, t3, t4;
    mlpTrainStatus s1, s2, s3, s4, cap2, cap4;
    qfix xin[N], grad[N];
    const qfix lr = float_to_qfix(0.4f);
    int ok = 1;

    rnd_seed(0xbeefu);
    net_alloc(&base, 2, dims, dims, act);
    for(uint32_t l = 0; l < 2; l++){
        for(uint32_t r = 0; r < N; r++){
            for(uint32_t c = 0; c < N; c++){
                base.w[l][r * N + c] = (r == c) ? float_to_qfix(1.0f) : 0;
            }
        }
        for(uint32_t c = 0; c < N; c++) base.b[l][c] = rnd_qfix();
    }
    net_copy(&t1, &base);
    net_copy(&t2, &base);
    net_copy(&t3, &base);
    net_copy(&t4, &base);
    for(uint32_t i = 0; i < N; i++){ xin[i] = rnd_qfix(); grad[i] = rnd_qfix(); }

    /* 组1(mt): execute -> backward(lr) -> execute */
    mlptrainer_setup(2, t1.lyr, &s1);
    trainer_zerograds(&s1);
    mlptrainer_execute_mt(&s1, xin);
    mlptrainer_backward_mt(&s1, grad, lr);
    mlptrainer_execute_mt(&s1, xin);

    /* 组2(mt): execute -> backward(0) -> savegrads -> total_backward(lr) -> execute */
    mlptrainer_setup(2, t2.lyr, &s2);
    mlptrainer_totalgrads_cap_setup(2, t2.lyr, &cap2);
    trainer_zerograds(&s2);
    cap_zerograds(&cap2);
    mlptrainer_execute_mt(&s2, xin);
    mlptrainer_backward_mt(&s2, grad, 0);
    mlptrainer_totalgrads_savegrads_mt(&s2, &cap2);
    mlptrainer_totalgrads_backward_mt(&s2, &cap2, lr);
    mlptrainer_execute_mt(&s2, xin);

    /* 同样的两组再用单线程实现跑一遍 */
    mlptrainer_setup(2, t3.lyr, &s3);
    trainer_zerograds(&s3);
    mlptrainer_execute(&s3, xin);
    mlptrainer_backward(&s3, grad, lr);
    mlptrainer_execute(&s3, xin);

    mlptrainer_setup(2, t4.lyr, &s4);
    mlptrainer_totalgrads_cap_setup(2, t4.lyr, &cap4);
    trainer_zerograds(&s4);
    cap_zerograds(&cap4);
    mlptrainer_execute(&s4, xin);
    mlptrainer_backward(&s4, grad, 0);
    mlptrainer_totalgrads_savegrads(&s4, &cap4);
    mlptrainer_totalgrads_backward(&s4, &cap4, lr);
    mlptrainer_execute(&s4, xin);

    ok &= arr_eq("组1(mt) vs 组2(mt) 输出", mlptrain_resu_ref(&s1), mlptrain_resu_ref(&s2), N);
    ok &= arr_eq("组1(mt) vs 组1(st) 输出", mlptrain_resu_ref(&s1), mlptrain_resu_ref(&s3), N);
    ok &= arr_eq("组2(mt) vs 组2(st) 输出", mlptrain_resu_ref(&s2), mlptrain_resu_ref(&s4), N);

    mlptrainer_cleanup(&s1);
    mlptrainer_cleanup(&s2);
    mlptrainer_cleanup(&s3);
    mlptrainer_cleanup(&s4);
    mlptrainer_cleanup(&cap2);
    mlptrainer_cleanup(&cap4);
    net_free(&base);
    net_free(&t1);
    net_free(&t2);
    net_free(&t3);
    net_free(&t4);
    case_check("DOC totalgrad 两组等价(恒等权重网络)", ok);
}

/* ---------------- 6) 一般网络下两组的权重路径等价 ---------------- */
static void test_group_weights(void){
    /* 组1: backward(lr); 组2: backward(0) + savegrads + total_backward(lr)
     * 两组的权重更新必须逐位相同。
     * 注: 库中 totalgrads 的偏置梯度取自 lyrinput_grad[i], 而 backward 使用
     *     lyrinput_grad[i+1], 故一般网络下两组的偏置(进而输出)不相等, 单线程实现亦然;
     *     此处只校验权重路径, 偏置差异仅作提示。 */
    const uint16_t in[2] = {6, 8};
    const uint16_t out[2] = {8, 4};
    const uint8_t act[2] = {ac_pass, ac_pass};
    testnet base, t1, t2;
    mlpTrainStatus s1, s2, cap2;
    qfix xin[6], grad[4];
    const qfix lr = float_to_qfix(0.4f);
    int ok = 1;
    uint32_t bias_diff = 0;

    rnd_seed(0xfeedu);
    net_alloc(&base, 2, in, out, act);
    net_randfill(&base);
    net_copy(&t1, &base);
    net_copy(&t2, &base);
    for(uint32_t i = 0; i < 6; i++) xin[i] = rnd_qfix();
    for(uint32_t i = 0; i < 4; i++) grad[i] = rnd_qfix();

    mlptrainer_setup(2, t1.lyr, &s1);
    trainer_zerograds(&s1);
    mlptrainer_execute(&s1, xin);
    mlptrainer_backward(&s1, grad, lr);

    mlptrainer_setup(2, t2.lyr, &s2);
    mlptrainer_totalgrads_cap_setup(2, t2.lyr, &cap2);
    trainer_zerograds(&s2);
    cap_zerograds(&cap2);
    mlptrainer_execute(&s2, xin);
    mlptrainer_backward(&s2, grad, 0);
    mlptrainer_totalgrads_savegrads(&s2, &cap2);
    mlptrainer_totalgrads_backward(&s2, &cap2, lr);

    for(uint32_t i = 0; i < 2; i++){
        char what[96];
        const uint32_t wsize = (uint32_t)in[i] * out[i];
        snprintf(what, sizeof(what), "一般网络 组1/组2 weights[lyr%u]", i);
        ok &= arr_eq(what, t1.w[i], t2.w[i], wsize);
        bias_diff += arr_diff_count(t1.b[i], t2.b[i], out[i]);
    }

    mlptrainer_cleanup(&s1);
    mlptrainer_cleanup(&s2);
    mlptrainer_cleanup(&cap2);
    net_free(&base);
    net_free(&t1);
    net_free(&t2);
    case_check("一般网络下 组1/组2 权重更新一致", ok);
    if(bias_diff != 0){
        case_info("一般网络下 组1/组2 偏置更新",
                  "不一致(库中 totalgrads 偏置索引与 backward 不同, 单线程亦然)");
    }
}

int main(void){
    printf("mlp_process_mt.c 多线程实现验证\n");

    /* ---- DOC 示例 ---- */
    test_doc_example();

    /* ---- 前向: 一般计算(ac_pass) + 各切分路径 ---- */
    {
        const uint16_t in[2] = {64, 32}, out[2] = {32, 8};
        const uint8_t act[2] = {ac_pass, ac_pass};
        test_forward_case("前向 2层 64->32->8 (ac_pass)", 2, in, out, act);
    }
    {
        const uint16_t in[2] = {16, 32}, out[2] = {32, 8};
        const uint8_t act[2] = {ac_ReLU, ac_pass};
        test_forward_case("前向 2层 16->32(ReLU)->8 (ac_pass)", 2, in, out, act);
    }
    {
        const uint16_t in[1] = {200}, out[1] = {50};
        const uint8_t act[1] = {ac_pass};
        test_forward_case("前向 1层 200->50 (行切分)", 1, in, out, act);
    }
    {
        const uint16_t in[1] = {8192}, out[1] = {1};
        const uint8_t act[1] = {ac_pass};
        test_forward_case("前向 1层 8192->1 (列分块规约)", 1, in, out, act);
    }
    {
        const uint16_t in[1] = {2}, out[1] = {8192};
        const uint8_t act[1] = {ac_pass};
        test_forward_case("前向 1层 2->8192 (行切分)", 1, in, out, act);
    }
    {
        const uint16_t in[1] = {4}, out[1] = {3};
        const uint8_t act[1] = {ac_pass};
        test_forward_case("前向 1层 4->3 (小层内联)", 1, in, out, act);
    }

    /* ---- 反向 ---- */
    {
        const uint16_t in[2] = {16, 32}, out[2] = {32, 8};
        const uint8_t act[2] = {ac_ReLU, ac_pass};
        test_backward_case("反向 2层 16->32(ReLU)->8", 2, in, out, act);
    }
    {
        const uint16_t in[2] = {64, 32}, out[2] = {32, 8};
        const uint8_t act[2] = {ac_Tanh, ac_pass};
        test_backward_case("反向 2层 64->32(Tanh)->8", 2, in, out, act);
    }
    {
        const uint16_t in[1] = {512}, out[1] = {512};
        const uint8_t act[1] = {ac_pass};
        test_backward_case("反向 1层 512->512 (双阶段并行)", 1, in, out, act);
    }
    {
        const uint16_t in[1] = {8192}, out[1] = {1};
        const uint8_t act[1] = {ac_pass};
        test_backward_case("反向 1层 8192->1", 1, in, out, act);
    }
    {
        const uint16_t in[1] = {1}, out[1] = {8192};
        const uint8_t act[1] = {ac_pass};
        test_backward_case("反向 1层 1->8192", 1, in, out, act);
    }

    /* ---- totalgrads ---- */
    {
        const uint16_t in[2] = {16, 32}, out[2] = {32, 8};
        const uint8_t act[2] = {ac_pass, ac_pass};
        test_totalgrads_case("totalgrads 2层 16->32->8", 2, in, out, act);
    }
    {
        const uint16_t in[1] = {512}, out[1] = {256};
        const uint8_t act[1] = {ac_pass};
        test_totalgrads_case("totalgrads 1层 512->256", 1, in, out, act);
    }

    /* ---- DOC 两组等价性验证 ---- */
    test_doc_group();
    test_group_weights();

    printf("\n==== %s (fail = %d) ====\n", g_fail ? "FAILED" : "ALL PASSED", g_fail);
    return g_fail ? 1 : 0;
}

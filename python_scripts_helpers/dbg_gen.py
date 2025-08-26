import numpy as np

#struct mat_init_para
class mat_init_para:
    def __init__(self):
        self.rows:int = 0
        self.cols:int = 0
        self.dataRaw:list[int] = []
    

    def __str__(self):
        return f"matrix {self.rows}x{self.cols}-> raw{self.dataRaw}"

#struct example
class example:
    def __init__(self):
        self.mat1:mat_init_para = mat_init_para()
        self.mat2:mat_init_para = mat_init_para()
        self.optype:int      #0:add 1:mulpty
        self.mat_resu_excepeted:mat_init_para = mat_init_para()

    def __str__(self):
        return f"mat1:{self.mat1}\n{"+" if self.optype == 0 else "x"}\nmat2:{self.mat2}\n=\nresu:{self.mat_resu_excepeted} "
    

#q16.16与float互转

def float_to_q1616(va : float) -> int:
    return int(np.clip(np.round(va * (1 << 16)),-2**31,2**31 - 1))

def q1616_to_float(va : int) -> float:
    return va / (1 << 16)

#矩阵生成
def gen_matrix(m : int,n : int):
    mat = np.random.uniform(-20,20,(m,n))
    mat_q16 = np.vectorize(float_to_q1616)(mat)
    return mat , mat_q16

#加法测试

def gen_add_test(m:int,n:int) -> example:
    matA , matA_q16 = gen_matrix(m,n)
    matB , matB_q16 = gen_matrix(m,n)

    ret:example = example()

    ret.mat1.cols = n
    ret.mat1.rows = m
    ret.mat2.cols = n
    ret.mat2.rows = m
    ret.optype = 0
    ret.mat_resu_excepeted.cols = n
    ret.mat_resu_excepeted.rows = m

   
    ret.mat1.dataRaw = matA_q16.flatten(order = "C").tolist()
    ret.mat2.dataRaw = matB_q16.flatten(order = "C").tolist()

    matResu = matA + matB
    matResu_q16 = np.vectorize(float_to_q1616)(matResu)

    ret.mat_resu_excepeted.dataRaw = matResu_q16.flatten(order = "C").tolist()

    return ret

def gen_mul_test(m:int,k:int,n:int) -> example:
    matA , matA_q16 = gen_matrix(m,k)
    matB , matB_q16 = gen_matrix(k,n)
    
    ret:example = example()

    ret.mat1.cols = k
    ret.mat1.rows = m
    ret.mat2.cols = n
    ret.mat2.rows = k
    ret.optype = 1
    ret.mat_resu_excepeted.cols = n
    ret.mat_resu_excepeted.rows = m

   
    ret.mat1.dataRaw = matA_q16.flatten(order = "C").tolist()
    ret.mat2.dataRaw = matB_q16.flatten(order = "C").tolist()

    matResu = np.dot(matA, matB)
    matResu_q16 = np.vectorize(float_to_q1616)(matResu)

    ret.mat_resu_excepeted.dataRaw = matResu_q16.flatten(order = "C").tolist()

    return ret

if __name__ == "__main__":
    m = 2
    n = 10
    k = 3    
    with open("moduledebug.cpp","w") as fwr:
        fwr.write("#include <stdio.h>\n#include \"matrix/matrix_static.h\"\n\n")
        fwr.write("int matrix_compare(const matrix_qfloat_data* a, const matrix_qfloat_data* b) {\n")
        fwr.write("\tif (a->rows != b->rows || a->cols != b->cols) return 1;\n")
        fwr.write("\tfor (int i = 0; i < a->rows * a->cols; i++) {\n")
        fwr.write("\t\tif (a->data[i] != b->data[i]) return 1;\n")
        fwr.write("\t}\n")
        fwr.write("\treturn 0;\n")
        fwr.write("}\n\n")
        fwr.write("int main(){\n")
        fwr.write("\tint error_count = 0;\n")
        fwr.write("\tmatrix_qfloat_data *mat1 = alloc_matrix_qfloat(),*mat2 = alloc_matrix_qfloat(),*excepted = alloc_matrix_qfloat(),*actual = alloc_matrix_qfloat();\n")
        fwr.write(f"\tmatrix_qfloat_init(mat1,{m},{n},0);\n")
        fwr.write(f"\tmatrix_qfloat_init(mat2,{m},{n},0);\n")
        fwr.write(f"\tmatrix_qfloat_init(excepted,{m},{n},0);\n")
        fwr.write(f"\tmatrix_qfloat_init(actual,{m},{n},0);\n\n")
        
        #加法
        for i in range(5):
            opt = gen_add_test(m,n)
            fwr.write(f"\tqfloat madd1_{i}[] = {{ {",".join(str(nu) for nu in opt.mat1.dataRaw)} }};\n")
            fwr.write(f"\tqfloat madd2_{i}[] = {{ {",".join(str(nu) for nu in opt.mat2.dataRaw)} }};\n")
            fwr.write(f"\tqfloat mresu_{i}[] = {{ {",".join(str(nu) for nu in opt.mat_resu_excepeted.dataRaw)} }};\n\n")
            fwr.write(f"\tmatrix_qfloat_reset(mat1,{m},{n},madd1_{i});\n")
            fwr.write(f"\tmatrix_qfloat_reset(mat2,{m},{n},madd2_{i});\n")
            fwr.write(f"\tmatrix_qfloat_reset(excepted,{m},{n},mresu_{i});\n")
            fwr.write(f"\tmatrix_qfloat_add(mat1, mat2, actual);\n\n")
            fwr.write(f"\tif(matrix_compare(excepted, actual) != 0) {{\n")
            fwr.write(f"\t\tprintf(\"Add test {i} failed!\\n\");\n")
            fwr.write(f"\t\terror_count++;\n")
            fwr.write(f"\t}}\n")
            fwr.write(f"\tDbgPrint_qfloat_matrix(excepted,\"excepted add {i}\");\n\tputc(\'\\n\',stdout);\n")
            fwr.write(f"\tDbgPrint_qfloat_matrix(actual,\"actual add {i}\");\n\tputs(\"\\n\");\n\n\n")
        
        #乘法    
        for i in range(5):
            opt = gen_mul_test(m,k,n)
            fwr.write(f"\tqfloat madd1_{i}_2[] = {{ {",".join(str(nu) for nu in opt.mat1.dataRaw)} }};\n")
            fwr.write(f"\tqfloat madd2_{i}_2[] = {{ {",".join(str(nu) for nu in opt.mat2.dataRaw)} }};\n")
            fwr.write(f"\tqfloat mresu_{i}_2[] = {{ {",".join(str(nu) for nu in opt.mat_resu_excepeted.dataRaw)} }};\n\n")
            fwr.write(f"\tmatrix_qfloat_reset(mat1,{m},{k},madd1_{i}_2);\n")
            fwr.write(f"\tmatrix_qfloat_reset(mat2,{k},{n},madd2_{i}_2);\n")
            fwr.write(f"\tmatrix_qfloat_reset(excepted,{m},{n},mresu_{i}_2);\n")
            fwr.write(f"\tmatrix_qfloat_mulpty(mat1, mat2, actual);\n\n")
            fwr.write(f"\tif(matrix_compare(excepted, actual) != 0) {{\n")
            fwr.write(f"\t\tprintf(\"Multiply test {i} failed!\\n\");\n")
            fwr.write(f"\t\terror_count++;\n")
            fwr.write(f"\t}}\n")
            fwr.write(f"\tDbgPrint_qfloat_matrix(excepted,\"excepted mulpty {i}\");\n\tputc(\'\\n\',stdout);\n")
            fwr.write(f"\tDbgPrint_qfloat_matrix(actual,\"actual mulpty {i}\");\n\tputs(\"\\n\");\n\n\n")
        
        fwr.write("\tprintf(\"Total errors: %d\\n\", error_count);\n")
        fwr.write("\treturn error_count;\n}")

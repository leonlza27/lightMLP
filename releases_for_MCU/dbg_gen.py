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
    m = 10
    n = 10
    k = 12    
    with open("moudledebug_matrix.cpp","w") as fwr:
        fwr.write("#include <stdio.h>\n#include \"matrix/matrix_static.h\"\nint main(){\n")
        fwr.write("\tmatrix_Q16_16_data *mat1 = alloc_matrix_Q16_16(),*mat2 = alloc_matrix_Q16_16(),*excepted = alloc_matrix_Q16_16(),*actual = alloc_matrix_Q16_16();\n")
        fwr.write(f"\tmatrix_Q16_16_init(mat1,{m},{n},0);\n")
        fwr.write(f"\tmatrix_Q16_16_init(mat2,{m},{n},0);\n")
        fwr.write(f"\tmatrix_Q16_16_init(excepted,{m},{n},0);\n")
        fwr.write(f"\tmatrix_Q16_16_init(actual,{m},{n},0);\n\n")
        
        #加法
        for i in range(5):
            opt = gen_add_test(m,n)
            fwr.write(f"\tf_q16_16 madd1_{i}[] = {{ {",".join(str(nu) for nu in opt.mat1.dataRaw)} }};\n")
            fwr.write(f"\tf_q16_16 madd2_{i}[] = {{ {",".join(str(nu) for nu in opt.mat2.dataRaw)} }};\n")
            fwr.write(f"\tf_q16_16 mresu_{i}[] = {{ {",".join(str(nu) for nu in opt.mat_resu_excepeted.dataRaw)} }};\n\n")
            fwr.write(f"\tmatrix_Q16_16_reset(mat1,{m},{n},madd1_{i});\n")
            fwr.write(f"\tmatrix_Q16_16_reset(mat2,{m},{n},madd2_{i});\n")
            fwr.write(f"\tmatrix_Q16_16_reset(excepted,{m},{n},mresu_{i});\n")
            fwr.write(f"\tmatrix_Q16_16_add(mat1, mat2, actual);\n\n")
            fwr.write(f"\tDbgPrint_Q16_16_matrix(excepted,\"excepted\");\n\tputc(\'\\n\',stdout);\n")
            fwr.write(f"\tDbgPrint_Q16_16_matrix(actual,\"actual\");\n\tputs(\"\\n\");\n\n\n")
        
        #乘法    
        for i in range(5):
            opt = gen_mul_test(m,k,n)
            fwr.write(f"\tf_q16_16 madd1_{i}_2[] = {{ {",".join(str(nu) for nu in opt.mat1.dataRaw)} }};\n")
            fwr.write(f"\tf_q16_16 madd2_{i}_2[] = {{ {",".join(str(nu) for nu in opt.mat2.dataRaw)} }};\n")
            fwr.write(f"\tf_q16_16 mresu_{i}_2[] = {{ {",".join(str(nu) for nu in opt.mat_resu_excepeted.dataRaw)} }};\n\n")
            fwr.write(f"\tmatrix_Q16_16_reset(mat1,{m},{n},madd1_{i}_2);\n")
            fwr.write(f"\tmatrix_Q16_16_reset(mat2,{m},{n},madd2_{i}_2);\n")
            fwr.write(f"\tmatrix_Q16_16_reset(excepted,{m},{n},mresu_{i}_2);\n")
            fwr.write(f"\tmatrix_Q16_16_mul(mat1, mat2, actual);\n\n")
            fwr.write(f"\tDbgPrint_Q16_16_matrix(excepted,\"excepted\");\n\tputc(\'\\n\',stdout);\n")
            fwr.write(f"\tDbgPrint_Q16_16_matrix(actual,\"actual\");\n\tputs(\"\\n\");\n\n\n")
        
            
        
        fwr.write("return 0;\n}")

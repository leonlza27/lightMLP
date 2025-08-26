import math

def generate_exp_lut(num_points, q_shift, min_val=-8, max_val=8):
    """生成exp函数的LUT表
    
    Args:
        num_points: 表的大小
        q_shift: 定点数移位位数(8或16)
        min_val: 最小值
        max_val: 最大值
        
    Returns:
        生成的LUT表(list)
    """
    lut = []
    step = (max_val - min_val) / (num_points - 1)
    
    for i in range(num_points):
        x = min_val + i * step
        exp_val = math.exp(x)
        
        # 转换为定点数
        q_val = int(exp_val * (1 << q_shift))
        
        # 限制在有效范围内
        max_q = (1 << (q_shift * 2 - 1)) - 1  # 考虑符号位
        q_val = max(-max_q, min(q_val, max_q))
        
        lut.append(q_val)
    
    return lut

def format_lut(lut, values_per_line=8):
    """格式化LUT表为C数组定义
    
    Args:
        lut: LUT表
        values_per_line: 每行显示的值数量
        
    Returns:
        格式化后的字符串
    """
    lines = []
    for i in range(0, len(lut), values_per_line):
        line_values = lut[i:i+values_per_line]
        line = ",".join([str(v) for v in line_values])
        if i + values_per_line < len(lut):
            line += ","
        lines.append(line)
    
    return "\n".join(lines)

# 生成LUT表并输出到文件
with open("exp_lut_values.txt", "w") as f:
    # 生成q8.8格式的128点LUT
    q8_lut = generate_exp_lut(128, 8)
    f.write("const f_q8_8 exp_lut_q8[128] = {\n")
    f.write(format_lut(q8_lut))
    f.write("\n};\n\n")

    # 生成q16.16格式的256点LUT
    q16_lut = generate_exp_lut(256, 16)
    f.write("const f_q16_16 exp_lut_q16[256] = {\n")
    f.write(format_lut(q16_lut))
    f.write("\n};\n")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# 添加构建路径到 Python 路径
build_path = os.path.join(os.path.dirname(__file__), '..', '..', 'build', '__out__', 'py_bind', 'Release')
if build_path not in sys.path:
    sys.path.insert(0, build_path)

try:
    # 测试导入 corepy.convop 作为子模块
    import corepy.convop as convop
    print("✓ 成功导入 corepy.convop 作为子模块")
    
    # 测试导入 corepy
    import corepy as core
    print("✓ 成功导入 corepy")
    
    # 测试通过 core.convop 访问
    print("✓ core.convop 可访问:", hasattr(core, 'convop'))
    
    # 测试导入 mbp16dpy
    import mbp16dpy as mlib
    print("✓ 成功导入 mbp16dpy")
    
    # 创建测试矩阵
    src = mlib.matrixbp(4, 4)
    src.fromlist([i for i in range(16)])
    print("✓ 创建源矩阵:")
    print(src)
    
    # 创建输出矩阵
    to = mlib.matrixbp(2, 2)
    print("✓ 创建输出矩阵")
    
    # 测试 averpool 操作
    print("测试 core.convop.averpool...")
    core.convop.averpool(src, (2, 2), 2, to)
    print("✓ averpool 操作成功")
    
    print("输出矩阵:")
    print(to)
    
    print("\n所有测试通过！convop 现在是 corepy 的子模块。")
    
except ImportError as e:
    print(f"✗ 导入错误: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ 运行错误: {e}")
    sys.exit(1)
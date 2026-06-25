import os
import sys

if os.path.exists("Debug") or os.path.exists("Release"): # 按程序特征排除msvc构建(windows)
    exit()

#===========================REQUESTS(for ai)================================

#.pyd/.dll为win下动态库
#由gcc/clang + cmake等含linux/unix命名习惯(libxxx.dll/pyd)
#win下无法为java加载(jni加载win下要求xxx.dll,实为libxxx.dll)
#所有平台python无法加载(要求xxx.pyd/.so/.dylib, 生成实为libxxx.pyd/so/dylib)

#工作目录于<build,参考<projroot>/CMakeLists.txt>/__out__
#python对接文件于py_bind文件夹,后缀为.pyd/.so,别动里面的mbpbuffer
#java对接文件位于java_bind目录,后缀为.dll(仅为windows去掉lib前缀其他保留)

#===========================================================================

def install():
    # 处理 Python 绑定
    py_bind_dir = "py_bind"
    if os.path.exists(py_bind_dir):
        for root, dirs, files in os.walk(py_bind_dir):
            for filename in files:
                if filename.startswith("lib") and (filename.endswith(".pyd") or filename.endswith(".so")):
                    if "mbpbuffer" not in filename:
                        new_name = filename[3:]
                        old_path = os.path.join(root, filename)
                        new_path = os.path.join(root, new_name)
                        try:
                            os.rename(old_path, new_path)
                            print(f"Renamed Python binding: {old_path} -> {new_path}")
                        except OSError as e:
                            print(f"Error renaming {old_path}: {e}")
    else:
        print(f"Directory {py_bind_dir} not found, skipping.")

    # 处理 Java 绑定
    java_bind_dir = "java_bind"
    if os.path.exists(java_bind_dir):
        if sys.platform == "win32":
            for root, dirs, files in os.walk(java_bind_dir):
                for filename in files:
                    if filename.startswith("lib") and filename.endswith(".dll"):
                        new_name = filename[3:]
                        old_path = os.path.join(root, filename)
                        new_path = os.path.join(root, new_name)
                        try:
                            os.rename(old_path, new_path)
                            print(f"Renamed Java binding: {old_path} -> {new_path}")
                        except OSError as e:
                            print(f"Error renaming {old_path}: {e}")
        else:
            print(f"Java bind renaming only supported on Windows, skipping.")
    else:
        print(f"Directory {java_bind_dir} not found, skipping.")

if __name__ == "__main__":
    install()
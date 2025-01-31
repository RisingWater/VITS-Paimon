from setuptools import setup
from Cython.Build import cythonize
import sys

# 获取 Python 安装目录
python_dir = sys.exec_prefix

# 构建包含路径
include_dirs = [f'{python_dir}/include', f'{python_dir}/include/cpython']

setup(
    name='monotonic_align',
    ext_modules=cythonize('core.pyx', compiler_directives={'language_level': 3}),
    include_dirs=include_dirs
)
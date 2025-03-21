#!/usr/bin/env python

from setuptools import dist
dist.Distribution().fetch_build_eggs(['cython', 'numpy'])

import os
import platform
import numpy as np

from Cython.Build import cythonize
from setuptools import find_packages
from distutils.core import setup
from distutils.extension import Extension

# https://stackoverflow.com/questions/11013851/speeding-up-build-process-with-distutils
import os
import multiprocessing
try:
    from concurrent.futures import ThreadPoolExecutor as Pool
except ImportError:
    from multiprocessing.pool import ThreadPool as LegacyPool

    # To ensure the with statement works. Required for some older 2.7.x releases
    class Pool(LegacyPool):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()
            self.join()


def build_extensions(self):
    """Function to monkey-patch
    distutils.command.build_ext.build_ext.build_extensions

    """
    self.check_extensions_list(self.extensions)

    try:
        num_jobs = os.cpu_count()
    except AttributeError:
        num_jobs = multiprocessing.cpu_count()

    with Pool(num_jobs) as pool:
        results = pool.map(self.build_extension, self.extensions)
        list(results)

def compile(
    self, sources, output_dir=None, macros=None, include_dirs=None,
    debug=0, extra_preargs=None, extra_postargs=None, depends=None,
):
    """Function to monkey-patch distutils.ccompiler.CCompiler"""
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
        output_dir, macros, include_dirs, sources, depends, extra_postargs
    )
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

    for obj in objects:
        try:
            src, ext = build[obj]
        except KeyError:
            continue
        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

    # Return *all* object filenames, not just the ones we just built.
    return objects


from distutils.ccompiler import CCompiler
from distutils.command.build_ext import build_ext
build_ext.build_extensions = build_extensions


def _compile_and_check(c_content, compiler_args = []):
    import os, tempfile, subprocess, shutil
    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)
    filename = r'test.c'
    with open(filename, 'w') as file:
        file.write(c_content)
    with open(os.devnull, 'w') as fnull:
        args = [os.environ.get("CC") or 'cc']
        args += compiler_args
        args.append(filename)
        result = subprocess.call(args, stdout=fnull, stderr=fnull)
    os.chdir(curdir)
    #clean up
    shutil.rmtree(tmpdir)
    return result == 0

def _check_openmp():
    # see http://openmp.org/wp/openmp-compilers/
    omp_test = \
        r"""
#include <omp.h>
#include <stdio.h>
int main() {
#pragma omp parallel
printf("Hello from thread %d, nthreads %d\n", omp_get_thread_num(), omp_get_num_threads());
}
"""
    return _compile_and_check(omp_test, ['-fopenmp', '-lgomp'])

def _check_avx2():
    # The MIT License (MIT)

    # Copyright (c) 2014 Anders Høst

    # Permission is hereby granted, free of charge, to any person obtaining a copy of
    # this software and associated documentation files (the "Software"), to deal in
    # the Software without restriction, including without limitation the rights to
    # use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
    # the Software, and to permit persons to whom the Software is furnished to do so,
    # subject to the following conditions:

    # The above copyright notice and this permission notice shall be included in all
    # copies or substantial portions of the Software.

    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
    # FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
    # COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
    # IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
    # CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


    import platform
    import os
    import ctypes
    from ctypes import c_uint32, c_int, c_long, c_ulong, c_size_t, c_void_p, POINTER, CFUNCTYPE

    # Posix x86_64:
    # Three first call registers : RDI, RSI, RDX
    # Volatile registers         : RAX, RCX, RDX, RSI, RDI, R8-11

    # Windows x86_64:
    # Three first call registers : RCX, RDX, R8
    # Volatile registers         : RAX, RCX, RDX, R8-11

    # cdecl 32 bit:
    # Three first call registers : Stack (%esp)
    # Volatile registers         : EAX, ECX, EDX

    _POSIX_64_OPC = [
        0x53,  # push   %rbx
        0x89,
        0xF0,  # mov    %esi,%eax
        0x89,
        0xD1,  # mov    %edx,%ecx
        0x0F,
        0xA2,  # cpuid
        0x89,
        0x07,  # mov    %eax,(%rdi)
        0x89,
        0x5F,
        0x04,  # mov    %ebx,0x4(%rdi)
        0x89,
        0x4F,
        0x08,  # mov    %ecx,0x8(%rdi)
        0x89,
        0x57,
        0x0C,  # mov    %edx,0xc(%rdi)
        0x5B,  # pop    %rbx
        0xC3,  # retq
    ]

    _WINDOWS_64_OPC = [
        0x53,  # push   %rbx
        0x89,
        0xD0,  # mov    %edx,%eax
        0x49,
        0x89,
        0xC9,  # mov    %rcx,%r9
        0x44,
        0x89,
        0xC1,  # mov    %r8d,%ecx
        0x0F,
        0xA2,  # cpuid
        0x41,
        0x89,
        0x01,  # mov    %eax,(%r9)
        0x41,
        0x89,
        0x59,
        0x04,  # mov    %ebx,0x4(%r9)
        0x41,
        0x89,
        0x49,
        0x08,  # mov    %ecx,0x8(%r9)
        0x41,
        0x89,
        0x51,
        0x0C,  # mov    %edx,0xc(%r9)
        0x5B,  # pop    %rbx
        0xC3,  # retq
    ]

    _CDECL_32_OPC = [
        0x53,  # push   %ebx
        0x57,  # push   %edi
        0x8B,
        0x7C,
        0x24,
        0x0C,  # mov    0xc(%esp),%edi
        0x8B,
        0x44,
        0x24,
        0x10,  # mov    0x10(%esp),%eax
        0x8B,
        0x4C,
        0x24,
        0x14,  # mov    0x14(%esp),%ecx
        0x0F,
        0xA2,  # cpuid
        0x89,
        0x07,  # mov    %eax,(%edi)
        0x89,
        0x5F,
        0x04,  # mov    %ebx,0x4(%edi)
        0x89,
        0x4F,
        0x08,  # mov    %ecx,0x8(%edi)
        0x89,
        0x57,
        0x0C,  # mov    %edx,0xc(%edi)
        0x5F,  # pop    %edi
        0x5B,  # pop    %ebx
        0xC3,  # ret
    ]

    is_windows = os.name == "nt"
    is_64bit = ctypes.sizeof(ctypes.c_voidp) == 8


    class CPUID_struct(ctypes.Structure):
        _fields_ = [(r, c_uint32) for r in ("eax", "ebx", "ecx", "edx")]


    class CPUID(object):
        def __init__(self):
            if platform.machine() not in ("AMD64", "x86_64", "x86", "i686"):
                raise SystemError("Only available for x86")

            if is_windows:
                if is_64bit:
                    # VirtualAlloc seems to fail under some weird
                    # circumstances when ctypes.windll.kernel32 is
                    # used under 64 bit Python. CDLL fixes this.
                    self.win = ctypes.CDLL("kernel32.dll")
                    opc = _WINDOWS_64_OPC
                else:
                    # Here ctypes.windll.kernel32 is needed to get the
                    # right DLL. Otherwise it will fail when running
                    # 32 bit Python on 64 bit Windows.
                    self.win = ctypes.windll.kernel32
                    opc = _CDECL_32_OPC
            else:
                opc = _POSIX_64_OPC if is_64bit else _CDECL_32_OPC

            size = len(opc)
            code = (ctypes.c_ubyte * size)(*opc)

            if is_windows:
                self.win.VirtualAlloc.restype = c_void_p
                self.win.VirtualAlloc.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.c_ulong,
                    ctypes.c_ulong,
                ]
                self.addr = self.win.VirtualAlloc(None, size, 0x1000, 0x40)
                if not self.addr:
                    raise MemoryError("Could not allocate RWX memory")
            else:
                self.libc = ctypes.cdll.LoadLibrary(None)
                self.libc.valloc.restype = ctypes.c_void_p
                self.libc.valloc.argtypes = [ctypes.c_size_t]
                self.addr = self.libc.valloc(size)
                if not self.addr:
                    raise MemoryError("Could not allocate memory")

                self.libc.mprotect.restype = c_int
                self.libc.mprotect.argtypes = [c_void_p, c_size_t, c_int]
                ret = self.libc.mprotect(self.addr, size, 1 | 2 | 4)
                if ret != 0:
                    raise OSError("Failed to set RWX")

            ctypes.memmove(self.addr, code, size)

            func_type = CFUNCTYPE(None, POINTER(CPUID_struct), c_uint32, c_uint32)
            self.func_ptr = func_type(self.addr)

        def __call__(self, eax, ecx=0):
            struct = CPUID_struct()
            self.func_ptr(struct, eax, ecx)
            return struct.eax, struct.ebx, struct.ecx, struct.edx

        def __del__(self):
            if is_windows:
                self.win.VirtualFree.restype = c_long
                self.win.VirtualFree.argtypes = [c_void_p, c_size_t, c_ulong]
                self.win.VirtualFree(self.addr, 0, 0x8000)
            elif self.libc:
                # Seems to throw exception when the program ends and
                # libc is cleaned up before the object?
                self.libc.free.restype = None
                self.libc.free.argtypes = [c_void_p]
                self.libc.free(self.addr)
    
    try:
        # Invoke CPUID instruction with eax 0x7
        # ECX bit 5: AVX2 support
        # For more information, refer to https://en.wikipedia.org/wiki/CPUID
        input_eax = 0x7
        output_eax, output_ebx, output_ecx, output_edx = CPUID()(input_eax)
        bits = bin(output_ebx)[::-1]
        avx2_support = bits[5]
        return avx2_support == '1'
    
    except Exception as e:
        with open('error.log', 'w') as f:
            f.write(f"Failed to check AVX2 support: {e}")
        return False

def _check_neon():
    neon_test = r"""
#include <arm_neon.h>
int main() {
    const uint16x8_t constant = { 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF };
    const uint32x4_t vadded = vpaddlq_u16(constant);
    return 0;
}
    """
    return _compile_and_check(neon_test, ["-mfpu=neon"])



extra_compile_args = []
extra_link_args = []
if platform.system() != 'Windows':
    extra_compile_args.append("-std=c++11")
    if _check_openmp():
        extra_compile_args.append('-fopenmp')
        extra_link_args.append('-lgomp')

    if _check_avx2():
        extra_compile_args.append("-DUSE_AVX2")
        extra_compile_args.append("-mavx2")
        extra_compile_args.append("-mfma")
    if platform.machine().lower().startswith("arm") and _check_neon():
        extra_compile_args.append("-DUSE_NEON")
        extra_compile_args.append("-mfpu=neon")

else:
    extra_compile_args.append("/openmp")
    if _check_avx2():
        extra_compile_args.append("/DUSE_AVX2")
        extra_compile_args.append("/arch:AVX2")
        extra_compile_args.append("/fp:fast")


cpp_sources = [
    "src/timer.cpp",
    "src/parallel.cpp",
    "src/fast-slic.cpp",
    "src/cca.cpp",
    "src/context.cpp",
    "src/context-impl.cpp",
    "src/lsc.cpp",
    "src/lsc-builder.cpp",
]


setup(
    name="fast-slic",
    version="0.4.0",
    description="Fast Slic Superpixel Implementation",
    author="Alchan Kim",
    author_email="a9413miky@gmail.com",
    setup_requires = ["cython", "numpy"],
    install_requires=["numpy"],
    python_requires=">=3.5",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    packages=find_packages(),
    ext_modules=cythonize(
        [
            Extension(
                "cfast_slic",
                include_dirs=[np.get_include()],
                sources=cpp_sources + ["cfast_slic.pyx"],
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                language="c++",
            ),
            Extension(
                "csimple_crf",
                include_dirs=[np.get_include()],
                sources=["src/simple-crf.cpp", "csimple_crf.pyx"],
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                language="c++",
            )
        ]
    )
)

from distutils.core import setup, Extension

#include_dirs=['/home/odroid/realsense/Maghoumi/include/librealsense'], //Maybe needed for libs

module1 = Extension('lrs',
                    include_dirs=['/usr/local/include/librealsense','/usr/include/boost'],
                    runtime_library_dirs = ['/usr/local/lib'],
                    libraries = ['realsense'],
                    sources = ['lrsmodule.c','segmenter.cpp'],
                    extra_compile_args = ['-std=c++11'])

setup (name = 'lrsPackage',
       version = '1.0',
       description = 'Realsense module by Patrick',
       ext_modules = [module1])
from distutils.core import setup, Extension

#include_dirs=['/home/odroid/realsense/Maghoumi/include/librealsense'], //Maybe needed for libs

module1 = Extension('lrs',
                    include_dirs=['/usr/local/include/librealsense'],
                    runtime_library_dirs = ['/usr/local/lib'],
                    libraries = ['realsense'],
                    sources = ['lrsmodule.c'])

setup (name = 'lrsPackage',
       version = '1.0',
       description = 'Realsense module by Patrick',
       ext_modules = [module1])

#module2 = Extension('pcl',
#                    include_dirs=[ '/usr/local/include/pcl-1.7','/usr/include/eigen3', 
#                                  '/usr/include/vtk-5.10', '/usr/include', '/usr/lib', 'usr/local/lib' /usr/local/lib/libpcl_common.so /usr/lib/libvtkGenericFiltering.so.5.10.1 /usr/lib/libvtkGeovis.so.5.10.1 /usr/lib/libvtkCharts.so.5.10.1 /usr/local/lib/libpcl_io.so],
#                    runtime_library_dirs = ['/usr/local/lib'],
#                    libraries = ['boost_filesystem', 'boost_thread', 'boost_date_time', 'boost_iostreams', 'boost_serialization', 'boost_chrono', 'boost_atomic', 'boost_regex', 'pthread' , 'pcl_common', ''boost_system', 'boost_filesystem', 'boost_thread', 'boost_date_time', 'boost_iostreams', 'boost_serialization', 'boost_chrono', 'boost_atomic', 'boost_regex', 'pthread', 'boost_system', 'boost_filesystem', 'boost_thread', 'boost_date_time', 'boost_iostreams', 'boost_serialization', 'boost_chrono', 'boost_atomic', 'boost_regex', 'pthread', 'm' ],
#                    sources = ['pclmodule.cpp'])

module2 = Extension('pcl',
                    include_dirs=[ '/usr/local/include/pcl-1.7','/usr/include/eigen3', 
                                  '/usr/include/vtk-5.10', '/usr/include', '/usr/lib', 'usr/local/lib'],
                    extra_link_args=['-lboost_system', '-lpcl_io'],
#		    extra_link_args=['-L/usr/local/lib -rdynamic -lboost_system -lboost_filesystem -lboost_thread -lboost_date_time -lboost_iostreams -lboost_serialization -lboost_chrono -lboost_atomic -lboost_regex -lpthread /usr/local/lib/libpcl_common.so /usr/lib/libvtkGenericFiltering.so.5.10.1 /usr/lib/libvtkGeovis.so.5.10.1 /usr/lib/libvtkCharts.so.5.10.1 /usr/local/lib/libpcl_io.so -lboost_system -lboost_filesystem -lboost_thread -lboost_date_time -lboost_iostreams -lboost_serialization -lboost_chrono -lboost_atomic -lboost_regex -lpthread /usr/local/lib/libpcl_common.so /usr/local/lib/libpcl_octree.so -lboost_system -lboost_filesystem -lboost_thread -lboost_date_time -lboost_iostreams -lboost_serialization -lboost_chrono -lboost_atomic -lboost_regex -lpthread /usr/local/lib/libpcl_common.so /usr/local/lib/libpcl_io.so /usr/local/lib/libpcl_octree.so /usr/lib/libvtkViews.so.5.10.1 /usr/lib/libvtkInfovis.so.5.10.1 /usr/lib/libvtkWidgets.so.5.10.1 /usr/lib/libvtkVolumeRendering.so.5.10.1 /usr/lib/libvtkHybrid.so.5.10.1 /usr/lib/libvtkParallel.so.5.10.1 /usr/lib/libvtkRendering.so.5.10.1 /usr/lib/libvtkImaging.so.5.10.1 /usr/lib/libvtkGraphics.so.5.10.1 /usr/lib/libvtkIO.so.5.10.1 /usr/lib/libvtkFiltering.so.5.10.1 /usr/lib/libvtkCommon.so.5.10.1 -lm /usr/lib/libvtksys.so.5.10.1 -ldl -Wl,-rpath,/usr/local/lib:/usr/lib/openmpi/lib -Wl,-rpath-link,/usr/lib/openmpi/lib -Wl,--no-allow-shlib-undefined'], 
                    sources = ['pclmodule.cpp'])

setup (name = 'pclPackage',
       version = '1.0',
       description = 'PCL module by Patrick',
       ext_modules = [module2])

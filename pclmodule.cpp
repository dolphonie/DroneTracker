/* Example of embedding Python in another program */

#include "Python.h"
#include <numpy/arrayobject.h>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>


int main(int argc, char **argv)
{
    
}

/* A static module */

/* 'self' is not used */
extern "C" {
    void initpcl(void); /* Forward */

    static PyObject *
	   pcl_processCloud(PyObject *self, PyObject* args)
    {

	   PyArrayObject npPC;
	   PyArg_ParseTuple(args, "O", &npPC);
	   //Convert to c array
	   float *pc[];
	   npy_intp dims[2];
	   PyArray_AsCArray(&npPC, (void *)pc, dims, 2, NPY_FLOAT32);
	   return Py_BuildValue("i", 1);
    }

    static PyObject *
	   pcl_test(PyObject *self, PyObject* args)
    {
	   pcl::PointCloud<pcl::PointXYZ> cloud;

	   //// Fill in the cloud data
	   cloud.width = 5;
	   cloud.height = 1;
	   cloud.is_dense = false;
	   cloud.points.resize(cloud.width * cloud.height);

	   for (size_t i = 0; i < cloud.points.size(); ++i)
	   {
		  cloud.points[i].x = 1024 * rand() / (RAND_MAX + 1.0f);
		  cloud.points[i].y = 1024 * rand() / (RAND_MAX + 1.0f);
		  cloud.points[i].z = 1024 * rand() / (RAND_MAX + 1.0f);
	   }

	   pcl::io::savePCDFileASCII("test_pcd.pcd", cloud);
	   std::cerr << "Saved " << cloud.points.size() << " data points to test_pcd.pcd." << std::endl;

	   for (size_t i = 0; i < cloud.points.size(); ++i)
		  std::cerr << "    " << cloud.points[i].x << " " << cloud.points[i].y << " " << cloud.points[i].z << std::endl;
	   return Py_BuildValue("i", 1);
	   
    }
    static PyMethodDef pcl_methods[] = {
	   { "processCloud",             pcl_processCloud,      METH_VARARGS,
	   "Processes point cloud and returns quad" },
	   { "test",             pcl_test,      METH_NOARGS,
	   "Processes point cloud and returns quad" },
	   { NULL,              NULL }           /* sentinel */
    };

    void
	   initpcl(void)
    {
	   PyImport_AddModule("pcl");
	   Py_InitModule("pcl", pcl_methods);
	   import_array();//Starts numpy C-API
    }
}

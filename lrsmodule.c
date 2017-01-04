/* Example of embedding Python in another program */

#include "Python.h"
#include <numpy/arrayobject.h>
#include <librealsense/rs.h>

#define DEBUG_PRINT 0

rs_context * ctx;
rs_device * dev;
uint16_t one_meter;
rs_error * e = 0;
rs_intrinsics depth_intrin, color_intrin;
rs_extrinsics depth_to_color;


int check_error(void)
{
    if (e)
    {
	   const char* failedF = rs_get_failed_function(e);
	   const char* failedArgs = rs_get_failed_args(e);
	   const char* errMsg = rs_get_error_message(e);
	   int i = strlen(failedF)+ strlen(failedArgs)+ strlen(errMsg);
	   char buff[i+90];
	   strcpy(buff, "rs_error was raised when calling ");
	   strcat(buff, failedF);
	   strcat(buff, failedArgs);
	   strcat(buff, errMsg);
	   PyErr_SetString(PyExc_ZeroDivisionError, buff);
	   return 1;
    }
    return 0;
}


void initlrs(void); /* Forward */

int main(int argc, char **argv)
{
    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(argv[0]);

    /* Initialize the Python interpreter.  Required. */
    Py_Initialize();

    /* Add a static module */
    initlrs();

    /* Define sys.argv.  It is up to the application if you
    want this; you can also leave it undefined (since the Python
    code is generally not a main program it has no business
    touching sys.argv...)

    If the third argument is true, sys.path is modified to include
    either the directory containing the script named by argv[0], or
    the current working directory.  This can be risky; if you run
    an application embedding Python in a directory controlled by
    someone else, attackers could put a Trojan-horse module in the
    directory (say, a file named os.py) that your application would
    then import and run.
    */
    PySys_SetArgvEx(argc, argv, 0);

    /* Do some application specific code */
    printf("Hello, brave new world\n\n");

    /* Execute some Python statements (in module __main__) */
    PyRun_SimpleString("import sys\n");
    PyRun_SimpleString("print sys.builtin_module_names\n");
    PyRun_SimpleString("print sys.modules.keys()\n");
    PyRun_SimpleString("print sys.executable\n");
    PyRun_SimpleString("print sys.argv\n");

    /* Note that you can call any public function of the Python
    interpreter here, e.g. call_object(). */

    /* Some more application specific code */
    printf("\nGoodbye, cruel world\n");

    /* Exit, cleaning up the interpreter */
    Py_Exit(0);
    /*NOTREACHED*/
}

/* A static module */



static PyObject *
lrs_startStream(PyObject *self, PyObject* args)
{
    import_array();//Starts numpy C-API


    /* Create a context object. This object owns the handles to all connected realsense devices. */
    rs_context * ctx = rs_create_context(RS_API_VERSION, &e);
    if(check_error()) return NULL;
    
    printf("There are %d connected RealSense devices.\n", rs_get_device_count(ctx, &e));
    check_error();
    if (rs_get_device_count(ctx, &e) == 0) return NULL;

    /* This tutorial will access only a single device, but it is trivial to extend to multiple devices */
    dev = rs_get_device(ctx, 0, &e);
    if (check_error()) return NULL;
    printf("\nUsing device 0, an %s\n", rs_get_device_name(dev, &e));
    if (check_error()) return NULL;
    printf("    Serial number: %s\n", rs_get_device_serial(dev, &e));
    if (check_error()) return NULL;
    printf("    Firmware version: %s\n", rs_get_device_firmware_version(dev, &e));
    if (check_error()) return NULL;

    /* Configure depth to run at VGA resolution at 30 frames per second */
    rs_enable_stream_preset(dev, RS_STREAM_DEPTH, RS_PRESET_BEST_QUALITY, &e);
    if (check_error()) return NULL;
    rs_enable_stream_preset(dev, RS_STREAM_COLOR, RS_PRESET_BEST_QUALITY, &e);
    if (check_error()) return NULL;
    
    rs_start_device(dev, &e);
    if (check_error()) return NULL;

    /*depth_scale = rs_get_device_depth_scale(dev, &e);
    if (check_error()) return NULL;*/

    rs_get_stream_intrinsics(dev, RS_STREAM_DEPTH, &depth_intrin, &e);
    if (check_error()) return NULL;
    rs_get_device_extrinsics(dev, RS_STREAM_DEPTH, RS_STREAM_COLOR, &depth_to_color, &e);
    if (check_error()) return NULL;
    rs_get_stream_intrinsics(dev, RS_STREAM_COLOR, &color_intrin, &e);
    if (check_error()) return NULL;

    one_meter = (uint16_t)(1.0f / rs_get_device_depth_scale(dev, &e));

    return Py_BuildValue("i", one_meter);
}

static PyObject *
lrs_stopStream(PyObject *self, PyObject* args)
{
    rs_stop_device(dev, &e);
    if (check_error()) return NULL;
    return Py_BuildValue("i", 1);
}

/* 'self' is not used */
static PyObject *
lrs_getFrame(PyObject *self, PyObject* args)
{
    rs_wait_for_frames(dev, &e);
    /* Retrieve depth data, which was previously configured as a 640 x 480 image of 16-bit depth values */
    uint16_t * depthPointer = (uint16_t *)(rs_get_frame_data(dev, RS_STREAM_DEPTH, &e));
    if (check_error()) return NULL;
    uint16_t*  colorPointer = (uint16_t *)(rs_get_frame_data(dev, RS_STREAM_COLOR_ALIGNED_TO_DEPTH, &e));
    if (check_error()) return NULL;

    if (DEBUG_PRINT) {
	   uint16_t* fPCopy = depthPointer;
	   int  numNonZero = 0;
	   for (int i = 0; i < 640 * 480; i++) {
		  int pixel = *fPCopy++;
		  if (pixel != 0) {
			 numNonZero++;
		  }
	   }
	   printf("Number of nonzero pixels in frame: %d\n", numNonZero);
    }

    int depthDims[2] = { depth_intrin.height, depth_intrin.width };
    PyArrayObject* depthFrame =   PyArray_SimpleNewFromData(2, depthDims, NPY_UINT16, (void*) depthPointer);

    int colorDims[3] = {depth_intrin.height, depth_intrin.width, 3};
    PyArrayObject* colorFrame = PyArray_SimpleNewFromData(3, colorDims, NPY_UINT8, (void*)colorPointer);
    return Py_BuildValue("(O,O)", PyArray_Return(depthFrame), PyArray_Return(colorFrame));
}


static PyMethodDef lrs_methods[] = {
    { "getFrame", lrs_getFrame, METH_NOARGS,
    "get Intel Realsense Frame from R200" },
    {"startStream", lrs_startStream, METH_NOARGS,
    "Initialize RealSense camera. Returns 1 if success, NULL if failure" },
    {"stopStream", lrs_stopStream, METH_NOARGS,
    "Stop RealSense camera. Returns 1 if success, NULL if failure. DEVICE NEEDS REBOOT AFTER"},
    { NULL,              NULL }           /* sentinel */
};

void
initlrs(void)
{
    PyImport_AddModule("lrs");
    Py_InitModule("lrs", lrs_methods);
}

/* Example of embedding Python in another program */

#include "Python.h"
#include <numpy/arrayobject.h>
#include <librealsense/rs.h>
#include <librealsense/rsutil.h>

#define DEBUG_PRINT 0

rs_context * ctx;
rs_device * dev;
float depth_scale;
rs_error * e = 0;
rs_intrinsics depth_intrin, color_intrin;
rs_extrinsics depth_to_color;

//extern int getQuadCenter();

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

//int main(int argc, char **argv)
//{
//    
//}

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

    depth_scale = rs_get_device_depth_scale(dev, &e);
    if (check_error()) return NULL;

    return Py_BuildValue("f", depth_scale);
}

static PyObject *
lrs_stopStream(PyObject *self, PyObject* args)
{
    rs_stop_device(dev, &e);
    if (check_error()) return NULL;
    return Py_BuildValue("i", 1);
}

//Returns (depth frame as Numpy Array, color frame as Numpy Array)
static PyObject *
lrs_getFrame(PyObject *self, PyObject* args)
{
    rs_wait_for_frames(dev, &e);
    /* Retrieve depth data, which was previously configured as a 640 x 480 image of 16-bit depth values */
    uint16_t * depthPointer = (uint16_t *)(rs_get_frame_data(dev, RS_STREAM_DEPTH, &e));
    if (check_error()) return NULL;

    //Get color data
    uint16_t*  colorPointer = (uint16_t *)(rs_get_frame_data(dev, RS_STREAM_COLOR_ALIGNED_TO_DEPTH, &e));
    if (check_error()) return NULL;

    uint16_t* dCopy = depthPointer;
    int  numNonZero = 0;
    for (int i = 0; i < depth_intrin.width * depth_intrin.height; i++) {
	   int pixel = *dCopy++;
	   if (pixel != 0) {
		  numNonZero++;
	   }
    }
    if(DEBUG_PRINT)  printf("Number of nonzero pixels in frame: %d\n", numNonZero);

    float pointCloud[numNonZero][4];
    dCopy = depthPointer;
    //Generate point cloud
    int dx, dy;
    int index = 0;
    for (dy = 0; dy < depth_intrin.height; ++dy)
    {
	   for (dx = 0; dx < depth_intrin.width; ++dx)
	   {
		  /* Retrieve the 16-bit depth value and map it into a depth in meters */
		  uint16_t depth_value = dCopy[dy * depth_intrin.width + dx];
		  float depth_in_meters = depth_value * depth_scale;
		  pointCloud[index][0] = depth_in_meters;

		  /* Skip over pixels with a depth value of zero, which is used to indicate no data */
		  if (depth_value == 0) continue;

		  /* Map from pixel coordinates in the depth image to pixel coordinates in the color image */
		  float depth_pixel[2] = { (float)dx, (float)dy };
		  float depth_point[3];// , color_point[3], color_pixel[2];//Maybe later
		  rs_deproject_pixel_to_point(depth_point, &depth_intrin, depth_pixel, depth_in_meters);
		  
		  
		  //Patrick code: Subject to RUD
		  for (int i = 1; i < 4; i++) {
			 pointCloud[index][i] = depth_point[i-1];
		  }
		  index++;
	   }
    }
    


    if (DEBUG_PRINT)  printf("Populated frame\n");

    //Depth x y z
    int pcDims[2] = { numNonZero, 4 };
    PyArrayObject* pointCloudNP = PyArray_SimpleNewFromData(2, pcDims, NPY_FLOAT32, (void*)&pointCloud);
    
    //Pack depth and color data into Numpy arrays
    int depthDims[2] = { depth_intrin.height, depth_intrin.width };
    PyArrayObject* depthFrame =   PyArray_SimpleNewFromData(2, depthDims, NPY_UINT16, (void*) depthPointer);

    int colorDims[3] = {depth_intrin.height, depth_intrin.width, 3};
    PyArrayObject* colorFrame = PyArray_SimpleNewFromData(3, colorDims, NPY_UINT8, (void*)colorPointer);
    return Py_BuildValue("(O,O,O)", PyArray_Return(depthFrame), PyArray_Return(colorFrame), PyArray_Return(pointCloudNP));
    //return Py_BuildValue("(O,O)", PyArray_Return(depthFrame), PyArray_Return(colorFrame));
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

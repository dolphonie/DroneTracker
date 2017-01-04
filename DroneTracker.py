import lrs
import time
import numpy as np
import cv2

DEBUG_PRINT = False
MAX_RANGE_DISP = 5#meters


print lrs.startStream()


#while True:
#    lrs.getFrame()
#    time.sleep(2)

#np.savetxt("img.txt", lrs.getFrame(), fmt="%d")


cv2.namedWindow('depth', cv2.WINDOW_NORMAL)
cv2.namedWindow('color', cv2.WINDOW_NORMAL)

while True:
    comped = lrs.getFrame()
    if DEBUG_PRINT:
        print type(comped)
        print np.shape(comped)

    rawDepth = comped[0]
    rangedDepth = np.empty(np.shape(rawDepth),dtype = np.uint16)
    np.divide(rawDepth,MAX_RANGE_DISP*1000/255,rangedDepth);
    im = np.array(rangedDepth, dtype = np.uint8)
    cv2.imshow('depth',im)

    color = comped[1]
    
    if DEBUG_PRINT: 
        print np.shape(color)

    cv2.imshow('color', color)

    cv2.waitKey(100)

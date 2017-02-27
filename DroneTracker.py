import lrs
import time
import numpy as np
import cv2

DEBUG_PRINT = True
MAX_RANGE_DISP = 5#meters
NOT_FOUND_SENTINEL_VALUE = -99


print lrs.startStream()


#while True:
#    lrs.getFrame()
#    time.sleep(2)


#for i in range(10):
#    lrs.getFrame()

#np.savetxt("img.txt", lrs.getFrame()[2], fmt="%f")


cv2.namedWindow('depth', cv2.WINDOW_NORMAL)
cv2.namedWindow('color', cv2.WINDOW_NORMAL)

while True:
    comped = lrs.getFrame()
    if DEBUG_PRINT:
        print 'xPos: ' + str(comped[3])
        print 'yPos: ' + str(comped[4])
        print 'xPixel: ' + str(comped[6])
        print 'yPixel: ' + str(comped[7])

    rawDepth = comped[0]
    rangedDepth = np.empty(np.shape(rawDepth),dtype = np.uint16)
    np.divide(rawDepth,MAX_RANGE_DISP*1000/255,rangedDepth)
    im = np.array(rangedDepth, dtype = np.uint8)
    cv2.imshow('depth',im)

    color = comped[1]
    if comped[5]!=NOT_FOUND_SENTINEL_VALUE:
        cv2.circle(color, (comped[6],comped[7]),30,(255,0,0),-1)

    #if DEBUG_PRINT: 
        #print np.shape(color)

    cv2.imshow('color', color)

    cv2.waitKey(100)

import lrs
import time
import numpy as np
import cv2

print lrs.startDepth()


#while True:
#    lrs.getFrame()
#    time.sleep(2)

#np.savetxt("img.txt", lrs.getFrame(), fmt="%d")


cv2.namedWindow('image', cv2.WINDOW_NORMAL)

while True:
    im = np.array(lrs.getFrame(), dtype = np.uint8)
    threshed = cv2.adaptiveThreshold(im, 2000, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
    print "img generated"
    cv2.imshow('image',threshed)
    cv2.waitKey(100)

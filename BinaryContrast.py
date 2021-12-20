import cv2 as cv 
import numpy as np
from numpy import int16, uint8, log2
from PIL import Image
import time

#grab laptop camera as object
camera = cv.VideoCapture(0)
#initial frame object
ret, img_initial = camera.read()
#resize ratio to decrease of pixels (changeable)
resize_ratio = 1
#pixel brightness change threshold (changeable)
pixel_thresh = 30

#initial temp frame 
img_initial = cv.resize(img_initial,(1280//resize_ratio,720//resize_ratio))
img_initial = cv.flip(img_initial,1)
img_initial = cv.cvtColor(img_initial, cv.COLOR_BGR2GRAY)

#craete temp 2s array of pixels
temp_cont_frame = np.zeros((img_initial.shape[0], img_initial.shape[1]))

#pixel changer function
def change_pixels(image,image_initial):
    #iterate through a frame to compare each pixel of current frame to temp frame
    temp_cont_frame[...] = 0
    dif_frame = image.astype(np.int) - image_initial
    temp_cont_frame[dif_frame > pixel_thresh] = 1
    temp_cont_frame[dif_frame < -(pixel_thresh)] = 1
    return temp_cont_frame
    
#-----------------MAIN LOOP------------------
while True:
    #current frame object
    ret, img = camera.read()

    #current frame 
    img = cv.resize(img,(1280//resize_ratio,720//resize_ratio))
    img = cv.flip(img,1)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    #call function to edit pixels
    cv.imshow("Video Feed", change_pixels(img, img_initial))
    #update temp frame to current frame
    img_initial = img
    
    if cv.waitKey(1) == ord("q"):
        break  
cv.destroyAllWindows()
camera.release()

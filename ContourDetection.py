import cv2 as cv 
import numpy as np
from numpy import histogram, int16, uint8, log2
from PIL import Image
import time

camera = cv.VideoCapture(0)
resize_ratio = 1

ret, img_initial = camera.read()
img_initial = cv.resize(img_initial,(1280//resize_ratio,720//resize_ratio))
img_initial = cv.flip(img_initial,1)
img_initial = cv.cvtColor(img_initial, cv.COLOR_BGR2GRAY)

kernel = np.ones((4,4),np.uint8)
kernel_conv = np.array(([-1,-1,-1],[-1,8,-1],[-1,-1,-1]),np.float32)

def convolve(image_initial, image):
    conv = cv.filter2D(image, -1, kernel_conv, borderType=cv.BORDER_CONSTANT)
    diff = cv.absdiff(image_initial,image)
    ret, thresh = cv.threshold(diff, 30, 255, cv.THRESH_BINARY)
    dilated = cv.dilate(thresh, kernel, iterations=1)
    contours, hierarchy = cv.findContours(dilated.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    valid_contours = []
    for cntr in contours:
        x,y,w,h = cv.boundingRect(cntr)
        valid_contours.append(cntr)
        cv.drawContours(conv, valid_contours, -1, (255,255,255), 2)
    return conv
    
#-----------------MAIN LOOP------------------
while True:
    #get image and setup
    ret, img = camera.read()
    img = cv.resize(img,(1280//resize_ratio,720//resize_ratio))
    img = cv.flip(img,1)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.GaussianBlur(img, (3,3), 0)


    #display image
    cv.imshow("Video Feed", convolve(img_initial, img))
    img_initial=img


    #kill
    if cv.waitKey(1) == ord("q"):
        break  
cv.destroyAllWindows()
camera.release()
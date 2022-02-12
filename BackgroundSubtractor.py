import cv2 as cv 
import numpy as np
from numpy import int16, uint8, log2
from PIL import Image
import time

camera = cv.VideoCapture(0)
resize_ratio = 1
backSubtractor = cv.createBackgroundSubtractorKNN()

def change_pixels(image):
    backSub = backSubtractor.apply(image)
    (cnts,_) = cv.findContours(backSub.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        x,y,w,h = cv.boundingRect(cnt)
    return backSub
    
#-----------------MAIN LOOP------------------
while True:
    ret, img = camera.read()

    img = cv.resize(img,(1280//resize_ratio,720//resize_ratio))
    img = cv.flip(img,1)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (3,3), 0)
    cv.imshow("Video Feed", change_pixels(img))
    
    if cv.waitKey(1) == ord("q"):
        break  
cv.destroyAllWindows()
camera.release()
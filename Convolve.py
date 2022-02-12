import cv2 as cv 
import numpy as np
from numpy import int16, uint8, log2
from PIL import Image
import time

camera = cv.VideoCapture(0)
resize_ratio = 1

kernel = np.array(([-1,-1,-1],[-1,8,-1],[-1,-1,-1]),np.float32)

faceCascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")


def convolve(image):
    faces = faceCascade.detectMultiScale(image, 1.3, 5)  
    for(x,y,w,h) in faces:
        image = cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
    conv = cv.filter2D(image, -1, kernel, borderType=cv.BORDER_CONSTANT)
    return conv
    
#-----------------MAIN LOOP------------------
while True:
    #get image and setup
    ret, img = camera.read()
    img = cv.resize(img,(1280//resize_ratio,720//resize_ratio))
    img = cv.flip(img,1)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img,(3,3),0)
    #display image
    cv.imshow("Video Feed", convolve(img))

    #kill
    if cv.waitKey(1) == ord("q"):
        break  
cv.destroyAllWindows()
camera.release()
import numpy as np
import cv2 as cv

#changeable variables
resize_ratio = 1 #imagesizing 
thresh = .55 #object detection threshold 
nms_threshold = .9 #0.1=no supress, 1=high supress 
kernel = np.array(([-1,-1,-1],[-1,8,-1],[-1,-1,-1]),np.float32) #convolusion kernel

#video capture object
camera = cv.VideoCapture(0)
classNames = []
classFile = '/Users/keshavshankar/VSCODEPROJECTS/DVS_EMULATOR/ObjectDetector/coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

#get file paths
weightsPath = "/Users/keshavshankar/VSCODEPROJECTS/DVS_EMULATOR/ObjectDetector/frozen_inference_graph.pb"
configPath = "/Users/keshavshankar/VSCODEPROJECTS/DVS_EMULATOR/ObjectDetector/objectModel.pbtxt"

#network object setup
network = cv.dnn_DetectionModel(weightsPath,configPath)
network.setInputSize(320,320)
network.setInputScale(1.0/ 127.5)
network.setInputMean((127.5, 127.5, 127.5))
network.setInputSwapRB(True)


def translate(image):#takes the image and resizes,flips,blurs,and gray scales
    image = cv.resize(image, (1280//resize_ratio,720//resize_ratio))
    image = cv.flip(image, 1)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.GaussianBlur(image, (3,3), 0)
    return image

#-------convolve function------ 
def convolve(image):#takes the image then convolves the image using the convolusion kernel 
    image = cv.filter2D(image, -1, kernel, borderType=cv.BORDER_CONSTANT)
    #image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,25,16)
    return image

#-----object detection function-----
def objectDetection(imageReg,convImg,personCount,checkInitial):#takes the convolved image, identifies objects, bounds objects with box and displays confidence of each prediction
    #objdet
    #convImg = cv.flip(convImg,1)
    classIds, confidences, bbox = network.detect(convImg,confThreshold=thresh)
    bbox = list(bbox)
    confidences = list(np.array(confidences).reshape(1,-1)[0])
    confidences = list(map(float,confidences))
    indices = cv.dnn.NMSBoxes(bbox,confidences,thresh,nms_threshold)
    if len(classIds) != 0:
        for i in indices:
            i = i[0]
            box = bbox[i]
            confidence = str(round(confidences[i]*100,2))
            x,y,w,h = box[0],box[1],box[2],box[3]
            cv.rectangle(convImg, (x,y), (x+w,y+h), color=(255,255,255), thickness=2)
            cv.putText(convImg, "Predicition: "+classNames[classIds[i][0]-1]+" "+confidence+"%",(x+10,y+20), cv.FONT_HERSHEY_PLAIN,1,color=(255,255,255),thickness=1)
            #count people passed
            if classNames[classIds[i][0]-1] == "person":
                check = True
                if checkInitial == False and check == True:
                    personCount += 1
            else:
                check = False
            checkInitial = check
    cv.putText(convImg, "People Detected: "+str(personCount),(1050,50), cv.FONT_HERSHEY_PLAIN,1,color=(255,255,255),thickness=1)
    return convImg,checkInitial,personCount

#counter
checkInitial = False 
personCount = 0

#------MAIN LOOP---------
while True:
    success,img = camera.read()
    img = cv.GaussianBlur(img, (3,3),0)
    imgTranslated = translate(img)
    convolusionImg = convolve(imgTranslated)
    convolusionImg = cv.cvtColor(convolusionImg,cv.COLOR_GRAY2BGR)
    storage = objectDetection(img,convolusionImg,personCount,checkInitial)
    (imageOut, checkI, count) = storage
    checkInitial = checkI
    personCount = count
    cv.imshow("Video Feed", imageOut)
    if cv.waitKey(1) == ord("q"):
        break  
cv.destroyAllWindows()
camera.release()
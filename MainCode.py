##################################################################
# MainCode.py
# Author: Paulius Brickus
# Date: 30/04/2021
# 
# Description: This code is the main code for the project it does object detection and and and lane detection  
#
# References:
#        - Evan Juras
#        - Harrison Kinsley / Sentdex
#        - Christopher Barnatt / ExplainingComputer.com
#        - OpenCv Python Tutorials
#        - www.docs.opencv.org
#        - https://pysource.com/
#        - Murtaza Hassan
#        
# Links:
#     - https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi
#     - https://github.com/sentdex/pygta5/
#     - https://www.explainingcomputers.com/rasp_pi_robotics.html
#     - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
#     - https://www.docs.opencv.org/master/da/d22/tutorial_py_canny.html
#     - https://pysource.com/2018/03/07/lines-detection-with-hough-transform-opencv-3-4-with-python-3-tutorial-21/
#     - https://www.murtazahassan.com
import cv2
import numpy as np
from numpy import asarray
from Motor import *            
PWM=Motor()

video = cv2.VideoCapture(0)
curveList = []
avgVal = 10

def WarpPerspective(img,):
    point1 = np.float32([[60,240] ,[580,240] ,[1,480] ,[639,480]]) # setting up points 
    point2 = np.float32([[0,0] ,[400,0] ,[0,600] ,[400,600]])
    
    matrix = cv2.getPerspectiveTransform(point1,point2)
    imgWarp = cv2.warpPerspective(img, matrix, (400,600))
    return imgWarp

    
#def canny(image):
#    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#    blur = cv2.GaussianBlur(gray, (5, 5), 0)
#    canny = cv2.Canny(blur, 50, 150)
#    return canny

def thres(img):
   # h_min = 0,h_max = 255,s_min = 0,s_max=56,v_min=36,v_max=255
    imgHsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    lowerWhite = np.array([0,0,107])             #min H,S,V Values
    upperWhite = np.array([255,47,255])        #max H,S,V Values
    maskedImage = cv2.inRange(imgHsv,lowerWhite,upperWhite)
    return maskedImage

def RoadDetection(img, minPer = 0.1, region = 1):
    
    if region ==1:                         # if region is one do pixel summation for whole image 
        ArrayValue = np.sum(img, axis = 0)
    else:                                 # else do it for the region defined 
        ArrayValue = np.sum(img[img.shape[0]//region:,:],axis = 0)
        #print(ArrayValue)
    maxValue = np.max(ArrayValue)
    minValue = minPer*maxValue  # mimimum value to be allowed as the road is min percentage * max Value of array
    #print(maxValue)
    indexArray = np.where(ArrayValue >= minValue)
    baseP = int(np.average(indexArray))
    #print(basePoint)
    imgHist = np.zeros((img.shape[0],img.shape[1], 3), np.uint8)# creating an empty image giving it the shape of the image
    
    for x, intensity in enumerate(ArrayValue):
        cv2.line(imgHist,(x,img.shape[0]),(x,img.shape[0]-intensity//255//region),(0,255,0),1)
        cv2.circle(imgHist,(baseP,img.shape[0]),20,(0,0,255),cv2.FILLED)
    return baseP, imgHist

    

while True:
    _, og = video.read()
    img = cv2.flip(og,-1)
    img2 = img.copy()
    cv2.circle(img, (60,240),5,(0,0,255), -1)
    cv2.circle(img, (580,240),5,(0,0,255), -1)
    cv2.circle(img, (1,480),5,(0,0,255), -1)
    cv2.circle(img, (639,480),5,(0,0,255), -1)
   
    # Warp
    imgOutput = WarpPerspective(img)
    # edge detection
    #canny_image = canny(imgOutput)
    HSVthres  = thres(imgOutput)
    # Histogram
    midPoint, imgHist = RoadDetection(HSVthres,minPer =0.5, region =4)
    basePoint, imgHist = RoadDetection(HSVthres,minPer =0.9, )
    curveRaw = basePoint - midPoint
    #print(basePoint-midPoint)
    curveList.append(curveRaw)
    if len(curveList)>avgVal:
        curveList.pop(0)
    curve = int(sum(curveList)/len(curveList))

    #lines = cv2.HoughLinesP(canny_image, 1, np.pi/150, 10, maxLineGap=50)
   # if lines is not None:
       # for line in lines:
           # x1, y1, x2, y2 = line[0]
          #  cv2.line(imgOutput, (x1, y1), (x2, y2), (0, 255, 0), 8)
    #Object detection
    #resize = cv2.resize(img2, (320,240))
    resize = cv2.resize(img2, (640,480))
    imgOD = asarray(resize)
    stopsign_cascade = cv2.CascadeClassifier('/home/pi/Desktop/L8_Project_SDC/stopsign_classifier.xml')
    car_cascade = cv2.CascadeClassifier('home/pi/Desktop/L8_Project_SDC/cars.xml')
    
    gray = cv2.cvtColor(imgOD,cv2.COLOR_BGR2GRAY)
    stopsign = stopsign_cascade.detectMultiScale(gray, 1.1, 5)
    #car =  car_cascade.detectMultiScale(gray, 1.3, 3)
    for (x,y,w,h) in stopsign:
        cv2.rectangle(imgOD,(x,y),(x+w,y+h),(255,255,0),3)
        print("Stop")
   # for (x,y,w,h) in car:
   #     cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),3)
   #     print("car")
    cv2.imshow("Result",imgOD)
    
    
    #for (x,y,w,h) in stopsign:
    #    cv2.rectangle(imgOD,(x,y),(x+w,y+h),(255,255,0),2)
    #cv2.imshow("Result",imgOD)
    print(basePoint)
    cv2.imshow("original Image", img)
    cv2.imshow("Out Image", imgOutput)
    #cv2.imshow("Output Image", canny_image)
    cv2.imshow("thres",HSVthres)
    cv2.imshow("histogram",imgHist)
 
    print (curve)
    if len(stopsign) != 0:          
        print("Stop Sign Detected")
        PWM.setMotorModel(0,0,0,0)
        break
        
       # print (curve)
    elif basePoint <=100:
        #print("Left")
        PWM.setMotorModel(1000,2000,-2000,-2000)
    elif basePoint >=285:
       # print("right")
        PWM.setMotorModel(-2000,-2000,2000,2000)
    elif curve <= 14 and curve >=-14:
        #print("forwad")
        PWM.setMotorModel(-1200,-1200,-1200,-1200)
    elif curve <= 30 and curve >= 15:
        #print("Right light")
        PWM.setMotorModel(-2000,-2000,2000,2000)
    elif curve <= 85 and curve >= 31:
        #print("right")
        PWM.setMotorModel(-2000,-2000,2000,2000)
    elif curve >= -30 and curve <= -15:
        #print("left light")
        PWM.setMotorModel(1000,2500,-2500,-2500)
    elif curve >= -85 and curve <= -31:
        #print("left")
        PWM.setMotorModel(1000,2500,-2500,-2500)
      
    else:
        PWM.setMotorModel(0,0,0,0)
        

        
    key = cv2.waitKey(1)
    if key == 27:
        PWM.setMotorModel(0,0,0,0)
         #mdev.move(0,0,0)
    
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()
#mdev.move(0,0)
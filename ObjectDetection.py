##################################################################
# Objectdetection.py
# Author: Paulius Brickus
# Date: 30/04/2021
# 
# Description: this code shows both the stop sign and car detection at the same time   
#
# References:
#        - Evan Juras
#        - Harrison Kinsley / Sentdex
#        - Christopher Barnatt / ExplainingComputer.com
#        - OpenCv Python Tutorials
#        - www.docs.opencv.org
#        - https://pysource.com/
#        - Murtaza Hassan
#        - stopsign cascade 
#        - harr cascade for car
#        
# Links:
#     - https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi
#     - https://github.com/sentdex/pygta5/
#     - https://www.explainingcomputers.com/rasp_pi_robotics.html
#     - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
#     - https://www.docs.opencv.org/master/da/d22/tutorial_py_canny.html
#     - https://pysource.com/2018/03/07/lines-detection-with-hough-transform-opencv-3-4-with-python-3-tutorial-21/
#     - https://www.murtazahassan.com
#     - https://github.com/markgaynor/stopsigns/blob/master/stopsign_classifier.xml
#     - https://github.com/andrewssobral/vehicle_detection_haarcascades/blob/master/cars.xml
import cv2
import numpy
from numpy import asarray

video = cv2.VideoCapture(0)

stopsign_cascade = cv2.CascadeClassifier('/home/pi/Desktop/objectDetect/OD2/stopsign_classifier.xml')
car_cascade = cv2.CascadeClassifier('/home/pi/Desktop/objectDetect/OD2/cars.xml')



while True:
    _, og = video.read()
    img = cv2.flip(og,-1)
    resize = cv2.resize(img, (640,480))
    image = asarray(resize)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    stopsign = stopsign_cascade.detectMultiScale(gray, 1.2, 3)
    car =  car_cascade.detectMultiScale(gray, 1.3, 3)
    #print(stopsign)
    for (x,y,w,h) in stopsign:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),3)
        print("Stop")
    for (x,y,w,h) in car:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),3)
        print("car")
    #stopSign = Detecter(image,stopsign_cascade,car_cascade)
    #Car = Detecter2(image,car_cascade)
    cv2.imshow("Result",image)
    key = cv2.waitKey(1)
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()
import cv2
import time

imagePath = "D:/facerec/testimg/img.jpg"


#faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

image = cv2.imread(imagePath)
cv2.imshow("Faces found", image)
cv2.waitKey(2000)


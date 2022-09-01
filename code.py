import cv2
import time

imagePath = 'D://facerec//abbatest.png'


faceCascade = cv2.CascadeClassifier('D:/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')

image = cv2.imread(imagePath)
cv2.imshow("Faces found", image)
cv2.waitKey(2000)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Faces found", gray)
cv2.waitKey(2000)



faces = faceCascade.detectMultiScale(
gray, 
scaleFactor = 1.1,
minNeighbors = 5,
minSize = (30, 30),
maxSize = (1620,1080),
flags = cv2.CASCADE_SCALE_IMAGE
)



print ("Found {0} faces!".format(len(faces)))


#draw a rectangle around the faces
for(x,y,w,h) in faces:
	cv2.rectangle(image,(x,y), (x+w, y+h), (0,255,0), 2)

cv2.imshow("Faces found", image)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('messigray.png',img)
    cv2.destroyAllWindows()
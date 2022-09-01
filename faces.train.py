import os
import numpy as np
from PIL import Image
import cv2
import time
import pickle


faceCascade = cv2.CascadeClassifier('D:/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')
BASE_DIR = os.path.dirname(os.path.abspath("D:/facerec/known"))
image_dir = os.path.join(BASE_DIR, "known")

recognizer = cv2.face.LBPHFaceRecognizer_create()


current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file)
			label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
			#print(label,path)
			if label in label_ids:
				pass
			else:
				label_ids[label] = current_id
				current_id += 1

			id_ = label_ids[label]
			#print(label_ids)
			#y_labels.append(label)#some number
			#x_train.append(path)#verrfiy this image, turn it in to a  NUMPY array, GRAY
			pil_image = Image.open(path).convert("L")
			image_array = np.array(pil_image, "uint8")
			#print(image_array)
			faces = faceCascade.detectMultiScale(image_array, 1.5,  5)


			for(x,y,w,h) in faces :
				roi = image_array[y: y+h, x: x+w]
				x_train.append(roi)
				y_labels.append(id_)

#print(y_labels)
#print(x_train)


with open("labels.pickle", 'wb') as f:
	pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")

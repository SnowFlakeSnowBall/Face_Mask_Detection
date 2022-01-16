import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


faceCascade = cv2.CascadeClassifier("C:\\Users\\Mongol\\Desktop\\cv\\haarcascade_frontalface_default.xml")
model = load_model("C:\\Users\\Mongol\\Desktop\\cv\\model.h5")


cap = cv2.VideoCapture(0)

while (1):

	_, frame = cap.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray,
										scaleFactor = 1.1,
										minNeighbors = 5,
										minSize = (60, 60),
										flags = cv2.CASCADE_SCALE_IMAGE)

	faces_list = []
	preds = []
	print(faces)
	for (x, y, w, h) in faces:
		face_frame = frame[y:y + h, x:x + w]
		face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
		face_frame = cv2.resize(face_frame, (160, 160))
		face_frame = img_to_array(face_frame)
		face_frame = np.expand_dims(face_frame, axis = 0)
		face_frame =  preprocess_input(face_frame)
		#faces_list.append(face_frame)
		faces_list = face_frame
		if len(faces_list) > 0:
			preds = model.predict(faces_list)
		for pred in preds:
			(mask, withoutMask) = pred
	  
		label = "No Mask" if mask > withoutMask else "Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		label = "{}: {}%".format(label, int(max(mask, withoutMask) * 100))
		cv2.putText(frame, label, (x, y - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
		cv2.rectangle(frame, (x, y), (x + w, y + h),color, 3)

	cv2.imshow("frame", frame)
	cv2.waitKey(50)




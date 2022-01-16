import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

faceCascade = cv2.CascadeClassifier("C:\\Users\\Mongol\\Desktop\\cv\\haarcascade_frontalface_default.xml")
model = load_model("C:\\Users\\Mongol\\Desktop\\cv\\model.h5")

def face_mask_detector(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_t = faceCascade.detectMultiScale(gray,
                                        scaleFactor = 1.1,
                                        minNeighbors = 5,
                                        minSize = (60, 60),
                                        flags = cv2.CASCADE_SCALE_IMAGE)
    w_t = 0
    h_t = 0
    for i in range(len(faces_t)):
        w_t += faces_t[i][2] 
        h_t += faces_t[i][3]
    w_t_f = w_t // len(faces_t) - 15
    h_t_f = h_t // len(faces_t) - 15
    print(w_t_f, h_t_f)

    faces = faceCascade.detectMultiScale(gray,
                                        scaleFactor = 1.1,
                                        minNeighbors = 5,
                                        minSize = (w_t_f, h_t_f),
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


    return frame

input_image = cv2.imread("C:\\Users\\Mongol\\Desktop\\cv\\mask.jpeg")
output = face_mask_detector(input_image)
cv2.imshow("Mongol", output)
cv2.waitKey(5000)








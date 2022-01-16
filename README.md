# Face_Mask_Detection

## How to use:

**1. Install reuirements;**

**2. PLatforms:**
- For Windows: Run like usual python programm;
- For MacOS: change path in code for .xml, .h5, .photo:
```
faceCascade = cv2.CascadeClassifier("C:\\Users\\Mongol\\Desktop\\cv\\haarcascade_frontalface_default.xml")
model = load_model("C:\\Users\\Mongol\\Desktop\\cv\\model.h5")
input_image = cv2.imread("C:\\Users\\Mongol\\Desktop\\cv\\mask.jpeg")
```
- For Linux: Has not been tested

**3. Allow the programm to use webcam (if required)**

## FILES:

**CSV.py** - Works with photo. 

**RT_MD_OCV.py** - Works with webcam. 

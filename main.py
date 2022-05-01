from ctypes import resize
import streamlit as st
import numpy as np
from cv2 import cv2
import tensorflow as tf
import tensorflow_hub as hub
from mtcnn import MTCNN

    
    
def detect_faces(resized,model):
    resized = np.array([resized])
    resized = resized / 255.0
    
    
    model_predictions = model.predict(resized)
    print(model_predictions)
    return np.argmax(model_predictions[0])

 
st.title("Face-Mask Detector App")
face_cascade = cv2.CascadeClassifier('./model/haarcascade_frontalface.xml')

img_file_buffer = st.camera_input("Take a picture")
detector = MTCNN()


if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    cv2_col = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    model = tf.keras.models.load_model('./model/model.h5',custom_objects={'KerasLayer':hub.KerasLayer} )
    print("Model Has Been Loaded")
    savedModel = model.load_weights('./model/modelweights.h5')
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    X = []
    detected_faces = detector.detect_faces(cv2_col)
    print("Detected Faces")
    print(detected_faces)
    
    if(len(detected_faces)==0):
        st.write("No Face Detected")
    else:
        mask_detector_values = [
            "withmask",
            "withoutmask",
            "maskweared_incorrect"
        ]
    
        for face in detected_faces:
            (x,y,w,h) = face['box']
            if face['confidence'] < 0.7: 
                continue
            cv2_img = cv2.rectangle(cv2_col,(x,y),(x+w,y+h),(255,0,0),2)
            roi_color = cv2_img[y:y+h, x:x+w]
            resized = cv2.resize(roi_color, (224,224))            
            val = detect_faces(resized,model)
            cv2.putText(cv2_col, f'{mask_detector_values[val]}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        st.image(cv2_col)
    


    

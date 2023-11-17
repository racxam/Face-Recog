import cv2
from keras.models import model_from_json
import numpy as np
# from keras_preprocessing.image import load_img
json_file = open("Trainedmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("Trainedmodel.h5")
haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

labels = {0 : 'Angry', 1 : 'Disgust', 2 : 'Fear', 3 : 'Happy', 4 : 'Neutral', 5 : 'Sad', 6 : 'Surprise'}

def emotion_detector(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(image,1.3,5)

    for (p,q,r,s) in faces:
        image = gray[q:q+s,p:p+r]
        cv2.rectangle(image,(p,q),(p+r,q+s),(0,255,0),2)
        image = cv2.resize(image,(48,48))
        img = extract_features(image)
        pred = model.predict(img)
        prediction_label = labels[pred.argmax()]
        print("Predicted Output:", prediction_label)
        return {"emotion": prediction_label}
  



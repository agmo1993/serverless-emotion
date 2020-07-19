import numpy as np
import cv2
from urllib.request import urlopen
from processing import data_processing
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''
      Level | Level for Humans | Level Description                  
 -------|------------------|------------------------------------ 
  0     | DEBUG            | [Default] Print all messages       
  1     | INFO             | Filter out INFO messages           
  2     | WARNING          | Filter out INFO & WARNING messages 
  3     | ERROR            | Filter out all messages 
'''



def evaluate_sample_model(image_path):
    """Evaluates a test video from path using following models:
    `sample_model.hdf5` pre-built model to detect facial expression/emotion.

    :param path: an absolute path to the test video
    :type path: String
    """
    folder = '/Users/haiho/PycharmProjects/Serverless-emotion/serverless-emotion/cv/model/sample_model.hdf5'
    classifier = load_model(folder)
    classes = {0: 'Angry', 1: 'Disgust', 2: 'Fearful', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprised'}


    # grab the image
    req = urlopen(image_path)
    image_arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image = cv2.imdecode(image_arr, 3)

    if image is None:
        return "no image found"

    rects, roi = data_processing.face_det_crop_resize(image)

    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = roi.astype("float") / 255.0
    # make a prediction on the ROI, then lookup the class
    try:
        roi = img_to_array(roi)
    except:
        return "no face detected"
    roi = np.expand_dims(roi, axis=0)
    preds = classifier.predict(roi)[0]

    classes = {0: 'Happy', 1: 'Neutral', 2: 'Sad'}
    preds = preds[3:6]
    label = classes[preds.argmax()]
    print('label: {} | prediction: {}'.format(label, preds))

    return label, preds.argmax()

def evaluate_model_smile(image_path):
    """Evaluates a test video from path using following models:
    `haarcascade_frontalface_alt.xml` pre-built model to detect face(s) in the video
    `haarcascade_smile.xml` pre-built model to detect smile within face's zone
    References: https://github.com/opencv/opencv/tree/master/data/haarcascades

    :param path: an absolute path to the test video
    :type path: String
    """
    faceCascade = cv2.CascadeClassifier('/Users/haiho/PycharmProjects/Serverless-emotion/serverless-emotion/cv/model/haarcascade_frontalface_alt.xml')
    smileCascade= cv2.CascadeClassifier('/Users/haiho/PycharmProjects/Serverless-emotion/serverless-emotion/cv/model/haarcascade_smile.xml')

    # grab the image
    req = urlopen(image_path)
    image_arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image = cv2.imdecode(image_arr, 3)

    if image is None:
        return "no image found"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.equalizeHist(gray)

    faces = faceCascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=7, minSize=(200, 200))
    status = "no face"
    for (x,y,w,h) in faces:
        status = "face"
        face_gray = gray[y:y+h, x:x+w]
        smiles = smileCascade.detectMultiScale(face_gray, scaleFactor=2, minNeighbors=20, minSize=(40, 25))
        if len(smiles) != 0:
            status = "smiling"
        else:
            status = "no smiling"
        print(status)
    print(status)
    return status

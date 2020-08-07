import numpy as np
import cv2
from urllib.request import urlopen
from processing import data_processing
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from fastai.vision import *
from fastai.torch_core import *
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import requests

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
    folder = '../model/sample_model.hdf5'
    classifier = load_model(folder)

    # Load the image
    req = urlopen(image_path)
    image_arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image = cv2.imdecode(image_arr, 3)
    if image is None:
        print("no image found")
        return "no image found"

    # Detect the face
    faceCascade = cv2.CascadeClassifier('../model/haarcascade_frontalface_alt.xml')
    faces = faceCascade.detectMultiScale(image, 1.3, 5)
    if len(faces) == 0:
        print("no face found")
        return "no face found"

    (x,y,w,h) = faces[0]
    face_clip = image[y:y+h, x:x+w]  #cropping the face in image
    face_resized = cv2.resize(face_clip, (350, 350))  #resizing image
    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    face_float = face_gray.astype("float") / 255.0
    try:
        roi = img_to_array(face_float)
    except:
        print("image to array error")
        return "image to array error"

    # Detect the smile
    if is_smiling(face_resized):
        print('smile')
        compensate_probability = [0.1, 0.1, 0.0]
    else:
        print('no smile')
        compensate_probability = [0.0, 0.1, 0.1]

    roi = np.expand_dims(roi, axis=0)
    preds = classifier.predict(roi)[0]

    classes = {0: 'Happy', 1: 'Neutral', 2: 'Sad'}
    preds = preds[3:6]
    for index, pred in enumerate(preds):
        if pred < 0.7:
            preds[index] += compensate_probability[index]
    # preds = preds + [p_happy, p_neutral, p_sad]
    label = classes[preds.argmax()]
    print('label: {} | prediction: {}'.format(label, preds))

    return label, preds.argmax()

def is_smiling(image):
    smileCascade= cv2.CascadeClassifier('../model/haarcascade_smile.xml')
    smiles = smileCascade.detectMultiScale(image, scaleFactor=1.8, minNeighbors=20)#, minSize=(25, 25))
    if len(smiles) > 0:
        return True
    return False

def evaluate_model_smile(image_path):
    """Evaluates a test video from path using following models:
    `haarcascade_frontalface_alt.xml` pre-built model to detect face(s) in the video
    `haarcascade_smile.xml` pre-built model to detect smile within face's zone
    References: https://github.com/opencv/opencv/tree/master/data/haarcascades

    :param path: an absolute path to the test video
    :type path: String
    """
    faceCascade = cv2.CascadeClassifier('../model/haarcascade_frontalface_alt.xml')
    smileCascade= cv2.CascadeClassifier('../model/haarcascade_smile.xml')

    # grab the image
    req = urlopen(image_path)
    image_arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image = cv2.imdecode(image_arr, 3)

    if image is None:
        return "no image found"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=7, minSize=(150,150))

    if len(faces) == 0:
        return "no face detected"

    for (x,y,w,h) in faces:
        face_gray = gray[y:y+h, x:x+w]
        smiles = smileCascade.detectMultiScale(face_gray, scaleFactor=2, minNeighbors=20, minSize=(40, 25))
        if len(smiles) != 0:
            status = "smiling"
        else:
            status = "no smiling"
        # overlay our detected emotion on our pic
        # print(status)
        # cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
        # cv2.putText(gray, status, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        # cv2.imshow("Emotion Detector", gray)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    # print(status)
    return status


def evaluate_fastai_model(image_path):
    model_dir = "/Users/haiho/PycharmProjects/Serverless-emotion/serverless-emotion/cv/model"
    learn = load_learner(model_dir, "fastai_model.pkl")

    # Read image from url
    response = requests.get(image_path, stream=True)
    im = Image.open(io.BytesIO(response.content))
    print(type(im))

    pred_class, pred_idx, outputs = learn.predict(im)
    result = dict()
    result["emotion"] = learn.data.classes[pred_idx]
    result["confident level"] = float(max(outputs))
    print(result)
    return result

def evaluate_fastai_model1(video_path):
    model_dir = "/Users/haiho/PycharmProjects/Serverless-emotion/serverless-emotion/cv/model"
    learn = load_learner(model_dir, "fastai_model.pkl")
    vs = cv2.VideoCapture(video_path)
    while True:
        ret, frame = vs.read()
        if frame is None:
            break
        rects, roi = data_processing.face_det_crop_resize(frame)
        t = torch.tensor(np.ascontiguousarray(np.flip(roi, 2)).transpose(2,0,1)).float()/255
        img = vision.Image(t)
        pred_class, pred_idx, outputs = learn.predict(img)
        result = dict()
        result["emotion"] = learn.data.classes[pred_idx]
        result["confident level"] = float(max(outputs))
        print(result)
        label_position = (rects[0][0] + int((rects[0][1]/2)), abs(rects[0][2] - 10))
        cv2.putText(frame, result["emotion"], label_position , cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)
        cv2.imshow("Emotion Detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    return
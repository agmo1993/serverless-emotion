import numpy as np
import cv2
from urllib.request import urlopen
##from processing import data_processing
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

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
    folder = '/home/app/function/sample_model.hdf5'
    classifier = load_model(folder)
    
    # Load the image
    req = urlopen(image_path)
    image_arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image = cv2.imdecode(image_arr, 3)
    if image is None:
        #print("no image found")
        return "no image found", 0

    # Detect the face
    faceCascade = cv2.CascadeClassifier('/home/app/function/haarcascade_frontalface_alt.xml')
    faces = faceCascade.detectMultiScale(image, 1.3, 5)
    if len(faces) == 0:
        #print("no face found")
        return "no face found", 0

    (x,y,w,h) = faces[0]
    face_clip = image[y:y+h, x:x+w]  #cropping the face in image
    face_resized = cv2.resize(face_clip, (350, 350))  #resizing image
    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    face_float = face_gray.astype("float") / 255.0
    try:
        roi = img_to_array(face_float)
    except:
        #print("image to array error")
        return "image to array error", 0

    # Detect the smile
    if is_smiling(face_resized):
        #print('smile')
        compensate_probability = [0.1, 0.1, 0.0]
    else:
        #print('no smile')
        compensate_probability = [0.0, 0.1, 0.1]

    roi = np.expand_dims(roi, axis=0)
    preds = classifier.predict(roi)[0]

    classes = {0: 'Happy', 1: 'Neutral', 2: 'Sad'}
    preds = preds[3:6]
    for index, pred in enumerate(preds):
        if pred < 0.7:
            preds[index] += compensate_probability[index]
    
    label = classes[preds.argmax()]
    #print('label: {} | prediction: {}'.format(label, preds))
    preds = preds[preds.argmax()]
    return label, preds

def is_smiling(image):
    smileCascade= cv2.CascadeClassifier('/home/app/function/haarcascade_smile.xml')
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
    faceCascade = cv2.CascadeClassifier('/home/app/function/haarcascade_frontalface_alt.xml')
    smileCascade= cv2.CascadeClassifier('/home/app/function/haarcascade_smile.xml')

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

def face_det_crop_resize(img_gray):
    """Returns a resized cropped (350, 350) grayscale image.

    :param img_gray: an original grayscale image
    :type img_gray: cv2 image
    :return: a cropped (350, 350) grayscale image
    :rtype: cv2 image
    """
    faceCascade = cv2.CascadeClassifier('/home/app/function/haarcascade_frontalface_alt.xml')
    faces = faceCascade.detectMultiScale(img_gray, 1.3, 5)
    rects = []
    for (x,y,w,h) in faces:
        face_clip = img_gray[y:y+h, x:x+w]  #cropping the face in image
        img_gray_resize = cv2.resize(face_clip, (350, 350))  #resizing image
        rects.append((x,w,y,h))
    return rects, img_gray_resize

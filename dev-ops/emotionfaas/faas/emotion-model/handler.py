import numpy as np
from urllib.request import urlopen
import cv2
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from evaluating import model_evaluation
import json

def handle(req):
    """handle a request to the function
    Args:
        req (str): request body
    """
    url = req
    image = Image.open(urlopen(url))
    width, height = image.size
    labels, prediction = model_evaluation.evaluate_sample_model(url)

    return json.dumps({"prediction": labels, "Confidence level" : str(round(prediction,2))})

##def evaluate_sample_model(image_path):
##    """Evaluates a test video from path using following models:
##    `sample_model.hdf5` pre-built model to detect facial expression/emotion.
##
##    :param path: an absolute path to the test video
##    :type path: String
##    """
##    folder = '/home/app/function/sample_model.hdf5'
##    classifier = load_model(folder)
##    classes = {0: 'Angry', 1: 'Disgust', 2: 'Fearful', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprised'}
##
##
##    # grab the image
##    req = urlopen(image_path)
##    image_arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
##    image = cv2.imdecode(image_arr, 3)
##
##    if image is None:
##        return "no image found"
##
##    rects, roi = face_det_crop_resize(image)
##
##    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
##    roi = roi.astype("float") / 255.0
##    # make a prediction on the ROI, then lookup the class
##    try:
##        roi = img_to_array(roi)
##    except:
##        return "no face detected"
##    roi = np.expand_dims(roi, axis=0)
##    preds = classifier.predict(roi)[0]
##
##    classes = {0: 'Happy', 1: 'Neutral', 2: 'Sad'}
##    preds = preds[3:6]
##    label = classes[preds.argmax()]
##    print('label: {} | prediction: {}'.format(label, preds))
##
##    return label, preds[preds.argmax()]

##def face_det_crop_resize(img_gray):
##    """Returns a resized cropped (350, 350) grayscale image.
##
##    :param img_gray: an original grayscale image
##    :type img_gray: cv2 image
##    :return: a cropped (350, 350) grayscale image
##    :rtype: cv2 image
##    """
##    faceCascade = cv2.CascadeClassifier('/home/app/function/haarcascade_frontalface_alt.xml')
##    faces = faceCascade.detectMultiScale(img_gray, 1.3, 5)
##    rects = []
##    for (x,y,w,h) in faces:
##        face_clip = img_gray[y:y+h, x:x+w]  #cropping the face in image
##        img_gray_resize = cv2.resize(face_clip, (350, 350))  #resizing image
##        rects.append((x,w,y,h))
##    return rects, img_gray_resize

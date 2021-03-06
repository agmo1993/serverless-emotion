import cv2

def face_det_crop_resize(img_gray):
    """Returns a resized cropped (350, 350) grayscale image.

    :param img_gray: an original grayscale image
    :type img_gray: cv2 image
    :return: a cropped (350, 350) grayscale image
    :rtype: cv2 image
    """
    faceCascade = cv2.CascadeClassifier('../model/haarcascade_frontalface_alt.xml')
    faces = faceCascade.detectMultiScale(img_gray, 1.3, 5)
    if len(faces) > 0:
        for (x,y,w,h) in faces:
            face_clip = img_gray[y:y+h, x:x+w]  #cropping the face in image
            img_gray_resize = cv2.resize(face_clip, (224, 224))  #resizing image
        return img_gray_resize
    return None

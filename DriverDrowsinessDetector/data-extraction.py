import cv2
import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np

# vidcap = cv2.VideoCapture('raw_data/1-FemaleNoGlasses.avi')



# Function to convert RGB Image to GrayScale
def convertImageToGray(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.imwrite(path, image)

def face_coordinates(path):

    ds_factor = 0.5
    face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    if face.empty():
        raise IOError('Unable to load the face cascade classifier xml file')
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_rects = face.detectMultiScale3(gray, minSize=(90,45), minNeighbors=14, outputRejectLevels=True)
    print(face_rects, '  this is face recs')
    return face_rects[0][0]

def mouth_coordinates(path):

    ds_factor = 0.5

    mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    if mouth_cascade.empty():
        raise IOError('Unable to load the mouth cascade classifier xml file')
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mouth_rects = mouth_cascade.detectMultiScale3(gray, minSize=(90,45), minNeighbors=14, outputRejectLevels=True)

    return mouth_rects[0][0]

def crop_face(path, output):
    img = cv2.imread(path)
    print(face_coordinates(path))
    x,y,w,h = face_coordinates(path)
    crop_img = img[y:y+h, x:x+w]
    cv2.imwrite(output, crop_img)

def crop_mouth(path, output):
    img = cv2.imread(path)
    x,y,w,h = mouth_coordinates(path)
    crop_img = img[y:y+h, x:x+w]
    cv2.imwrite(output, crop_img)



def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        return cv2.imwrite("raw_data/cropped_gray/image"+str(count)+".jpg", image)     # save frame as JPG file
    return hasFrames
# sec = 0
# frameRate = 4 #//it will capture image in each 2 second
# count=1
# success = getFrame(sec)
# while success:
#     count = count + 1
#     sec = sec + frameRate
#     sec = round(sec, 2)
#     success = getFrame(sec)
var = "/Users/kimfriedel/code/Pr4t3/DriverDrowsinessDetector/raw_data/cropped_gray/image1.jpg"


path_departure = '/Users/kimfriedel/code/Pr4t3/DriverDrowsinessDetector/raw_data/cropped_gray/image1.jpg'
path_arrival = '/Users/kimfriedel/code/Pr4t3/DriverDrowsinessDetector/raw_data/face_cropped/image2.jpg'
crop_face(path_departure, path_arrival)

path_final = '/Users/kimfriedel/code/Pr4t3/DriverDrowsinessDetector/raw_data/mouth_cropped/mouth2.jpg'
crop_mouth(path_arrival, path_final)

# # load and display an image with Matplotlib
# from matplotlib import image

# # load image as pixel array
# image = image.imread(var)
# # summarize shape of the pixel array
# print(image.dtype)
# print(image.shape)
# # display the array of pixels as an image
# pyplot.imshow(image)
# pyplot.show()

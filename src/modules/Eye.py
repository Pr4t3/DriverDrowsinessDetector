#imports
from tensorflow.keras.models import load_model
import cv2
import numpy as np

PATH_EYE_MODEL = 'models/model_eyes.h5'

class Eye:

    '''
    arguments:
    -- position: left or right
    '''

    def __init__(self, position, image):
        self.model = load_model(PATH_EYE_MODEL)
        self.classifier = cv2.CascadeClassifier(cv2.data.haarcascades + f'haarcascade_{position}eye_2splits.xml')
        self.image = image
        self.x, self.y, self.w, self.h = self.set_eye_coordinates()
        self.eye_image = self.image[self.y:self.y+self.h,self.x:self.x+self.w]
        self.open = self.set_open()


    def set_eye_coordinates(self):
        '''
        retruns the coordinates of a detected eye
        in a frame'''

        self.eyes = self.classifier.detectMultiScale(self.image)
        print(self.eyes, 'hier sind augen')
        if len(self.eyes) > 0:
            return self.eyes[0] #x,y,w,h -> coordinates in frame
        else:
            return [0,0,24,24]

    def set_open(self):
        '''
        sets the class attr open to True of False, based on model's prediction'''

        eye = cv2.resize(self.image,(24,24))
        eye = eye/255
        eye= eye.reshape(24,24,-1)
        eye = np.expand_dims(eye,axis=0)
        prediction = self.model.predict_classes(eye)
        if prediction[0] == 1:
            self.open = True
        else:
            self.open = False

    def draw_rectangle(self, color=(0,0,255), thickness=3):

        cv2.rectangle(self.image, (self.x,self.y) , (self.x+self.w,self.y+self.h) , color , thickness)
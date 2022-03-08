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

    def __init__(self, position, frame):
        self.frame = frame
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.model = load_model(PATH_EYE_MODEL)
        self.classifier = cv2.CascadeClassifier(cv2.data.haarcascades + f'haarcascade_{position}eye_2splits.xml')
        self.x, self.y, self.w, self.h = self.set_eye_coordinates()
        self.eye_image = self.gray[self.y:self.y+self.h,self.x:self.x+self.w]
        self.open = self.set_open()


    def set_eye_coordinates(self):
        '''
        retruns the coordinates of a detected eye
        in a frame'''

        self.eyes = self.classifier.detectMultiScale(self.gray)

        if len(self.eyes) > 0:
            return self.eyes[0] #x,y,w,h -> coordinates in frame
        else:
            return [0,0,24,24]

    def set_open(self):
        '''
        sets the class attr open to True of False, based on model's prediction'''

        eye = cv2.resize(self.eye_image,(24,24))
        eye = eye/255
        eye= eye.reshape(24,24,-1)
        eye = np.expand_dims(eye,axis=0)
        #prediction = self.model.predict_classes(eye)
        prediction_results = self.model.predict(eye)
        prediction = np.argmax(prediction_results, axis=1)

        if prediction[0] == 1:
            return True
        else:
            return False

    def draw_rectangle(self, color=(0,0,255), thickness=3):

        cv2.rectangle(self.frame, (self.x,self.y) , (self.x+self.w,self.y+self.h) , color , thickness)

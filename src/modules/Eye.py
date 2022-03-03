#imports
from keras.models import load_model
import cv2

PATH_EYE_MODEL = 'models/model_eyes.h5'

class Eye:

    '''
    arguments:
    -- position: left or right
    '''

    def __init__(self, position, image):
        self.open = self.set_open()
        self.model = load_model(PATH_EYE_MODEL)
        self.classifier = cv2.CascadeClassifier(cv2.data.haarcascade + f'\haarcascade_{position}eye_2splits.xml')
        self.image = image
        self.x, self.y, self.w, self.h = self.set_eye_coordinates()
        self.eye_image = self.image[self.y:self.y+self.h,self.x:self.x+self.w]

    def set_eye_coordinates(self):
        '''
        retruns the coordinates of a detected eye
        in a frame'''

        self.eyes = self.classifier.detectMultiScale(self.image)
        return self.eyes[0] #x,y,w,h -> coordinates in frame

    def set_open(self):
        '''
        sets the class attr open to True of False, based on model's prediction'''
        pred = self.model.predict_classes(self.eye_image)
        if pred:
            self.open = True
        else:
            self.open = False

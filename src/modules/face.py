from tensorflow.keras.models import load_model

import cv2
import numpy as np
from datetime import datetime


#module import
from src.modules.eye import Eye
from src.modules.tilt import Tilt
PATH = 'models/model_yawn.h5'

# #TODO  self.drowsy_score = 0
#         self.yawn_score = 0


class Face:
    '''
    arguments:
    -- frame: captured frame (webcam image for instance)
    '''
    def __init__(self, frame,
                 face_classifier=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'),
                 ):
        self.frame = frame
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.face_classifier = face_classifier
        self.model = load_model(PATH) ## TODO has to be the right model
        self.drowsy = False #TODO rethink
        self.x, self.y, self.w, self.h = self.face_set_coordinates()
        self.face_image = self.gray[self.y:self.y+self.h,self.x:self.x+self.w]
        self.left_eye = self.create_left_eye()
        self.right_eye = self.create_right_eye()
        self.yawn = self.set_yawn()
        self.tilt = Tilt(self.frame, 20)
        # self.is_tilt = self.tilt.set_tilt()



    def face_set_coordinates(self, minNeighbors=5, scaleFactor=1.1, minSize=(25,25)):
        '''
        retruns the coordinates of a detected face
        in a gray'''


        try:
            self.faces = self.face_classifier.detectMultiScale(self.gray,
                                                          minNeighbors=minNeighbors,
                                                              scaleFactor=scaleFactor,
                                                              minSize=minSize)
            return self.faces[0] #x,y,w,h -> coordinates in frame
        except:
            return [0,0,24,24]

    def set_yawn(self):
        '''
        sets yawn variable based on model's prediction'''

        face = cv2.resize(self.face_image,(24,24))
        face = face/255
        face= face.reshape(24,24,-1)
        face = np.expand_dims(face,axis=0)

        #prediction = self.model.predict_classes(face)
        prediction_results = self.model.predict(face)
        prediction = np.argmax(prediction_results, axis=1)

        # if prediction[0] == 1:
        #     print('*************** Yawning', datetime.now())
        if prediction[0] == 1:
            return False
        else:
            return True

    def create_right_eye(self):
        '''
        detects eyes and instanciate eyes classes'''

        return Eye('right', self.frame)

    def create_left_eye(self):
        '''
        detects eyes and instanciate eyes classes'''

        return Eye('left', self.frame)

    def draw_rectangle(self, color=(68,214,44), thickness=5):
        '''
        draws a rectangle from the face'''
        cv2.rectangle(self.frame, (self.x,self.y) , (self.x+self.w,self.y+self.h) , color , thickness)

    # def set_tilt(self):
    #     '''
    #     evaluates it the face is tilting'''

    #     return self.tilt.is_tilt


# closed=0
# open=1

# no_yawn= 0
# yawn = 1

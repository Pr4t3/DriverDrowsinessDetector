#imports
from keras.models import load_model
import cv2

#module import
from src.modules.Eye import Eye

PATH = 'models/model_yawn.h5'

# #TODO  self.drowsy_score = 0
#         self.yawn_score = 0


class Face:
    '''
    arguments:
    -- frame: captured frame (webcam image for instance)
    '''
    def __init__(self, frame,
                 face_classifier=cv2.CascadeClassifier(cv2.data.haarcascade + '\haarcascade_frontalface_alt.xml'),
                 ):
        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.face_classifier = face_classifier
        self.model = load_model(PATH) ## TODO has to be the right model
        self.yawn = self.set_yawn()
        self.drowsy = False #TODO rethink
        self.x, self.y, self.w, self.h = self.face_set_coordinates()
        self.face_image = frame[self.y:self.y+self.h,self.x:self.x+self.w]
        self.left_eye = self.create_left_eye('TODO')
        self.right_eye = self.create_right_eye('TODO')


    def face_set_coordinates(self, minNeighbors=5, scaleFactor=1.1, minSize=(25,25)):
        '''
        retruns the coordinates of a detected face
        in a frame'''
        self.faces = self.face_classifier.detectMultiScale(self.frame,
                                                              minNeighbors=minNeighbors,
                                                              scaleFactor=scaleFactor,
                                                              minSize=minSize)
        return self.faces[0] #x,y,w,h -> coordinates in frame

    def set_yawn(self):
        '''
        sets yawn variable based on model's prediction'''
        pred = self.model.predict_classes(self.face_image)
        if pred:
            self.yawn = True
        else:
            self.yawn = False

    def create_right_eye():
        '''
        detects eyes and instanciate eyes classes'''
        pass
        #return Eye('TODO')

    def create_left_eye():
        '''
        detects eyes and instanciate eyes classes'''
        pass
        #return Eye('TODO')

    def draw_rectangle():
        pass
        #cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 ) #TODO finish

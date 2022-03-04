from src.modules.Face import Face
from src.modules.Eye import Eye

import cv2
from pygame import mixer

# mixer.init()
# sound = mixer.Sound('alarm.wav')
test_im = cv2.imread('/Users/kimfriedel/code/Pr4t3/DriverDrowsinessDetector/DriverDrowsinessDetector/1318.jpg')

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# face = Face(test_im)

# score = 0
# ret, frame = cap.read()
# print(type(frame), frame.shape)
# print(type(test_im), test_im.shape)

# cap.release()
# cv2.destroyAllWindows()

while (True):

    open_eye_label = 'Open'
    ret, frame = cap.read() #TODO what is ret
    height, width = frame.shape[:2]

    face = Face(frame)

    #TODO toggle to turn off rectangle around face
    face.draw_rectangle()
    face.left_eye.draw_rectangle()

    if not face.left_eye.open and not face.left_eye.open:
        open_eye_label = 'Closed'

    cv2.putText(frame, open_eye_label, (10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)


    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

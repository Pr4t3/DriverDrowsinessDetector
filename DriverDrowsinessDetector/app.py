from src.modules.Face import Face


import cv2
from pygame import mixer

# mixer.init()
# sound = mixer.Sound('alarm.wav')
test_im = cv2.imread('/Users/kimfriedel/code/Pr4t3/DriverDrowsinessDetector/DriverDrowsinessDetector/1318.jpg')

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL


while (True):

    open_eye_label = 'Eyes Open'
    yawn_label = 'Not Yawning'
    tilt_label = 'Not Tilting'
    ret, frame = cap.read() #TODO what is ret
    height, width = frame.shape[:2]

    face = Face(frame)

    #TODO toggle to turn off rectangle around face
    face.draw_rectangle()
    face.left_eye.draw_rectangle()
    face.right_eye.draw_rectangle()

    if not face.left_eye.open and not face.left_eye.open:
        open_eye_label = 'Eyes Closed'

    cv2.putText(frame, open_eye_label, (10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

    if face.yawn:
        yawn_label = 'Yawning'

    cv2.putText(frame, yawn_label, (10,height-40), font, 1,(255,255,255),1,cv2.LINE_AA)

    if face.tilt:
        tilt_label = 'Tilting'

    cv2.putText(frame, tilt_label, (10,height-60), font, 1,(255,255,255),1,cv2.LINE_AA)


    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

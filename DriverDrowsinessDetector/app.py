
from src.modules.Face import Face

import streamlit as st
import cv2
#from pygame import mixer

# mixer.init()
# sound = mixer.Sound('DriverDrowsinessDetector/src/utils/alarm.wav')
test_im = cv2.imread('/Users/kimfriedel/code/Pr4t3/DriverDrowsinessDetector/DriverDrowsinessDetector/1318.jpg')

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
cache_len = 10
eye_cache = [0] * cache_len
yawn_cache = [0] * cache_len
tilt_cache = [0] * cache_len
danger_score = 0
run = st.checkbox('Run')

eye_button = st.checkbox("Show eye Detection")
yawn_button = st.checkbox("Show face Detection")
tilt_button = st.checkbox("Show tilt Detection")
confidence_threshold = st.slider('Alert threshold', 0.05,1.0, 0.8, 0.1)

FRAME_WINDOW = st.image([])

while (run):
    DANGER_SCORE = sum(eye_cache) + sum(tilt_cache)
    open_eye_label = 'Eyes Open'
    yawn_label = 'Not Yawning'
    tilt_label = 'Not Tilting'
    _,frame = cap.read()
    height, width = frame.shape[:2]

    face = Face(frame)

    #TODO toggle to turn off rectangle around face
    if yawn_button:
        face.draw_rectangle()

    if eye_button:
        face.left_eye.draw_rectangle()
        face.right_eye.draw_rectangle()

    if tilt_button:
        face.tilt.draw_face_axis()

    if not face.left_eye.open and not face.right_eye.open:
        open_eye_label = 'Eyes Closed'
        del eye_cache[0]
        eye_cache.append(1)
    else:
        del eye_cache[0]
        eye_cache.append(0)


    cv2.putText(frame, open_eye_label, (10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

    # Enabling yawn detection
    if yawn_button:
        face.draw_rectangle()
        cv2.putText(frame, yawn_label, (10,height-40), font, 1,(255,255,255),1,cv2.LINE_AA)
    if face.yawn:
        yawn_label = 'Yawning'
        del yawn_cache[0]
        yawn_cache.append(1)
    else:
        del yawn_cache[0]
        yawn_cache.append(0)




    if face.tilt.tilt:
        tilt_label = 'Tilting'
        del tilt_cache[0]
        tilt_cache.append(1)
    else:
        del tilt_cache[0]
        tilt_cache.append(0)


    if sum(eye_cache)/cache_len >= confidence_threshold:
        # sound.play()
        cv2.putText(frame, f'ALARM! Score: {DANGER_SCORE}', (10,height-80), font, 1,(255,0,0),1,cv2.LINE_AA)

    if sum(tilt_cache)/cache_len >= confidence_threshold:
        # sound.play()
        cv2.putText(frame, f'ALARM! Score: {DANGER_SCORE}', (10,height-80), font, 1,(255,0,0),1,cv2.LINE_AA)

    cv2.putText(frame, tilt_label, (10,height-60), font, 1,(255,255,255),1,cv2.LINE_AA)

    if DANGER_SCORE >= 8:
        bar_color = (0, 255, 0)
    elif DANGER_SCORE >= 5:
        bar_color = (255, 255, 0)
    else:
        bar_color = (255, 0, 0)

    cv2.line(frame, (180,height-20), (200+(DANGER_SCORE*10),height-20), bar_color, 4, cv2.LINE_AA)



    print(tilt_cache)
    FRAME_WINDOW.image(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# cap.release()
# cv2.destroyAllWindows()

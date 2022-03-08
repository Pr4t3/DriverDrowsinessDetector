import streamlit as st
from modules.face import Face

import cv2
from pygame import mixer


mixer.init()
#sound = mixer.Sound('src/utils/alarm.wav')


st.title("Driver Drowsiness Detector App")
run = st.radio('Webcam:', ('On', 'Off'))

FRAME_WINDOW = st.image([])
cam = cv2.VideoCapture(0)



while run:

    _, frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
else:
    st.write("Stopped")
    cam.release()

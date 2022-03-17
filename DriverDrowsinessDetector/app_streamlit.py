
from streamlit_webrtc import (
    AudioProcessorBase,
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
import cv2
from pygame import mixer
import streamlit as st
import av
import numpy as np
from src.modules.Face import Face


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
try:
    mixer.init()
    sound = mixer.Sound('DriverDrowsinessDetector/src/utils/alarm.wav')
except:
    print('no mixer')
    sound = None


font = cv2.FONT_HERSHEY_COMPLEX_SMALL
cache_len = 10
eye_cache = [0] * cache_len
yawn_cache = [0] * cache_len
tilt_cache = [0] * cache_len
DANGER_SCORE = 0

st.title("Driver Drowsiness Detector App")



def main():


    object_detection_page = "Real time drowsiness detection"

    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        [
            object_detection_page
        ],
    )
    st.subheader(app_mode)


    if app_mode == object_detection_page:
        app_object_detection()



def app_object_detection():


    eye_button = st.checkbox("Show eye Detection")
    yawn_button = st.checkbox("Show face Detection")
    tilt_button = st.checkbox("Show tilt Detection")

    DEFAULT_CONFIDENCE_THRESHOLD = 0.8

    confidence_threshold = st.slider(
        "Alert Threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
    )







    class Drowsiness(VideoProcessorBase):

        def __init__(self) -> None:
            pass

        def _annotate_image(self, frame):

            DANGER_SCORE = sum(eye_cache) + sum(tilt_cache)
            open_eye_label = 'Eyes Open'
            yawn_label = 'Not Yawning'
            tilt_label = 'Not Tilting'

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


            # filling the eyes quene
            if not face.left_eye.open and not face.right_eye.open:
                open_eye_label = 'Eyes Closed'
                del eye_cache[0]
                eye_cache.append(1)
            else:
                del eye_cache[0]
                eye_cache.append(0)

            cv2.putText(frame, open_eye_label, (10,height-20), font, 1.51,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(frame, open_eye_label, (10,height-20), font, 1.5,(255,255,255),1,cv2.LINE_AA)

            # Enabling yawn detection
            #make the font more bolt
            cv2.putText(frame, yawn_label, (10,height-50), font, 1.51,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(frame, yawn_label, (10,height-50), font, 1.5,(255,255,255),1,cv2.LINE_AA)

            #filling the yawn quene
            if face.yawn:
                yawn_label = 'Yawning'
                del yawn_cache[0]
                yawn_cache.append(1)
            else:
                del yawn_cache[0]
                yawn_cache.append(0)



            # filling the yawn quene
            if face.tilt.tilt:
                tilt_label = 'Tilting'
                del tilt_cache[0]
                tilt_cache.append(1)
            else:
                del tilt_cache[0]
                tilt_cache.append(0)



            # triggering alarm
            if sum(eye_cache)/cache_len >= confidence_threshold:
                if sound:
                    sound.play()
                cv2.putText(frame, f'ALARM! Score: {DANGER_SCORE}', (10,height-120), font, 1,(0,0,255),1,cv2.LINE_AA)

            if sum(tilt_cache)/cache_len >= confidence_threshold:
                if sound:
                    sound.play()
                cv2.putText(frame, f'ALARM! Score: {DANGER_SCORE}', (10,height-120), font, 1,(0,0,255),1,cv2.LINE_AA)


            #make the font more bolt
            cv2.putText(frame, tilt_label, (10,height-80), font, 1.51,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(frame, tilt_label, (10,height-80), font, 1.5,(255,255,255),1,cv2.LINE_AA)

            # bar color according to Danger score - red is dangerous, green if not
            if DANGER_SCORE >= 8:
                bar_color = (0, 0, 255)
            elif DANGER_SCORE >= 5:
                bar_color = (0, 127, 255)
            else:
                bar_color = (0, 255, 0)


            # Detection Bar
            cv2.rectangle(frame, (300,height-15), (450 + (DANGER_SCORE*10), height-35), bar_color, -1)
            cv2.putText(frame, 'Drowsiness', (310,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

            return frame

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            '''
            webrtc receiving function
            returns the image displayed'''
            image = frame.to_ndarray(format="bgr24")

            annotated_image = self._annotate_image(image)

            return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

    webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=Drowsiness,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )



if __name__ == '__main__':
    main()

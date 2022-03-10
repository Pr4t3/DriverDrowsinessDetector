from src.modules.Face import Face
import streamlit as st
import cv2
from pygame import mixer

import av
import cv2

mixer.init()
sound = mixer.Sound('src/utils/alarm.wav')

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
cache_len = 10
eye_cache = [0] * cache_len
yawn_cache = [0] * cache_len
tilt_cache = [0] * cache_len
danger_score = 0

st.title("Driver Drowsiness Detector App")
run = st.radio('Webcam:', ('On', 'Off'))






class VideoProcessor:
    def __init__(self):
        pass


    def recv(self, frame):
        eye_button = st.checkbox("Toggle eyes")
        yawn_button = st.checkbox("Toggle yawn")
        tilt_button = st.checkbox("Toggle tilt")

        frm = frame.to_ndarray(format="bgr24")

        danger_score = sum(eye_cache) + sum(tilt_cache)
        open_eye_label = 'Eyes Open'
        yawn_label = 'Not Yawning'
        tilt_label = 'Not Tilting'
        frame = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]

        face = Face(frame)

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

        if face.yawn:
            yawn_label = 'Yawning'
            del yawn_cache[0]
            yawn_cache.append(1)
        else:
            del yawn_cache[0]
            yawn_cache.append(0)


        cv2.putText(frame, yawn_label, (10,height-40), font, 1,(255,255,255),1,cv2.LINE_AA)

        if face.tilt.tilt:
            tilt_label = 'Tilting'
            del tilt_cache[0]
            tilt_cache.append(1)
        else:
            del tilt_cache[0]
            tilt_cache.append(0)


        if sum(eye_cache)/cache_len >= 0.8:
            sound.play()
            cv2.putText(frame, f'ALARM! Score: {danger_score}', (10,height-80), font, 1,(255,0,0),1,cv2.LINE_AA)

        if sum(tilt_cache)/cache_len >= 0.8:
            sound.play()
            cv2.putText(frame, f'ALARM! Score: {danger_score}', (10,height-80), font, 1,(255,0,0),1,cv2.LINE_AA)

        cv2.putText(frame, tilt_label, (10,height-60), font, 1,(255,255,255),1,cv2.LINE_AA)



        return av.VideoFrame.from_ndarray(frame, format='bgr24')

def app_video_filters():
    """Video transforms with OpenCV"""

    class OpenCVVideoProcessor(VideoProcessorBase):
        type: Literal["noop", "cartoon", "edges", "rotate"]

        def __init__(self) -> None:
            self.type = "noop"

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            if self.type == "noop":
                pass
            elif self.type == "cartoon":
                # prepare color
                img_color = cv2.pyrDown(cv2.pyrDown(img))
                for _ in range(6):
                    img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
                img_color = cv2.pyrUp(cv2.pyrUp(img_color))

                # prepare edges
                img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img_edges = cv2.adaptiveThreshold(
                    cv2.medianBlur(img_edges, 7),
                    255,
                    cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY,
                    9,
                    2,
                )
                img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

                # combine color and edges
                img = cv2.bitwise_and(img_color, img_edges)
            elif self.type == "edges":
                # perform edge detection
                img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
            elif self.type == "rotate":
                # rotate image
                rows, cols, _ = img.shape
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
                img = cv2.warpAffine(img, M, (cols, rows))

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="opencv-filter",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=OpenCVVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.type = st.radio(
            "Select transform type", ("noop", "cartoon", "edges", "rotate")
        )

    st.markdown(
        "This demo is based on "
        "https://github.com/aiortc/aiortc/blob/2362e6d1f0c730a0f8c387bbea76546775ad2fe8/examples/server/server.py#L34. "  # noqa: E501
        "Many thanks to the project."
    )

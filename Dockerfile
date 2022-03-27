#FROM python:3.8.12-buster
FROM tensorflow/tensorflow
# EXPOSE 8501
#ENV PYTHONPATH "/"

# COPY models/model_yawn.h5 /models/model_yawn.h5
# COPY models/model_eyes.h5 /models/model_eyes.h5
# COPY models/lite-model_movenet_singlepose_thunder_3.tflite /models/lite-model_movenet_singlepose_thunder_3.tflite

COPY requirements.txt /requirements.txt
COPY DriverDrowsinessDetector /DriverDrowsinessDetector
# COPY src /src

RUN pip install --upgrade pip
RUN pip install -r requirements.txt



CMD streamlit run --server.port=$PORT DriverDrowsinessDetector/app.py

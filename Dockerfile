#FROM python:3.8.12-buster
FROM tensorflow/tensorflow

ENV PYTHONPATH "/"

COPY models/model_yawn.h5 /models/model_yawn.h5
COPY models/model_eyes.h5 /models/model_eyes.h5
COPY models/lite-model_movenet_singlepose_thunder_3.tflite /models/lite-model_movenet_singlepose_thunder_3.tflite

COPY requirements.txt /requirements.txt
#COPY DriverDrowsinessDetector /DriverDrowsinessDetector
COPY src /src
COPY streamlitt-app.py /streamlitt-app.py

RUN pip install --upgrade pip
RUN pip install --upgrade -r requirements.txt

EXPOSE 8501

CMD streamlit run streamlitt-app.py

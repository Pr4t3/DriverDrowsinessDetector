#imports
import tensorflow as tf
import numpy as np
import cv2
import os
from src.utils.edges import EDGES
from sympy import Point, Segment

PATH = os.getcwd()
MODEL_PATH = os.path.join(PATH,'models/lite-model_movenet_singlepose_thunder_3.tflite')


class Tilt:
    '''
    arguments:
    -- frame: frame from webcam or camera
    -- tilt_threshold: threshold value for determaning if head is tilted
    based on y-coordinates of the eyes'''

    def __init__(self, frame, tilt_threshold):
        self.frame = frame
        self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_image = self.reshape_image(frame)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.keypoints_with_scores = self.create_keypoints()
        self.tilt = self.eval_tilt(tilt_threshold)

    def reshape_image(self, frame):
        '''
        bringing the image in the right shape for prediction'''
        img = frame.copy()
        img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256,256)
        return tf.cast(img, dtype=tf.float32)

    def create_keypoints(self):
        '''
        creating the keypoints of the body/face'''
        self.interpreter.set_tensor(self.input_details[0]['index'], np.array(self.input_image))
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])

    def draw_connections(self, confidence_threshold):
        '''
        drawing lines between keypoints
        arguments:
        -- confidence_threshold:  '''
        y, x, c = self.frame.shape
        shaped = np.squeeze(np.multiply(self.keypoints_with_scores, [y,x,1]))

        for edge, color in EDGES.items():
            p1, p2 = edge
            y1, x1, c1 = shaped[p1]
            y2, x2, c2 = shaped[p2]

            if (c1 > confidence_threshold) & (c2 > confidence_threshold):
                cv2.line(self.frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)

    def draw_keypoints(self, confidence_threshold):
        '''
        drawing circles for each keypoint
        arguments:
        -- confidence_threshold:  '''
        y, x, c = self.frame.shape
        shaped = np.squeeze(np.multiply(self.keypoints_with_scores, [y,x,1]))

        for kp in shaped:
            ky, kx, kp_conf = kp
            if kp_conf > confidence_threshold:
                cv2.circle(self.frame, (int(kx), int(ky)), 4, (0,255,0), -1)

    def eval_tilt(self, tilt_threshold):
        '''
        evaluates to True if the head is tilted more than threshold value'''
        left_eye = self.keypoints_with_scores[0][0][1]
        right_eye = self.keypoints_with_scores[0][0][2]

        left_eye_y = np.array(left_eye[:2]*self.frame.shape[:2]).astype(int)[0]
        right_eye_y = np.array(right_eye[:2]*self.frame.shape[:2]).astype(int)[0]

        print('Ycoord left eye', left_eye_y)
        return True if abs(left_eye_y-right_eye_y) > tilt_threshold else False

    def draw_face_axis(self):

        if self.tilt:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        x_left, y_left = self.keypoints_with_scores[0][0][1][:2]*self.frame.shape[:2]

        x_right, y_right = self.keypoints_with_scores[0][0][2][:2]*self.frame.shape[:2]

        p1, p2 = Point(x_left, y_left), Point(x_right, y_right)

        s1 = Segment(p1, p2)
        perpendicularBisector = s1.perpendicular_bisector()
        perp_1, perp_2 = perpendicularBisector.points

        perp_eval_1, perp_eval_2 = ((perp_1.coordinates[0].evalf(), perp_1.coordinates[1].evalf()),
                                    (perp_2.coordinates[0].evalf(), perp_2.coordinates[1].evalf()))

        print(perp_eval_1)
        cv2.line(self.frame, perp_eval_1, perp_eval_2, color, 2)

    if __name__ == '__main__':
        pass

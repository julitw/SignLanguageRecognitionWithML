import cv2
import numpy as np
import os
from mediapipe import solutions as mp

class MediapipeHandler:
    def __init__(self):
        self.mp_holistic = mp.holistic
        self.mp_drawing = mp.drawing_utils
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def mediapipe_detection(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def draw_styled_landmarks(self, image, results):
        draw_spec_face = self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1)
        draw_spec_pose = self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4)
        draw_spec_left_hand = self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4)
        draw_spec_right_hand = self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4)

        self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION,
                                       draw_spec_face, draw_spec_face)
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                       draw_spec_pose, draw_spec_pose)
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                       draw_spec_left_hand, draw_spec_left_hand)
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                       draw_spec_right_hand, draw_spec_right_hand)
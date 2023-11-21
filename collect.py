import cv2
import numpy as np
import os
from mediapipe import solutions as mp

from MediapipeHandler import MediapipeHandler
from DataCollector import DataCollector

def main():
    word = 'have'  
    actions = np.array([word])
    DATA_PATH = os.path.join(word)
    DATA_PATH_FRAMES = os.path.join(word + 'frames')
    no_sequences = 30
    sequence_length = 30

    cap = cv2.VideoCapture(0)
    
    mediapipe_handler = MediapipeHandler()
    data_collector = DataCollector(word, no_sequences, sequence_length)
    data_collector.create_directories(DATA_PATH, DATA_PATH_FRAMES)

    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                frame_filename = f'action_{action}_seq_{sequence}_frame_{frame_num}.jpg'
                frames_path = os.path.join(DATA_PATH_FRAMES, frame_filename)
                data_collector.save_frame(frame, frames_path)
                image, results = mediapipe_handler.mediapipe_detection(frame)
                mediapipe_handler.draw_styled_landmarks(image, results)

                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}',
                                (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}',
                                (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)

                keypoints = DataCollector.extract_keypoints(results)
                data_collector.save_keypoints(keypoints, DATA_PATH, frame_num, sequence)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import os, numpy as np, cv2

class DataCollector:
    def __init__(self, action, no_sequences, sequence_length):
        self.action = action
        self.no_sequences = no_sequences
        self.sequence_length = sequence_length

    def create_directories(self, data_path, data_path_frames):
        for sequence in range(self.no_sequences):
            os.makedirs(os.path.join(data_path, self.action, str(sequence)), exist_ok=True)
            os.makedirs(os.path.join(data_path_frames, self.action, str(sequence)), exist_ok=True)

    def save_frame(self, frame, frames_path):
        cv2.imwrite(frames_path, frame)

    def save_keypoints(self, keypoints, data_path, sequence, frame_num):
        npy_path = os.path.join(data_path, self.action, str(sequence), str(frame_num))
        np.save(npy_path, keypoints)

    def extract_keypoints(results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
        return np.concatenate([pose, face, lh, rh])
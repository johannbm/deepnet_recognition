import face_recognition
import cv2
import time

class FaceLandmarks:

    def __init__(self):
        self.performance_stats = {}
        self.landmarks = None

    def update_facial_landmarks(self, face, location):
        self.landmarks = self.get_facial_landmarks(face, location)

    def get_facial_landmarks(self, face, location, debug=True):
        time_start = time.time()
        landmarks = face_recognition.face_landmarks(face, location)

        if debug:
            time_end = time.time()
            self.performance_stats["Landmarks"] = time_end-time_start

        return landmarks

    def draw_eyes(self, landmarks, frame):
        for landmark in landmarks:
            left_eye = landmark['left_eye']
            right_eye = landmark['right_eye']
            self.draw_eye(frame, left_eye)
            self.draw_eye(frame, right_eye)

    def draw_landmarks(self, landmarks, frame, radius=1, color=(0, 255, 100, 255)):
        for landmark in landmarks:
            for location in landmark:
                for point in landmark[location]:
                    cv2.circle(frame, point, radius, color)

    def draw_eye(self, frame, eye_marks, width=1, fill=(0, 0, 255, 255)):
        eyelines_tuples = [(0,1), (2,3), (4,3), (0,-1), (1,-1), (2,-2)]
        for x,y in eyelines_tuples:
            cv2.line(frame, eye_marks[x], eye_marks[y], color=fill, thickness=width)

    def show_landmarks(self, frame):
        landmarks_frame = frame.copy()
        self.draw_eyes(self.landmarks, landmarks_frame)
        self.draw_landmarks(self.landmarks, landmarks_frame)
        cv2.imshow("Landmarks", landmarks_frame)

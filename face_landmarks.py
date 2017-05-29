import numpy as np
import cv2
import math
import imutils
import time
import face_recognition as fr
import wink_queue as wq

class WinkClassifier:

    def __init__(self):
        self.knn = self.load_knn('eye_wink_3.npz')
        self.landmarks = None
        self.performance_stats = {"Landmarks": 0}
        self.wink_queue = wq.WinkQueue()

    def load_data(self, file_name):
        with np.load(file_name) as data:
            train = np.float32(data['train'])
            train_labels = np.float32(data['train_labels'])

            return train, train_labels


    def load_knn(self, path):
        data, labels = self.load_data(path)
        return self.train_knn(data, labels)

    def train_knn(self, data, labels):
        knn = cv2.KNearest()
        knn.train(data, labels)

        return knn

    def classify_current(self):
        if self.landmarks is not None:
            feature = self.extract_features_3(self.landmarks)
            ret, result, neighbours, dist = self.knn.find_nearest(np.float32(feature), k=3)
            self.wink_queue.add(result[0])
        return None

    def extract_features_3(self, landmarks):

        features = []
        for landmark in landmarks:

            feature = []
            eyebrow_distance_l = self.extract_eyebrow_eye_distance(landmark['left_eyebrow'], landmark['left_eye'])
            eyebrow_distance_r = self.extract_eyebrow_eye_distance(landmark['right_eyebrow'], landmark['right_eye'])

            angles_l, height_l = self.extract_eye_features(landmark['left_eye'])
            angles_r, height_r = self.extract_eye_features(landmark['right_eye'])

            feature.append(math.fabs(eyebrow_distance_l - eyebrow_distance_r))
            feature.append(math.fabs(angles_r['left'] - angles_r['right']))
            feature.append(math.fabs(angles_l['left'] - angles_l['right']))

            feature.append(math.fabs(height_l - height_r))

            features.append(feature)

        return features

    def extract_eye_features(self, eye_landmarks):
        # Vectors
        baseline_lr_v = self.create_vector(eye_landmarks[0], eye_landmarks[3])
        baseline_rl_v = self.create_vector(eye_landmarks[3], eye_landmarks[0])
        left_up_v = self.create_vector(eye_landmarks[0], eye_landmarks[1])
        left_down_v = self.create_vector(eye_landmarks[0], eye_landmarks[-1])
        right_up_v = self.create_vector(eye_landmarks[3], eye_landmarks[2])
        right_down_v = self.create_vector(eye_landmarks[3], eye_landmarks[4])

        angles = {}
        angles["top_left"] = self.angle_between(baseline_lr_v, left_up_v)
        angles["top_right"] = self.angle_between(baseline_rl_v, right_up_v)
        angles["bottom_right"] = self.angle_between(baseline_rl_v, right_down_v)
        angles["bottom_left"] = self.angle_between(baseline_lr_v, left_down_v)
        angles["left"] = self.angle_between(left_up_v, left_down_v)
        angles["right"] = self.angle_between(right_up_v, right_down_v)

        height = math.fabs(eye_landmarks[1][1] - eye_landmarks[-1][1])

        return angles, height

    def extract_eyebrow_eye_distance(self, landmarks_eyebrow, landmarks_eye):
        # average eyebrow height
        landmarks_eyebrow = landmarks_eyebrow[1:-1]
        landmarks_eye = landmarks_eye[1:3]
        height = sum([l[1] for l in landmarks_eyebrow]) / len(landmarks_eyebrow)
        eye_height = sum([l[1] for l in landmarks_eye]) / len(landmarks_eye)
        distance = math.fabs(height - eye_height)
        return distance

    def create_vector(self, p2, p1):
        x_diff = p1[0] - p2[0]
        y_diff = -(p1[1] - p2[1])
        return np.array([x_diff, y_diff])

    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::
        """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


    def update_facial_landmarks(self, face, location):
        if len(location) > 0:
            self.landmarks = self.get_facial_landmarks(face, location)
        else:
            self.landmarks = None

    @staticmethod
    def crop_face(image, face_locations):
        top, right, bottom, left = face_locations[0]
        return image[top:bottom, left:right]

    @staticmethod
    def resize_face(image):
        image = imutils.resize(image, height=800)
        shape = image.shape
        location = [(0, shape[0], shape[1], 0)]
        return image, location

    def get_facial_landmarks(self, face, location, debug=True):
        time_start = time.time()
        face_image = self.crop_face(face, location)
        face_image, location = self.resize_face(face_image)
        landmarks = fr.face_landmarks(face_image, location)

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
        if self.landmarks is not None:
            landmarks_frame = frame.copy()
            self.draw_eyes(self.landmarks, landmarks_frame)
            self.draw_landmarks(self.landmarks, landmarks_frame)
            cv2.imshow("Landmarks", landmarks_frame)




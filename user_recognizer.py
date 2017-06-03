import time
import nodejs_input
import mirror_messenger
import cv2
import deepnet_face_recognition
import os
import inspect
import opencv_modules


class UserRecognizer:

    def __init__(self, face_recognition_algorithm, conf):
        self.algorithm = face_recognition_algorithm
        self.face_recognizer = self.load_face_recognition_algorithm(conf)

        self.time_since_face_recognized = 0
        self.logout_time = conf["logout_time"]
        self.consecutive_detection_limit = conf["consecutive_detections"]
        self.detection_index = 0

        self.current_user = None
        self.recent_faces_recognized = []
        self.reset_recognized_faces()
        self.messenger = mirror_messenger.MirrorMessenger(conf["rpi_IP"])

    def load_face_recognition_algorithm(self, conf):
        if self.algorithm == 4:
            path_to_file = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
            model = deepnet_face_recognition.DeepNetRecognizer(path_to_file + "/images/", conf)
        else:
            model = opencv_modules.FaceRecModel(algorithm=self.algorithm)
        return model

    @staticmethod
    def get_algorithm_text(algorithm):
        threshold = opencv_modules.FaceRecModel.get_threshold(algorithm)
        if algorithm == 1:
            return "LBPH Classifier     Threshold: {0}".format(threshold)
        elif algorithm == 2:
            return "Fisherface Classifier     Threshold: {0}".format(threshold)
        elif algorithm == 3:
            return "Eigenface Classifier     Threshold: {0}".format(threshold)
        else:
            return "Google Facenet"

    def get_performance_stats(self):
        return self.face_recognizer.get_average_stats()

    def reset_recognized_faces(self):
        self.recent_faces_recognized = [[-1]] * self.consecutive_detection_limit

    def recognize_face(self, frame, face_locations):
        """
        checks whether there is a known face in the frame
        :param frame: current frame
        :param face_locations: a list of (top, right, bottom, left)? tuples
        :return: a list of (name, index) tuples for the found faces
        """
        indexes = self.face_recognizer.recognize_face(frame, face_locations)
        self.update_detection_list(indexes)

        return indexes

    def update_detection_list(self, indexes):
        """
        updates the list and index of recent faces detected
        :param indexes: list indexes to update
        :return: None
        """
        if len(indexes) > 0:
            self.recent_faces_recognized[self.detection_index] = indexes
            self.detection_index = (self.detection_index + 1) % self.consecutive_detection_limit
            self.time_since_face_recognized = time.time()

    def check_login(self):
        """
        checks if the conditions for a login-event are met, if so, sends login-event to nodeJS
        conditions:
            1. list of recently detected faces must be same for all elements
            2. the new user to login must differ from user already logged in
            3. list of recently detected faces must be full
        :return: index of logged in user or False
        """
        same_user, index = self.are_all_same_user(self.recent_faces_recognized, self.consecutive_detection_limit)

        if same_user:
            if self.current_user is None and index > 0:
                self.current_user = index
                nodejs_input.to_node("login", {"user": self.current_user})
                self.messenger.send_to_mirror("login", {"user": self.current_user})

                return self.current_user
        return False

    def check_logout(self):
        """
        checks if the conditions for a logout-event are met, if so send logout-event to nodeJS
        conditions:
            1. the elapsed time from when the last face was recognized must be greater than self.logout_time
            2. a user must be logged in
        :return: True or False. A user either logged out, or didn't
        """
        user = False
        if time.time() - self.time_since_face_recognized > self.logout_time and self.current_user is not None:
            nodejs_input.to_node("logout", {"user": self.current_user})
            self.messenger.send_to_mirror("logout", {"user": self.current_user})

            self.reset_recognized_faces()
            user = self.current_user
            self.current_user = None
        return user

    def are_all_same_user(self, users, detection_limit):
        """
        Detects if the same user has been detected detection_limit amount of times in a row
        :param users: A nested list of users indexes [[1, 2], [1]...]
        :param detection_limit: number of consecutive detections for logint to trigger
        :return: True or False
        """
        user_count = {}
        for i in range(self.consecutive_detection_limit):
            for j in users[i]:
                if user_count.has_key(j):
                    user_count[j] += 1
                else:
                    user_count[j] = 1

        for key in user_count.keys():
            if user_count[key] == detection_limit:
                return True, key

        return False, -1

    def are_all_elements_equal(self, l):
        """
        checks if all elements in the list are equal
        :param l: list to check
        :return: True if all elements are equal, otherwise False
        """
        return l[1:] == l[:-1]

    def show_recognized_face(self, image_frame, face_locations, face_names):
        frame = image_frame.copy()
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
            #font = cv2.FONT_HERSHEY_DUPLEX
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            cv2.putText(frame, str(name), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            # cv2.putText(frame, "Detection: " + str(round(performance_stats["Detection"], 5)), (10, 20), font, 1.0, (255, 255, 255), 1)
            # cv2.putText(frame, "Encoding: " + str(round(performance_stats["Encoding"], 5)), (10, 40), font, 1.0, (255, 255, 255), 1)
            # cv2.putText(frame, "Matching: " + str(round(performance_stats["Matching"], 5)), (10, 60), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)


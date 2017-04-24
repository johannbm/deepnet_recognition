import face_recognition as fr
import cv2
import os
import inspect
import time
import nodejs_input

class DeepNetRecognizer:

    def __init__(self, image_folder):
        self.known_face_encodings = self.initialize_face_encoding(image_folder)
        self.known_face_names = self.initialize_face_names(image_folder)
        self.this_script_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

        self.time_since_face_recognized = 0
        self.logout_time = 3
        self.consecutive_detection_limit = 5
        self.detection_index = 0

        self.current_user = None
        self.recent_faces_recognized = []
        self.reset_recognized_faces()
        self.performance_stats = {}


    def reset_recognized_faces(self):
        self.recent_faces_recognized = [-1] * self.consecutive_detection_limit

    def initialize_face_encoding(self, image_folder):
        """
        creates an encoding of length 128 using deepnet
        :param image_folder: folder of images to encode
        :return: list of encodings
        """
        images = [fr.load_image_file(os.path.join(image_folder, image)) for image in os.listdir(image_folder)]
        return [fr.face_encodings(encoding) for encoding in images]

    def initialize_face_names(self, image_folder):
        """
        Extracts the name of the images in image_folder
        :param image_folder: folder of images to encode
        :return: list of image names without extension
        """
        return [os.path.splitext(image)[0] for image in os.listdir(image_folder)]

    def encode_face(self, frame, face_locations, debug=True):
        """
        creates encoding for the face(s) given by face_locations
        :param frame: current frame
        :param face_locations: a list of (top, right, bottom, left)? tuples
        :param debug: whether execution time should be recorded
        :return: a list 128-length encodings of found faces
        """
        time_start = time.time()
        face_encodings = fr.face_encodings(frame, face_locations)

        if debug:
            time_end = time.time()
            self.performance_stats["Encoding"] = time_end - time_start

        return face_encodings

    def compare_face(self, new_encoding, debug=True):
        """
        compares the given encoding with the set of known faces
        :param new_encoding: encoding to be classified
        :param debug: whether execution time should be recorded
        :return: a boolean list of whether the given index face was found
        """
        time_start = time.time()
        match = fr.compare_faces(self.known_face_encodings, new_encoding)

        if debug:
            time_end = time.time()
            self.performance_stats["Matching"] = time_end - time_start

        return match

    def get_corresponding_user(self, match_list):
        """
        looks up the names corresponding to the given list of matches
        :param match_list: a boolean list indicating the indexes of found faes
        :return: a list of (name, index) tuples
        """
        names = []
        for i in range(len(match_list)):
            if match_list[i]:
                names.append((self.known_face_names[i], i))
                self.update_detection_list(i)
        if len(names) == 0:
            names.append(("Unknown", -1))
        return names

    def recognize_face(self, frame, face_locations):
        """
        checks whether there is a known face in the frame
        :param frame: current frame
        :param face_locations: a list of (top, right, bottom, left)? tuples
        :return: a list of (name, index) tuples for the found faces
        """
        new_encodings = self.encode_face(frame, face_locations)
        users = []
        for new_encoding in new_encodings:
            match_list = self.compare_face(new_encoding)
            users.extend(self.get_corresponding_user(match_list))
        return users

    def are_all_elements_equal(self, l):
        """
        checks if all elements in the list are equal
        :param l: list to check
        :return: True if all elements are equal, otherwise False
        """
        return l[1:] == l[:-1]

    def update_detection_list(self, index):
        """
        updates the list and index of recent faces detected
        :param index: list index to update
        :return: None
        """
        self.recent_faces_recognized[self.detection_index] = index
        self.detection_index = (self.detection_index + 1) % self.consecutive_detection_limit
        self.time_since_face_recognized = time.time()

    def check_login(self):
        """
        checks if the conditions for a login-event are met, if so, sends login-event to nodeJS
        conditions:
            1. list of recently detected faces must be same for all elements
            2. the new user to login must differ from user already logged in
            3. list of recently detected faces must be full
        :return: None
        """
        if (self.recent_faces_recognized[0] != -1 and
                self.are_all_elements_equal(self.recent_faces_recognized) and
                    self.current_user != self.recent_faces_recognized[0]):
            self.current_user = self.recent_faces_recognized[0]
            nodejs_input.to_node("login", {"user": self.current_user + 1})

    def check_logout(self):
        """
        checks if the conditions for a logout-event are met, if so send logout-event to nodeJS
        conditions:
            1. the elapsed time from when the last face was recognized must be greater than self.logout_time
            2. a user must be logged in
        :return: None
        """
        if time.time() - self.time_since_face_recognized > self.logout_time and self.current_user is not None:
            nodejs_input.to_node("logout", {"user": self.current_user + 1})
            self.reset_recognized_faces()
            self.current_user = None







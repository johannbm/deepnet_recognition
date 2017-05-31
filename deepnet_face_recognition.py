import face_recognition as fr
import cv2
import os
import inspect
import time
import nodejs_input
import mirror_messenger


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
        self.performance_stats = {"Matching": [], "Facenet_encoding": []}
        self.messenger = mirror_messenger.MirrorMessenger()

    def reset_recognized_faces(self):
        self.recent_faces_recognized = [-1] * self.consecutive_detection_limit

    def get_average_stats(self):
        return {"Matching": sum(self.performance_stats["Matching"]) / float(len(self.performance_stats["Matching"])),
                "Facenet_encoding": sum(self.performance_stats["Facenet_encoding"]) / float(len(self.performance_stats["Facenet_encoding"]))}


    def initialize_face_encoding(self, image_folder):
        """
        creates an encoding of length 128 using deepnet
        :param image_folder: folder of images to encode
        :return: list of encodings
        """
        images = [fr.load_image_file(os.path.join(image_folder, image)) for image in self.get_sorted_directory(image_folder)]
        return [fr.face_encodings(encoding)[0] for encoding in images]

    def initialize_face_names(self, image_folder):
        """
        Extracts the name of the images in image_folder
        :param image_folder: folder of images to encode
        :return: list of image names without extension
        """
        return [os.path.splitext(image)[0] for image in self.get_sorted_directory(image_folder)]

    def get_sorted_directory(self, image_folder):
        return sorted(os.listdir(image_folder))

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
            self.performance_stats["Facenet_encoding"].append(time_end - time_start)

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
            self.performance_stats["Matching"].append(time_end - time_start)

        return match

    def get_corresponding_user(self, match_list):
        """
        looks up the names corresponding to the given list of matches
        :param match_list: a boolean list indicating the indexes of found faces
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

    def get_corresponding_index(self, match_list):
        """
        looks up the names corresponding to the given list of matches
        :param match_list: a boolean list indicating the indexes of found faces
        :return: a list of match_list indexes that were true (1-indexed)
        """
        indexes = [i+1 for i in range(len(match_list)) if match_list[i]]
        return indexes if len(indexes) > 0 else [-1]


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
            users.extend(self.get_corresponding_index(match_list))
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








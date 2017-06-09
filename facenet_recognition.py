import face_recognition as fr
import os
import inspect
import time
import utility


class FacenetRecognizer:

    def __init__(self, image_folder, conf):
        self.number_of_faces = conf["num_faces"]

        self.known_face_encodings, self.known_face_names = self.initialize_face_encoding(image_folder)
        self.this_script_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

        self.performance_stats = {"Matching": [], "Facenet_encoding": []}

    def get_average_stats(self):
        """
        Calculates the average of the accumulated statistics
        :return: Dictionary with the keys "Matching" and "Facenet_encoding" for the average
                matching speed and encoding speed respectively
        """
        return {"Matching": utility.list_avg(self.performance_stats["Matching"]),
                "Facenet_encoding": utility.list_avg(self.performance_stats["Facenet_encoding"])}

    def initialize_face_encoding(self, image_folder):
        """
        creates an encoding of length 128 using deepnet
        :param image_folder: folder of images to encode
        :return: list of encodings
        """
        images = []
        names = []
        for directory in utility.get_sorted_directory(image_folder):
            dir_path = os.path.join(image_folder, directory)
            name = os.path.splitext(dir_path)[0]
            person_images = utility.get_sorted_directory(dir_path)
            for i in range(self.number_of_faces):
                images.append(fr.load_image_file(os.path.join(dir_path, person_images[i])))
                names.append(name)
        return [fr.face_encodings(encoding)[0] for encoding in images], names

    def initialize_face_names(self, image_folder):
        """
        Extracts the name of the images in image_folder
        :param image_folder: folder of images to encode
        :return: list of image names without extension
        """
        return [os.path.splitext(image)[0] for image in utility.get_sorted_directory(image_folder)]

    def encode_face(self, frame, face_locations, debug=False):
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

    def compare_face(self, new_encoding, debug=False):
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
        if len(names) == 0:
            names.append(("Unknown", -1))
        return names

    def get_corresponding_index(self, match_list):
        """
        looks up the names corresponding to the given list of matches
        :param match_list: a boolean list indicating the indexes of found faces
        :return: a list of match_list indexes that were true (1-indexed)
        """
        indexes = [(i//self.number_of_faces)+1 for i in range(len(match_list)) if match_list[i]]

        return [utility.most_common_element(indexes)] if len(indexes) > 0 else [-1]

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








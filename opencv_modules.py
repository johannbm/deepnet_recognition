import cv2
import os
import time
import utility


class FaceRecModel:

    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.training_file_dir = "./models/"
        self.training_file = ""
        self.model = None
        self.load()
        self.FACE_WIDTH = 92
        self.FACE_HEIGHT = 112
        self.performance_stats = {"Matching": []}

    @staticmethod
    def get_threshold(algorithm):
        if algorithm == 1:
            return 110
        elif algorithm == 2:
            return 600
        else:
            return 3000

    def load(self):
        threshold = self.get_threshold(self.algorithm)
        if self.algorithm == 1:
            self.model = cv2.createLBPHFaceRecognizer(threshold=threshold)
            self.training_file = "training_lbp.xml"
        elif self.algorithm == 2:
            self.model = cv2.createFisherFaceRecognizer(threshold=threshold)
            self.training_file = "training_fisher.xml"
        else:
            self.model = cv2.createEigenFaceRecognizer(threshold=threshold)
            self.training_file = "training_eigen.xml"

        self.model.load(os.path.join(self.training_file_dir, self.training_file))

    def get_average_stats(self):
        return {"Matching": utility.list_avg(self.performance_stats["Matching"])}

    def recognize_face(self, frame, locations):
        start_time = time.time()
        names = []
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for location in locations:
            x, y, w, h = utility.convert_dlib_location_to_opencv(location)

            if self.algorithm == 1:
                #crop = utility.basic_crop_dlib(frame, location)
                crop = self.crop(frame, x, y, w, h)
            else:
                crop = self.resize(self.crop(frame, x, y, w, h))
            label, confidence = self.model.predict(crop)
            names.append(label)

        self.performance_stats["Matching"].append(time.time() - start_time)
        return names

    def crop(self, image, x, y, w, h):
        """Crop box defined by x, y (upper left corner) and w, h (width and height)
        to an image with the same aspect ratio as the face training data.  Might
        return a smaller crop if the box is near the edge of the image.
        """
        crop_height = int((self.FACE_HEIGHT / float(self.FACE_WIDTH)) * w)
        midy = y + h / 2
        y1 = max(0, midy - crop_height / 2)
        y2 = min(image.shape[0] - 1, midy + crop_height / 2)
        return image[y1:y2, x:x + w]

    def resize(self, image):
        """Resize a face image to the proper size for training and detection.
        """
        return cv2.resize(image, (self.FACE_WIDTH, self.FACE_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
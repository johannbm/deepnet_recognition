import cv2
import time
import os
import face_recognition
import utility


class BackgroundExtractor:

    def __init__(self, first_frame, conf, path_to_file):
        """
        :param first_frame: initial frame for initializing background model
        :param conf: configuration file
        :param path_to_file: directory of working directory
        :param face_detection_algorithm: face detection algorithm to use:
            algorithm-Choices:
                1: use dlib's frontal face detector
                2: use LBP cascade detector
                3: use haar cascade detector
        """
        gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        self.avg = gray.copy().astype("float")
        self.current_frame = self.avg
        self.motion_threshold = conf["motion_threshold"]

        self.do_bg_subtraction = conf["do_bgsub"]
        self.face_detection_method = None
        self.face_cascade = None
        self.face_detection_algorithm = None
        self.load_face_detection_algorithm(path_to_file, conf)

        self.negative_seconds_limit = conf["consecutive_negative_seconds_limit"]
        self.previous_positive_detection = time.time() - self.negative_seconds_limit
        self.min_area = conf["min_area"]
        self.show_feed = conf["show_video"]
        self.is_dynamic = conf["dynamic_background"]
        self.performance_stats = {"Detection": [], "Total_detection_time": []}

    def load_face_detection_algorithm(self, path_to_file, conf):
        """
        Loads the given face_detection algorithm and loads its cascade file if necessary

        :param algorithm: face detection algorithm to use:
        Choices:
            1: use dlib's frontal face detector
            2: use LBP cascade detector
            3: use haar cascade detector
        :param path_to_file: directory of working directory
        :param conf: configuration file
        :return: None
        """
        algorithm = conf["detection_algorithm"]
        self.face_detection_algorithm = algorithm
        if algorithm == 1:
            self.face_detection_method = self.find_faces_dlib
        elif algorithm == 2:
            self.face_detection_method = self.find_faces_haar
            self.face_cascade = cv2.CascadeClassifier(os.path.join(path_to_file, conf["lbp_cascade_path"]))
        elif algorithm == 3:
            self.face_detection_method = self.find_faces_haar
            self.face_cascade = cv2.CascadeClassifier(os.path.join(path_to_file, conf["haar_cascade_path"]))

    @staticmethod
    def get_algorithm_text(algorithm):
        """
        Retrieves the textualized name of current detection algorithm
        :param algorithm: integer of algorithm
        :return: name of detection algorithm
        """
        if algorithm == 1:
            return "Dlib HOG"
        elif algorithm == 2:
            return "LBP Cascade"
        else:
            return "Haar Cascade"

    def get_performance_stats(self):
        return {"Detection": utility.list_avg(self.performance_stats["Detection"]),
                "Total_detection": utility.list_avg(self.performance_stats["Total_detection_time"])}

    def get_potential_regions(self, frame):
        """
        Extract regions that differ from background model.
        Small regions are filtered out by property self.min_area
        Toggle conf.json properties "blur", "average", "contour", "detection" to display those phases
        :param frame: input image to analyze
        :return: A list of tuples in the form (cropped_image, original_bounding box)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        self.current_frame = gray

        if self.show_feed["blur"]:
            cv2.imshow('Blurred', gray)

        frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(self.avg))

        if self.show_feed["average"]:
            cv2.imshow('Average', cv2.convertScaleAbs(self.avg))

        thresh = cv2.threshold(frame_delta, self.motion_threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        if self.show_feed["contour"]:
            cv2.imshow("Countor", thresh)

        (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)

        areas = [x for x in cnts if cv2.contourArea(x) > self.min_area]
        cropped_images = []
        for a in areas:
            x, y, w, h = cv2.boundingRect(a)
            cropped_images.append((frame[y:y + h, x:x + w], (x, y, w, h)))

            if self.show_feed["detection"]:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                cv2.imshow('Video CONTOUR', frame)

        return cropped_images

    def find_faces_haar(self, frame):
        """
        Returns bounding box of faces found using this instances haar-cascade
        This operation's execution speed has key "haar_detection" or "lbp_detection" depending on the algorithm
        :param frame: image to analyze
        :return: List of bounding box tuples (x, y, w, h)
        """
        start = time.time()
        faces = self.face_cascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        self.performance_stats["Detection"].append(time.time() - start)
        return faces

    def find_faces_dlib(self, frame):
        """
        Finds bounding box for faces using dlibs frontal face detector
        This operations execution speed has key "dlib_detection"
        :param frame: image to analyze
        :return: List of bounding box tuples in the form (x, y, w, h)
        """
        start = time.time()
        faces = face_recognition.face_locations(frame)
        self.performance_stats["Detection"].append(time.time() - start)
        opencv_faces = [utility.convert_dlib_location_to_opencv(x) for x in faces]
        return opencv_faces

    def get_bounding_box(self, frame):
        """
        returns the bounding box (opencv format) of potential face regions
        :param frame: image to analyze for faces
        :return: List of bounding boxes in (x, y, w, h)
        """
        start_time = time.time()
        potential_areas = self.get_potential_regions(frame)
        bounding_boxes = []

        if len(potential_areas) == 0 and self.is_dynamic:
            self.accumulate_background()

        for area, bounding_box in potential_areas:
            faces = self.face_detection_method(area)

            #update background model if noe faces found
            if len(faces) == 0 and self.is_dynamic:
                self.accumulate_background()

            bounding_boxes.extend(self.get_face_bounds(faces, bounding_box))
        if len(bounding_boxes) > 0:
            self.performance_stats["Total_detection_time"].append(time.time() - start_time)
        return bounding_boxes

    def detect_face(self, frame):
        """
        Detects faces in the given image using the parameters given during class initialization
        :param frame: image to search for faces in
        :return: list of face locations in dlib-format
        """
        return self.get_bounding_box(frame) if self.do_bg_subtraction else self.detect_face_basic(frame)

    def detect_face_basic(self, frame):
        """
        Detects the face without doing background substitution
        :param frame: image in which to search for a face
        :return: A list of face locations in dlib-format
        """
        time_s = time.time()
        faces = utility.convert_opencv_location_to_dlib(self.face_detection_method(frame))
        self.performance_stats["Total_detection_time"].append(time.time() - time_s)
        return faces

    def accumulate_background(self):
        """
        If self.negative_seconds_limit has passed since previous positive face detection
        accumulate the background
        :return: None
        """
        if time.time()-self.previous_positive_detection > self.negative_seconds_limit:
            cv2.accumulateWeighted(self.current_frame, self.avg, 0.01)

    def get_face_bounds(self, faces, bounding_box):
        """
        Translates the coordinates of the face found in a sub-regions to the face's actual
        coordinates in the original frame
        :param faces: List of face bounds as found in the cropped image of its bounding box
        :param bounding_box: List of the original bounding_boxes in which the face was found
        :return:
        """
        bounding_boxes = []
        for face in faces:
            (x1, y1, w1, h1) = face
            x, y, w, h = bounding_box
            top = y+y1
            right = x+x1+w1
            bottom = y+y1+h1
            left = x+x1
            bounding_boxes.append((top, right, bottom, left))
            self.previous_positive_detection = time.time()
        return bounding_boxes


'''
if __name__ == "__main__":
    conf = json.load(open('conf.json'))
    path_to_file = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

    video_capture = VideoStream(usePiCamera=conf["use_rpi_camera"] > 0).start()
    time.sleep(2.0)

    bgsub = BackgroundExtractor(video_capture.read(), conf, path_to_file)
    face_cascade = cv2.CascadeClassifier('cascades/lbpcascade_frontalface.xml')
    debug=True
    try:
        while True:
            frame = video_capture.read()
            frame = imutils.resize(frame, width=500)

            #potentialAreas = bgsub.getPotentialRegions(frame)

            #face_locations = bgsub.getBoundingBox(frame)

            regions = bgsub.getBoundingBox(frame)
            for x,y,w,h in regions:
                cv2.rectangle(frame, (h, x), (y, w), (100, 0, 100), 2)

            print "negative frames count: " + str(time.time()-bgsub.previous_positive_detection)

            if conf["show_video"]["detection"]:
                cv2.imshow('Face detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


                    # all_faces = []
                    # for area, bounding_box in potentialAreas:
                    #     faces = face_cascade.detectMultiScale(
                    #         area,
                    #         scaleFactor=1.1,
                    #         minNeighbors=5,
                    #         minSize=(30, 30),
                    #         flags=cv2.cv.CV_HAAR_SCALE_IMAGE
                    #     )
                    #     if conf["show_video"]["detection"]:
                    #         for face in faces:
                    #             (x1, y1, w1, h1) = face
                    #             x, y, w, h = bounding_box
                    #             cv2.rectangle(frame, (x+x1, y+y1), (x+x1 + w1, y+y1 + h1), (0, 255, 0), 2)
                    #             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
    except KeyboardInterrupt:
        pass
'''


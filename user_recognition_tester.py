import face_recognition
import cv2
import time
import background_subtractor as bgsub
import imutils
import json
import sys
import inspect
import os
import signal
from imutils.video import VideoStream
import deepnet_face_recognition
import face_landmarks
import opencv_modules
import user_recognizer


def get_face_locations_dlib(frame):
    return face_recognition.face_locations(frame)


def get_face_locations_bgsub(frame, bg_sub_model):
    return bg_sub_model.getBoundingBox(frame)


def get_face_locations(frame, bg_sub_model, debug=True):
    time_start = time.time()
    face_areas = get_face_locations_bgsub(frame, bg_sub_model)

    if debug:
        time_end = time.time()
        performance_stats["Detection"] = time_end-time_start
    return face_areas

path_to_file = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
conf = json.load(open(path_to_file + '/conf.json'))

model = deepnet_face_recognition.DeepNetRecognizer(path_to_file + "/images/")
#model = opencv_modules.FaceRecModel(algorithm=1)
dnr = user_recognizer.UserRecognizer(model)

# Initialize some variables

process_this_frame = True
performance_stats = {}


def test_video(filename, bg_sub_model):
    cap = cv2.VideoCapture(filename)
    cap.set(6, 5)
    fps = 30
    limit_fps = False

    face_locations = []
    face_names = []

    frame_counter = 0
    login_frames = []
    logout_frames = []

    success = True

    while success:
        start_time = time.time()
        success, frame = cap.read()

        if not success:
            continue

        frame = imutils.resize(frame, width=500)

        if bg_sub_model is None:
            bg_sub_model = bgsub.BackgroundExtractor(frame, conf, path_to_file)

        # Only process every other frame of video to save time
        if process_this_frame:
            face_locations = get_face_locations(frame, bg_sub_model)
            face_names = dnr.recognize_face(frame, face_locations)
            login = dnr.check_login()
            logout = dnr.check_logout()
            if login:
                login_frames.append((login, frame_counter))
            if logout:
                logout_frames.append((logout, frame_counter))

        # process_this_frame = not process_this_frame

        if conf["show_video"]["recognition"]:
            dnr.show_recognized_face(frame, face_locations, face_names)
        else:
            print performance_stats
        execution_time = time.time() - start_time
        if limit_fps:
            if execution_time < 1 / float(fps):
                time.sleep((1 / float(fps)) - execution_time)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_counter += 1

    return login_frames, logout_frames


print test_video('../test_data/training_videos/002_2.mp4', None)


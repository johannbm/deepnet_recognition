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


def to_node(type, message):
    # convert to json and print (node helper will read from stdout)
    try:
        print(json.dumps({type: message}))
    except Exception:
        pass
    # stdout has to be flushed manually to prevent delays in the node helper communication
    sys.stdout.flush()


def shutdown(self, signum):
    to_node("status", 'Shutdown: Cleaning up camera...')
    camera.stop()
    quit()



def get_face_locations_dlib(frame):
    return face_recognition.face_locations(frame)


def get_face_locations_bgsub(frame):
    return bg_sub_model.getBoundingBox(frame)


def get_face_locations(frame, debug=True):
    time_start = time.time()
    face_areas = get_face_locations_bgsub(frame)

    if debug:
        time_end = time.time()
        performance_stats["Detection"] = time_end-time_start
    return face_areas


def get_facial_landmarks(face, location, debug=True):
    time_start = time.time()
    landmarks = face_recognition.face_landmarks(face, location)

    if debug:
        time_end = time.time()
        performance_stats["Landmarks"] = time_end-time_start

    return landmarks


def update_detection_list(index):
    global detection_index
    recent_detections[detection_index] = index
    detection_index = (detection_index + 1) % conf["consecutive_detections"]


def draw_eyes(landmarks, frame):
    for landmark in landmarks:
        left_eye = landmark['left_eye']
        right_eye = landmark['right_eye']
        draw_eye(frame, left_eye)
        draw_eye(frame, right_eye)


def draw_landmarks(landmarks, frame, radius=1, color=(0,255,100,255)):
    for landmark in landmarks:
        for location in landmark:
            for point in landmark[location]:
                cv2.circle(frame, point, radius, color)

def draw_eye(frame, eye_marks, width=1, fill=(0, 0, 255, 255)):
    eyelines_tuples = [(0,1), (2,3), (4,3), (0,-1), (1,-1), (2,-2)]
    for x,y in eyelines_tuples:
        cv2.line(frame, eye_marks[x], eye_marks[y], color=fill, thickness=width)

def show_recognized_face(image_frame):
    frame = image_frame.copy()
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name[0], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        #cv2.putText(frame, "Detection: " + str(round(performance_stats["Detection"], 5)), (10, 20), font, 1.0, (255, 255, 255), 1)
        #cv2.putText(frame, "Encoding: " + str(round(performance_stats["Encoding"], 5)), (10, 40), font, 1.0, (255, 255, 255), 1)
        #cv2.putText(frame, "Matching: " + str(round(performance_stats["Matching"], 5)), (10, 60), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

def show_landmarks(frame):
    landmarks_frame = frame.copy()
    draw_eyes(landmarks, landmarks_frame)
    draw_landmarks(landmarks, landmarks_frame)
    cv2.imshow("Landmarks", landmarks_frame)


signal.signal(signal.SIGINT, shutdown)

path_to_file = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

conf = json.load(open(path_to_file + '/conf.json'))

camera = VideoStream(usePiCamera=conf["use_rpi_camera"] > 0).start()
time.sleep(conf["camera_warmup_time"])

dnr = deepnet_face_recognition.DeepNetRecognizer(path_to_file + "/images/")

# Initialize some variables
face_locations = []
face_names = []
process_this_frame = True
bg_sub_model = None
performance_stats = {}


print "Init recognition"
while True:
    # Grab a single frame of video
    frame = camera.read()
    frame = imutils.resize(frame, width=500)

    if bg_sub_model is None:
        bg_sub_model = bgsub.BackgroundExtractor(frame, conf, path_to_file)

    # Only process every other frame of video to save time
    if process_this_frame:
        face_locations = get_face_locations(frame)
        face_names = dnr.recognize_face(frame, face_locations)
        landmarks = get_facial_landmarks(frame, face_locations)

        dnr.check_login()
        dnr.check_logout()

    process_this_frame = not process_this_frame
    if conf["show_video"]["landmarks"]:
        show_landmarks(frame)

    if conf["show_video"]["recognition"]:
        show_recognized_face(frame)
    else:
        print performance_stats

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


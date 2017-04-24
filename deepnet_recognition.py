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




signal.signal(signal.SIGINT, shutdown)

path_to_file = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

conf = json.load(open(path_to_file + '/conf.json'))

camera = VideoStream(usePiCamera=conf["use_rpi_camera"] > 0).start()
time.sleep(conf["camera_warmup_time"])

dnr = deepnet_face_recognition.DeepNetRecognizer(path_to_file + "/images/")
fl = face_landmarks.FaceLandmarks()

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
        fl.update_facial_landmarks(frame, face_locations)

        dnr.check_login()
        dnr.check_logout()

    process_this_frame = not process_this_frame
    if conf["show_video"]["landmarks"]:
        fl.show_landmarks(frame)

    if conf["show_video"]["recognition"]:
        dnr.show_recognized_face(frame, face_locations, face_names)
    else:
        print performance_stats

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

